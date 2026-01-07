import json
import os
import glob
import time
import random
import numpy as np
import argparse
import math
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

class AlternatingBestWorstTTTS:
    """
    Implements the Alternating Best-Worst Top-Two Thompson Sampling algorithm
    adapted for ad selection where each ad is treated as an arm.
    """

    def __init__(self, k, beta=0.5, conf_target=0.95, check_every=20, k_post_samples=2000, soft_weight=1.0):
        """
        Initializes the algorithm with a uniform Dirichlet prior for each arm (ad).

        Args:
            k (int): The number of arms (ads).
            beta (float, optional): The tuning parameter. Defaults to 0.5.
            conf_target (float, optional): Confidence threshold for stopping. Defaults to 0.95.
            check_every (int, optional): Check confidence every N steps. Defaults to 20.
            k_post_samples (int, optional): Monte Carlo samples for confidence estimation. Defaults to 2000.
            soft_weight (float, optional): Weight for soft label updates. Defaults to 1.0.
        """
        if not 0 < k:
            raise ValueError("Number of arms k must be a positive integer.")
        if not 0.0 < beta < 1.0:
            raise ValueError("Beta must be between 0 and 1.")

        self.k = k
        self.beta = beta
        self.conf_target = conf_target
        self.check_every = check_every
        self.k_post_samples = k_post_samples
        self.soft_weight = soft_weight
        # Initialize with Dirichlet(1,1,1,1,1) prior for each arm (uniform over 5 outcomes)
        self.num_outcomes = 5  # Outcomes: 1, 2, 3, 4, 5
        self.dirichlet_params = np.ones((k, self.num_outcomes))  # Shape: (k, 5)

    def _get_contenders(self, objective='best'):
        """
        Identifies the top two contenders for a given objective.

        Args:
            objective (str): Either 'best' or 'worst'.

        Returns:
            tuple: A tuple containing the primary and secondary contender arms.
        """
        # Calculate expected value for each arm based on Dirichlet posterior
        # Expected value = sum(outcome_value * probability) for outcomes 1,2,3,4,5
        outcome_values = np.array([1, 2, 3, 4, 5])
        expected_values = np.zeros(self.k)
        
        for arm in range(self.k):
            # Expected probabilities from Dirichlet posterior
            prob_vector = self.dirichlet_params[arm] / np.sum(self.dirichlet_params[arm])
            expected_values[arm] = np.dot(outcome_values, prob_vector)

        if objective == 'best':
            # Identify the arm with the highest expected value
            primary_contender = np.argmax(expected_values)
        else:  # objective == 'worst'
            # Identify the arm with the lowest expected value
            primary_contender = np.argmin(expected_values)

        secondary_contender = primary_contender
        # Continue sampling until a different arm is found
        while secondary_contender == primary_contender:
            # Re-sample from Dirichlet and recalculate expected values
            sampled_expected_values = np.zeros(self.k)
            for arm in range(self.k):
                sampled_probs = np.random.dirichlet(self.dirichlet_params[arm])
                sampled_expected_values[arm] = np.dot(outcome_values, sampled_probs)
            
            if objective == 'best':
                secondary_contender = np.argmax(sampled_expected_values)
            else:  # objective == 'worst'
                secondary_contender = np.argmin(sampled_expected_values)

        return primary_contender, secondary_contender

    def select_arm(self, timestep):
        """
        Selects an arm (ad) to pull based on the alternating best/worst objective.

        Args:
            timestep (int): The current timestep, used to determine the objective.

        Returns:
            int: The index of the arm (ad) to pull.
        """
        if timestep % 2 != 0:  # Odd timestep: find the best arm
            objective = 'best'
        else:  # Even timestep: find the worst arm
            objective = 'worst'

        primary, secondary = self._get_contenders(objective=objective)

        # Randomize between the top two contenders with probability beta
        if np.random.random() < self.beta:
            return primary
        else:
            return secondary

    def update(self, arm_pulled, outcome):
        """
        Updates the Dirichlet posterior distribution for the arm that was pulled.

        Args:
            arm_pulled (int): The index of the arm that was measured.
            outcome: either an int in {1..5} or a length-5 prob vector summing to 1.
        """
        # vector path
        if isinstance(outcome, (list, tuple, np.ndarray)):
            vec = np.asarray(outcome, dtype=float)
            if vec.shape != (self.num_outcomes,):
                raise ValueError("Outcome prob vector must have length 5.")
            s = vec.sum()
            if s <= 0 or not np.isfinite(vec).all() or (vec < 0).any():
                raise ValueError("Outcome prob vector must be nonnegative and sum>0.")
            self.dirichlet_params[arm_pulled] += self.soft_weight * vec / s  # fractional counts
            return

        # discrete path
        if outcome not in [1, 2, 3, 4, 5]:
            raise ValueError("Outcome must be 1..5.")
        self.dirichlet_params[arm_pulled, outcome - 1] += 1

    def get_estimated_best_arm(self):
        """Returns the arm with the highest expected value."""
        outcome_values = np.array([1, 2, 3, 4, 5])
        expected_values = np.zeros(self.k)
        
        for arm in range(self.k):
            prob_vector = self.dirichlet_params[arm] / np.sum(self.dirichlet_params[arm])
            expected_values[arm] = np.dot(outcome_values, prob_vector)
            
        return np.argmax(expected_values)

    def get_estimated_worst_arm(self):
        """Returns the arm with the lowest expected value."""
        outcome_values = np.array([1, 2, 3, 4, 5])
        expected_values = np.zeros(self.k)
        
        for arm in range(self.k):
            prob_vector = self.dirichlet_params[arm] / np.sum(self.dirichlet_params[arm])
            expected_values[arm] = np.dot(outcome_values, prob_vector)
            
        return np.argmin(expected_values)

    def posterior_best_prob(self):
        """Estimate posterior probability that each arm is optimal (max mean) via Monte Carlo."""
        outcome_values = np.array([1, 2, 3, 4, 5])
        
        # Draw K samples per arm from Dirichlet; compute sampled means
        sampled_means = np.zeros((self.k_post_samples, self.k))
        
        for k_sample in range(self.k_post_samples):
            for arm in range(self.k):
                sampled_probs = np.random.dirichlet(self.dirichlet_params[arm])
                sampled_means[k_sample, arm] = np.dot(outcome_values, sampled_probs)
        
        # Find winners for each sample
        winners = np.argmax(sampled_means, axis=1)
        counts = np.bincount(winners, minlength=self.k)
        probs = counts / self.k_post_samples
        
        return probs

    def posterior_worst_prob(self):
        """Estimate posterior probability that each arm is worst (min mean) via Monte Carlo."""
        outcome_values = np.array([1, 2, 3, 4, 5])
        
        # Draw K samples per arm from Dirichlet; compute sampled means
        sampled_means = np.zeros((self.k_post_samples, self.k))
        
        for k_sample in range(self.k_post_samples):
            for arm in range(self.k):
                sampled_probs = np.random.dirichlet(self.dirichlet_params[arm])
                sampled_means[k_sample, arm] = np.dot(outcome_values, sampled_probs)
        
        # Find worst arms for each sample
        worst_arms = np.argmin(sampled_means, axis=1)
        counts = np.bincount(worst_arms, minlength=self.k)
        probs = counts / self.k_post_samples
        
        return probs

    def check_confidence(self, objective):
        """Check if confidence target is reached for given objective."""
        if objective == 'best':
            probs = self.posterior_best_prob()
            est_best = np.argmax(probs)
            return probs[est_best] >= self.conf_target, est_best
        else:  # objective == 'worst'
            probs = self.posterior_worst_prob()
            est_worst = np.argmax(probs)
            return probs[est_worst] >= self.conf_target, est_worst

def pil_to_part(img: Image.Image, mime="image/png"):
    """Convert PIL Image to Part for maximum compatibility."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)

def score_with_logprobs(ad_image, client, model_id):
    """
    Ask Gemini for effectiveness score with logprobs, returning comprehensive results.
    """
    
    # Direct ad effectiveness evaluation prompt (no persona involved)
    score_only_prompt = """TASK:
Evaluate this advertisement's effectiveness for engaging users and driving clicks on social media platforms.

Return only one item from ["1","2","3","4","5"] for ad effectiveness.

Effectiveness Scale Definition:
1: Extremely Ineffective. Users would actively ignore or be annoyed by this ad.
2: Ineffective. Users would likely scroll past without a second thought.
3: Mediocre. Mixed reaction - some users might engage, others won't.
4: Effective. The ad is compelling and has a good chance of engaging users.
5: Extremely Effective. The ad is highly engaging and would likely drive significant clicks.

Consider factors like:
- Visual appeal and composition
- Emotional impact and messaging
- Brand clarity and memorability
- Call-to-action effectiveness

No explanation. Just the score."""

    resp = client.models.generate_content(
        model=model_id,
        contents=[pil_to_part(ad_image), score_only_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": ["1", "2", "3", "4", "5"]},
            response_logprobs=True,
            logprobs=5,             # must be in range [0,5] when response_logprobs=True
            temperature=0,
            max_output_tokens=1,
        ),
    )
    lp = resp.candidates[0].logprobs_result
    probs = {str(i): 0.0 for i in range(1, 6)}

    # read top candidates for the first (and only) token
    for cand in lp.top_candidates[0].candidates:
        t = cand.token.strip()
        if t in probs:
            probs[t] = math.exp(cand.log_probability)

    # ensure the chosen token is included
    ch = lp.chosen_candidates[0]
    t = ch.token.strip()
    if t in probs and probs[t] == 0.0:
        probs[t] = math.exp(ch.log_probability)

    # renormalize
    total = sum(probs.values())
    for k in probs:
        probs[k] = probs[k] / total if total > 0 else 0.2

    expected = sum(int(k) * v for k, v in probs.items())
    
    # Return comprehensive result with just logprobs and scores
    # Use 1-5 keys for score_probs to match user expectation
    result = {
        "effectiveness_score": expected,
        "score_probs": {str(i+1): probs[str(i+1)] for i in range(5)}  # 1-5 keys for scores
    }
    
    return result

def predict_ad_effectiveness_with_gemini(ad_image, client, model_id):
    """Use Gemini to predict ad effectiveness with direct evaluation and logprobs."""
    
    try:
        # Use score_with_logprobs for all evaluation
        result = score_with_logprobs(ad_image, client, model_id)
        
        # Validate and process the effectiveness score
        effectiveness_score = float(result["effectiveness_score"])
        effectiveness_score = max(1.0, min(5.0, effectiveness_score))  # Ensure bounds 1-5
        result["effectiveness_score"] = effectiveness_score
        
        return result
        
    except Exception as e:
        print(f"Error predicting ad effectiveness: {e}")
        return {
            "effectiveness_score": 3.0,
            "score_probs": {str(i+1): 0.2 for i in range(5)}
        }

def load_ad_images(campaign_dir, scenario_name):
    """Load the actual ad images from the campaign directory."""
    ad_images = {}
    
    # Find all ad images in the directory that match the pattern
    pattern = os.path.join(campaign_dir, f"{scenario_name.lower()}_ad_*.png")
    ad_files = glob.glob(pattern)
    ad_files.sort()  # Ensure consistent ordering
    
    for i, image_path in enumerate(ad_files):
        try:
            ad_images[i] = Image.open(image_path)  # 0-indexed for arms
            print(f"Loaded ad image {i+1} as arm {i}: {image_path}")
        except Exception as e:
            print(f"Error loading ad image {image_path}: {e}")
            ad_images[i] = None
    
    return ad_images

def run_dttts_simulation(ad_images, client, model_id, scenario_name, num_timesteps=500, conf_target=0.95, check_every=20, k_post_samples=2000):
    """Run the Double TTTS simulation for ad selection."""
    
    num_ads = len(ad_images)
    dttts = AlternatingBestWorstTTTS(k=num_ads, beta=0.5, conf_target=conf_target, check_every=check_every, k_post_samples=k_post_samples)
    
    # Track results
    results = {
        "simulation_metadata": {
            "scenario_name": scenario_name,
            "num_timesteps": num_timesteps,
            "num_ads": num_ads,
            "algorithm": "Double TTTS (Alternating Best-Worst) with Gemini Logprobs",
            "evaluation_method": "direct_gemini_evaluation"
        },
        "timestep_results": [],
        "final_statistics": {}
    }
    
    arm_pull_counts = np.zeros(num_ads, dtype=int)
    total_rewards = np.zeros(num_ads)
    
    # Track confidence stopping
    best_confidence_reached = False
    worst_confidence_reached = False
    best_identified_at = None
    worst_identified_at = None
    identified_best_arm = None
    identified_worst_arm = None
    
    print(f"Starting Double TTTS simulation with {num_timesteps} timesteps")
    print(f"Available ads (arms): {num_ads}")
    print("-" * 50)
    
    for timestep in range(1, num_timesteps + 1):
        # Step 1: Use Double TTTS to select an ad (arm)
        selected_ad = dttts.select_arm(timestep)
        
        # Step 2: Get the ad image for the selected ad
        if ad_images[selected_ad] is None:
            print(f"Warning: Ad {selected_ad+1} image not available, skipping timestep {timestep}")
            continue
            
        # Step 3: Get prediction for this ad using Gemini with logprobs
        prediction = predict_ad_effectiveness_with_gemini(
            ad_images[selected_ad], 
            client,
            model_id
        )
        
        # Step 4: Get effectiveness score (this is our multinomial outcome 1-5)
        effectiveness_score = prediction.get("effectiveness_score", 3)
        
        # Step 5: Update the TTTS algorithm
        if "score_probs" in prediction:
            # Convert dictionary to array in 1-5 order
            prob_array = [prediction["score_probs"][str(i+1)] for i in range(5)]
            dttts.update(selected_ad, np.array(prob_array, dtype=float))
        else:
            dttts.update(selected_ad, int(round(effectiveness_score)))
        
        # Step 6: Track results
        arm_pull_counts[selected_ad] += 1
        total_rewards[selected_ad] += effectiveness_score
        
        objective = "BEST" if timestep % 2 != 0 else "WORST"
        
        timestep_result = {
            "timestep": timestep,
            "objective": objective,
            "selected_ad": selected_ad + 1,  # 1-indexed for display
            "effectiveness_score": effectiveness_score,
            "score_probs": prediction.get("score_probs", None)
        }
        
        results["timestep_results"].append(timestep_result)
        
        # Print progress
        if timestep <= 10 or timestep % 50 == 0:
            print(f"Step {timestep:4d} (Find {objective:5}): Ad {selected_ad+1}, Score {effectiveness_score}")
        
        # Check confidence every check_every steps
        if timestep % dttts.check_every == 0:
            # Check best arm confidence
            if not best_confidence_reached:
                best_confident, est_best = dttts.check_confidence('best')
                if best_confident:
                    best_confidence_reached = True
                    best_identified_at = timestep
                    identified_best_arm = est_best
                    print(f"✓ Best arm confidence reached at timestep {timestep}: arm {est_best + 1}")
            
            # Check worst arm confidence
            if not worst_confidence_reached:
                worst_confident, est_worst = dttts.check_confidence('worst')
                if worst_confident:
                    worst_confidence_reached = True
                    worst_identified_at = timestep
                    identified_worst_arm = est_worst
                    print(f"✓ Worst arm confidence reached at timestep {timestep}: arm {est_worst + 1}")
            
            # Stop if both objectives reached confidence
            if best_confidence_reached and worst_confidence_reached:
                print(f"Both best and worst arm confidence reached. Stopping at timestep {timestep}")
                break
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Calculate final statistics
    estimated_best_ad = dttts.get_estimated_best_arm()
    estimated_worst_ad = dttts.get_estimated_worst_arm()
    
    # Calculate posterior means (expected values) for each arm
    outcome_values = np.array([1, 2, 3, 4, 5])
    posterior_means = np.zeros(num_ads)
    for arm in range(num_ads):
        prob_vector = dttts.dirichlet_params[arm] / np.sum(dttts.dirichlet_params[arm])
        posterior_means[arm] = np.dot(outcome_values, prob_vector)
    
    # Calculate average effectiveness scores
    avg_effectiveness = np.divide(total_rewards, arm_pull_counts, out=np.zeros_like(total_rewards), where=arm_pull_counts!=0)
    
    results["final_statistics"] = {
        "arm_pull_counts": [int(x) for x in arm_pull_counts],
        "total_rewards": [float(x) for x in total_rewards],
        "avg_effectiveness_scores": [float(x) for x in avg_effectiveness],
        "posterior_means": [float(x) for x in posterior_means],
        "estimated_best_ad": int(estimated_best_ad + 1),  # 1-indexed
        "estimated_worst_ad": int(estimated_worst_ad + 1),  # 1-indexed
        "final_dirichlet_params": [[float(x) for x in row] for row in dttts.dirichlet_params],
        "confidence_results": {
            "conf_target": conf_target,
            "best_confidence_reached": best_confidence_reached,
            "worst_confidence_reached": worst_confidence_reached,
            "best_identified_at": int(best_identified_at) if best_identified_at is not None else None,
            "worst_identified_at": int(worst_identified_at) if worst_identified_at is not None else None,
            "identified_best_arm": int(identified_best_arm + 1) if identified_best_arm is not None else None,
            "identified_worst_arm": int(identified_worst_arm + 1) if identified_worst_arm is not None else None,
            "actual_timesteps": int(timestep)
        }
    }
    
    return results

def save_results(results, output_dir, scenario_name):
    """Save DTTTS simulation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{scenario_name.lower()}_dttts_gemini_results.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved DTTTS Gemini results to: {output_file}")
    
    # Print summary
    stats = results["final_statistics"]
    print("\n" + "="*60)
    print("DOUBLE TTTS WITH GEMINI LOGPROBS SIMULATION SUMMARY")
    print("="*60)
    print(f"Evaluation method: {results['simulation_metadata']['evaluation_method']}")
    print(f"Total timesteps: {stats['confidence_results']['actual_timesteps']}")
    print(f"Pulls per ad: {stats['arm_pull_counts']}")
    print(f"Avg effectiveness per ad: {[f'{rate:.3f}' for rate in stats['avg_effectiveness_scores']]}")
    print(f"Posterior means: {[f'{mean:.3f}' for mean in stats['posterior_means']]}")
    print(f"Estimated best ad: {stats['estimated_best_ad']}")
    print(f"Estimated worst ad: {stats['estimated_worst_ad']}")

def main():
    """Main execution function for DTTTS ad selection with Gemini evaluation."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DTTTS Ad Selection Algorithm with Gemini Evaluation')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced timesteps')
    args = parser.parse_args()
    
    # Load environment variables and validate Vertex AI requirements
    load_dotenv()
    project = os.getenv("GOOGLE_PROJECT_ID")
    location = os.getenv("GOOGLE_LOCATION")
    
    if not all([project, location]):
        print("❌ Missing required environment variables:")
        print("   Please set GOOGLE_PROJECT_ID and GOOGLE_LOCATION in .env file")
        return
    
    # Use Vertex AI client
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location
    )
    print(f"Using Vertex AI client: project={project}, location={location}")
        
    model_id = "gemini-2.5-flash"
    
    # Configuration
    SCENARIO_NAME = "terraexplorer"
    NUM_TIMESTEPS = 10000  # Number of Double TTTS timesteps
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    campaign_dir = os.path.join(base_dir, "campaign_output", f"{SCENARIO_NAME.lower()}_campaign_output")
    output_dir = os.path.join(base_dir, "campaign_output", f"{SCENARIO_NAME.lower()}_campaign_output")
    
    # Test mode (command line flag or environment variable)
    test_mode = args.test or os.getenv("TEST_MODE", "false").lower() == "true"
    if test_mode:
        print("Running in TEST MODE - using reduced timesteps")
        NUM_TIMESTEPS = 5
        print(f"Test mode: {NUM_TIMESTEPS} timesteps")
    
    print("Loading ad images...")
    ad_images = load_ad_images(campaign_dir, SCENARIO_NAME)
    
    print("Running Double TTTS simulation with Gemini logprobs evaluation...")
    results = run_dttts_simulation(ad_images, client, model_id, SCENARIO_NAME, NUM_TIMESTEPS)
    
    print("Saving results...")
    save_results(results, output_dir, SCENARIO_NAME)
    
    print("Double TTTS with Gemini evaluation simulation complete!")

if __name__ == "__main__":
    main()