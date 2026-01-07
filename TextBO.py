import json
import os
import glob
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import google.genai as genai
from google.genai import types
import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from io import BytesIO

class TextBOOptimizer:
    """
    Text-Based Optimization using Best-of-N Bayesian Optimization (TextBO)

    Implements the TextBO algorithm from the paper:
    "Bayesian Optimization in Language space: An Eval-Efficient AI Self-improvement Framework"

    Unlike ad_gen.py which uses creative briefs, TextBO starts with an initial prompt
    and optimizes it through gradient-based exploration with persona feedback.
    """
    
    def __init__(
        self,
        api_key: str,
        project_id: str,
        location: str,
        scenario_name: str,
        initial_prompt: str,
        output_dir: str,
        persona_dir: str = None
    ):
        """Initialize T-BoN optimizer.
        
        Args:
            api_key: Google Gemini API key
            project_id: Google Cloud Project ID
            location: Google Cloud Project Location
            scenario_name: Campaign scenario name
            initial_prompt: Starting prompt to optimize
            output_dir: Directory to save results
            persona_dir: Directory containing persona files for evaluation
        """
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.scenario_name = scenario_name
        self.initial_prompt = initial_prompt
        self.output_dir = output_dir
        self.persona_dir = persona_dir or os.path.join("text_simulation", "text_personas")
        
        # Configure APIs
        self.vertex_client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )
        self.image_model = "imagen-4.0-ultra-generate-preview-06-06"
        self.text_model_id = "gemini-2.5-flash"
        self.critic_model_id = "gemini-2.5-flash"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimization history: List of (prompt, generated_content, evaluation_score)
        self.history = []
        
        # Reproducibility: Base seed for deterministic experiments
        self.base_seed = random.randint(0, 1000000)
        print(f"Base seed for reproducibility: {self.base_seed}")

        # Load persona data for evaluation
        self.train_personas, self.test_personas = self._load_persona_data()

    def _load_persona_data(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load persona data from text files and split into training and test sets."""
        if not os.path.exists(self.persona_dir):
            print(f"Warning: Persona directory {self.persona_dir} not found")
            return {}, {}

        # Get and sort all persona files to ensure consistent ordering
        persona_files = sorted(glob.glob(os.path.join(self.persona_dir, "pid_*_mega_persona.txt")))
        
        if not persona_files:
            print("Warning: No persona files found.")
            return {}, {}

        # Split files into training and test sets based on index
        split_index = int(len(persona_files) * 0.8)
        train_files = persona_files[:split_index]
        test_files = persona_files[split_index:]

        def load_personas_from_files(files: List[str]) -> Dict[str, str]:
            loaded_personas = {}
            for file_path in files:
                persona_id = os.path.basename(file_path).split('_')[1]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        loaded_personas[persona_id] = content
                except Exception as e:
                    print(f"Error loading persona {persona_id}: {e}")
            return loaded_personas

        print("Loading training personas...")
        train_personas = load_personas_from_files(train_files)
        print("Loading testing personas...")
        test_personas = load_personas_from_files(test_files)

        print(f"Loaded {len(train_personas)} personas for training.")
        print(f"Loaded {len(test_personas)} personas for testing.")
        
        return train_personas, test_personas
    
    def analyze_performance_patterns(self, history: List[Tuple]) -> str:
        """Analyze historical performance patterns to extract visual insights.
        
        Args:
            history: List of (prompt, image_path, score) tuples from previous iterations
            
        Returns:
            Analysis of visual patterns that correlate with performance
        """
        if not history:
            return "No historical data available for pattern analysis."
        
        # Use bottom 5 and top 5 performing iterations for focused visual analysis
        if len(history) < 10:
            # If we have fewer than 10 iterations, use lower half as worst and upper half as best
            sorted_history = sorted(history, key=lambda x: x[2])  # Sort by score (index 2)
            mid_point = len(sorted_history) // 2
            bottom_half = sorted_history[:mid_point] if mid_point > 0 else []
            top_half = sorted_history[mid_point:] if len(sorted_history) > mid_point else sorted_history
            recent_history = bottom_half + top_half
        else:
            # Sort by score and take bottom 5 and top 5
            sorted_history = sorted(history, key=lambda x: x[2])  # Sort by score (index 2)
            bottom_5 = sorted_history[:5]   # Worst performing
            top_5 = sorted_history[-5:]     # Best performing
            recent_history = bottom_5 + top_5
        
        # Prepare multimodal prompt content for analysis
        content_parts = []
        
        # Add analysis instruction
        analysis_instruction = """You are an expert at analyzing visual patterns in advertising performance.

VISUAL ANALYSIS TASK:
I will show you images from the lowest-scoring and highest-scoring ad iterations. 
Your task is to identify specific visual patterns that distinguish effective from ineffective ads.

VISUAL EXAMPLES - BEST VS WORST PERFORMING:
"""
        content_parts.append(analysis_instruction)
        
        # Add each historical iteration with its image and analysis
        for i, (prompt, image_path, score) in enumerate(recent_history):
            # Create fine-grained performance labels with individual rankings
            if len(history) < 10:
                # For fewer than 10 iterations, use lower/upper half logic
                sorted_history = sorted(history, key=lambda x: x[2])
                mid_point = len(sorted_history) // 2
                if i < mid_point:
                    overall_rank = sorted_history.index((prompt, image_path, score)) + 1
                    performance_label = f"RANK #{overall_rank}/{len(history)} - LOWER HALF"
                else:
                    overall_rank = sorted_history.index((prompt, image_path, score)) + 1
                    performance_label = f"RANK #{overall_rank}/{len(history)} - UPPER HALF"
            else:
                # For 10+ iterations, use bottom 5 and top 5 with fine-grained ranking
                sorted_history = sorted(history, key=lambda x: x[2])
                if i < 5:
                    overall_rank = sorted_history.index((prompt, image_path, score)) + 1
                    rank_in_bottom = i + 1
                    performance_label = f"RANK #{overall_rank}/{len(history)} - {rank_in_bottom} WORST PERFORMING"
                else:
                    overall_rank = sorted_history.index((prompt, image_path, score)) + 1
                    rank_in_top = i - 4
                    performance_label = f"RANK #{overall_rank}/{len(history)} - {rank_in_top} BEST PERFORMING"
            
            iteration_text = f"""
{performance_label} (Score: {score:.3f}/5.0):
Prompt excerpt: {prompt[:200]}...

"""
            content_parts.append(iteration_text)
            
            # Load and add the actual image if path exists
            if isinstance(image_path, str) and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        content_parts.append(img.copy())  # .copy() to ensure image persists after file closes
                except Exception as e:
                    print(f"      Warning: Could not load image {image_path}: {e}")
                    content_parts.append("[Image could not be loaded]")
            else:
                content_parts.append("[Image not available for this iteration]")
        
        # Add analysis task instruction
        task_instruction = """

Based on your visual analysis, identify patterns that correlate with higher effectiveness scores:
1. Visual composition and framing differences
2. Lighting conditions and mood variations
3. Color palettes and visual tone patterns
4. Subject positioning and action effectiveness
5. Brand integration approaches
6. Environmental and atmospheric elements

RESPONSE FORMAT:
Provide a structured analysis of visual patterns observed, focusing on what distinguishes high-performing from low-performing ads.
"""

        content_parts.append(task_instruction)
        
        try:
            generation_config = types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for consistent analysis
                top_p=0.9
            )
            response = self.vertex_client.models.generate_content(
                model=self.critic_model_id,
                contents=content_parts,
                config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error analyzing performance patterns: {e}")
            return "Unable to analyze performance patterns due to error."

    def generate_improvement_suggestions(self, current_prompt: str, performance_analysis: str = None) -> str:
        """Generate specific improvement suggestions for the current prompt.
        
        Args:
            current_prompt: Current prompt to improve
            performance_analysis: Optional analysis of performance patterns from history
            
        Returns:
            Specific, actionable improvement suggestions
        """
        if performance_analysis:
            improvement_prompt = f"""You are an expert at optimizing image generation prompts for advertising effectiveness.

CURRENT PROMPT TO IMPROVE:
{current_prompt}

PERFORMANCE ANALYSIS FROM PREVIOUS ITERATIONS:
{performance_analysis}

TASK: Based on the performance analysis above, generate specific, actionable improvements to make the current prompt more effective.

Focus on implementing the successful visual patterns identified in the analysis while avoiding the ineffective elements.

Provide 3-5 specific, implementable suggestions for improvement. (It can be addition of a new prompt part, deletion of existing promt part, or rewriting a prompt part).
Each suggestion should reference insights from the performance analysis.

RESPONSE FORMAT:
1. [Specific improvement suggestion based on performance analysis]
2. [Specific improvement suggestion based on performance analysis]
3. [Specific improvement suggestion based on performance analysis]
[etc.]

Be concrete and actionable. Focus on changes that will meaningfully improve ad effectiveness based on the analysis."""
        else:
            improvement_prompt = f"""You are an expert at optimizing image generation prompts for advertising effectiveness.

CURRENT PROMPT TO IMPROVE:
{current_prompt}

TASK: Generate specific, actionable improvements to make this ad prompt more effective.
Focus on elements that will increase engagement and appeal to the target audience.

Consider improvements in:
1. Visual composition and framing
2. Emotional appeal and messaging
3. Color palette and lighting
4. Subject positioning and action
5. Brand integration and logo placement
6. Text overlay space and readability

Provide 3-5 specific, implementable suggestions for improvement.
Each suggestion should be concrete and actionable.

RESPONSE FORMAT:
1. [Specific improvement suggestion]
2. [Specific improvement suggestion] 
3. [Specific improvement suggestion]
[etc.]

Be concise but specific. Focus on changes that will meaningfully improve ad effectiveness."""

        try:
            generation_config = types.GenerateContentConfig(
                temperature=0.8,  # Add randomness for diverse gradient sampling
                top_p=0.9       # Nucleus sampling for controlled diversity
            )
            response = self.vertex_client.models.generate_content(
                model=self.critic_model_id,
                contents=improvement_prompt,
                config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating improvement suggestions: {e}")
            return "Improve visual appeal and emotional connection with target audience."

    def generate_textual_gradient(self, current_prompt: str, history: List[Tuple], 
                                 cached_analysis: str = None) -> str:
        """Generate textual gradient (suggested improvements) for current prompt.
        
        Args:
            current_prompt: Current prompt to improve
            history: List of (prompt, image_path, score) tuples from previous iterations
            cached_analysis: Pre-computed performance analysis to avoid redundant computation
            
        Returns:
            Textual gradient (improvement suggestions)
        """
        if not history:
            # No history, use basic gradient generation
            return self.generate_improvement_suggestions(current_prompt)
        
        # Use cached analysis if provided, otherwise compute it
        if cached_analysis is not None:
            performance_analysis = cached_analysis
        else:
            # Analyze performance patterns from history (expensive operation)
            performance_analysis = self.analyze_performance_patterns(history)
        
        # Generate targeted improvements based on analysis
        return self.generate_improvement_suggestions(current_prompt, performance_analysis)
    
    
    def apply_gradient(self, current_prompt: str, gradient: str, max_retries: int = 3) -> str:
        """Apply textual gradient to create improved prompt.
        
        Args:
            current_prompt: Original prompt
            gradient: Improvement suggestions
            max_retries: Maximum number of retry attempts
            
        Returns:
            Updated prompt incorporating the improvements
        """
        apply_prompt = f"""You are an expert at revising image generation prompts based on improvement suggestions.

ORIGINAL PROMPT:
{current_prompt}

IMPROVEMENT SUGGESTIONS:
{gradient}

TASK: Rewrite the prompt incorporating the improvement suggestions while maintaining the core message and structure.

REQUIREMENTS:
- Keep the prompt structure and format similar to the original
- Integrate the improvement suggestions naturally
- Maintain coherence and readability
- Ensure the prompt is optimized for image generation
- Keep the prompt length reasonable (not too long)

Return ONLY the revised prompt, no explanations or additional text."""

        for attempt in range(max_retries):
            try:
                # Use low temperature for stable, consistent rewrites
                generation_config = types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for stable rewrites
                    top_p=0.95      # Slightly constrained sampling
                )
                response = self.vertex_client.models.generate_content(
                    model=self.text_model_id,
                    contents=apply_prompt,
                    config=generation_config
                )
                return response.text.strip()
            except Exception as e:
                print(f"Error applying gradient (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    return current_prompt  # Return original if all retries fail
    
    def generate_image(self, prompt: str, iteration: int, gradient_step: int, candidate: int, trajectory: int = 0, max_retries: int = 3) -> Optional[str]:
        """Generate image using Imagen 4 with retry logic.
        
        Args:
            prompt: Text prompt for image generation
            iteration: Current iteration number
            gradient_step: Current gradient step within iteration
            candidate: Candidate number within gradient step
            trajectory: Trajectory number (0 for sequential mode)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Path to generated image file, or None if failed after all retries
        """
        filename = f"textbo_{self.scenario_name.lower()}_traj{trajectory:02d}_iter{iteration:02d}_grad{gradient_step:02d}_cand{candidate:02d}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"  Generating image for candidate {candidate}")
                else:
                    print(f"  Retry {attempt}/{max_retries-1} for candidate {candidate}")
                
                response = self.vertex_client.models.generate_images(
                    model=self.image_model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                        person_generation="ALLOW_ADULT",
                    ),
                )
                
                # Check if response contains valid image data and save using public API
                if (response and 
                    getattr(response, "generated_images", None) and 
                    len(response.generated_images) > 0 and
                    hasattr(response.generated_images[0], "image") and
                    response.generated_images[0].image):
                    
                    try:
                        # Use public API methods instead of private _pil_image field
                        image_obj = response.generated_images[0].image
                        
                        # Try different public methods to get image bytes
                        image_bytes = None
                        if hasattr(image_obj, 'get_bytes') and callable(image_obj.get_bytes):
                            image_bytes = image_obj.get_bytes()
                        elif hasattr(image_obj, 'bytes') and image_obj.bytes:
                            image_bytes = image_obj.bytes
                        elif hasattr(image_obj, '_pil_image') and image_obj._pil_image:
                            # Fallback to private field if public methods unavailable
                            image_obj._pil_image.save(filepath)
                            print(f"    Saved (fallback): {filename}")
                            return filepath
                        
                        if image_bytes:
                            # Convert bytes to PIL Image and save
                            pil_image = Image.open(BytesIO(image_bytes))
                            pil_image.save(filepath)
                            print(f"    Saved: {filename}")
                            return filepath
                        else:
                            error_msg = "Could not extract image bytes using public API"
                    
                    except Exception as save_error:
                        error_msg = f"Error saving image: {save_error}"
                else:
                    error_msg = "No valid image data returned"
                    if response and hasattr(response, "generated_images"):
                        if not response.generated_images:
                            error_msg = "Empty generated_images list"
                        elif not hasattr(response.generated_images[0], "image"):
                            error_msg = "No image attribute in response"
                        elif not response.generated_images[0].image:
                            error_msg = "Image attribute is None"
                    
                    print(f"    Attempt {attempt + 1}: {error_msg}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait 2^attempt seconds
                        wait_time = 2 ** attempt
                        print(f"    Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                print(f"    Attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 2^attempt seconds  
                    wait_time = 2 ** attempt
                    print(f"    Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                continue
        
        print(f"    Failed to generate image for candidate {candidate} after {max_retries} attempts")
        return None
    
    def _single_comparison(self, image1: Image.Image, image2: Image.Image) -> int:
        """Single head-to-head comparison between two images.
        
        Args:
            image1: First image for comparison
            image2: Second image for comparison
            
        Returns:
            0 if image1 wins, 1 if image2 wins
        """
        import re
        
        comparison_prompt = """You are evaluating two advertisement images for mobile Instagram ads. 
Which image would be more effective at engaging users and driving clicks?

CRITICAL: Return exactly 1 or 2 with no other text.
- Return 1 if the first image is more effective
- Return 2 if the second image is more effective

Your response must be exactly one character: either 1 or 2."""

        try:
            contents = [image1, image2, comparison_prompt]
            
            # Try up to 2 attempts for valid response
            for attempt in range(2):
                response = self.vertex_client.models.generate_content(
                    model=self.critic_model_id,
                    contents=contents
                )
                response_text = response.text.strip()
                
                # Strict regex matching for exactly "1" or "2"
                match = re.match(r'^(1|2)$', response_text)
                if match:
                    result = int(match.group(1))
                    return 0 if result == 1 else 1
                
                if attempt == 0:
                    # First attempt failed, modify prompt for retry
                    comparison_prompt = """PREVIOUS RESPONSE WAS INVALID. 

You are evaluating two advertisement images for mobile Instagram ads.
Which image would be more effective at engaging users and driving clicks?

RESPOND WITH ONLY THE NUMBER 1 OR 2. NOTHING ELSE.
1 = first image is better
2 = second image is better

Your response:"""
                    contents = [image1, image2, comparison_prompt]
            
            # Both attempts failed, fall back to random
            print(f"        Invalid response after 2 attempts: '{response_text}'. Using random fallback.")
            return random.randint(0, 1)
                
        except Exception as e:
            print(f"        Error in single comparison: {e}")
            return random.randint(0, 1)  # Random fallback
    
    def _pairwise_compare(self, candidate1: Tuple[str, str, Optional[str]], 
                         candidate2: Tuple[str, str, Optional[str]], K: int = 3) -> int:
        """Compare two candidates using majority voting across K evaluations.
        
        Args:
            candidate1: (prompt, image_path, image_path) tuple
            candidate2: (prompt, image_path, image_path) tuple
            K: Number of comparisons to run (default=3)
            
        Returns:
            0 if candidate1 wins majority, 1 if candidate2 wins majority
        """
        prompt1, _, image_path1 = candidate1
        prompt2, _, image_path2 = candidate2
        
        # Check image paths exist
        if not image_path1 or not os.path.exists(image_path1):
            return 1  # candidate2 wins if candidate1 has no image
        if not image_path2 or not os.path.exists(image_path2):
            return 0  # candidate1 wins if candidate2 has no image
        
        # Run K comparisons and count votes
        votes_candidate1 = 0
        votes_candidate2 = 0
        
        for i in range(K):
            if i > 0:
                time.sleep(0.5)  # Rate limiting between comparisons
            
            try:
                # Open images with context managers for each comparison
                with Image.open(image_path1) as image1, Image.open(image_path2) as image2:
                    winner = self._single_comparison(image1, image2)
                    if winner == 0:
                        votes_candidate1 += 1
                    else:
                        votes_candidate2 += 1
            except Exception as e:
                print(f"        Error loading images for comparison {i+1}: {e}")
                # Random vote on error to avoid bias
                if random.randint(0, 1) == 0:
                    votes_candidate1 += 1
                else:
                    votes_candidate2 += 1
        
        return 0 if votes_candidate1 > votes_candidate2 else 1

    def tournament_select(self, candidates: List[Tuple[str, str, Optional[str]]], tournament_seed: int = None) -> Tuple[str, str, Optional[str]]:
        """Select best candidate using knockout tournament with pairwise comparisons.
        
        Args:
            candidates: List of (prompt, image_path, image_path) tuples
            tournament_seed: Seed for reproducible tournament bracket generation
            
        Returns:
            Best (prompt, image_path, image_path) tuple
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # Filter out candidates without valid images
        valid_candidates = []
        for candidate in candidates:
            prompt, _, image_path = candidate
            if image_path and os.path.exists(image_path):
                valid_candidates.append(candidate)
        
        if len(valid_candidates) == 0:
            return candidates[0]  # Fallback to first if no valid images
        elif len(valid_candidates) == 1:
            return valid_candidates[0]
        
        # Set up local RNG for reproducible tournament bracket generation
        local_rng = random.Random(tournament_seed) if tournament_seed is not None else random.Random()
        if tournament_seed is not None:
            print(f"        Using seed {tournament_seed} for tournament bracket")
        
        # Shuffle candidates to randomize initial bracket arrangement
        current_round = valid_candidates.copy()
        local_rng.shuffle(current_round)
        round_num = 1
        
        print(f"        Starting tournament with {len(current_round)} candidates")
        
        while len(current_round) > 1:
            next_round = []
            pairs_in_round = len(current_round) // 2
            
            print(f"        Round {round_num}: {len(current_round)} candidates → {pairs_in_round} matches")
            
            # Process pairs
            for i in range(0, len(current_round) - 1, 2):
                candidate1 = current_round[i]
                candidate2 = current_round[i + 1]
                
                print(f"          Match {i//2 + 1}: Comparing candidates...")
                winner_idx = self._pairwise_compare(candidate1, candidate2)
                
                if winner_idx == 0:
                    next_round.append(candidate1)
                    print(f"            Winner: Candidate {i + 1}")
                else:
                    next_round.append(candidate2)
                    print(f"            Winner: Candidate {i + 2}")
            
            # Handle odd number of candidates (bye)
            if len(current_round) % 2 == 1:
                bye_candidate = current_round[-1]
                next_round.append(bye_candidate)
                print(f"          Bye: Last candidate advances automatically")
            
            current_round = next_round
            round_num += 1
        
        winner = current_round[0]
        print(f"        Tournament winner selected!")
        return winner
    
    def evaluate_candidate(self, image_path: str, prompt: str, num_personas: int = 400, iteration_seed: int = None, use_test_set: bool = False, return_individual_scores: bool = False) -> tuple:
        """Evaluate candidate using persona feedback simulation.

        Args:
            image_path: Path to generated image
            prompt: Prompt used to generate the image
            num_personas: Number of personas to sample for evaluation (only for training set)
            iteration_seed: Seed for reproducible persona sampling
            use_test_set: If True, use the entire test set for evaluation. Otherwise, sample from the training set.
            return_individual_scores: If True, return (avg_score, individual_scores_dict). Otherwise, return just avg_score.

        Returns:
            If return_individual_scores=False: Average effectiveness score (1-5 scale)
            If return_individual_scores=True: Tuple of (avg_score, dict of persona_id -> effectiveness_score)
        """
        if not image_path or not os.path.exists(image_path):
            print(f"    Warning: Image not found for evaluation: {image_path}")
            if return_individual_scores:
                return 3.0, {}  # Default mediocre score with no individual scores
            else:
                return 3.0  # Default mediocre score
        
        try:
            # Load image
            with Image.open(image_path) as image:
                if use_test_set:
                    # Use test set for evaluation, but respect num_personas limit
                    personas_to_evaluate = self.test_personas
                    if len(personas_to_evaluate) == 0:
                        print("    Warning: No test personas available for evaluation.")
                        return 3.0
                    
                    # Set seed for reproducible persona sampling from test set
                    rng = random.Random(iteration_seed + 2000) if iteration_seed is not None else random.Random()
                    if iteration_seed is not None:
                        print(f"    Using seed {iteration_seed + 2000} for test persona sampling")
                    
                    # Sample from test set respecting num_personas limit
                    sampled_personas = rng.sample(list(personas_to_evaluate.items()), 
                                                min(num_personas, len(personas_to_evaluate)))
                    print(f"    Evaluating with {len(sampled_personas)} sampled personas from the test set...")
                else:
                    # Sample from the training set for optimization
                    personas_to_evaluate = self.train_personas
                    if len(personas_to_evaluate) == 0:
                        print("    Warning: No training personas available for evaluation.")
                        return 3.0
                    
                    # Set seed for reproducible persona sampling
                    rng = random.Random(iteration_seed + 1000) if iteration_seed is not None else random.Random()
                    if iteration_seed is not None:
                        print(f"    Using seed {iteration_seed + 1000} for persona sampling")
                    
                    sampled_personas = rng.sample(list(personas_to_evaluate.items()), 
                                                min(num_personas, len(personas_to_evaluate)))
                    print(f"    Evaluating with {len(sampled_personas)} sampled personas from the training set...")
                
                # Store probability distributions for soft weighting - always use distributions
                all_score_probs = []
                individual_scores = {}  # Store individual persona scores

                for i, (persona_id, persona_text) in enumerate(sampled_personas):
                    if i % 10 == 0:
                        print(f"      Evaluating persona {i+1}/{len(sampled_personas)}")

                    try:
                        result = self._predict_persona_effectiveness_with_logprobs(
                            persona_text, persona_id, image, prompt
                        )
                        # Always use probability distributions - no discrete fallback
                        if "score_probs" in result:
                            # Convert dictionary to array in 1-5 order for soft weighting
                            prob_array = [result["score_probs"][str(i+1)] for i in range(5)]
                            all_score_probs.append(np.array(prob_array, dtype=float))
                            # Store individual effectiveness score
                            individual_scores[persona_id] = result["effectiveness_score"]
                        else:
                            # If no probs available, use uniform distribution instead of discrete
                            print(f"      Warning: No score_probs for persona {persona_id}, using uniform distribution")
                            all_score_probs.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float))
                            individual_scores[persona_id] = 3.0
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        print(f"      Error evaluating persona {persona_id}: {e}")
                        # Always use uniform distribution on error - never discrete
                        all_score_probs.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float))
                        individual_scores[persona_id] = 3.0

                # Always calculate using soft weights - no discrete fallback
                # Aggregate all probability distributions
                aggregated_probs = np.mean(all_score_probs, axis=0)
                # Calculate expected value using outcome values 1,2,3,4,5
                outcome_values = np.array([1, 2, 3, 4, 5])
                avg_score = np.dot(outcome_values, aggregated_probs)

                print(f"    Average effectiveness score: {avg_score:.3f}")

                if return_individual_scores:
                    return avg_score, individual_scores
                else:
                    return avg_score
            
        except Exception as e:
            print(f"    Error in evaluation: {e}")
            if return_individual_scores:
                return 3.0, {}
            else:
                return 3.0
    
    def pil_to_part(self, img: Image.Image, mime="image/png"):
        """Convert PIL Image to Part for maximum compatibility."""
        buf = BytesIO()
        img.save(buf, format="PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)

    def _predict_persona_effectiveness_with_logprobs(self, persona_text: str, persona_id: str, 
                                                    ad_image: Image.Image, prompt: str) -> Dict:
        """Predict ad effectiveness for a single persona with logprobs."""
        
        # Truncate persona text if too long
        max_persona_length = 50000
        if len(persona_text) > max_persona_length:
            persona_sample = persona_text[:max_persona_length] + "\n\n[Note: Content truncated]"
        else:
            persona_sample = persona_text
        
        # First get logprobs for the score distribution
        score_only_prompt = f"""PERSONA DATA:
{persona_sample}

TASK:
Return only one item from ["1","2","3","4","5"] for ad effectiveness.
Effective Score Scale Definition:
1: Extremely Unlikely. The persona would actively ignore or be annoyed by this ad.
2: Unlikely. The persona would likely scroll past without a second thought.
3: Mediocre. It is hard to decide whether the personal would click or don't click.
4: Likely. The persona is intrigued and has a good chance of clicking to learn more.
5: Extremely Likely. The persona is the ideal target; a click is almost certain.
No explanation. Just the score."""

        try:
            resp = self.vertex_client.models.generate_content(
                model=self.critic_model_id,
                contents=[self.pil_to_part(ad_image), score_only_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="text/x.enum",
                    response_schema={"type": "STRING", "enum": ["1", "2", "3", "4", "5"]},
                    response_logprobs=True,
                    logprobs=5,
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
            result = {
                "persona_id": persona_id,
                "effectiveness_score": expected,
                "score_probs": {str(i+1): probs[str(i+1)] for i in range(5)}  # 1-5 keys for scores
            }
            
            return result
            
        except Exception as e:
            print(f"        Error predicting for persona {persona_id}: {e}")
            return {
                "persona_id": persona_id,
                "effectiveness_score": 3.0,
                "score_probs": {str(i+1): 0.2 for i in range(5)}
            }
    
    def _predict_persona_effectiveness(self, persona_text: str, persona_id: str, 
                                     ad_image: Image.Image, prompt: str) -> float:
        """Predict ad effectiveness for a single persona."""
        result = self._predict_persona_effectiveness_with_logprobs(persona_text, persona_id, ad_image, prompt)
        return float(result["effectiveness_score"])
    
    def optimize(self,
                 T: int = 10,          # Number of optimization iterations
                 G: int = 3,           # Gradient steps per iteration
                 N: int = 4,           # Candidates per gradient step
                 eval_personas: int = 50,  # Personas to use for evaluation
                 k: int = 1,           # Number of parallel trajectories (1 = non-parallel)
                 initial_ads: List[Tuple[int, float, str, str]] = None  # Initial ads for parallel trajectories
                ) -> Tuple[str, float, Optional[str], str]:
        """Run TextBO optimization algorithm.
        
        Args:
            T: Number of optimization iterations
            G: Gradient steps per iteration  
            N: Number of candidates per gradient step
            eval_personas: Number of personas for evaluation
            k: Number of parallel trajectories (1 = non-parallel, >1 = parallel)
            initial_ads: List of (ad_index, score, prompt, image_path) for parallel trajectories
            
        Returns:
            Tuple of (best_prompt, best_score, best_image_path, optimization_log_path)
        """
        if k == 1:
            print(f"🚀 Starting TextBO optimization for {self.scenario_name}")
        else:
            print(f"🚀 Starting Parallel TextBO optimization for {self.scenario_name}")
        print(f"Parameters: T={T}, G={G}, N={N}, eval_personas={eval_personas}, k={k}")
        print(f"Initial prompt length: {len(self.initial_prompt)} characters")
        
        if k == 1:
            return self._optimize_sequential(T, G, N, eval_personas)
        else:
            return self._optimize_parallel(T, G, N, eval_personas, k, initial_ads)

    def _optimize_sequential(self, T: int, G: int, N: int, eval_personas: int) -> Tuple[str, float, Optional[str], str]:
        """Run sequential T-BoN optimization (original algorithm)."""
        current_prompt = self.initial_prompt
        current_score = None  # Track score of current prompt
        optimization_log = []
        
        for t in range(1, T + 1):
            # Generate iteration seed for reproducibility
            iteration_seed = self.base_seed + t * 100
            print(f"\n=== ITERATION {t}/{T} (seed: {iteration_seed}) ===")
            
            # Cache performance analysis once per iteration (expensive operation)
            cached_analysis = None
            
            # G gradient descent steps
            winner_image_path = None  # Track last valid winner across gradient steps
            for g in range(1, G + 1):
                print(f"\n--- Gradient Step {g}/{G} ---")
                
                # Generate N candidate variants
                candidates = []
                
                for i in range(1, N + 1):
                    print(f"Generating candidate {i}/{N}...")
                    
                    # Generate textual gradient using cached analysis
                    gradient = self.generate_textual_gradient(current_prompt, self.history, cached_analysis)
                    
                    # Apply gradient to create new prompt
                    new_prompt = self.apply_gradient(current_prompt, gradient)
                    
                    # Generate image from new prompt
                    image_path = self.generate_image(new_prompt, t, g, i, trajectory=0)
                    
                    # Store only the image path, don't keep PIL images open
                    candidates.append((new_prompt, image_path, image_path))
                    
                    time.sleep(1)  # Rate limiting
                
                # Tournament selection to pick best candidate
                print(f"Selecting best candidate from {len(candidates)} options...")
                tournament_seed = iteration_seed + g * 10  # Different seed per gradient step
                step_winner_prompt, step_winner_image, step_winner_image_path = self.tournament_select(candidates, tournament_seed)
                
                # Update winner only if we got a valid path
                if step_winner_image_path and os.path.exists(step_winner_image_path):
                    winner_image_path = step_winner_image_path
                    # Store the candidate prompt but don't update current_prompt yet
                    candidate_prompt = step_winner_prompt
                    print(f"Selected candidate with image: {winner_image_path}")
                else:
                    print(f"No valid image from gradient step {g}, keeping previous winner")
                    candidate_prompt = current_prompt
            
            # Evaluate the final prompt from this iteration
            print(f"\nEvaluating iteration {t} result...")
            if winner_image_path and os.path.exists(winner_image_path):
                score = self.evaluate_candidate(winner_image_path, candidate_prompt, eval_personas, iteration_seed)
            else:
                score = 3.0  # Default score if no successful candidates
            
            # Only update current_prompt if new score exceeds previous score
            if current_score is None or score > current_score:
                print(f"New score ({score:.3f}) exceeds previous score ({current_score:.3f} if not None), updating prompt")
                current_prompt = candidate_prompt
                current_score = score
            else:
                print(f"New score ({score:.3f}) does not exceed previous score ({current_score:.3f}), keeping previous prompt")
            
            # Update history with the evaluated candidate (store image path, not PIL object)
            self.history.append((candidate_prompt, winner_image_path, score))
            
            iteration_log = {
                "iteration": t,
                "prompt": candidate_prompt,
                "score": score,
                "image_path": winner_image_path if any(candidates) else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "iteration_seed": iteration_seed,
                "tournament_seed": tournament_seed if 'tournament_seed' in locals() else None,
                "performance_analysis": cached_analysis
            }
            optimization_log.append(iteration_log)
            
            print(f"Iteration {t} complete. Score: {score:.3f}")
            
            # Analyze performance patterns at the end of iteration for next iteration
            if self.history and t < T:  # Don't analyze after the last iteration
                print("Analyzing performance patterns from history for next iteration...")
                self.analyze_performance_patterns(self.history)
            
            # Save intermediate results (basic structure for now)
            log_path = os.path.join(self.output_dir, f"textbo_{self.scenario_name.lower()}_optimization_log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "scenario": self.scenario_name,
                    "initial_prompt": self.initial_prompt,
                    "parameters": {"T": T, "G": G, "N": N, "eval_personas": eval_personas},
                    "iterations": optimization_log,
                    "final_score": score,
                    "final_prompt": current_prompt
                }, f, indent=2, ensure_ascii=False)
        
        # Find best result
        best_iteration = max(optimization_log, key=lambda x: x["score"])
        best_prompt = best_iteration["prompt"]
        best_score = best_iteration["score"]
        best_image_path = best_iteration.get("image_path")
        
        # Copy best image to dedicated file if it exists
        best_image_final_path = None
        if best_image_path and os.path.exists(best_image_path):
            import shutil
            best_image_final_path = os.path.join(self.output_dir, f"best_ad_image_{self.scenario_name.lower()}.png")
            shutil.copy2(best_image_path, best_image_final_path)
            print(f"Best performing image saved: {best_image_final_path}")
        
        # Update final log with best image information
        final_log_data = {
            "scenario": self.scenario_name,
            "initial_prompt": self.initial_prompt,
            "parameters": {"T": T, "G": G, "N": N, "eval_personas": eval_personas},
            "base_seed": self.base_seed,
            "iterations": optimization_log,
            "final_score": score,
            "final_prompt": current_prompt,
            "best_result": {
                "iteration": best_iteration["iteration"],
                "score": best_score,
                "prompt": best_prompt,
                "original_image_path": best_image_path,
                "best_image_path": best_image_final_path,
                "timestamp": best_iteration["timestamp"],
                "iteration_seed": best_iteration.get("iteration_seed"),
                "tournament_seed": best_iteration.get("tournament_seed")
            }
        }
        
        # Save updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(final_log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ TextBO optimization complete!")
        print(f"Best score achieved: {best_score:.3f} (iteration {best_iteration['iteration']})")
        if best_image_path:
            print(f"Best image: {os.path.basename(best_image_path)}")
        print(f"Final prompt length: {len(best_prompt)} characters")
        print(f"Optimization log saved: {log_path}")
        
        return best_prompt, best_score, best_image_path, log_path

    def _optimize_parallel(self, T: int, G: int, N: int, eval_personas: int, k: int, initial_ads: List[Tuple[int, float, str, str]] = None) -> Tuple[str, float, Optional[str], str]:
        """Run parallel T-BoN optimization (Algorithm 2 from paper).
        
        Args:
            T: Number of optimization iterations
            G: Gradient steps per iteration
            N: Number of candidates per gradient step
            eval_personas: Number of personas for evaluation
            k: Number of parallel trajectories
            initial_ads: List of (ad_index, score, prompt, image_path) for k initial trajectories
        """
        # Initialize k trajectories using provided initial ads or fallback to single prompt
        if initial_ads and len(initial_ads) >= k:
            trajectories = [{
                'prompt': initial_ads[j][2],  # Use prompt from initial_ads[j]
                'image_path': initial_ads[j][3] if os.path.exists(initial_ads[j][3]) else None,
                'score': initial_ads[j][1]    # Use initial score
            } for j in range(k)]
            print(f"Initialized {k} trajectories with worst {k} ads as starting points")
        else:
            # Fallback to using the same initial prompt for all trajectories
            trajectories = [{
                'prompt': self.initial_prompt,
                'image_path': None,
                'score': None
            } for _ in range(k)]
            print(f"Warning: Not enough initial ads provided, using same initial prompt for all {k} trajectories")
        
        # Shared history H across all trajectories
        H = []
        # Meta-reflection R updated at end of each iteration
        R = None
        
        optimization_log = []
        
        # Initialization: Only evaluate and log worst (1st) and kth worst ads to shared history
        print("\n=== INITIALIZATION ===")
        print(f"Evaluating boundary ads: worst (1st) and {k}th worst for shared history...")
        
        # Initialize all trajectories but only evaluate boundary cases
        for j in range(k):
            print(f"Setting up trajectory {j+1}/{k}...")
            
            # All trajectories should have existing ads from initial_ads by design
            if trajectories[j]['image_path'] and trajectories[j]['score'] is not None:
                print(f"Trajectory {j+1} initialized with existing ad (score: {trajectories[j]['score']:.3f})")
            else:
                print(f"Warning: Trajectory {j+1} missing initial ad data")
                trajectories[j]['score'] = 3.0  # Default score
        
        # Only evaluate and log the worst (j=0) and kth worst (j=k-1) to shared history H
        boundary_indices = [0]  # Always evaluate worst
        if k > 1:
            boundary_indices.append(k-1)  # Also evaluate kth worst if k > 1
        
        for j in boundary_indices:
            print(f"Evaluating boundary ad: trajectory {j+1} ({'worst' if j == 0 else f'{k}th worst'})...")
            
            if trajectories[j]['image_path'] and trajectories[j]['score'] is not None:
                # Use existing evaluation and add to shared history
                H.append((trajectories[j]['prompt'], trajectories[j]['image_path'], trajectories[j]['score']))
                print(f"Added trajectory {j+1} to shared history (score: {trajectories[j]['score']:.3f})")
            else:
                # Fallback: generate and evaluate if missing
                initial_image_path = self.generate_image(trajectories[j]['prompt'], 0, 0, j+1, trajectory=j+1)
                
                if initial_image_path and os.path.exists(initial_image_path):
                    initial_score = self.evaluate_candidate(initial_image_path, trajectories[j]['prompt'], eval_personas)
                    trajectories[j]['image_path'] = initial_image_path
                    trajectories[j]['score'] = initial_score
                    H.append((trajectories[j]['prompt'], initial_image_path, initial_score))
                    print(f"Trajectory {j+1} evaluated and added to shared history (score: {initial_score:.3f})")
                else:
                    print(f"Warning: Failed to generate image for trajectory {j+1}")
                    trajectories[j]['score'] = 3.0
                    H.append((trajectories[j]['prompt'], None, 3.0))
        
        print(f"Shared history initialized with {len(H)} boundary evaluations (worst and {k}th worst)")
        
        # Main optimization loop
        for t in range(1, T + 1):
            iteration_seed = self.base_seed + t * 100
            print(f"\n=== ITERATION {t}/{T} (seed: {iteration_seed}) ===")
            
            # Process all k trajectories in parallel
            iteration_results = []
            
            for j in range(k):
                print(f"\n--- Trajectory {j+1}/{k} ---")
                current_prompt = trajectories[j]['prompt']
                current_score = trajectories[j]['score']  # Track current score for this trajectory
                
                # G gradient descent steps for this trajectory
                winner_image_path = trajectories[j]['image_path']
                for g in range(1, G + 1):
                    print(f"Gradient Step {g}/{G} for trajectory {j+1}...")
                    
                    # Generate N candidate variants
                    candidates = []
                    
                    for i in range(1, N + 1):
                        # Generate textual gradient conditioned on shared R
                        gradient = self.generate_textual_gradient(current_prompt, H, R)
                        
                        # Apply gradient to create new prompt
                        new_prompt = self.apply_gradient(current_prompt, gradient)
                        
                        # Generate image from new prompt
                        image_path = self.generate_image(new_prompt, t, g, i, trajectory=j+1)
                        
                        candidates.append((new_prompt, image_path, image_path))
                        time.sleep(0.5)  # Rate limiting
                    
                    # Tournament selection to pick best candidate
                    tournament_seed = iteration_seed + j * 100 + g * 10
                    step_winner_prompt, _, step_winner_image_path = self.tournament_select(candidates, tournament_seed)
                    
                    # Update winner image path but store candidate prompt
                    if step_winner_image_path and os.path.exists(step_winner_image_path):
                        candidate_prompt = step_winner_prompt
                        winner_image_path = step_winner_image_path
                    else:
                        candidate_prompt = current_prompt
                
                # Evaluate final prompt for this trajectory
                print(f"Evaluating trajectory {j+1} result...")
                if winner_image_path and os.path.exists(winner_image_path):
                    score = self.evaluate_candidate(winner_image_path, candidate_prompt, eval_personas, iteration_seed + j)
                else:
                    score = 3.0  # Default score if no successful candidates
                
                # Only update current_prompt if new score exceeds previous score
                if current_score is None or score > current_score:
                    print(f"Trajectory {j+1}: New score ({score:.3f}) exceeds previous score ({current_score:.3f} if not None), updating prompt")
                    current_prompt = candidate_prompt
                    current_score = score
                else:
                    print(f"Trajectory {j+1}: New score ({score:.3f}) does not exceed previous score ({current_score:.3f}), keeping previous prompt")
                
                # Store trajectory results
                trajectories[j] = {
                    'prompt': current_prompt,
                    'image_path': winner_image_path,
                    'score': current_score  # Store the current (best so far) score
                }
                
                iteration_results.append({
                    "trajectory": j+1,
                    "iteration": t,
                    "prompt": candidate_prompt,  # Log the evaluated candidate
                    "candidate_score": score,    # Score of the evaluated candidate
                    "current_prompt": current_prompt,  # The actual prompt being used
                    "current_score": current_score,    # The score of current prompt
                    "image_path": winner_image_path,
                    "iteration_seed": iteration_seed + j
                })
                
                print(f"Trajectory {j+1} iteration {t} complete. Score: {score:.3f}")
            
            # Synchronized update: merge evaluations from all trajectories into H
            for j in range(k):
                # Add the evaluated candidate to shared history (not necessarily the current prompt)
                candidate_result = iteration_results[j]
                H.append((candidate_result['prompt'], candidate_result['image_path'], candidate_result['candidate_score']))
            
            # Refresh meta-reflection R based on updated H
            if len(H) >= 3:  # Only compute meta-reflection if we have enough history
                print("Updating meta-reflection from shared history...")
                # Use a subset of recent history for meta-reflection to avoid context limits
                recent_history = H[-min(10, len(H)):]
                R = self.analyze_performance_patterns(recent_history)
            
            # Log iteration results
            optimization_log.extend(iteration_results)
            
            # Save intermediate results
            log_path = os.path.join(self.output_dir, f"textbo_{self.scenario_name.lower()}_parallel_optimization_log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "scenario": self.scenario_name,
                    "initial_prompt": self.initial_prompt,
                    "parameters": {"T": T, "G": G, "N": N, "eval_personas": eval_personas, "k": k},
                    "iterations": optimization_log,
                    "shared_history_length": len(H)
                }, f, indent=2, ensure_ascii=False)
        
        # Find best result among the final states of k trajectories
        best_trajectory = max(trajectories, key=lambda x: x["score"])
        best_prompt = best_trajectory["prompt"]
        best_score = best_trajectory["score"]
        best_image_path = best_trajectory.get("image_path")
        
        # Find corresponding iteration info for logging
        best_iteration = None
        for entry in reversed(optimization_log):  # Search from latest entries
            if entry["current_prompt"] == best_prompt and entry["current_score"] == best_score:
                best_iteration = entry
                break
        
        # Copy best image to dedicated file if it exists
        best_image_final_path = None
        if best_image_path and os.path.exists(best_image_path):
            import shutil
            best_image_final_path = os.path.join(self.output_dir, f"best_ad_image_{self.scenario_name.lower()}_parallel.png")
            shutil.copy2(best_image_path, best_image_final_path)
            print(f"Best performing image saved: {best_image_final_path}")
        
        # Update final log with best results
        final_log_data = {
            "scenario": self.scenario_name,
            "initial_prompt": self.initial_prompt,
            "parameters": {"T": T, "G": G, "N": N, "eval_personas": eval_personas, "k": k},
            "base_seed": self.base_seed,
            "iterations": optimization_log,
            "shared_history_length": len(H),
            "best_result": {
                "trajectory": best_iteration["trajectory"],
                "iteration": best_iteration["iteration"],
                "score": best_score,
                "prompt": best_prompt,
                "original_image_path": best_image_path,
                "best_image_path": best_image_final_path,
                "iteration_seed": best_iteration.get("iteration_seed")
            }
        }
        
        # Save updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(final_log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Parallel TextBO optimization complete!")
        print(f"Best score achieved: {best_score:.3f} (trajectory {best_iteration['trajectory']}, iteration {best_iteration['iteration']})")
        if best_image_path:
            print(f"Best image: {os.path.basename(best_image_path)}")
        print(f"Final prompt length: {len(best_prompt)} characters")
        print(f"Optimization log saved: {log_path}")
        
        return best_prompt, best_score, best_image_path, log_path


def find_worst_performing_ads(scenario_name: str, k: int = 1) -> List[Tuple[int, float, str, str]]:
    """Find the worst k performing ads and their prompts for a given scenario.
    
    Args:
        scenario_name: Name of the advertising scenario
        k: Number of worst ads to return
        
    Returns:
        List of tuples (ad_index, score, prompt, image_path) sorted from worst to best
    """
    # Load DTTTS results to find worst performing ad
    results_path = f"campaign_output/{scenario_name.lower()}_campaign_output/{scenario_name.lower()}_dttts_results.json"
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    posterior_means = results["final_statistics"]["posterior_means"]
    
    # Load corresponding prompts
    prompts_path = f"campaign_output/{scenario_name.lower()}_campaign_output/generated_prompts.json"
    
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Create list of (index, score, prompt, image_path) tuples
    ad_data = []
    for idx, score in enumerate(posterior_means):
        prompt = prompts[idx]["prompt"]
        image_filename = f"{scenario_name.lower()}_ad_{idx+1:02d}.png"
        image_path = f"campaign_output/{scenario_name.lower()}_campaign_output/{image_filename}"
        ad_data.append((idx, score, prompt, image_path))
    
    # Sort by score (ascending) to get worst ads first
    ad_data.sort(key=lambda x: x[1])
    
    # Return worst k ads
    worst_k_ads = ad_data[:min(k, len(ad_data))]
    
    print(f"Found worst {len(worst_k_ads)} performing ads:")
    for i, (idx, score, _, image_path) in enumerate(worst_k_ads):
        print(f"  {i+1}. Index {idx} with score {score:.3f} - {image_path}")
    
    return worst_k_ads


def find_worst_performing_ad(scenario_name: str) -> Tuple[int, float, str, str]:
    """Find the worst performing ad and its prompt for a given scenario.
    
    Args:
        scenario_name: Name of the advertising scenario
        
    Returns:
        Tuple of (ad_index, score, prompt, image_path)
    """
    worst_ads = find_worst_performing_ads(scenario_name, k=1)
    return worst_ads[0]


def find_best_performing_ad(scenario_name: str) -> Tuple[int, float, str, str]:
    """Find the best performing ad and its prompt for a given scenario.
    
    Args:
        scenario_name: Name of the advertising scenario
        
    Returns:
        Tuple of (ad_index, score, prompt, image_path)
    """
    # Load DTTTS results to find best performing ad
    results_path = f"campaign_output/{scenario_name.lower()}_campaign_output/{scenario_name.lower()}_dttts_results.json"
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    posterior_means = results["final_statistics"]["posterior_means"]
    best_ad_idx = posterior_means.index(max(posterior_means))
    best_score = posterior_means[best_ad_idx]
    
    # Load corresponding prompt
    prompts_path = f"campaign_output/{scenario_name.lower()}_campaign_output/generated_prompts.json"
    
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    best_prompt = prompts[best_ad_idx]["prompt"]
    
    # Construct path to corresponding image file
    image_filename = f"{scenario_name.lower()}_ad_{best_ad_idx+1:02d}.png"
    image_path = f"campaign_output/{scenario_name.lower()}_campaign_output/{image_filename}"
    
    print(f"Found best performing ad at index {best_ad_idx} with score {best_score:.3f}")
    print(f"Corresponding image: {image_path}")
    return best_ad_idx, best_score, best_prompt, image_path


def _parse_args(argv):
    """Parse command line arguments with Unicode dash normalization."""
    # normalize Unicode dashes to ASCII hyphen
    argv = [a.replace('—', '-').replace('–', '-') for a in argv]
    p = argparse.ArgumentParser(prog="TextBO")
    p.add_argument('--test', '-t', action='store_true', help='run in test mode')
    p.add_argument('--gepa', action='store_true', help='run in GEPA mode')
    p.add_argument('--parallel', '-p', type=int, default=1, metavar='K', help='run in parallel mode with K trajectories (default: 1)')
    return p.parse_args(argv[1:])


def main(test_mode: bool = False, gepa_mode: bool = False, parallel_k: int = 1):
    """Main function to run T-BoN optimization."""
    
    # Choose and announce mode FIRST
    if test_mode:
        T = 2           # 2 optimization iterations
        G = 2           # 2 gradient steps per iteration
        N = 2           # 2 candidates per gradient step
        eval_personas = 2  # 2 personas for evaluation
        print("🧪 RUNNING IN TEST MODE")
        print(f"Parameters: T={T}, G={G}, N={N}, eval_personas={eval_personas}")
    elif gepa_mode:
        T = 10          # 10 optimization iterations
        G = 1           # 1 gradient step per iteration
        N = 1           # 1 candidate per gradient step
        eval_personas = 200  # 200 personas for evaluation
        print("🧬 RUNNING IN GEPA MODE")
        print(f"Parameters: T={T}, G={G}, N={N}, eval_personas={eval_personas}")
    else:
        T = 10          # 10 optimization iterations
        G = 5           # 5 gradient steps per iteration  
        N = 5           # 5 candidates per gradient step
        eval_personas = 200  # 200 personas for evaluation
        print("🚀 RUNNING IN PRODUCTION MODE")
        print(f"Parameters: T={T}, G={G}, N={N}, eval_personas={eval_personas}")
    
    # Then load env and validate only what Vertex actually needs
    load_dotenv()
    PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID") 
    LOCATION = os.getenv("GOOGLE_LOCATION")
    
    # For vertexai=True, API key is not required
    if not all([PROJECT_ID, LOCATION]):
        print("❌ Missing required environment variables:")
        print("   Please set GOOGLE_PROJECT_ID and GOOGLE_LOCATION in .env file")
        return
    
    # Configuration
    SCENARIO_NAME = "syncflow"
    #SCENARIO_NAME = "terraexplorer"
    
    try:
        # Find worst performing ad(s) as starting point(s)
        if parallel_k > 1:
            print(f"🔍 Finding worst {parallel_k} performing ads for {SCENARIO_NAME}...")
            worst_ads = find_worst_performing_ads(SCENARIO_NAME, parallel_k)
            # Use the single worst for comparison purposes
            worst_ad_idx, worst_score, initial_prompt, initial_image_path = worst_ads[0]
        else:
            print(f"🔍 Finding worst performing ad for {SCENARIO_NAME}...")
            worst_ad_idx, worst_score, initial_prompt, initial_image_path = find_worst_performing_ad(SCENARIO_NAME)
            worst_ads = None
        
        # Find best performing ad for comparison
        print(f"🔍 Finding best performing ad for {SCENARIO_NAME}...")
        best_ad_idx, best_score, best_prompt, best_image_path = find_best_performing_ad(SCENARIO_NAME)
        
        print(f"Starting optimization from ad index {worst_ad_idx} (score: {worst_score:.3f})")
        print(f"Best DTTTS ad is at index {best_ad_idx} (score: {best_score:.3f})")
        print(f"Initial prompt length: {len(initial_prompt)} characters")
        
        # Set up output directory
        if test_mode:
            mode_suffix = "_test"
        elif gepa_mode:
            mode_suffix = "_gepa"
        else:
            mode_suffix = ""
        
        # Add parallel suffix if using parallel mode
        if parallel_k > 1:
            mode_suffix += f"_parallel_k{parallel_k}"

        output_dir = os.path.join("campaign_output", f"{SCENARIO_NAME.lower()}_textbo_output{mode_suffix}")
        
        # Copy initial image to output directory with proper naming
        if initial_image_path and os.path.exists(initial_image_path):
            import shutil
            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
            initial_image_filename = f"textbo_{SCENARIO_NAME.lower()}_init.png"
            initial_image_dest = os.path.join(output_dir, initial_image_filename)
            shutil.copy2(initial_image_path, initial_image_dest)
            print(f"📸 Initial image copied: {initial_image_filename}")
        else:
            print(f"⚠️  Warning: Initial image not found at {initial_image_path}")
        
        # Initialize T-BoN optimizer
        optimizer = TextBOOptimizer(
            api_key=None,  # Not needed for vertexai=True
            project_id=PROJECT_ID,
            location=LOCATION,
            scenario_name=SCENARIO_NAME,
            initial_prompt=initial_prompt,
            output_dir=output_dir
        )
        
        # Evaluate best DTTTS ad + worst 5 DTTTS ads on the test set before optimization
        print("\n=== PRE-OPTIMIZATION EVALUATION ON TEST SET ===")

        # Find worst 5 DTTTS ads for evaluation
        print("🔍 Finding worst 5 performing ads for pre-optimization evaluation...")
        worst_5_ads = find_worst_performing_ads(SCENARIO_NAME, k=5)

        # Store individual persona scores for all pre-optimization evaluations
        pre_optimization_scores = {}

        # Evaluate best DTTTS ad
        print("\n--- Evaluating best DTTTS ad on the test set ---")
        best_dttts_test_score, best_dttts_individual = optimizer.evaluate_candidate(
            best_image_path, best_prompt, eval_personas, use_test_set=True, return_individual_scores=True
        )
        print(f"Best DTTTS ad score on test set: {best_dttts_test_score:.3f}")
        pre_optimization_scores["best_dttts_ad"] = {
            "aggregate_score": best_dttts_test_score,
            "ad_index": best_ad_idx,
            "individual_scores": best_dttts_individual
        }

        # Evaluate worst 5 DTTTS ads
        for rank, (ad_idx, dttts_score, ad_prompt, ad_image_path) in enumerate(worst_5_ads, start=1):
            print(f"\n--- Evaluating {rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'} worst DTTTS ad on the test set ---")
            test_score, individual_scores = optimizer.evaluate_candidate(
                ad_image_path, ad_prompt, eval_personas, use_test_set=True, return_individual_scores=True
            )
            print(f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'} worst ad score on test set: {test_score:.3f}")
            pre_optimization_scores[f"{rank}_worst_dttts_ad"] = {
                "aggregate_score": test_score,
                "ad_index": ad_idx,
                "individual_scores": individual_scores
            }

        # Keep references for backward compatibility
        initial_test_score = pre_optimization_scores["1_worst_dttts_ad"]["aggregate_score"]

        # Add initial evaluation to history for more informed optimization
        print("--- Adding initial evaluation to optimization history ---")
        # Choose a path that will exist later (use the copied file if available)
        seed_image_path = initial_image_dest if os.path.exists(initial_image_dest) else initial_image_path
        optimizer.history.append((initial_prompt, seed_image_path, float(initial_test_score)))
        print(f"Initial evaluation added to history for pattern analysis")

        # Run optimization
        best_prompt, best_score_on_train, best_image_path, log_path = optimizer.optimize(
            T=T,
            G=G,
            N=N,
            eval_personas=eval_personas,
            k=parallel_k,
            initial_ads=worst_ads if parallel_k > 1 else None
        )
        
        # Evaluate the final optimized ad on the test set
        print("\n=== POST-OPTIMIZATION EVALUATION ON TEST SET ===")
        print("--- Evaluating final optimized ad on the test set ---")
        final_test_score, final_individual_scores = optimizer.evaluate_candidate(
            best_image_path, best_prompt, eval_personas, use_test_set=True, return_individual_scores=True
        )
        print(f"Final ad score on test set: {final_test_score:.3f}")

        # Store post-optimization scores
        post_optimization_scores = {
            "final_optimized_ad": {
                "aggregate_score": final_test_score,
                "individual_scores": final_individual_scores
            }
        }

        # Update the log file with test set results and individual scores
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r+') as f:
                log_data = json.load(f)

                # Add aggregate test set results for backward compatibility
                log_data["test_set_results"] = {
                    "best_dttts_ad_test_score": best_dttts_test_score,
                    "initial_ad_test_score": initial_test_score,
                    "optimized_ad_test_score": final_test_score,
                    "improvement_vs_worst_dttts": final_test_score - initial_test_score,
                    "improvement_vs_best_dttts": final_test_score - best_dttts_test_score
                }

                # Add improvements vs all worst 5 ads
                for rank in range(1, 6):
                    if f"{rank}_worst_dttts_ad" in pre_optimization_scores:
                        worst_score = pre_optimization_scores[f"{rank}_worst_dttts_ad"]["aggregate_score"]
                        log_data["test_set_results"][f"improvement_vs_{rank}_worst_dttts"] = final_test_score - worst_score

                # Add detailed individual persona scores
                log_data["pre_optimization_individual_scores"] = pre_optimization_scores
                log_data["post_optimization_individual_scores"] = post_optimization_scores

                f.seek(0)
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                f.truncate()

        print(f"\n🎯 OPTIMIZATION COMPLETE")
        print(f"Initial score (from DTTTS): {worst_score:.3f}")
        print(f"Final score (on train set sample): {best_score_on_train:.3f}")
        print("=" * 60)
        print("PERFORMANCE ON UNSEEN TEST SET:")
        print(f"Best DTTTS ad score:          {best_dttts_test_score:.3f}")
        for rank in range(1, 6):
            if f"{rank}_worst_dttts_ad" in pre_optimization_scores:
                score = pre_optimization_scores[f"{rank}_worst_dttts_ad"]["aggregate_score"]
                print(f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'} worst DTTTS ad score:      {score:.3f}")
        print(f"Final TextBO optimized ad:    {final_test_score:.3f}")
        print("=" * 60)
        print("IMPROVEMENTS OVER DTTTS:")
        for rank in range(1, 6):
            if f"{rank}_worst_dttts_ad" in pre_optimization_scores:
                score = pre_optimization_scores[f"{rank}_worst_dttts_ad"]["aggregate_score"]
                improvement = final_test_score - score
                print(f"TextBO vs {rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'} worst:        {improvement:+.3f}")
        print(f"TextBO vs best DTTTS:         {final_test_score - best_dttts_test_score:+.3f}")
        print("=" * 60)
        print(f"Results saved in: {output_dir}")
        print(f"Individual persona scores saved in optimization log")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        raise


if __name__ == "__main__":
    import sys
    print("🤖 TextBO: TextGrad with Best-of-N Bayesian Optimization")
    print("=" * 60)
    
    args = _parse_args(sys.argv)
    print(f"Args parsed → test={args.test}, gepa={args.gepa}, parallel={args.parallel}")
    main(test_mode=args.test, gepa_mode=args.gepa, parallel_k=args.parallel)
