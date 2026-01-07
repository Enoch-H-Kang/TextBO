import json
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

def load_persona_data(persona_dir):
    """Load all persona data from text files."""
    personas = {}
    persona_files = glob.glob(os.path.join(persona_dir, "pid_*_prompt.txt"))
    
    for file_path in persona_files:
        persona_id = os.path.basename(file_path).split('_')[1]
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            personas[persona_id] = content
    
    return personas


def predict_ad_click_with_gemini(persona_text, persona_id, ad_image, model):
    """Use Gemini 2.5 Flash to predict if persona will click on ad based on their survey responses and the actual ad image."""
    
    # Truncate persona text if too long to avoid token limits
    max_persona_length = 50000  # Reasonable limit for Gemini context
    if len(persona_text) > max_persona_length:
        persona_sample = persona_text[:max_persona_length] + "\n\n[Note: Additional survey responses truncated for length]"
    else:
        persona_sample = persona_text
    
    # Create a comprehensive prompt for Gemini
    prompt = f"""You are analyzing whether a person will click on a mobile Instagram advertisement based on their detailed survey responses.

PERSONA DATA (Survey Responses):
{persona_sample}

ADVERTISEMENT CONTEXT:
- Platform: Mobile Instagram
- Context: This is a mobile Instagram ad that appears in user's feed. That is, it is a native ad that blends with other content.
- The actual ad image is provided above for your analysis

TASK:
Based on this person's survey responses and the visual content of the ad image shown, pessimistically predict the effectiveness of ad from ``Highly effective, effective, mediocre, less effective, not effective'', which corresponds to score [5,4,3,2,1]].
REQUIRED OUTPUT FORMAT:
Your response must be valid JSON in exactly this format:
{{
    "reasoning": "Detailed explanation based on survey responses and ad visual elements",
    "effectiveness_score": X
}}

Effective Score Scale Definition:
1: Extremely Unlikely. The persona would actively ignore or be annoyed by this ad.
2: Unlikely. The persona would likely scroll past without a second thought.
3: Mediocre. It is hard to decide whether the personal would click or don't click.
4: Likely. The persona is intrigued and has a good chance of clicking to learn more.
5: Extremely Likely. The persona is the ideal target; a click is almost certain.
Important: Only respond with the JSON object, no other text."""

    try:
        # Create content list with image and prompt
        contents = [ad_image, prompt]
        response = model.generate_content(contents)
        
        # Parse the JSON response
        response_text = response.text.strip()
        
        # Clean up response if it has markdown formatting
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        result = json.loads(response_text)
        
        # Add persona_id to result
        result["persona_id"] = persona_id
        
        # Validate and process the effectiveness score
        if "effectiveness_score" not in result:
            result["effectiveness_score"] = 3  # Default to "Mediocre"
        
        effectiveness_score = int(result["effectiveness_score"])
        effectiveness_score = max(1, min(5, effectiveness_score))  # Ensure bounds 1-10
        result["effectiveness_score"] = effectiveness_score
        
        if "reasoning" not in result:
            result["reasoning"] = "Standard analysis based on survey responses"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error for persona {persona_id}: {e}")
        print(f"Raw response: {response.text[:200]}...")
        return {
            "persona_id": persona_id,
            "effectiveness_score": 3,
            "reasoning": f"Error parsing LLM response for persona {persona_id}: {str(e)}"
        }
    except Exception as e:
        print(f"Error predicting for persona {persona_id}: {e}")
        return {
            "persona_id": persona_id,
            "effectiveness_score": 3,
            "reasoning": f"Error in prediction for persona {persona_id}: {str(e)}"
        }

def load_ad_images(campaign_dir, scenario_name):
    """Load the actual ad images from the campaign directory."""
    ad_images = {}
    
    for ad_num in range(1, 4):
        image_path = os.path.join(campaign_dir, f"{scenario_name.lower()}_ad_{ad_num:02d}.png")
        if os.path.exists(image_path):
            try:
                ad_images[ad_num] = Image.open(image_path)
                print(f"Loaded ad image {ad_num}: {image_path}")
            except Exception as e:
                print(f"Error loading ad image {ad_num}: {e}")
                ad_images[ad_num] = None
        else:
            print(f"Ad image {ad_num} not found: {image_path}")
            ad_images[ad_num] = None
    
    return ad_images

def generate_ad_predictions(personas, ad_images, model, scenario_name):
    """Generate predictions for all personas for all ads using actual ad images."""
    
    results = {}
    
    for ad_num in range(1, 4):
        print(f"\nProcessing Ad {ad_num}...")
        results[f"ad_{ad_num:02d}"] = []
        
        if ad_images[ad_num] is None:
            print(f"  Skipping Ad {ad_num} - image not available")
            continue
        
        for i, (persona_id, persona_text) in enumerate(personas.items()):
            if i % 25 == 0:
                print(f"  Processing persona {i+1}/{len(personas)} (ID: {persona_id})")
            
            try:
                prediction = predict_ad_click_with_gemini(
                    persona_text,
                    persona_id, 
                    ad_images[ad_num],
                    model
                )
                results[f"ad_{ad_num:02d}"].append(prediction)
                
                # Print occasional successful predictions
                if i % 50 == 0 and prediction:
                    effectiveness = prediction.get("effectiveness_score", 3)
                    print(f"    Sample result: Persona {persona_id} effectiveness score: {effectiveness}/5")
                
            except Exception as e:
                print(f"    Error processing persona {persona_id}: {e}")
                # Add fallback prediction
                fallback_prediction = {
                    "persona_id": persona_id,
                    "effectiveness_score": 3,
                    "reasoning": f"Error in processing: {str(e)}"
                }
                results[f"ad_{ad_num:02d}"].append(fallback_prediction)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.2)
    
    return results

def save_results(results, output_dir, scenario_name):
    """Save results as separate JSON files for each ad."""
    os.makedirs(output_dir, exist_ok=True)
    
    for ad_key, predictions in results.items():
        output_file = os.path.join(output_dir, f"{scenario_name.lower()}_{ad_key}_predictions.json")
        
        # Calculate summary statistics based on effectiveness scores
        total_personas = len(predictions)
        effectiveness_scores = [p.get('effectiveness_score', 4) for p in predictions]
        avg_effectiveness = sum(effectiveness_scores) / total_personas
        
        output_data = {
            "ad_campaign": scenario_name,
            "ad_number": int(ad_key.split('_')[1]),
            "analysis_metadata": {
                "total_personas_analyzed": total_personas,
                "average_effectiveness_score": round(avg_effectiveness, 2),
                "effectiveness_scale": "1=Extremely Unlikely, 2=Unlikely, 3=Medicre, 4=Likely, 5=Extremely Likely",
                "ad_context": "Mobile Instagram Advertisement",
                "analysis_date": "2025-07-23"
            },
            "persona_predictions": predictions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {ad_key} predictions for {total_personas} personas")

def main():
    """Main execution function."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY missing (set in .env)")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # ------------------------------------------------------------------
    # 1. SPECIFY SCENARIO NAME
    #    Change this value to analyze ads for a different product.
    # ------------------------------------------------------------------
    SCENARIO_NAME = "SyncFlow"
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_dir = os.path.join(base_dir, "text_simulation", "text_simulation_input")
    campaign_dir = os.path.join(base_dir, "campaign_output", f"{SCENARIO_NAME.lower()}_campaign_output")
    output_dir = os.path.join(base_dir, "campaign_output", f"{SCENARIO_NAME.lower()}_campaign_output")
    
    print("Loading persona data...")
    personas = load_persona_data(persona_dir)
    print(f"Loaded {len(personas)} personas")
    
    # For testing, limit to first 10 personas
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    if test_mode:
        print("Running in TEST MODE - using only first 3 personas")
        personas = dict(list(personas.items())[:3])
        print(f"Test mode: {len(personas)} personas")
    
    print("Loading ad images...")
    ad_images = load_ad_images(campaign_dir, SCENARIO_NAME)
    
    print("Generating click predictions using Gemini 2.5 Flash with actual ad images...")
    print(f"This will make approximately {len(personas) * 3} API calls to Gemini")
    results = generate_ad_predictions(personas, ad_images, model, SCENARIO_NAME)
    
    print("Saving results...")
    save_results(results, output_dir, SCENARIO_NAME)
    
    print("Analysis complete! Results saved in:", output_dir)

if __name__ == "__main__":
    main()