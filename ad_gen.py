from google import genai
from google.genai import types
import google.generativeai as legacy_genai
from PIL import Image
import os
import time
from typing import List, Dict
import json
from dotenv import load_dotenv


class ImageGenerator:
    def __init__(
        self,
        api_key: str,
        project_id: str,
        location: str,
        scenario_name: str,
        creative_brief: str,
        output_dir: str,
        max_retries: int = 5,
    ):
        """Initializes the ad image generator using Vertex AI.

        Args:
            api_key: Your Google Gemini API key (for the text model).
            project_id: Your Google Cloud Project ID.
            location: Your Google Cloud Project Location (e.g., "us-central1").
            scenario_name: The name of the campaign or product (e.g., "GreenBite").
            creative_brief: A string containing the creative brief for the campaign.
            output_dir: The directory where images and logs will be saved.
            max_retries: Maximum number of retry attempts for failed image generation.
        """
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.scenario_name = scenario_name
        self.creative_brief = creative_brief
        self.output_dir = output_dir
        self.max_retries = max_retries

        # --- Configure APIs ---
        # Vertex AI client for Imagen 4
        self.client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )
        #self.image_model = "imagen-4.0-generate-preview-06-06"
        self.image_model = "imagen-4.0-ultra-generate-preview-06-06"

        # Legacy genai client for Gemini text model
        legacy_genai.configure(api_key=api_key)
        self.text_model = legacy_genai.GenerativeModel("gemini-2.5-pro")

        os.makedirs(output_dir, exist_ok=True)

    def generate_image_prompt(self, image_number: int) -> str:
        """Asks Gemini to produce a detailed image prompt based on the creative brief."""

        prompt_requirements = """
        Generate a creative, structured and descriptive prompt for a generative AI model (e.g., Imagen) that will produce a brand-aligned, and scroll-stopping advertisement image suitable for an Instagram feed.
        A successful prompt must be constructed using the following components and principles:
        0. (Most important!) Prompt NEVER includes any description of kids. Kids are not allowed to generate in Imagen 4. Even if creative_brief includes description of "family" or "kids", never include kids in the prompt. Only create adults if humans are included.
        1.  Key Message (The "Why"): This is the foundational element that guides all other components. It defines the core idea or feeling the ad must communicate. Before writing the rest of the prompt, clearly articulate the message the ad will deliver. This message will act as the "North Star" for all subsequent creative choices. It may be desirable to put the message as the text overlay.
        2. Core Components (The "What"):
        - This should completely depend on the key message. 
        - Scene & Environment: Based on the key message, establish a clear, relatable setting that aligns with the brand's lifestyle appeal. 
        - Action & Narrative: Based on the key message and the scene, describe a dynamic but clear action or interaction to create a sense of a captured moment and tell a micro-story.
        - Composition & Framing: Specify the camera shot, angle, and framing (e.g., "low-angle shot," "dynamic medium shot," "close-up on the shoe").
        3. Stylistic Qualities (The "How"):
        - This should completely depend on the key message and the scene.
        - Photography Style: Define the overall aesthetic (e.g., "photorealistic," "cinematic," "professional product photography," "lifestyle action shot").
        - Lighting: Be specific about the lighting to set the mood (e.g., "warm golden hour light," "bright morning sun," "dramatic side-lighting").
        - Color Palette & Tone: Guide the color scheme and emotional feel (e.g., "vibrant and energetic colors," "empowering and motivational tones," "clean and modern palette").
        - Atmosphere & Feeling: Aim to evoke a specific feeling aligned with the brand (e.g., "a feeling of effortless performance," "an atmosphere of vibrant energy," "a sense of supreme comfort").
        4. What to Avoid:
        - Vagueness: Use specific, descriptive terms instead of "nice" or "good."
        - Contradictory Elements: Ensure all elements work together harmoniously.
        - Over-stuffing: Focus on a single, clear message without too many competing objects.
        5. What to include:
        - Logo: create a logo based on creative brief, and naturally place it.
        - Text: If you are including a text message, prompt for "negative space for text overlay".
        """

        prompt_request = f"""
        Based on this creative brief:
        {self.creative_brief}

        Create a detailed, specific image generation prompt for ad image #{image_number}.

        Requirements:\n{prompt_requirements}

        Return **only** the image generation prompt with no additional explanation.
        """

        try:
            response = self.text_model.generate_content(
                prompt_request,
                generation_config={"temperature": 2.0}
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating prompt for image {image_number}: {e}")
            return "Error"


    def generate_image(self, prompt: str, image_number: int, max_retries: int = None) -> str:
        """Generates a single image using Imagen 4 on Vertex AI with retry logic."""
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"  🔄 Retry attempt {attempt} for image {image_number}")
                    time.sleep(5)  # Longer delay between retries
                else:
                    print(f"Generating image {image_number} → {prompt[:90]}…")

                response = self.client.models.generate_images(
                    model=self.image_model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                        person_generation="ALLOW_ADULT",
                        safety_filter_level="BLOCK_ONLY_HIGH",
                    ),
                )
                
                # Log the full response for debugging
                print(f"  📋 FULL IMAGEN 4 API RESPONSE for image {image_number} (attempt {attempt + 1}):")
                try:
                    # Convert response to dict for JSON serialization
                    response_dict = {}
                    # Skip deprecated Pydantic internal attributes
                    skip_attrs = {'model_computed_fields', 'model_fields', 'model_config', 'model_extra', 'model_fields_set'}
                    for attr in dir(response):
                        if not attr.startswith('_') and attr not in skip_attrs:
                            try:
                                value = getattr(response, attr)
                                if not callable(value):
                                    response_dict[attr] = str(value)
                            except Exception:
                                pass
                    print(f"     {json.dumps(response_dict, indent=6, default=str)}")
                except Exception as e:
                    print(f"     Raw response object: {response}")
                    print(f"     Error serializing response: {e}")
                print(f"  📋 END FULL RESPONSE")

                if response and getattr(response, "generated_images", None):
                    if (response.generated_images and 
                        len(response.generated_images) > 0 and 
                        hasattr(response.generated_images[0], 'image') and
                        hasattr(response.generated_images[0].image, '_pil_image') and
                        response.generated_images[0].image._pil_image is not None):
                        
                        # Use the scenario_name for the filename
                        filename = f"{self.scenario_name.lower()}_ad_{image_number:02d}.png"
                        filepath = os.path.join(self.output_dir, filename)

                        response.generated_images[0].image._pil_image.save(filepath)

                        print(f"  ✓ Saved → {filepath}")
                        return filepath
                    else:
                        print(f"  ❌ Image bytes not set for image {image_number} (attempt {attempt + 1}/{max_retries}) - technical issue")
                        # Try to extract additional response details
                        if hasattr(response, 'error') and response.error:
                            print(f"     Response error: {response.error}")
                        if hasattr(response, 'status') and response.status:
                            print(f"     Response status: {response.status}")
                        if hasattr(response, 'message') and response.message:
                            print(f"     Response message: {response.message}")
                        if attempt < max_retries - 1:
                            continue
                else:
                    print(f"  ⚠️ No images returned from API (attempt {attempt + 1}/{max_retries}) - possible rate limit or technical issue")
                    # Try to extract error details from empty/null response
                    if response and hasattr(response, 'error') and response.error:
                        print(f"     Response error: {response.error}")
                    if response and hasattr(response, 'status') and response.status:
                        print(f"     Response status: {response.status}")
                    if response and hasattr(response, 'message') and response.message:
                        print(f"     Response message: {response.message}")
                    if attempt < max_retries - 1:
                        continue
                        
            except Exception as e:
                print(f"  ❌ IMAGEN 4 ERROR for image {image_number} (attempt {attempt + 1}/{max_retries}):")
                print(f"     Error type: {type(e).__name__}")
                print(f"     Error message: {str(e)}")
                
                # Try to extract additional error details from the exception
                if hasattr(e, 'details'):
                    print(f"     Error details: {e.details}")
                if hasattr(e, 'code'):
                    print(f"     Error code: {e.code}")
                if hasattr(e, 'status_code'):
                    print(f"     Status code: {e.status_code}")
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    print(f"     HTTP status: {e.response.status_code}")
                if hasattr(e, 'args') and len(e.args) > 1:
                    print(f"     Additional args: {e.args[1:]}")
                
                if attempt < max_retries - 1:
                    continue
        
        print(f"  💀 Failed to generate image {image_number} after {max_retries} attempts")
        return None

    def _save_generation_log(self, images_info: List[Dict]):
        """Saves a final JSON log with all generated image data."""
        log_path = os.path.join(self.output_dir, "generation_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    # Use the scenario_name for the campaign title
                    "campaign": f"{self.scenario_name} Ad Campaign",
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_images": len(images_info),
                    "images": images_info,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"📋 Final log saved → {log_path}")

    # ------------------------------------------------------------------
    # STEP 1: GENERATE AND SAVE PROMPTS
    # ------------------------------------------------------------------
    def generate_and_save_prompts(self, num_prompts: int = 10) -> str:
        """Generates N prompts and saves them to a single JSON file."""
        print(f"--- Step 1: Generating {num_prompts} Prompts ---")
        prompts_data = []
        for i in range(1, num_prompts + 1):
            print(f"  [{i}/{num_prompts}] Crafting prompt...")
            prompt_text = self.generate_image_prompt(i)
            if prompt_text != "Error":
                prompts_data.append({"image_number": i, "prompt": prompt_text})
            time.sleep(1)  # Respect text model rate limits

        prompts_filepath = os.path.join(self.output_dir, "generated_prompts.json")
        with open(prompts_filepath, "w", encoding="utf-8") as f:
            json.dump(prompts_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Successfully generated {len(prompts_data)} prompts.")
        print(f"📁 Prompts saved to: {prompts_filepath}")
        return prompts_filepath

    # ------------------------------------------------------------------
    # STEP 2: GENERATE IMAGES FROM SAVED PROMPTS
    # ------------------------------------------------------------------
    def generate_images_from_file(self, prompts_filepath: str) -> List[Dict]:
        """Generates images based on a JSON file of prompts."""
        try:
            with open(prompts_filepath, "r", encoding="utf-8") as f:
                prompts_to_process = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Prompts file not found at {prompts_filepath}")
            return []

        print(
            f"\n--- Step 2: Generating Images from {os.path.basename(prompts_filepath)} ---"
        )

        generated_images_info = []
        total_prompts = len(prompts_to_process)

        for idx, prompt_data in enumerate(prompts_to_process):
            image_num = prompt_data.get("image_number")
            prompt_text = prompt_data.get("prompt")

            print(f"\n[{idx + 1}/{total_prompts}] Processing image #{image_num}")

            if not prompt_text:
                print(f"  ⚠️ Skipping image #{image_num} due to missing prompt text.")
                continue

            image_path = self.generate_image(prompt=prompt_text, image_number=image_num)

            generated_images_info.append(
                {
                    "image_number": image_num,
                    "prompt": prompt_text,
                    "image_path": image_path,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            time.sleep(3)  # Respect image model rate limits

        self._save_generation_log(generated_images_info)
        print(f"\n✅ Finished generating {len(generated_images_info)} images!")
        print(f"📁 All outputs saved in: {self.output_dir}")
        return generated_images_info


# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------
def main():
    """Main function to run the two-step ad generation process."""
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    LOCATION = os.getenv("GOOGLE_LOCATION")

    if not all([API_KEY, PROJECT_ID, LOCATION]):
        print("❌ One or more required environment variables are missing.")
        print("   Please set GOOGLE_API_KEY, GOOGLE_PROJECT_ID, and GOOGLE_LOCATION in a .env file.")
        return

    # ------------------------------------------------------------------
    # 1. SPECIFY SCENARIO NAME
    #    Change this value to generate ads for a different product.
    # ------------------------------------------------------------------
    SCENARIO_NAME = "oasis"

    print(f"🚀 Starting campaign generation for: {SCENARIO_NAME}")
    
    # 2. DEFINE THE CREATIVE BRIEF (Dynamically uses the SCENARIO_NAME)
    creative_brief = f"""
   Product: "Oasis Eco-Lodge," a secluded, luxury resort operating in harmony with its natural surroundings. Background: The luxury travel market often equates opulence with excess. A growing segment of affluent travelers seeks experiences that are both exclusive and responsible. The Challenge: Redefine luxury as a seamless integration with nature, promising an experience that is restorative for both the guest and the environment. Target Audience: High-income travelers (40-65) seeking unique, restorative experiences that are both luxurious and environmentally conscious. Core Insight: For those who have everything, true luxury is not more things, but a deeper connection to something pure and serene. Single-Minded Proposition (SMP): Rediscover tranquility in a luxury that respects nature. Reasons to Believe (RTB): Secluded private bungalows; farm-to-table dining with locally sourced ingredients; carbon-neutral operations. Desired Response: Think: "This is a truly special escape, not just another five-star hotel." Feel: Serene, exclusive, and rejuvenated. Do: Book a stay. Brand Personality & Tone: Elegant, peaceful, and understated.
    """

    NUM_IMAGES = int(os.getenv("NUM_AD_IMAGES", "64"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    OUTPUT_DIR = os.path.join("campaign_output", f"{SCENARIO_NAME.lower()}_campaign_output")

    # Initialize the generator with scenario-specific details
    generator = ImageGenerator(
        api_key=API_KEY,
        project_id=PROJECT_ID,
        location=LOCATION,
        scenario_name=SCENARIO_NAME,
        creative_brief=creative_brief,
        output_dir=OUTPUT_DIR,
        max_retries=MAX_RETRIES,
    )

    try:
        # Step 1: Check if prompts already exist, if not generate them
        prompts_json_path = os.path.join(OUTPUT_DIR, "generated_prompts.json")
        
        if os.path.exists(prompts_json_path):
            print(f"📁 Found existing prompts file: {prompts_json_path}")
            print("⏭️  Skipping prompt generation...")
        else:
            # Generate and store the JSON file with N prompts
            prompts_json_path = generator.generate_and_save_prompts(
                num_prompts=NUM_IMAGES
            )

        # Step 2: Generate images based on the created JSON file
        generator.generate_images_from_file(prompts_filepath=prompts_json_path)

    except Exception as e:
        print(f"\nA fatal error occurred: {e}")


if __name__ == "__main__":
    print("🤖 Ad Image Generator (Vertex AI - Imagen 4)")
    print("=" * 60)
    main()
