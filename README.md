

# TextBO

This document provides detailed explanations of TextBO in "Bayesian optimization in language space: An eval-efficient AI self-improvement framework" (https://arxiv.org/abs/2511.12063). For the installation guide for the Twin-2k-500 dataset we utilize, please refer to https://github.com/TianyiPeng/Twin-2K-500-Mega-Study .

> **Note:** The scripts in this repository expect a `text_simulation/` directory with persona inputs and, for some workflows, pre-generated campaign assets. That directory is not included here, so you will need to generate from  https://github.com/TianyiPeng/Twin-2K-500-Mega-Study yourself (see **Required input folders** below).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the repository root with:

```bash
GOOGLE_API_KEY=...
GOOGLE_PROJECT_ID=...
GOOGLE_LOCATION=...
OPENAI_API_KEY=...            # Optional (TextBO-GPT.py)
OPENAI_TEXT_MODEL=gpt-5-mini  # Optional
OPENAI_VISION_MODEL=gpt-5-mini
NUM_AD_IMAGES=64              # Optional
MAX_RETRIES=5                 # Optional
TEST_MODE=false               # Optional
```

## Required input folders

Several scripts read from fixed paths. Ensure these exist before running:

* `campaign_output/<scenario>_campaign_output/`  
  * Generated ad images named like `<scenario>_ad_01.png`, `<scenario>_ad_02.png`, ...
  * A `generated_prompts.json` file created by `ad_gen.py`
* `text_simulation/text_simulation_input/`  
  * Persona prompt files named `pid_*_prompt.txt` (used by `DTTTS.py` and `ad_simulator.py`)
* `text_simulation/text_personas/`  
  * Persona summary files named `pid_*_mega_persona.txt` (used by `TextBO.py`)

## Core Optimization Algorithms

### `TextBO.py`
**Purpose**: TextBO with Gemini (Best-of-N Bayesian Optimization)

Implements the TextBO algorithm for optimizing prompts in language space using Gemini models. Key features:
- **Class**: `TextBOOptimizer`
- **Optimization Method**: Iteratively refines prompts through gradient-based exploration
- **Evaluation**: Uses persona-based feedback from digital twin data (train/test split)
- **Models**: Gemini 2.5 Flash for text generation and criticism, Imagen 4 for image generation
- **Output**: Optimization history with prompts, generated content, and evaluation scores

**Use Case**: Start with an initial prompt and optimize it based on digital twin persona feedback for maximum effectiveness.

### `TextBO-GPT.py`
**Purpose**: TextBO with OpenAI GPT (Multi-Model Implementation)

Similar to TextBO.py but uses both OpenAI and Gemini models:
- **Class**: `TextBOOptimizer`
- **Key Difference**: Uses OpenAI's GPT models (configurable via env: `OPENAI_TEXT_MODEL`, `OPENAI_VISION_MODEL`)
- **Evaluation**: Direct Gemini 2.5 Pro evaluation without persona-based feedback
- **Visual Pattern Analysis**: Analyzes top and bottom performing iterations to extract visual insights
- **Hybrid Approach**: Combines GPT for text generation with Gemini for image generation

**Use Case**: When you want to leverage OpenAI's GPT models with direct evaluation instead of persona simulation.

---

## Thompson Sampling Implementations

### `DTTTS.py`
**Purpose**: Digital Twin Thompson Sampling (Gemini-based)

Implements **Alternating Best-Worst Top-Two Thompson Sampling** for advertisement selection:
- **Class**: `AlternatingBestWorstTTTS`
- **Algorithm**: Multi-armed bandit with Dirichlet priors for each ad (arm)
- **Outcomes**: 5-point effectiveness scale (1=not effective to 5=highly effective)
- **Features**:
  - Beta parameter tuning (default: 0.5)
  - Confidence-based stopping (default: 95%)
  - Soft label updates with configurable weights
  - Monte Carlo sampling for confidence estimation
- **Integration**: Uses Gemini for ad effectiveness evaluation with digital twin personas

**Use Case**: Efficiently identify the best-performing advertisement from multiple candidates using minimal persona evaluations.

### `DTTTS-gemini.py`
**Purpose**: Digital Twin Thompson Sampling (Gemini variant)

Alternative implementation of the Thompson Sampling algorithm with Gemini-specific optimizations:
- **Class**: `AlternatingBestWorstTTTS`
- **Similar to DTTTS.py** but may have different evaluation or configuration strategies
- **Focus**: Optimized for Gemini API usage patterns


## Advertisement Generation & Simulation

### `ad_gen.py`
**Purpose**: Creative Brief-Based Ad Image Generator

Generates advertisement images using creative briefs as input:
- **Class**: `ImageGenerator`
- **Workflow**:
  1. Takes a creative brief describing the campaign
  2. Uses Gemini 2.5 Pro to generate detailed image prompts
  3. Creates images using Imagen 4.0 Ultra
  4. Applies brand-aligned, scroll-stopping visual principles
- **Key Prompt Requirements**:
  - No children/kids (Imagen 4 restriction)
  - Key message as foundation
  - Scene, action, composition guidelines
  - Stylistic qualities (photography style, lighting, color palette)
  - Logo placement
- **Configuration**:
  - `GOOGLE_API_KEY`, `GOOGLE_PROJECT_ID`, `GOOGLE_LOCATION`
  - `NUM_AD_IMAGES` (default: 64)
  - `MAX_RETRIES` (default: 5)

**Use Case**: Generate multiple ad image variations from a campaign creative brief for testing.

### `ad_simulator.py`
**Purpose**: Digital Twin Ad Effectiveness Prediction

Simulates how digital twin personas respond to advertisement images:
- **Function**: `predict_ad_click_with_gemini()`
- **Input**:
  - Persona survey response data
  - Advertisement image
- **Output**: JSON with:
  - `reasoning`: Detailed explanation of prediction
  - `effectiveness_score`: 1-5 scale (1=extremely unlikely to click, 5=extremely likely)
- **Model**: Gemini 2.5 Flash for multimodal analysis (text + image)
- **Context**: Mobile Instagram native ads

**Effectiveness Scale**:
- 1: Persona would actively ignore/be annoyed
- 2: Would scroll past without thought
- 3: Mediocre - uncertain if they'd click
- 4: Intrigued - good chance of clicking
- 5: Ideal target - click almost certain

**Use Case**: Evaluate how well an ad resonates with specific persona profiles before real-world deployment.


## Running the scripts

### Download the dataset
```bash
python download_dataset.py
```
This writes data into `data/mega_persona_json/` and `data/mega_persona_summary_text/`.

### Generate ads
```bash
python ad_gen.py
```
*Update `SCENARIO_NAME` and the creative brief in `ad_gen.py` before running.*

### Simulate ad performance
```bash
python ad_simulator.py
```
*Update `SCENARIO_NAME` in `ad_simulator.py` to point at the correct campaign output.*

### Run DTTTS selection
```bash
python DTTTS.py        # includes personas + images
python DTTTS-gemini.py # uses Gemini logprobs only
```
*Update `SCENARIO_NAME` in each script as needed. Use `--test` to reduce timesteps.*

### Run TextBO optimization
```bash
python TextBO.py --test
python TextBO.py --gepa
python TextBO.py --parallel 3
```
*`TextBO.py` looks for DTTTS outputs in `campaign_output/<scenario>_campaign_output/` and personas in `text_simulation/text_personas/`.*

```bash
python TextBO-GPT.py --test
```


## Configuration Files after you install the Twin-2k-500 dataset

### `text_simulation/configs/openai_config.yaml`
OpenAI API configuration for simulations:
```yaml
provider: "openai"
model_name: "gpt-4.1-mini-2025-04-14"  # or fine-tuned model
temperature: 0.0
max_tokens: 16384
max_retries: 10
num_workers: 300  # Adjust based on rate limits
force_regenerate: false
max_personas: 5  # Testing limit
```

### `text_simulation/configs/gemini_config.yaml`
Gemini API configuration for simulations:
```yaml
provider: "gemini"
num_workers: 30
max_retries: 3
llm_config:
  model_name: "gemini-2.5-pro-preview-06-05"
  temperature: 0.0
  max_tokens: 18192
```

```

### Advertisement Optimization Workflow
```bash
# 1. Generate ad variations from creative brief
poetry run python ad_gen.py

# 2. Evaluate ads using digital twin personas
poetry run python ad_simulator.py

# 3. Run Thompson Sampling to find best ad
poetry run python DTTTS.py

# OR: Use TextBO to optimize ad prompts
poetry run python TextBO.py --initial-prompt "Your starting prompt"
```

---

## Key Differences Between Files

| File | Primary Model | Evaluation Method | Use Case |
|------|--------------|-------------------|----------|
| `TextBO.py` | Gemini | Persona feedback (train/test split) | Persona-driven prompt optimization |
| `TextBO-GPT.py` | OpenAI GPT + Gemini | Direct Gemini evaluation | Multi-model prompt optimization |
| `DTTTS.py` | Gemini | Thompson Sampling with personas | Best ad selection (standard) |
| `DTTTS-gemini.py` | Gemini | Thompson Sampling | Gemini-optimized variant |
| `ad_gen.py` | Gemini + Imagen | Creative brief → Images | Bulk ad image generation |
| `ad_simulator.py` | Gemini 2.5 Flash | Persona + image → score | Individual ad evaluation |

---

## Environment Variables Summary

**Required for most scripts**:
- `GOOGLE_API_KEY` - Gemini API key
- `GOOGLE_PROJECT_ID` - GCP project ID
- `GOOGLE_LOCATION` - GCP region (e.g., "us-central1")

**Optional OpenAI**:
- `OPENAI_API_KEY` - For OpenAI GPT models
- `OPENAI_TEXT_MODEL` - Model name (default: "gpt-5-mini")
- `OPENAI_VISION_MODEL` - Vision model (default: "gpt-5-mini")

**Optional Configuration**:
- `TEST_MODE` - Enable test mode (true/false)
- `NUM_AD_IMAGES` - Number of ads to generate (default: 64)
- `MAX_RETRIES` - API retry attempts (default: 5)

Store these in a `.env` file in the repository root (already gitignored for security).
