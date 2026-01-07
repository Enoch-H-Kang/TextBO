

# TextBO

This document provides detailed explanations of TextBO in "Bayesian optimization in language space: An eval-efficient AI self-improvement framework" (https://arxiv.org/abs/2511.12063). For the installation guide for the Twin-2k-500 dataset we utilize, please refer to https://github.com/TianyiPeng/Twin-2K-500-Mega-Study

## Research Context

This repository implements methods for **Bayesian Optimization in Language Space** for the Twin-2K-500 dataset. The code combines:
1. **Advertisement Optimization**: Using TextBO (Text-Based Bayesian Optimization) to generate and optimize advertising content
2. **Thompson Sampling**: Multi-armed bandit algorithms for efficient ad selection

---

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


## Text Simulation Pipeline (text_simulation/ directory)

### `text_simulation/convert_persona_to_text.py`
**Purpose**: Convert JSON Persona Data to Text Format

Transforms structured JSON persona data into readable text format suitable for LLM consumption:
- Extracts survey responses from JSON
- Formats as human-readable persona profiles
- Used as input for digital twin simulations

### `text_simulation/batch_convert_personas.py`
**Purpose**: Batch Process Multiple Personas

Wrapper script to convert multiple persona JSON files to text format in parallel:
- Processes entire directories of persona files
- Efficient bulk conversion
- Progress tracking with tqdm

### `text_simulation/convert_question_json_to_text.py`
**Purpose**: Convert Question Data to Text Format

Converts survey question JSON structures into text format:
- Extracts question text, options, and metadata
- Formats for LLM simulation inputs
- Maintains question structure and response options

### `text_simulation/create_text_simulation_input.py`
**Purpose**: Combine Personas with Questions

Creates paired input files for simulations:
- **Combines**: Text persona profiles + Survey questions
- **Output**: Simulation-ready input files
- **Structure**: Each file contains persona context + new questions to answer
- **Directory**: `text_simulation_input/`

### `text_simulation/run_LLM_simulations.py`
**Purpose**: Execute Digital Twin Simulations

Main simulation runner that generates digital twin responses:
- **Input**: Combined persona-question files
- **Process**: Uses LLM to answer questions as if the persona is responding
- **Models**: Configurable (OpenAI GPT or Gemini)
- **Configuration Files**:
  - `configs/openai_config.yaml`
  - `configs/gemini_config.yaml`
- **Features**:
  - Concurrent workers for parallel processing
  - Retry logic for failed API calls
  - Token limit management
  - Force regenerate option
- **Output**: Simulated responses in `text_simulation_output/`

### `text_simulation/llm_helper.py`
**Purpose**: LLM Interaction Helper Functions

Utility library for LLM API interactions:
- **Functions**:
  - OpenAI API client setup (requires `OPENAI_API_KEY`)
  - Gemini API client setup (requires `GOOGLE_API_KEY`)
  - Rate limiting and retry logic
  - Response parsing and validation
  - Error handling for API failures
- **Models Supported**: OpenAI GPT, Google Gemini
- **Configuration**: Loads parameters from YAML config files

### `text_simulation/postprocess_responses.py`
**Purpose**: Analyze and Validate Simulation Results

Post-processing and analysis of digital twin simulation outputs:
- **Verification**: Checks response quality and completeness
- **Analysis**: Computes statistics on simulation accuracy
- **Comparison**: Validates digital twin responses against ground truth
- **Reporting**: Generates summary statistics and error reports
- **Quality Control**: Identifies failed or inconsistent simulations

---

## Configuration Files

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

---

## Workflow Examples

### Complete Digital Twin Simulation Workflow
```bash
# 1. Download dataset
poetry run python download_dataset.py

# 2. Convert personas to text (done via pipeline)
# 3. Convert questions to text (done via pipeline)
# 4. Create simulation inputs (done via pipeline)

# 5. Run simulations
./scripts/run_pipeline.sh --max_personas=5  # Test with 5 personas
./scripts/run_pipeline.sh                   # Full run (2058 personas)
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
| `DTTTS_r.py` | Gemini | Thompson Sampling | Refined/enhanced version |
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

