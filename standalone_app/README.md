# VLM Detections - Standalone Testing Application

This standalone application provides a Gradio-based interface for rapid prototyping, testing, and experimentation with vision-language models for object detection, entity property recognition, and relation detection.

**Note:** This is the development/testing version with support for multiple adapters. For production ROS deployments, see the main README in the parent directory.

## Features

- **Interactive Testing UI**: Gradio interface for real-time model evaluation
- **Multi-Model Support**: Test various VLM adapters (OWL-ViT, GroundingDINO, Florence-2, Qwen-VL, InternVL, OpenAI Vision, T-Rex, Cosmos-Reason1)
- **Flexible Input**: Webcam, image upload, or video processing
- **Prompt Experimentation**: Custom prompts or template-based with per-model history
- **Parameter Tuning**: Dynamic generation parameter controls (temperature, max_tokens, etc.)
- **Batch Processing**: Run experiments over image/video datasets with parameter sweeps
- **Visualization**: Real-time detection overlays and entity property displays

## Quick Start

### Installation

1. **Create Python environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run standalone app:**
```bash
python -m vlm_detections.app
# Or from the standalone_app directory:
python app.py
```

4. **Access the interface:**
   - Open the URL printed in the terminal (typically `http://127.0.0.1:7860`)
   - Select webcam or upload image/video
   - Choose model and configure parameters
   - Test prompts and visualize results

**Notes:**
- First run downloads model weights (may take several minutes)
- GPU (CUDA) highly recommended for real-time performance
- CPU fallback available but significantly slower

## Interface Overview

### Main Tabs

**Detection Tab:**
- Input source selection (webcam/upload)
- Model and variant selection
- Confidence threshold adjustment
- Detection visualization with bounding boxes

**Gestures & Emotions Tab:**
- Entity property detection display
- Person gesture recognition (waving, pointing, etc.)
- Emotion recognition (happy, sad, angry, etc.)

**Relations Tab:**
- Entity relation detection display
- Spatial relations (looking_at, pointing_at, etc.)
- Social relations (talking_to, standing_next_to, etc.)

**Chat Tab** (for supported models):
- Multi-turn conversation interface
- Image/video context retention
- Text-only chat mode

### Configuration Controls

**Model Selection:**
- Dropdown with all registered adapters
- Variant selection (model size/checkpoint)
- Device choice (auto/cuda/cpu)

**Prompt Management:**
- System prompt templates (model-specific)
- User prompt templates with variables
- Custom prompt override
- Per-model prompt history

**Generation Parameters:**
- Temperature (sampling randomness)
- Max tokens (output length)
- Top-p (nucleus sampling)
- Model-specific parameters (num_beams, do_sample, etc.)

**Florence-2 Specific:**
- Task selection (OD, CAPTION, OCR, REGION_CAPTION, etc.)
- Task-specific output formatting

## Prompt Management

The standalone app includes a comprehensive prompt management system via `PromptManager` that provides template-based prompt rendering with variable substitution.

### Prompt Templates

Prompts are organized in `config/prompts/` with per-model YAML files:

**Structure:**
```yaml
# config/prompts/qwen2_5_vl.yaml
system:
  default: "You are a vision AI. Respond only with valid JSON."
  reasoning: "Think step-by-step. Output your reasoning, then JSON."
user:
  detection: "Detect these objects: {classes}. Image size: {width}x{height}."
  gesture: "Identify gestures performed by people in the image."
```

### Variable Substitution

User prompts support the following variables:
- `{classes}`: Comma-separated class list
- `{width}`: Image width in pixels
- `{height}`: Image height in pixels
- `{threshold}`: Confidence threshold

**Example rendering:**
```python
from prompt_manager import PromptManager

PromptManager.load_all()
PromptManager.select("Qwen2.5-VL", "default", "detection", override=False, "")
prompt = PromptManager.render_user_prompt(
    "Qwen2.5-VL", 
    ["person", "cup"], 
    custom_text="",
    width=1920, 
    height=1080, 
    threshold=0.25
)
# Result: "Detect these objects: person, cup. Image size: 1920x1080."
```

### UI Integration

The Gradio interface provides:
- **System prompt dropdown**: Select from model-specific templates
- **User prompt dropdown**: Select task-specific templates
- **Override checkbox**: Use custom text instead of template
- **Custom prompt input**: Free-form text when override enabled
- **Per-model history**: Automatically saves selections in `session_state.json`

### Cosmos-Reason1 Prompts

Cosmos models use a special prompt structure stored in `config/cosmos/prompts/`:
- Reasoning-focused system prompts
- Structured output formatting
- Multi-step reasoning chains

## Batch Processing

Run experiments over image/video datasets with automated parameter sweeps.

### Create Experiment Config

Example configuration (`experiment_config.json`):

```json
{
  "input": {
    "type": "folder",
    "folder": "assets/test_images",
    "include_subdirs": true
  },
  "model": {
    "name": "Qwen2.5-VL",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "device": "cuda",
    "threshold": 0.25
  },
  "class_sets": [
    ["person", "cup", "laptop"],
    ["gesture", "emotion"]
  ],
  "instruction_prompts": [
    "Detect objects and return JSON with bbox_2d, label, score.",
    "Identify gestures and emotions. Return JSON with entity, property, value."
  ],
  "generation_param_sweeps": {
    "temperature": {"values": [0.0, 0.5, 1.0]},
    "max_new_tokens": {"min": 200, "max": 600, "step": 200}
  },
  "output": {
    "base_dir": "experiments/results",
    "save_visualizations": true,
    "save_raw_text": true,
    "results_filename": "results.json"
  }
}
```

### Run Experiment

```bash
python batch_infer.py --config experiment_config.json
```

### Output Structure

```
experiments/results/20250120_143022/
├── results.json                    # Aggregated results
├── viz/                            # Annotated images
│   ├── image001_temp0.0_tokens200.jpg
│   └── image002_temp0.5_tokens400.jpg
└── config.json                     # Experiment snapshot
```

### Legacy CLI Mode

Quick inference without config file:

```bash
python batch_infer.py assets/captures \
    --prompts "person, laptop, cup" \
    --model "Qwen2.5-VL" \
    --threshold 0.25
```

## Supported Models

| Model | Detection | Properties | Relations | Notes |
|-------|-----------|------------|-----------|-------|
| OWL-ViT | ✓ | - | - | Fast zero-shot detection |
| GroundingDINO | ✓ | - | - | High-accuracy grounding |
| Florence-2 | ✓ | - | - | Multi-task (OD, caption, OCR) |
| Qwen2.5-VL | ✓ | ✓ | ✓ | Generative VLM, JSON output |
| Qwen3-VL | ✓ | ✓ | ✓ | Latest Qwen vision model |
| InternVL3.5 | ✓ | ✓ | ✓ | High-resolution VLM |
| OpenAI Vision | ✓ | ✓ | ✓ | GPT-4V/4o API (requires key) |
| T-Rex | ✓ | - | - | Visual prompting detection |
| Cosmos-Reason1 | ✓ | ✓ | ✓ | NVIDIA reasoning VLM |

## State Persistence

The app uses `session_state.json` to persist configuration between runs:

- Model and variant selection
- Prompt selections and custom text
- Generation parameters
- Threshold and other settings

**Location:** Same directory as `app.py`

**Manual editing:** You can edit this file directly to set default configurations.

## Performance Tips

**GPU Memory Management:**
- Close unused models via the UI
- Use smaller variants for development (e.g., Qwen2.5-VL-2B vs. 7B)
- Enable model quantization where supported

**Inference Speed:**
- Lower `max_new_tokens` for faster responses
- Set `temperature: 0.0` for deterministic, faster generation
- Use specialized models (OWL-ViT, GroundingDINO) for pure detection tasks

**Batch Processing:**
- Disable `save_visualizations` for large experiments
- Use `separate_per_param_combo: false` for single output file
- Consider parameter sweep size vs. time tradeoffs

## Troubleshooting

**Model fails to load:**
```
RuntimeError: CUDA out of memory
```
→ Try smaller model variant or device: "cpu"

**No detections output:**
- Lower threshold (try 0.1 or 0.05)
- Check prompt format matches model expectations
- Verify classes in prompt match objects in image

**JSON parsing errors:**
```
WARNING: No valid JSON found in model output
```
→ Some models need specific prompt instructions for JSON output. Check model-specific prompt templates.

**Webcam not working:**
- Check camera permissions
- Try different camera index in code
- Test with image upload first

**Slow inference on CPU:**
- Expected behavior - VLMs are compute-intensive
- Consider using API-based models (OpenAI Vision) for CPU-only systems
- Use lightweight models (OWL-ViT) for faster CPU inference

## Advanced Usage

### Adding Custom Adapters

See main README for adapter interface documentation.

For private adapters, place them in a separate directory and import them in `app.py`:

```python
# In app.py, after existing imports
import sys
sys.path.insert(0, '/path/to/private_adapters')

from my_private_adapter import MyAdapter

# Register in MODEL_REGISTRY
MODEL_REGISTRY["MyModel"] = lambda model_id: MyAdapter(model_id=model_id)
```

### Custom Prompt Templates

Edit `prompts_dictionary.yaml` to add model-specific prompt templates:

```yaml
models:
  "MyModel":
    system:
      default: "You are a vision AI assistant."
      reasoning: "Think step-by-step before answering."
    user:
      detection: "Detect: {classes}. Image: {width}x{height}"
      custom: "Your custom prompt template here"
```

### Session State Schema

```json
{
  "model": "Qwen2.5-VL",
  "model_variant": "Qwen/Qwen2.5-VL-7B-Instruct",
  "threshold": 0.25,
  "classes_text": "person, cup",
  "prompt_text": "",
  "florence_task": "OD",
  "override_user_prompt": false,
  "device": "auto",
  "prompt_selections": {
    "Qwen2.5-VL": {
      "system": "detect_json",
      "user": "detect_json",
      "override": false,
      "override_text": ""
    }
  },
  "generation_params": {
    "Qwen2.5-VL": {
      "max_new_tokens": 400,
      "temperature": 0.0,
      "top_p": 1.0
    }
  }
}
```

## Development

**Project Structure:**
```
standalone_app/
├── app.py              # Main Gradio application
├── batch_infer.py      # Batch processing script
└── README.md           # This file
```

**Key Dependencies:**
- `gradio`: Web UI framework
- `transformers`: Hugging Face model library
- `torch`: PyTorch deep learning
- `opencv-python`: Image processing
- `pillow`: Image handling

## License

[Specify your license]

## Support

For issues, questions, or contributions:
- Check main project README
- Open GitHub issues
- Contact maintainers

---

**Related Documentation:**
- Main package README: `../README.md`
- ROS integration guide: `../README.md#ros-2-integration`
- Adapter development: `../README.md#adding-new-models`
