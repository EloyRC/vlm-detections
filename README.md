# VLM Detections - ROS 2 Vision-Language Perception Package

A modular ROS 2 package for vision-language model (VLM) based perception, supporting zero-shot object detection, entity property recognition (gestures, emotions), and relation detection.

**Key Features:**
- **Modular ROS 2 Architecture**: Separate nodes for prompt management and inference
- **Plugin-Based Adapters**: Dynamically load vision model adapters via YAML configuration
- **OpenAI-Compatible API Support**: Included adapter works with any OpenAI-compatible endpoint
- **Multi-Output Parsing**: Extracts bounding boxes, gestures, emotions, and spatial relations
- **Configuration-Driven**: YAML-based model selection and parameter tuning
- **Production-Ready**: Real-time performance metrics, diagnostics, and monitoring GUI

## Quick Start

### Installation

1. **Install ROS 2 dependencies:**
```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-vision-msgs
```

2. **Set up Python environment:**

The package requires Python dependencies that should be installed in a virtual environment to avoid conflicts with system packages.

```bash
# Create and activate virtual environment
cd /path/to/vlm-detections
python3 -m venv vlm_venv
source vlm_venv/bin/activate

# Install package dependencies
pip install -r requirements.txt

# Install colcon for building
pip install colcon-common-extensions lark
```

**Note:** The base package only requires minimal dependencies. If using the standalone testing app (see `standalone_app/README.md`), additional dependencies like `transformers` and `torch` may be needed.

3. **Build the package:**
```bash
cd /path/to/ros2_workspace

# Build using the virtual environment's Python
/path/to/vlm-detections/vlm_venv/bin/python -m colcon build --packages-select vlm_detections perception_pipeline_msgs --symlink-install

source install/setup.bash
```

4. **Configure your model:**

Edit `config/model_config.yaml`:
```yaml
model: "OpenAI Vision (API)"
model_variant: "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
threshold: 0.25
device: "auto"
generation_params:
  max_new_tokens: 400
  temperature: 0.0
```

Configure API endpoint in `config/adapters_config.yaml`:
```yaml
adapters:
  "OpenAI Vision (API)":
    module: "vlm_detections.adapters.openai_vision_adapter"
    class: "OpenAIVisionAdapter"
    constructor_args:
      base_url: "http://your-api-endpoint:8000/v1"
```

5. **Launch the detection node and prompt manager nodes:**
```bash
# Activate your virtual environment first
cd /path/to/vlm-detections
source vlm_venv/bin/activate

# Set API key (optional)
export OPENAI_API_KEY=your_api_key

# Launch
ros2 launch vlm_detections vlm_node.launch.py
```

6. **Launch the monitoring GUI (optional):**

In a separate terminal:
```bash
# Activate your virtual environment
cd /path/to/vlm-detections
source vlm_venv/bin/activate

# Launch GUI
ros2 launch vlm_detections vlm_gui.launch.py
```

The GUI provides real-time visualization of detections, performance metrics, and control to pause/resume inference.

## ROS 2 Nodes

The package provides three main ROS 2 nodes that work together:

### 1. PromptManagerNode (C++, `vlm_prompt_manager_cpp`)

**Purpose:** Manages prompt templates and publishes images with prompts for inference.

**Subscribes:**
- `/camera/image_raw` (or configured topic) - `sensor_msgs/CompressedImage`
- `/people` - `strawberry_ros_msgs/People` (optional, for annotation)
- `/faces` - `strawberry_ros_msgs/Faces` (optional, temporary solution)

**Publishes:**
- `/vlm_prompts` - `perception_pipeline_msgs/VLMPrompts`
  - Batched images with system/user prompts
  - Supports multiple prompt templates per image
- `/vlm_debug_image` - `sensor_msgs/Image` (debug visualization)

**Parameters:**
- `input_image_topic` (string, default: `/camera/image_raw`) - Input image topic
- `people_topic` (string, default: `/people`) - People detection topic for annotation
- `faces_topic` (string, default: `/faces`) - Faces detection topic for annotation
- `output_topic` (string, default: `/vlm_prompts`) - Output prompts topic
- `fps` (double, default: 5.0) - **Image sampling rate in Hz**
- `batch_capacity` (int, default: 1) - **Number of images to batch together**
- `prompts_dictionary` (string, default: "") - Path to prompts YAML file
- `enable_people_annotation` (bool, default: false) - Annotate images with people bounding boxes
- `enable_faces_annotation` (bool, default: false) - Annotate images with face bounding boxes (temporary)
- `people_sync_tolerance` (double, default: 0.2) - Time tolerance in seconds for syncing people data
- `debug_image_topic` (string, default: `/vlm_debug_image`) - Debug visualization topic

**Timing Configuration:**

The node provides fine-grained control over inference timing through two parameters:

- **fps**: Image sampling frequency
  - Higher fps = more frequent sampling = lower sample period
  - Example: fps=10.0 → sample every 0.1s
  
- **batch_capacity**: Number of images per inference call
  - batch_capacity=1: Publish immediately after each sample (real-time mode)
  - batch_capacity>1: Accumulate images before publishing (batch mode)
  
- **Effective inference period**: `(1/fps) × batch_capacity`
  - Example 1: fps=5.0, batch_capacity=1 → 0.2s inference period (5 Hz)
  - Example 2: fps=5.0, batch_capacity=4 → 0.8s inference period (1.25 Hz)
  - Example 3: fps=10.0, batch_capacity=2 → 0.2s inference period (5 Hz)

This decoupling allows you to:
- Control VLM inference rate without changing image sampling
- Process video sequences by batching frames
- Balance latency vs. temporal context

**Configuration:**
- `prompts_dictionary.yaml` - Prompt templates per model
- Loads from package share directory

**Responsibilities:**
- Subscribe to image sources
- Apply prompt templates with variable substitution
- Batch images with prompts
- Optionally annotate images with people/face bounding boxes
- Publish `VLMPrompts` messages

**Note:** Prompt augmentation capabilities (e.g., variable substitution, dynamic class lists) are currently work in progress. The node currently loads static prompt templates from the dictionary file.

### 2. VLMDetectionNode (Python, `vlm_ros_node`)

**Purpose:** Performs VLM inference on images with prompts, extracts structured outputs.

**Subscribes:**
- `/vlm_prompts` - `perception_pipeline_msgs/VLMPrompts`
  - Receives images + prompts from PromptManagerNode

**Publishes:**
- `/vlm_detections` - `perception_pipeline_msgs/VLMOutputs`
  - Structured detections, gestures, emotions, relations
- `/vlm_detections/image` - `sensor_msgs/Image`
  - Annotated image with bounding boxes
- `/vlm_detections/status` - `std_msgs/String`
  - JSON status (FPS, latency, detection count)
- `/diagnostics` - `diagnostic_msgs/DiagnosticArray`
  - System health metrics

**Services:**
- `/vlm_detections/set_pause` - `std_srvs/SetBool`
  - Pause/resume inference

**Parameters:**
- `prompts_topic` (string, default: `/vlm_prompts`) - Input topic
- `paused` (bool, default: false) - Start paused
- `output_image_topic` (string, default: `/vlm_detections/image`)
- `output_raw_topic` (string, default: `/vlm_detections`)
- `model_config_file` (string, default: "") - Custom config path
- `device` (string, default: "auto") - Inference device

**Responsibilities:**
- Load VLM adapter dynamically from configuration
- Receive VLMPrompts messages
- Perform inference with configured model
- Parse JSON outputs (detections, properties, relations)
- Publish structured ROS messages
- Monitor performance and publish diagnostics

### 3. VLMMonitorGUI (Python, `vlm_ros_gui`)

**Purpose:** Real-time monitoring and control interface.

**Subscribes:**
- `/vlm_detections` - Displays structured outputs
- `/vlm_detections/image` - Shows annotated images
- `/vlm_detections/status` - Monitors performance

**Publishes:**
- None (monitoring only)

**Services Called:**
- `/vlm_detections/set_pause` - Control inference

**Responsibilities:**
- Display real-time detections
- Show performance metrics (FPS, latency)
- Visualize entity properties and relations
- Provide pause/resume controls

## System Architecture

**Typical Pipeline:**

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────┐
│   Camera    │─────>│ PromptManager    │─────>│ VLMDetection     │─────>│ Downstream   │
│             │      │ Node (C++)       │      │ Node (Python)    │      │ Nodes        │
└─────────────┘      └──────────────────┘      └──────────────────┘      └──────────────┘
                              │                          │
                              ↓                          ↓
                        /vlm_prompts              /vlm_detections
                   (images + prompts)          (structured outputs)
                                                         │
                                                         ↓
                                                  /vlm_detections/image
                                                  (annotated image)
```

**With Monitoring GUI:**

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Camera    │─────>│ PromptManager    │─────>│ VLMDetection     │
│             │      │ Node             │      │ Node             │
└─────────────┘      └──────────────────┘      └─────────┬────────┘
                                                          │
                                                          ↓
                                                   ┌──────────────┐
                                                   │ VLMMonitor   │
                                                   │ GUI          │
                                                   └──────────────┘
```

## Package Structure

```
vlm_detections/
├── core/
│   ├── adapter_base.py        # BaseVisionAdapter interface
│   ├── runtime.py             # Dynamic adapter loading
│   ├── parsed_items.py        # Data structures
│   └── visualize.py           # Drawing utilities
├── adapters/
│   └── openai_vision_adapter.py  # OpenAI-compatible API adapter
├── utils/
│   ├── json_parser.py         # Extract JSON from mixed text
│   ├── bbox_parser.py         # Parse bounding boxes
│   ├── entity_parser.py       # Parse properties/relations
│   └── config_loader.py       # Load YAML configs
├── ros_node.py                # VLMDetectionNode implementation
└── ros_gui.py                 # VLMMonitorGUI implementation

config/                        # Package configuration
├── adapters_config.yaml       # Adapter registry
├── model_config.yaml          # Model selection & parameters
└── model_variants.yaml        # Available model variants

src/
└── prompt_manager_node.cpp    # PromptManagerNode (C++)
```

## Configuration Files

The ROS node uses three main YAML configuration files located in the `/config` directory at the package root:

### 1. model_config.yaml - Model Selection & Inference Parameters

Controls which model to use and inference settings. This is the main configuration file loaded by the ROS node.

**Location:** `config/model_config.yaml`

**Structure:**
```yaml
# Model selection
model: "OpenAI Vision (API)"           # Model name (must match entry in adapters_config.yaml)
model_variant: "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"  # Specific model variant

# Detection parameters
threshold: 0.25                        # Confidence threshold for detections

# Device configuration
device: "auto"                         # Inference device: auto, cuda, cpu

# Generation parameters (model-specific)
generation_params:
  max_new_tokens: 400                  # Maximum output tokens
  temperature: 0.0                     # Sampling temperature (0.0 = deterministic)
  top_p: 1.0                           # Nucleus sampling threshold
```

**Notes:**
- `generation_params` are model-specific and validated by each adapter

### 2. adapters_config.yaml - Plugin Adapter Registry

Defines available model adapters and their configuration. This enables the dynamic loading system.

**Location:** `config/adapters_config.yaml`

**Structure:**
```yaml
adapters:
  "OpenAI Vision (API)":               # Display name used in model selection
    module: "vlm_detections.adapters.openai_vision_adapter"  # Python module path
    class: "OpenAIVisionAdapter"       # Adapter class name
    constructor_args:                   # Optional: passed to __init__(**kwargs)
      base_url: "http://100.115.56.116:8000/v1"  # API endpoint
      # Can reference environment variables: "${API_KEY}"
```

**How It Works:**
- `runtime.py` reads this file and dynamically imports adapters using `importlib`
- Constructor args are passed to the adapter's `__init__()` method
- Environment variables can be referenced with `${VAR_NAME}` syntax
- New adapters can be added without modifying core code

**Adding Custom Adapters:**
```yaml
adapters:
  "MyCustomModel":
    module: "my_package.my_adapter"
    class: "MyCustomAdapter"
    constructor_args:
      api_key: "${MY_API_KEY}"
      timeout: 30
```

### 3. model_variants.yaml - Available Model Variants

Defines which model variants (versions/sizes) are available for each adapter.

**Location:** `config/model_variants.yaml`

**Structure:**
```yaml
"OpenAI Vision (API)":
  - "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
  - "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
```

**Purpose:**
- Validates that selected `model_variant` exists for the chosen `model`
- Provides dropdown options in GUI applications
- Used by `runtime.py` functions: `default_variant_for()`, `ensure_valid_variant()`

**Example with Multiple Models:**
```yaml
"OpenAI Vision (API)":
  - "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
  - "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

"MyCustomModel":
  - "myorg/model-v1"
  - "myorg/model-v2-large"
```

### Configuration Loading Priority

The ROS node loads configuration in this order:

1. **Custom config file** (if specified via `model_config_file` parameter)
2. **Package default** (`config/model_config.yaml`)
3. **Fallback defaults** (hardcoded in `runtime.py`)

Override at launch:
```bash
ros2 launch vlm-detections vlm_node.launch.py \
    model_config_file:=/path/to/custom_config.yaml
```

### Plugin-Based Adapter System

The package uses a dynamic adapter loading system that allows adding new models without modifying core code.

**Adapter Protocols:**

The package defines an adapter protocol for prompt-based vision-language models:

**Creating a Prompt-Based VLM adater:**

```python
from vlm_detections.core.adapter_base import PromptBasedVLM
from vlm_detections.core.parsed_items import Detection, EntityProperty, EntityRelation
from typing import List, Tuple
import numpy as np

class MyVLMAdapter(PromptBasedVLM):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        # Handle additional constructor args from config
        
    def name(self) -> str:
        return f"MyVLM ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        # Load vision-language model
        self.model = ...
        self.processor = ...

    def infer(
        self,
        image_bgr: np.ndarray,
        user_prompt: str,
        system_prompt: str = "",
        threshold: float = 0.1
    ) -> Tuple[List[Detection], List[EntityProperty], List[EntityRelation], str]:
        # Run inference with natural language prompts
        # Parse outputs using utils/bbox_parser.py and utils/entity_parser.py
        # Return detections, properties, relations, and raw text
        return detections, properties, relations, raw_text
```

**Register your adapter** in `config/adapters_config.yaml`:

```yaml
adapters:
  "MyModel":
    module: "my_adapters_package.my_adapter"
    class: "MyAdapter"
    constructor_args:  # Optional, passed to __init__
      base_url: "http://localhost:8000"
      api_key: "${MY_API_KEY}"  # Can use environment variables
```

**Add model variants** in `config/model_variants.yaml`:

```yaml
"MyModel":
  - "org/mymodel-small"
  - "org/mymodel-large"
```

This approach allows you to:
- Keep proprietary adapters in separate packages
- Load adapters from different sources without code changes
- Configure adapter-specific parameters via YAML
- Easily switch between public and private adapter sets

### Generation Parameters

Adapters expose model-specific generation parameters for fine-tuning:

**Common parameters** (subset varies by adapter):
- `max_new_tokens` / `max_tokens`: Maximum output length
- `temperature`: Sampling randomness (0.0 = deterministic, 1.0 = creative)
- `top_p`: Nucleus sampling threshold
- `num_beams`: Beam search width (Florence-2, InternVL)
- `do_sample`: Enable/disable sampling

**Implementation in adapters:**
```python
def generation_config_spec(self) -> Dict[str, Dict[str, object]]:
    return {
        "max_new_tokens": {"type": "int", "default": 400, "min": 16, "max": 4096, "step": 16},
        "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1},
        "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    }

def update_generation_params(self, params: Dict[str, object]) -> None:
    self.gen_params.update(params)
    # Apply in model.generate(**self.gen_params)
```

**ROS Integration:**
- Configuration via `model_config.yaml`
- Dynamic parameter updates via ROS parameter server

## Launch Options

**Important:** Always activate your virtual environment before launching ROS nodes:
```bash
source vlm_venv/bin/activate
```

**Launch detection node:**
```bash
ros2 launch vlm_detections vlm_node.launch.py
```

**Launch with custom config:**
```bash
ros2 launch vlm_detections vlm_node.launch.py \
    model_config_file:=/path/to/custom_config.yaml \
    device:=cuda
```

**Launch with monitoring GUI:**
```bash
ros2 launch vlm_detections vlm_gui.launch.py
```

## Message Types

### VLMPrompts (Input)

Published by PromptManagerNode, consumed by VLMDetectionNode:

```
VLMPrompts:
  header: std_msgs/Header
  images[]: sensor_msgs/Image          # Batch of images
  system_prompts[]: string             # System prompt per image
  user_prompts[]: string               # User prompt per image
  classes[]: string[]                  # Optional: classes per image
```

### VLMOutputs (Output)

Published by VLMDetectionNode:

```
VLMOutputs:
  header: std_msgs/Header
  outputs[]: VLMOutput                 # One per input image/prompt

VLMOutput:
  vlm_metadata: VLMMetadata            # Model name, prompts
  detections[]: Detection2D            # Bounding boxes
  person_gestures[]: PersonGesture     # Gestures (waving, pointing, etc.)
  person_emotions[]: PersonEmotion     # Emotions (happy, sad, etc.)
  actor_relations[]: ActorRelation     # Spatial relations
  threshold: float32
  raw_output: string                   # Raw model text
  caption: string                      # Optional caption
```

**Example Detection:**
```yaml
detections:
  - bbox: [x1, y1, x2, y2]
    label: "person"
    score: 0.95
    uuid: 1
```

**Example Gesture:**
```yaml
person_gestures:
  - gesture_id: 0  # WAVING
    person_uuid: 1
    person_trackid: 1
```

**Example Relation:**
```yaml
actor_relations:
  - relation_id: 2  # IS_LOOKING_AT
    subject_uuid: 1
    object_uuid: 2
```

## Output Parsing

VLMDetectionNode extracts structured data from VLM text outputs:

**1. Bounding Box Detections**
- Parsed by `utils/bbox_parser.py`
- Supports multiple JSON formats (Qwen-style, OpenAI-style)
- Example: `{"bbox_2d": [x1, y1, x2, y2], "label": "person", "score": 0.95}`

**2. Entity Properties (Gestures, Emotions)**
- Parsed by `utils/entity_parser.py`
- Example: `{"entity": "person_1", "property": "gesture", "value": "waving", "score": 0.95}`
- Mapped to `PersonGesture` and `PersonEmotion` messages

**3. Entity Relations**
- Parsed by `utils/entity_parser.py`
- Example: `{"subject": "person_1", "predicate": "looking_at", "object": "person_2"}`
- Mapped to `ActorRelation` message

**JSON Extraction:**
- `json_parser.py` extracts JSON from mixed text/code blocks
- Handles nested structures: `{"entities": [...], "properties": [...], "relations": [...]}`
- Robust to markdown formatting and code fences

## Troubleshooting

**Node not publishing outputs:**
- Check `/vlm_prompts` topic: `ros2 topic echo /vlm_prompts`
- Verify PromptManagerNode is running
- Check logs: `ros2 topic echo /rosout`

**No detections found:**
- Lower threshold in `config/model_config.yaml`
- Check raw output in logs or `/vlm_detections/status`
- Verify prompt format matches model expectations

**Model fails to load:**
- Check GPU memory: `nvidia-smi`
- Try CPU: `device: cpu` in config
- Verify API endpoint is accessible (for OpenAI adapter)

**Performance issues:**
- Monitor `/diagnostics` topic for latency metrics
- Reduce image resolution from camera
- Use smaller model variant
- Enable model quantization (if supported)

**Parsing errors:**
- Check logs for JSON extraction failures
- Verify model outputs valid JSON
- Test prompt with standalone app first (see `standalone_app/README.md`)



