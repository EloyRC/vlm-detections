from __future__ import annotations

from typing import List, Tuple, Protocol, Optional, Dict, Any, runtime_checkable
import numpy as np

from vlm_detections.core.parsed_items import Detection, EntityProperty, EntityRelation


# ============================================================================
# BASE PROTOCOL
# ============================================================================

@runtime_checkable
class BaseVisionAdapter(Protocol):
    """
    Root protocol for all vision model adapters.
    
    Common attributes:
    - model: The loaded model instance
    - processor: Tokenizer/processor for inputs (if applicable)
    - device: Target device ("cpu", "cuda", or "api")
    - model_id: Model identifier (e.g., Hugging Face model ID)
    """
    
    def name(self) -> str:
        """Return human-readable adapter name."""
        ...
    
    def load(self, device: str = "auto") -> None:
        """Load model weights and move to device.
        
        Args:
            device: Target device. "auto" selects CUDA if available, otherwise CPU.
        """
        ...


# ============================================================================
# BRANCH 1: Zero-Shot Object Detection Models
# ============================================================================

@runtime_checkable
class ZeroShotObjectDetector(BaseVisionAdapter, Protocol):
    """
    Protocol for models that perform zero-shot object detection.
    
    Characteristics:
    - Single image input only
    - Takes a list of class names as input
    - Returns structured detections (bboxes + labels + scores)
    - No natural language prompts
    - No batch/video support
    - Non-generative (no free-form text output)
    
    Examples: OWL-ViT, Grounding-DINO, Florence-2 (OD mode)
    """
    
    def infer(
        self,
        image_bgr: np.ndarray,
        classes: List[str],
        threshold: float,
    ) -> List[Detection]:
        """
        Run zero-shot detection on a single BGR image.
        
        Args:
            image_bgr: Input image in BGR format (OpenCV convention)
            classes: List of object class names to detect
            threshold: Confidence threshold for filtering detections
            
        Returns:
            List of Detection objects with bboxes, labels, and scores
        """
        ...


# ============================================================================
# BRANCH 2: Prompt-Based Vision-Language Models (VLMs)
# ============================================================================

@runtime_checkable
class PromptBasedVLM(BaseVisionAdapter, Protocol):
    """
    Protocol for generative vision-language models that accept prompts.
    
    Characteristics:
    - Accept natural language prompts (user + optional system prompt)
    - Support multiple input modalities: single image, batch, video
    - Generate structured output (JSON with detections, properties, relations)
    - Can produce free-form text descriptions
    - Support advanced features (reasoning, entity relationships)
    
    Examples: Qwen2.5-VL, Qwen3-VL, TRex, InternVL3.5, OpenAI Vision API, 
              Cosmos-Reason1, Florence-2 (generative modes)
    """
    
    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> Tuple[List[Detection], List[EntityProperty], List[EntityRelation], str]:
        """
        Run inference on a single BGR image with natural language prompts.
        
        Args:
            image_bgr: Input image in BGR format
            prompt: User prompt/instruction for the model
            system_prompt: Optional system-level instruction
            threshold: Confidence threshold for filtering detections
            
        Returns:
            Tuple of:
            - detections: List of Detection objects in absolute pixel coords
            - properties: List of EntityProperty (attributes/descriptions)
            - relations: List of EntityRelation (spatial/semantic relationships)
            - raw_output_text: Unprocessed model output text
        """
        ...
    
    def infer_from_batch(
        self,
        frames_with_ts: List[Tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        """
        Run a single inference call over a batch of timestamped images.
        
        Args:
            frames_with_ts: List of (BGR image, timestamp in seconds) tuples
            prompt: User prompt
            system_prompt: Optional system prompt
            threshold: Confidence threshold
            
        Returns:
            Raw textual output from model. Empty string if unsupported.
        """
        ...
    
    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        """
        Run inference using a video file as input at the requested fps.
        
        Args:
            video_path: Path to video file
            fps: Sampling rate (frames per second)
            prompt: User prompt
            system_prompt: Optional system prompt
            threshold: Confidence threshold
            
        Returns:
            Raw textual output from model. Empty string if unsupported.
        """
        ...
    
    # --- Optional generation parameter support (duck-typed) ---
    # If an adapter is generative (LLM / VLM that uses decoding strategies) it can optionally
    # expose a schema of tunable parameters for the UI via generation_config_spec().
    #
    # def generation_config_spec(self) -> Dict[str, Dict[str, Any]]:
    #     """Return schema of tunable generation parameters.
    #     
    #     Returns:
    #         Mapping of param_name -> metadata dict with keys:
    #         - type: "int", "float", or "bool"
    #         - default: default value
    #         - min, max, step: for numeric parameters
    #         - label: human-readable label
    #         - help: optional help text
    #     """
    #     ...
    #
    # def update_generation_params(self, params: Dict[str, Any]) -> None:
    #     """Store sanitized generation parameters for subsequent infer calls."""
    #     ...
    #
    # Adapters that do not implement these are treated as not supporting generation tuning.
