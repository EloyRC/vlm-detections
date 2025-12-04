"""Utilities for parsing bounding box detections from VLM text outputs."""

from __future__ import annotations

import logging
from typing import List

from vlm_detections.core.parsed_items import Detection
from vlm_detections.utils.json_parser import extract_json_objects

logger = logging.getLogger(__name__)


def parse_bboxes_qwen_style(text: str, threshold: float = 0.0) -> List[Detection]:
    """Parse bounding boxes from Qwen-style model output.
    
    This parser expects JSON objects with the following structure:
    - bbox_2d (preferred) or bbox/bounding_box/box: [x1, y1, x2, y2] in absolute pixels
    - label/class/name: string label for the detection
    - score/confidence: optional confidence score (0.0 to 1.0, defaults to 1.0)
    
    The text can be pure JSON or mixed with other content. This function will
    extract all valid JSON objects and parse detections from them.
    
    Handles nested structures like {"entities": [{...}, {...}]} or flat arrays.
    
    Args:
        text: Model output text containing JSON-formatted detections.
        threshold: Minimum confidence score to include a detection (default 0.0).
        
    Returns:
        List of Detection objects with absolute pixel coordinates.
        
    Example:
        >>> text = '[{"bbox_2d": [10, 20, 100, 200], "label": "cat", "score": 0.95}]'
        >>> detections = parse_bboxes_qwen_style(text, threshold=0.5)
        >>> len(detections)
        1
        >>> detections[0].label
        'cat'
    """
    detections: List[Detection] = []
    
    # Collect candidate objects to parse
    candidates = []
    for obj in extract_json_objects(text):
        # Check if this is a wrapper object with nested arrays
        if isinstance(obj, dict):
            # Look for common wrapper keys
            for key in ['entities', 'detections', 'objects', 'boxes', 'bboxes']:
                if key in obj and isinstance(obj[key], list):
                    candidates.extend([item for item in obj[key] if isinstance(item, dict)])
            # If no wrapper key found, treat the object itself as a candidate
            if not any(key in obj for key in ['entities', 'detections', 'objects', 'boxes', 'bboxes']):
                candidates.append(obj)
    
    # Parse each candidate object
    for obj in candidates:
        # Extract label with fallback chain
        label = (
            obj.get("label") 
            or obj.get("class") 
            or obj.get("name") 
            or "object"
        )
        
        # Extract score with fallback chain
        score_val = obj.get("score") or obj.get("confidence")
        try:
            score = float(score_val) if score_val is not None else 1.0
        except (TypeError, ValueError):
            score = 1.0
        
        # Skip if below threshold
        if score < threshold:
            continue
        
        # Extract bounding box with fallback chain
        box = (
            obj.get("bbox_2d") 
            or obj.get("bbox") 
            or obj.get("bounding_box") 
            or obj.get("box")
        )
        
        # Validate and parse bounding box
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        
        # Create detection
        detections.append(Detection(
            xyxy=(x1, y1, x2, y2),
            score=score,
            label=str(label)
        ))
    
    return detections


def parse_bboxes_openai_style(text: str, threshold: float = 0.0) -> List[Detection]:
    """Parse bounding boxes from OpenAI-style model output.
    
    This is essentially the same as Qwen-style parsing but maintained as a
    separate function for clarity and potential future divergence.
    
    Expected JSON structure:
    - bbox_2d (preferred) or bbox/bounding_box/box: [x1, y1, x2, y2] in absolute pixels
    - label/class/name: string label for the detection
    - score/confidence: optional confidence score (0.0 to 1.0, defaults to 1.0)
    
    Args:
        text: Model output text containing JSON-formatted detections.
        threshold: Minimum confidence score to include a detection (default 0.0).
        
    Returns:
        List of Detection objects with absolute pixel coordinates.
    """
    # Currently identical to Qwen-style parsing
    return parse_bboxes_qwen_style(text, threshold)


def clamp_detections_to_image(
    detections: List[Detection],
    image_width: int,
    image_height: int,
    min_box_size: float = 1.0
) -> List[Detection]:
    """Clamp detection bounding boxes to image boundaries and filter invalid boxes.
    
    Args:
        detections: List of Detection objects to clamp.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        min_box_size: Minimum box size (width and height) to keep (default 1.0).
        
    Returns:
        List of Detection objects with clamped coordinates and valid boxes.
    """
    clamped: List[Detection] = []
    
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        
        # Clamp to image bounds
        x1 = max(0.0, min(float(x1), float(image_width - 1)))
        y1 = max(0.0, min(float(y1), float(image_height - 1)))
        x2 = max(0.0, min(float(x2), float(image_width - 1)))
        y2 = max(0.0, min(float(y2), float(image_height - 1)))
        
        # Validate box size
        if (x2 - x1) >= min_box_size and (y2 - y1) >= min_box_size:
            clamped.append(Detection(
                xyxy=(x1, y1, x2, y2),
                score=det.score,
                label=det.label,
                polygon=det.polygon,
                text=det.text
            ))
    
    return clamped
