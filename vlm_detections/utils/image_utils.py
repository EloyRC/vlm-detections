"""Common utility functions for vision model adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from PIL import Image


def bgr_to_pil_rgb(image_bgr: np.ndarray) -> "Image.Image":
    """
    Convert OpenCV BGR image to PIL RGB image.
    
    Args:
        image_bgr: Image in BGR format (OpenCV convention), shape (H, W, 3)
        
    Returns:
        PIL Image in RGB format
    """
    from PIL import Image
    
    image_rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(image_rgb)


def pil_rgb_to_bgr(pil_image: "Image.Image") -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR format.
    
    Args:
        pil_image: PIL Image (will be converted to RGB if needed)
        
    Returns:
        NumPy array in BGR format, shape (H, W, 3)
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    rgb_array = np.array(pil_image)
    return rgb_array[:, :, ::-1]
