from __future__ import annotations

"""
GroundingDINO adapter

Notes:
- Official repo: https://github.com/IDEA-Research/GroundingDINO
- pip packages vary (e.g., groundingdino-py). You may need to install from source.
- Pre/post-processing requires text tokenization and box decoding with thresholds.

This adapter uses groundingdino.util.inference utilities when available,
with a graceful fallback and clear instructions if the package is missing.
"""

from typing import List
import numpy as np
from vlm_detections.core.adapter_base import ZeroShotObjectDetector, Detection
from vlm_detections.utils.image_utils import bgr_to_pil_rgb


class GroundingDINOAdapter(ZeroShotObjectDetector):
    def __init__(self, model_id: str | None = None):
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.model_id = model_id or "IDEA-Research/grounding-dino-tiny"

    def name(self) -> str:
        return f"GroundingDINO ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
        except Exception as e:
            raise RuntimeError(
                "GroundingDINO adapter requires 'transformers' (with Grounding DINO support) and 'torch'."
            ) from e

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def infer(
        self,
        image_bgr: np.ndarray,
        classes: List[str],
        threshold: float,
    ) -> List[Detection]:
        if self.model is None or self.processor is None:
            raise RuntimeError("GroundingDINO adapter not loaded. Call load() first.")

        from PIL import Image
        import torch

        pil = bgr_to_pil_rgb(image_bgr)

        # GroundingDINO expects class phrases separated by periods
        text = ". ".join(classes) + "." if classes else "object ."

        inputs = self.processor(images=pil, text=text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]]).to(self.device)  # (h, w)

        detections: List[Detection] = []

        post = None
        for candidate in [
            "post_process_grounded_object_detection",
            "post_process_grounding_object_detection",
            "post_process_object_detection",
        ]:
            if hasattr(self.processor, candidate):
                post = getattr(self.processor, candidate)
                break
        if post is None:
            raise RuntimeError("Processor lacks a post-processing method for GroundingDINO.")

        try:
            results = post(
                outputs=outputs,
                target_sizes=target_sizes,
                box_threshold=threshold,
                text_threshold=threshold,
            )
        except TypeError:
            results = post(outputs=outputs, target_sizes=target_sizes)

        result = results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")

        if boxes is None:
            return detections

        import numpy as _np
        boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else _np.asarray(boxes)
        scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else _np.asarray(scores)
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        for i, box in enumerate(boxes):
            score = float(scores[i]) if scores is not None else 1.0
            if score < threshold:
                continue
            label = str(labels[i]) if labels is not None else "object"
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(Detection((x1, y1, x2, y2), score, label))
        return detections
