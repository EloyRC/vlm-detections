from __future__ import annotations

from typing import List
import numpy as np
from vlm_detections.core.adapter_base import ZeroShotObjectDetector, Detection
from vlm_detections.utils.image_utils import bgr_to_pil_rgb


class OwlVitAdapter(ZeroShotObjectDetector):
    def __init__(self, model_id: str | None = None):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.model_id = model_id or "google/owlvit-base-patch32"

    def name(self) -> str:
        return f"OWL-ViT ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.processor = OwlViTProcessor.from_pretrained(self.model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def infer(
        self,
        image_bgr: np.ndarray,
        classes: List[str],
        threshold: float,
    ) -> List[Detection]:
        import torch
        from PIL import Image

        # OWL-ViT expects PIL RGB
        pil = bgr_to_pil_rgb(image_bgr)

        # OWL-ViT accepts multiple phrases; build nested list [queries]
        with torch.inference_mode():
            inputs = self.processor(text=[classes], images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            # Post-process to get boxes/scores/labels in absolute coords
            target_sizes = torch.tensor([pil.size[::-1]]).to(self.device)  # (h, w)
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
            boxes = results["boxes"].cpu().numpy()  # xyxy
            scores = results["scores"].cpu().numpy()
            labels_idx = results["labels"].cpu().numpy()

            detections: List[Detection] = []
            for box, score, li in zip(boxes, scores, labels_idx):
                if score < threshold:
                    continue
                label = classes[int(li)] if int(li) < len(classes) else str(int(li))
                detections.append(Detection(tuple(map(float, box)), float(score), label))
            return detections
