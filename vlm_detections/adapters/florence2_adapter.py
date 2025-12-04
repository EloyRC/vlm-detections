from __future__ import annotations

"""Florence-2 adapter

Supported tasks (prompt tokens) and expected outputs:
    <CAPTION>                        -> caption text
    <DETAILED_CAPTION>               -> detailed caption text
    <MORE_DETAILED_CAPTION>          -> very detailed caption text
    <OD>                             -> boxes, labels, scores
    <DENSE_REGION_CAPTION>           -> boxes + per-region captions
    <REGION_PROPOSAL>                -> boxes (no labels)
    <REFERRING_EXPRESSION_SEGMENTATION> -> polygon(s) / mask, label
    <REGION_TO_SEGMENTATION>         -> polygon(s) / mask
    <OCR>                            -> full image text
    <OCR_WITH_REGION>                -> boxes + text per region
    <CAPTION_TO_PHRASE_GROUNDING>    -> boxes grounded to phrases

We attempt to use processor.post_process_generation(task=..., image_size=...). The structure returned
varies by checkpoint but typically includes keys like: bboxes/boxes, labels, scores, polygons, texts, captions.
We normalize into Detection objects plus a raw JSON string.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

from vlm_detections.core.adapter_base import ZeroShotObjectDetector, PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations
from vlm_detections.utils.image_utils import bgr_to_pil_rgb


class Florence2Adapter(ZeroShotObjectDetector, PromptBasedVLM):
    # Task compatibility categories
    OBJECT_DETECTION_TASKS = {"OD", "REF"}
    PROMPT_BASED_TASKS = {
        "CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION",
        "DENSE_REGION_CAPTION", "REGION_PROPOSAL",
        "REFERRING_EXPRESSION_SEGMENTATION", "REGION_TO_SEGMENTATION",
        "OCR", "OCR_WITH_REGION", "CAPTION_TO_PHRASE_GROUNDING"
    }
    
    # TASK_METADATA maps task token (without brackets) to a short description and expected primary output type.
    TASK_METADATA: Dict[str, Dict[str, str]] = {
        "CAPTION": {"desc": "Concise image caption", "out": "text"},
        "DETAILED_CAPTION": {"desc": "Detailed caption", "out": "text"},
        "MORE_DETAILED_CAPTION": {"desc": "Very detailed caption", "out": "text"},
        "OD": {"desc": "Object detection (boxes)", "out": "boxes"},
        "DENSE_REGION_CAPTION": {"desc": "Region captions (boxes + text)", "out": "boxes+text"},
        "REGION_PROPOSAL": {"desc": "Region proposals (boxes)", "out": "boxes"},
        "REFERRING_EXPRESSION_SEGMENTATION": {"desc": "Segmentation from referring expression", "out": "polygons"},
        "REGION_TO_SEGMENTATION": {"desc": "Segmentation for region", "out": "polygons"},
        "OCR": {"desc": "Full image OCR text", "out": "text"},
        "OCR_WITH_REGION": {"desc": "OCR with region boxes", "out": "boxes+text"},
        "CAPTION_TO_PHRASE_GROUNDING": {"desc": "Phrase grounding", "out": "boxes"},
        "REF": {"desc": "Legacy referring expression boxes", "out": "boxes"},
    }
    SUPPORTED_TASKS = {
        "CAPTION",
        "DETAILED_CAPTION",
        "MORE_DETAILED_CAPTION",
        "OD",
        "DENSE_REGION_CAPTION",
        "REGION_PROPOSAL",
        "REFERRING_EXPRESSION_SEGMENTATION",
        "REGION_TO_SEGMENTATION",
        "OCR",
        "OCR_WITH_REGION",
        "CAPTION_TO_PHRASE_GROUNDING",
        "REF",  # legacy / simple referring expression
    }

    def __init__(self, model_id: str | None = None, task: str = "OD"):
        """
        task: "OD" or "REF". Controls which special tag is used (<OD> or <REF>) and how prompts are built.
        """
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.model_id = model_id or "florence-community/Florence-2-base-ft"
        task = (task or "OD").upper()
        if task not in self.SUPPORTED_TASKS:
            task = "OD"
        self.task = task
        self.task_tag = f"<{task}>" if not task.startswith("<") else task
        self.attn_impl = None
        self.gen_params = {}

    def set_task(self, task: str) -> None:
        t = (task or "OD").upper()
        if t not in self.SUPPORTED_TASKS:
            t = "OD"
        self.task = t
        self.task_tag = f"<{t}>"

    def name(self) -> str:
        return f"Florence-2 ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except Exception as e:
            raise RuntimeError(
                "Florence-2 adapter requires 'transformers' and 'torch'. Please install compatible versions."
            ) from e

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch_dtype = torch.bfloat16 if (self.device == "cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        from transformers import Florence2ForConditionalGeneration as _AM
        self.model = _AM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        # Initialize generation params
        spec = self.generation_config_spec()
        self.gen_params = {k: meta.get("default") for k, meta in spec.items()}

    def _build_text(self, classes: List[str], prompt: str) -> str:
        tag = self.task_tag
        p = (prompt or "").strip()
        if tag == "<OD>":
            return "<OD>"
        if tag == "<REF>":
            return f"<REF> {p}" if p else "<REF>"
        if tag in {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>", "<REGION_PROPOSAL>"}:
            return tag
        if tag in {"<DENSE_REGION_CAPTION>", "<CAPTION_TO_PHRASE_GROUNDING>"}:
            # Provide optional user prompt if present
            return f"{tag} {p}" if p else tag
        if tag in {"<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"}:
            # These require a reference phrase
            return f"{tag} {p}" if p else tag
        if tag == "<OCR_WITH_REGION>":
            return tag
        return tag

    def _postprocess(self, parsed_answer: Dict[str, Any], threshold: float) -> List[Detection]:
        out: List[Detection] = []
        try:
            # Common container may be under the task tag or root
            data = parsed_answer.get(self.task_tag) or parsed_answer
            # Caption-only tasks
            if self.task in {"CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION", "OCR"}:
                txt = data.get("caption") or data.get("text") or data.get("content") or ""
                if txt.strip():
                    out.append(Detection((0, 0, 1, 1), 1.0, self.task.lower(), text=txt.strip()))
                return out
            # Region caption (dense)
            if self.task == "DENSE_REGION_CAPTION":
                boxes = data.get("bboxes") or data.get("boxes") or []
                captions = data.get("captions") or data.get("texts") or data.get("labels") or []
                scores = data.get("scores") or [1.0] * len(boxes)
                for box, cap, sc in zip(boxes, captions, scores):
                    s = float(sc if sc is not None else 1.0)
                    if s < threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    out.append(Detection((x1, y1, x2, y2), s, "region", text=str(cap)))
                return out
            # Region proposal
            if self.task == "REGION_PROPOSAL":
                boxes = data.get("bboxes") or data.get("boxes") or []
                scores = data.get("scores") or [1.0] * len(boxes)
                for box, sc in zip(boxes, scores):
                    s = float(sc if sc is not None else 1.0)
                    if s < threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    out.append(Detection((x1, y1, x2, y2), s, "region"))
                return out
            # OCR with boxes
            if self.task == "OCR_WITH_REGION":
                boxes = data.get("bboxes") or data.get("boxes") or []
                texts = data.get("texts") or data.get("ocr") or [""] * len(boxes)
                scores = data.get("scores") or [1.0] * len(boxes)
                for box, txt, sc in zip(boxes, texts, scores):
                    s = float(sc if sc is not None else 1.0)
                    if s < threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    out.append(Detection((x1, y1, x2, y2), s, "text", text=str(txt)))
                return out
            # Phrase grounding
            if self.task == "CAPTION_TO_PHRASE_GROUNDING":
                boxes = data.get("bboxes") or data.get("boxes") or []
                phrases = data.get("phrases") or data.get("labels") or []
                scores = data.get("scores") or [1.0] * len(boxes)
                for box, phr, sc in zip(boxes, phrases, scores):
                    s = float(sc if sc is not None else 1.0)
                    if s < threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    out.append(Detection((x1, y1, x2, y2), s, str(phr)))
                return out
            # Segmentation tasks
            if self.task in {"REFERRING_EXPRESSION_SEGMENTATION", "REGION_TO_SEGMENTATION"}:
                polys = data.get("polygons") or data.get("polygon") or []
                labels_list = data.get("labels") or []
                base_label = data.get("label") or data.get("text") or self.task.lower()
                score = float(data.get("score") or 1.0)

                def to_pairs(seq):
                    # Accept flattened [x1,y1,x2,y2,...] or [[x,y],...]
                    if not seq:
                        return []
                    if isinstance(seq[0], (list, tuple)) and len(seq[0]) == 2:
                        return [(float(a), float(b)) for a, b in seq]
                    # flattened
                    if len(seq) % 2 == 0:
                        return [(float(seq[i]), float(seq[i+1])) for i in range(0, len(seq), 2)]
                    return []

                parts = []
                # Cases:
                # 1) polys == [ [ [x,y]... ] ]  (list with one polygon containing point pairs)
                # 2) polys == [ [x,y,x2,y2,...] ] (list containing flattened list)
                # 3) polys == [ [ [x,y]... ], [ [x,y]... ] ] (multi-part)
                if isinstance(polys, list) and polys:
                    first = polys[0]
                    if isinstance(first, list):
                        if first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
                            # case: [ [ [x,y],... ] , [ [x,y],... ] ]
                            for sub in polys:
                                parts.append(to_pairs(sub))
                        elif first and all(isinstance(v, (int, float)) for v in first):
                            # case: [ [x1,y1,x2,y2,...] ]
                            parts.append(to_pairs(first))
                        elif first and isinstance(first[0], list) and first[0] and all(isinstance(v, (int, float)) for v in first[0]):
                            # case: [ [ [x1,y1,x2,y2,...] ] ] (triple nest)
                            for sub in polys:
                                for inner in sub:
                                    parts.append(to_pairs(inner))
                    elif isinstance(first, (int, float)):
                        # flattened: [x1,y1,x2,y2,...]
                        parts.append(to_pairs(polys))
                # Build detections
                filtered = [p for p in parts if p]
                for idx, poly in enumerate(filtered):
                    label = labels_list[idx] if idx < len(labels_list) and isinstance(labels_list[idx], str) and labels_list[idx].strip() else base_label
                    flat_box = self._poly_to_box(poly)
                    out.append(Detection(flat_box, score, str(label), polygon=poly))
                return out
            # OD / REF fallback (boxes + labels)
            boxes = data.get("bboxes") or data.get("boxes") or []
            labels = data.get("labels") or ["object"] * len(boxes)
            scores = data.get("scores") or [1.0] * len(boxes)
            for box, lab, sc in zip(boxes, labels, scores):
                s = float(sc if sc is not None else 1.0)
                if s < threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                out.append(Detection((x1, y1, x2, y2), s, str(lab)))
        except Exception:
            pass
        return out

    @staticmethod
    def _poly_to_box(poly: List[List[float]] | List[tuple]) -> tuple[float, float, float, float]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    
    def _infer_internal(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        threshold: float,
    ) -> Tuple[List[Detection], str]:
        """Internal inference method used by both ZeroShotObjectDetector and PromptBasedVLM interfaces."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Florence-2 adapter not loaded. Call load() first.")

        from PIL import Image
        import torch

        pil = bgr_to_pil_rgb(image_bgr)
        h, w = image_bgr.shape[:2]

        inputs = self.processor(text=text_prompt, images=pil, return_tensors="pt").to(self.model.device, self.model.dtype)

        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": int(self.gen_params.get("max_new_tokens", 1024)),
                "num_beams": int(self.gen_params.get("num_beams", 3)),
            }
            # Optional sampling override (if user enables do_sample)
            if bool(self.gen_params.get("do_sample", False)):
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": float(self.gen_params.get("temperature", 1.0)),
                })
            
            generated_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only the newly generated tokens (exclude the prompt/input ids)
        # try:
        #     input_len = inputs["input_ids"].shape[1]
        #     gen_only = generated_ids[:, input_len:]
        # except Exception:
        #     gen_only = generated_ids

        gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            gen_text, task=self.task_tag, image_size=(w, h)
        )
        detections = self._postprocess(parsed_answer, threshold)
        # Clamp to image bounds and filter invalids
        out: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            x1 = max(0.0, min(float(x1), float(w - 1)))
            y1 = max(0.0, min(float(y1), float(h - 1)))
            x2 = max(0.0, min(float(x2), float(w - 1)))
            y2 = max(0.0, min(float(y2), float(h - 1)))
            if x2 > x1 and y2 > y1:
                out.append(Detection((x1, y1, x2, y2), det.score, det.label, polygon=det.polygon, text=det.text))
        
        return out, json.dumps(parsed_answer, indent=2, default=str)

    # ========================================================================
    # Protocol-specific methods
    # ========================================================================
    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> Tuple[List[Detection], List[EntityProperty], List[EntityRelation], str]:
        """PromptBasedVLM protocol: inference with natural language prompt."""
        # Validate task compatibility
        if self.task not in self.PROMPT_BASED_TASKS:
            raise ValueError(
                f"Task '{self.task}' is not compatible with prompt-based inference. "
                f"Compatible tasks: {sorted(self.PROMPT_BASED_TASKS)}"
            )
        
        # Build text prompt based on current task
        text = self._build_text([], prompt)
        detections, raw_output = self._infer_internal(image_bgr, text, threshold)
        
        # Parse entity properties and relations from output
        properties = parse_entity_properties(raw_output, threshold=threshold)
        relations = parse_entity_relations(raw_output, threshold=threshold)
        
        return detections, properties, relations, raw_output
    
    # ========================================================================
    # ZeroShotObjectDetector protocol method
    # ========================================================================
    def infer(
        self,
        image_bgr: np.ndarray,
        classes: List[str],
        threshold: float,
    ) -> List[Detection]:
        """
        ZeroShotObjectDetector protocol: object detection with class list.
        
        Args:
            image_bgr: Input image in BGR format
            classes: List of class names (ignored by Florence-2 OD)
            threshold: Confidence threshold for detections
            
        Returns:
            List of Detection objects
        """
        # Validate task compatibility
        if self.task not in self.OBJECT_DETECTION_TASKS:
            raise ValueError(
                f"Task '{self.task}' is not compatible with zero-shot object detection. "
                f"Compatible tasks: {sorted(self.OBJECT_DETECTION_TASKS)}. "
                f"Use infer_from_image() for prompt-based tasks."
            )
        
        # OD mode doesn't use class names in Florence-2, just the <OD> tag
        if classes:
            logger.warning("Florence-2 OD does not utilize class names; ignoring provided classes.")

        text = "<OD>"
        detections, _ = self._infer_internal(image_bgr, text, threshold)
        return detections

    def infer_from_batch(
        self,
        frames_with_ts: List[Tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        """Batch inference not supported for Florence-2."""
        return ""

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        """Video inference not supported for Florence-2."""
        return ""

    # --- Generation parameter support ---
    def generation_config_spec(self) -> dict[str, dict]:
        return {
            "max_new_tokens": {"type": "int", "default": 1024, "min": 32, "max": 4096, "step": 32, "label": "Max New Tokens"},
            "num_beams": {"type": "int", "default": 3, "min": 1, "max": 16, "step": 1, "label": "Beams"},
            "do_sample": {"type": "bool", "default": False, "label": "Enable Sampling"},
            "temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "label": "Temperature"},
        }

    def update_generation_params(self, params: dict[str, object]) -> None:
        spec = self.generation_config_spec()
        for k, v in params.items():
            if k in spec:
                self.gen_params[k] = v
