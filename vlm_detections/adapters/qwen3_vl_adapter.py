from __future__ import annotations

from typing import List, Optional
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from vlm_detections.core.adapter_base import PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.core.generative_vlm_base import GenerativeVLMBase
from vlm_detections.utils.image_utils import bgr_to_pil_rgb
from vlm_detections.utils.qwen_message_utils import (
    build_qwen_conversation_messages,
    build_qwen_messages,
    build_qwen_timed_messages,
    gather_pil_images,
    move_inputs_to_device,
    infer_video_qwen_style,
    prepare_qwen_inputs_with_length_guard,
    validate_qwen_token_length,
)
from vlm_detections.utils.bbox_parser import parse_bboxes_qwen_style, clamp_detections_to_image
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations


class Qwen3VLAdapter(PromptBasedVLM, GenerativeVLMBase):
    """Adapter for Qwen3-VL models (Hugging Face Transformers)."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.gen_params: dict[str, object] = {}
        self._gen_spec_cache: dict[str, dict] | None = None
        self._image_patch_size: int = 16

    def name(self) -> str:
        return f"Qwen3-VL ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()

        self._gen_spec_cache = None
        self._reset_generation_defaults()

    # --- Internal helpers -------------------------------------------------
    def _prepare_inputs(self, messages: List[dict], pil_images: List[Image.Image]):
        if self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            kwargs = {
                "return_video_kwargs": True,
                "return_video_metadata": True,
            }
            if self._image_patch_size:
                kwargs["image_patch_size"] = self._image_patch_size
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    [messages], **kwargs
                )
            except TypeError:
                kwargs.pop("image_patch_size", None)
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    [messages], **kwargs
                )
            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            processor_kwargs = {
                "text": [text],
                "return_tensors": "pt",
            }
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
            processor_kwargs["do_resize"] = False
            if video_metadatas is not None:
                processor_kwargs["video_metadata"] = video_metadatas
            if video_kwargs:
                processor_kwargs.update(video_kwargs)
            tokenizer = getattr(self.processor, "tokenizer", None)
            original_limit = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
            try:
                if tokenizer is not None and isinstance(original_limit, int) and 0 < original_limit < 1_000_000_000:
                    tokenizer.model_max_length = 1_000_000_000
                inputs = self.processor(**processor_kwargs)
            finally:
                if tokenizer is not None and original_limit is not None:
                    try:
                        tokenizer.model_max_length = original_limit
                    except Exception:
                        logger.debug("Failed to restore tokenizer max length", exc_info=True)
            validate_qwen_token_length(
                self.model,
                self.processor,
                self.name(),
                inputs,
                original_limit=original_limit,
            )
            return inputs, text
        except Exception:
            return prepare_qwen_inputs_with_length_guard(
                self.processor,
                self.model,
                self.name(),
                messages,
                pil_images,
            )

    @staticmethod
    def _rescale_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
        coords = [float(x1), float(y1), float(x2), float(y2)]
        max_abs = max(abs(c) for c in coords)
        if width > 0 and height > 0:
            if max_abs <= 1.0005:
                # Normalized [0,1]
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height
            elif max_abs <= 1000.0005:
                # Qwen3 convention (0..1000 grid)
                x1 = x1 / 1000.0 * width
                x2 = x2 / 1000.0 * width
                y1 = y1 / 1000.0 * height
                y2 = y2 / 1000.0 * height
        return x1, y1, x2, y2

    def _sanitize_detections(
        self,
        detections: List[Detection],
        width: int,
        height: int,
    ) -> List[Detection]:
        sanitized: List[Detection] = []
        w = float(max(width, 1))
        h = float(max(height, 1))
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            x1, y1, x2, y2 = self._rescale_box(x1, y1, x2, y2, width, height)
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1))
            y2 = max(0.0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                sanitized.append(Detection((x1, y1, x2, y2), det.score, det.label))
        return sanitized

    def _generate(self, inputs) -> List[str]:  # type: ignore
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")
        from torch import no_grad

        validate_qwen_token_length(self.model, self.processor, self.name(), inputs)
        inputs = move_inputs_to_device(inputs, self.device)

        max_new_tokens = int(self._get_generation_value("max_new_tokens"))
        n_return = int(self._get_generation_value("num_return_sequences"))
        do_sample = bool(self._get_generation_value("do_sample"))
        rep_penalty = float(self._get_generation_value("repetition_penalty"))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": n_return,
            "do_sample": do_sample,
            "repetition_penalty": rep_penalty,
        }
        if do_sample:
            temperature = float(self._get_generation_value("temperature"))
            if temperature <= 0.0:
                temperature = 1e-4
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": float(self._get_generation_value("top_p")),
                    "top_k": int(self._get_generation_value("top_k")),
                }
            )
        logger.info("Generating with kwargs: %s", gen_kwargs)
        with no_grad():
            generated = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated[:, input_len:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)

    def _generate_sequences(
        self,
        pil_images: List[Image.Image],
        prompt_text: str,
        system_prompt: Optional[str],
    ) -> List[str]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")

        messages = build_qwen_messages((system_prompt or "").strip(), pil_images, (prompt_text or "").strip())
        inputs, _ = self._prepare_inputs(messages, pil_images)
        return self._generate(inputs)

    # --- PromptBasedVLM API --------------------------------------------
    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ):
        pil = bgr_to_pil_rgb(image_bgr)
        height, width = image_bgr.shape[:2]

        decoded = self._generate_sequences([pil], prompt, system_prompt)
        first_text = decoded[0] if decoded else ""
        aggregated_text = "\n---\n".join(decoded) if len(decoded) > 1 else first_text

        raw_detections = parse_bboxes_qwen_style(first_text or "", threshold)
        detections = self._sanitize_detections(raw_detections, width, height)
        properties = parse_entity_properties(first_text or "", threshold=threshold)
        relations = parse_entity_relations(first_text or "", threshold=threshold)
        return detections, properties, relations, (aggregated_text or first_text or "")

    def infer_from_batch(
        self,
        frames_with_ts: List[tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")
        if not frames_with_ts:
            return ""

        pil_with_ts = [
            (Image.fromarray(img[:, :, ::-1]), float(ts))
            for img, ts in frames_with_ts
        ]
        pil_images = [img for img, _ in pil_with_ts]
        messages = build_qwen_timed_messages(
            (system_prompt or "").strip(),
            pil_with_ts,
            (prompt or "").strip(),
        )
        inputs, _ = self._prepare_inputs(messages, pil_images)
        decoded = self._generate(inputs)
        return "\n---\n".join(decoded) if len(decoded) > 1 else (decoded[0] if decoded else "")

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")
        return infer_video_qwen_style(
            system_prompt=(system_prompt or "").strip() or None,
            processor=self.processor,
            model=self.model,
            device=self.device,
            video_path=video_path,
            requested_fps=fps,
            prompt=prompt,
            max_new_tokens=int(self._get_generation_value("max_new_tokens")),
            image_patch_size=self._image_patch_size,
        )

    # --- Generation parameter support ------------------------------------
    def generation_config_spec(self) -> dict[str, dict]:
        return {
            "max_new_tokens": {"type": "int", "default": 400, "min": 16, "max": 4096, "step": 16, "label": "Max New Tokens"},
            "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "label": "Temperature"},
            "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Top-p"},
            "top_k": {"type": "int", "default": 50, "min": 1, "max": 2048, "step": 1, "label": "Top-k"},
            "do_sample": {"type": "bool", "default": False, "label": "Enable Sampling"},
            "num_return_sequences": {"type": "int", "default": 1, "min": 1, "max": 8, "step": 1, "label": "Return Seqs"},
            "repetition_penalty": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.01, "label": "Repetition Penalty"},
        }

    # Generation parameter management inherited from GenerativeVLMBase

    # --- Chat support -----------------------------------------------------
    def chat_infer(self, conversation, threshold: float = 0.0) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL adapter not loaded. Call load() first.")
        if isinstance(conversation, str):
            messages = build_qwen_messages(None, [], conversation)
        else:
            messages = conversation
        pil_images = gather_pil_images(messages)
        inputs, _ = self._prepare_inputs(messages, pil_images)
        decoded = self._generate(inputs)
        return decoded[0] if decoded else ""

    def build_chat_messages(self, history: List[dict], system_prompt: Optional[str], include_media: bool = True):
        return build_qwen_conversation_messages(
            (system_prompt or ""),
            history,
            include_media=include_media,
        )
