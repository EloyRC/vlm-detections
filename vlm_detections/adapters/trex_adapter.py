from __future__ import annotations

from typing import List, Tuple, Optional
import logging

import numpy as np
from PIL import Image

from vlm_detections.core.adapter_base import PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.core.generative_vlm_base import GenerativeVLMBase
from vlm_detections.utils.image_utils import bgr_to_pil_rgb
from vlm_detections.utils.qwen_message_utils import (
    build_qwen_conversation_messages,
    build_qwen_messages,
    build_qwen_timed_messages,
    move_inputs_to_device,
    infer_video_qwen_style,
    prepare_qwen_inputs_with_length_guard,
    validate_qwen_token_length,
)
from vlm_detections.utils.bbox_parser import parse_bboxes_qwen_style, clamp_detections_to_image
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations

logger = logging.getLogger(__name__)


class TRexAdapter(PromptBasedVLM, GenerativeVLMBase):
    """
    Adapter for IDEA-Research Rex-Thinker models (Qwen2.5-VL compatible interface).
    Expects caller-provided prompts (system and user) and avoids hidden global state.
    """

    def __init__(self, model_id: str = "IDEA-Research/Rex-Thinker-GRPO-7B"):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.gen_params: dict[str, object] = {}
        self._gen_spec_cache: dict[str, dict] | None = None

    def name(self) -> str:
        return f"TRex ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Choose dtype
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._gen_spec_cache = None
        self._reset_generation_defaults()

    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ):
        if self.model is None or self.processor is None:
            raise RuntimeError("TRex adapter not loaded. Call load() first.")

        # Convert image to PIL RGB
        pil = bgr_to_pil_rgb(image_bgr)
        h, w = image_bgr.shape[:2]

        messages = build_qwen_messages((system_prompt or "").strip(), [pil], (prompt or "").strip())
        inputs, _text = prepare_qwen_inputs_with_length_guard(
            self.processor,
            self.model,
            self.name(),
            messages,
            [pil],
        )

        decoded = self._generate(inputs)
        first_text = decoded[0] if decoded else ""
        aggregated_text = "\n---\n".join(decoded) if len(decoded) > 1 else first_text

        detections = parse_bboxes_qwen_style(first_text or "", threshold)
        clamped = clamp_detections_to_image(detections, w, h)
        properties = parse_entity_properties(first_text or "", threshold=threshold)
        relations = parse_entity_relations(first_text or "", threshold=threshold)
        return clamped, properties, relations, (aggregated_text or first_text or "")

    def infer_from_batch(
        self,
        frames_with_ts: List[tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        """Single call with multiple images passed through the chat template."""
        if self.model is None or self.processor is None:
            raise RuntimeError("TRex adapter not loaded. Call load() first.")
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
        inputs, _text = prepare_qwen_inputs_with_length_guard(
            self.processor,
            self.model,
            self.name(),
            messages,
            pil_images,
        )

        decoded = self._generate(inputs)
        if len(decoded) > 1:
            return "\n---\n".join(decoded)
        return decoded[0] if decoded else ""

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("TRex adapter not loaded. Call load() first.")
        max_new_tokens = int(self._get_generation_value("max_new_tokens"))
        return infer_video_qwen_style(
            system_prompt=(system_prompt or "").strip() or None,
            processor=self.processor,
            model=self.model,
            device=self.device,
            video_path=video_path,
            requested_fps=fps,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    # --- Generation parameter support ---
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

    # --- Shared generation helper ---
    def _generate(self, inputs) -> List[str]:  # type: ignore
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")
        from torch import no_grad
        validate_qwen_token_length(self.model, self.processor, self.name(), inputs)
        inputs = move_inputs_to_device(inputs, self.device)
        max_new_tokens = int(self._get_generation_value("max_new_tokens"))
        n_return = int(self._get_generation_value("num_return_sequences"))
        do_sample = bool(self._get_generation_value("do_sample"))
        rep_penalty = float(self._get_generation_value("repetition_penalty"))
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
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
        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        return decoded

    # --- Chat support ---
    def chat_infer(self, conversation, threshold: float = 0.0) -> str:
        """Multi-turn inference.

        conversation: either a pre-built messages list (preferred) or a raw string.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("TRex adapter not loaded. Call load() first.")
        if isinstance(conversation, str):
            messages = build_qwen_messages(None, [], conversation)
        else:
            messages = conversation  # already structured (includes system if any)
        # Gather images for fallback path
        from vlm_detections.utils.qwen_message_utils import gather_pil_images
        pil_images = gather_pil_images(messages)
        inputs, _txt = prepare_qwen_inputs_with_length_guard(
            self.processor,
            self.model,
            self.name(),
            messages,
            pil_images,
        )
        decoded = self._generate(inputs)
        return decoded[0] if decoded else ""

    def build_chat_messages(self, history: List[dict], system_prompt: Optional[str], include_media: bool = True):
        return build_qwen_conversation_messages(
            (system_prompt or ""),
            history,
            include_media=include_media,
        )
