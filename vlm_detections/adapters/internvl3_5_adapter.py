from __future__ import annotations

"""Adapter for OpenGVLab InternVL3.5 models.

Supports single-image, multi-image (concatenated tiles) and video frame sampling similar
to Qwen style adapters. Provides optional reasoning mode by injecting a system prompt
with <think> tags (mirrors TREX style reasoning capability as requested).

Model reference: OpenGVLab/InternVL3_5-8B

Notes:
- InternVL3.5 exposes a `.chat(tokenizer, pixel_values, question, generation_config, **kwargs)` API
  when loaded with `trust_remote_code=True`.
- We implement preprocessing (dynamic tiling + optional thumbnail) adapted from authors' script.
- For detection style tasks we rely on prompt templates to enforce JSON output; parsing logic
  mirrors Qwen/TRex adapters for bbox extraction.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import math

import numpy as np
from PIL import Image

from vlm_detections.core.adapter_base import PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.utils.image_utils import bgr_to_pil_rgb
from vlm_detections.utils.bbox_parser import parse_bboxes_qwen_style, clamp_detections_to_image
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations

if TYPE_CHECKING:  # pragma: no cover
    import torch  # only for type hints


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios, width: int, height: int, image_size: int):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    tiles_x = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_x) * image_size,
            (i // tiles_x) * image_size,
            ((i % tiles_x) + 1) * image_size,
            ((i // tiles_x) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _prepare_internvl_image_tensor(pil: Image.Image, input_size: int = 448, max_num: int = 12) -> 'torch.Tensor':
    import torch
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(pil, image_size=input_size, max_num=max_num, use_thumbnail=True)
    tensors = [transform(t) for t in tiles]
    return torch.stack(tensors)


class InternVL35Adapter(PromptBasedVLM):
    def __init__(self, model_id: str = "OpenGVLab/InternVL3_5-8B"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.gen_params: dict[str, object] = {}
        self.input_size = 448
        self.max_tiles = 12
        # Reasoning is controlled via prompt presets; no dedicated generation param.
        self.reasoning_enabled = False

    def name(self) -> str:
        return f"InternVL3.5 ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)

        spec = self.generation_config_spec()
        self.gen_params = {k: meta.get("default") for k, meta in spec.items()}

    # --- Core inference helpers ---
    def _build_question(self, base_user_prompt: str, include_image_token: bool = True) -> str:
        # InternVL expects '<image>' token preceding image-related question
        if include_image_token and '<image>' not in base_user_prompt:
            return f"<image>\n{base_user_prompt}".rstrip()
        return base_user_prompt

    def _apply_system_prompt(self, system_prompt: Optional[str]) -> None:
        # Apply caller-provided system prompt when supported by the model.
        sys_prompt = (system_prompt or "").strip()
        try:
            # InternVL uses model.system_message attribute similar to sample code
            self.model.system_message = sys_prompt  # type: ignore[attr-defined]
        except Exception:
            pass

    def _decode_generation(self, output_text: str) -> str:
        # The model.chat already returns a plain string; nothing special needed.
        return output_text or ""

    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ):
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("InternVL3.5 adapter not loaded. Call load() first.")

        pil = bgr_to_pil_rgb(image_bgr)
        h, w = image_bgr.shape[:2]
        pixel_values = _prepare_internvl_image_tensor(pil, input_size=self.input_size, max_num=self.max_tiles)
        pixel_values = pixel_values.to(self.model.dtype)  # type: ignore
        if self.device == 'cuda':
            pixel_values = pixel_values.cuda()

        # Render user prompt with placeholders
        rendered = (prompt or "").strip()
        question = self._build_question(rendered, include_image_token=True)
        self._apply_system_prompt(system_prompt)
        generation_config = {
            "max_new_tokens": int(self.gen_params.get("max_new_tokens", 512)),
            "do_sample": bool(self.gen_params.get("do_sample", False)),
            "temperature": float(self.gen_params.get("temperature", 0.0)),
            "top_p": float(self.gen_params.get("top_p", 1.0)),
        }
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)  # type: ignore
        text = self._decode_generation(response)
        detections = parse_bboxes_qwen_style(text, threshold)
        clamped = clamp_detections_to_image(detections, w, h)
        properties = parse_entity_properties(text, threshold=threshold)
        relations = parse_entity_relations(text, threshold=threshold)
        return clamped, properties, relations, text

    def infer_from_batch(
        self,
        frames_with_ts: List[tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("InternVL3.5 adapter not loaded. Call load() first.")
        if not frames_with_ts:
            return ""
        tensors = []
        widths = []
        heights = []
        for img, _ in frames_with_ts:
            pil = bgr_to_pil_rgb(img)
            heights.append(img.shape[0])
            widths.append(img.shape[1])
            t = _prepare_internvl_image_tensor(pil, input_size=self.input_size, max_num=self.max_tiles)
            tensors.append(t)
        pixel_values = torch.cat(tensors, dim=0).to(self.model.dtype)  # type: ignore
        if self.device == 'cuda':
            pixel_values = pixel_values.cuda()
        rendered = (prompt or "").strip()
        question = self._build_question(rendered, include_image_token=True)
        self._apply_system_prompt(system_prompt)
        generation_config = {
            "max_new_tokens": int(self.gen_params.get("max_new_tokens", 512)),
            "do_sample": bool(self.gen_params.get("do_sample", False)),
            "temperature": float(self.gen_params.get("temperature", 0.0)),
            "top_p": float(self.gen_params.get("top_p", 1.0)),
        }
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)  # type: ignore
        return self._decode_generation(response)

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        # Sample frames uniformly to approximately requested FPS up to a modest limit (e.g., 8 frames)
        # to avoid heavy memory usage.
        try:
            from decord import VideoReader, cpu
        except Exception:
            return "(decord not installed; video unsupported)"
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("InternVL3.5 adapter not loaded. Call load() first.")
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            return f"Failed to read video: {e}"
        video_len = len(vr)
        # Estimate number of frames based on provided fps vs average fps
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:
            native_fps = fps if fps > 0 else 8.0
        if fps <= 0:
            fps = min(4.0, native_fps)
        # target frames = min(video_len, requested duration * fps)
        duration_sec = video_len / native_fps if native_fps > 0 else video_len / 8.0
        target_frames = int(min(video_len, max(2, duration_sec * fps)))
        target_frames = max(2, min(target_frames, 16))  # cap at 16 frames
        indices = np.linspace(0, video_len - 1, target_frames).astype(int)
        frame_tensors = []
        for idx in indices:
            arr = vr[idx].asnumpy()
            pil = Image.fromarray(arr).convert('RGB')
            t = _prepare_internvl_image_tensor(pil, input_size=self.input_size, max_num=1)  # no tiling for speed
            frame_tensors.append(t)
        if not frame_tensors:
            return "(no frames extracted)"
        pixel_values = torch.cat(frame_tensors, dim=0).to(self.model.dtype)  # type: ignore
        if self.device == 'cuda':
            pixel_values = pixel_values.cuda()
        frame_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frame_tensors))])
        base_user = (prompt or "").strip()
        question = frame_prefix + base_user
        self._apply_system_prompt(system_prompt)
        generation_config = {
            "max_new_tokens": int(self.gen_params.get("max_new_tokens", 512)),
            "do_sample": bool(self.gen_params.get("do_sample", False)),
            "temperature": float(self.gen_params.get("temperature", 0.0)),
            "top_p": float(self.gen_params.get("top_p", 1.0)),
        }
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)  # type: ignore
        return self._decode_generation(response)

    # --- Generation parameter spec ---
    def generation_config_spec(self) -> dict[str, dict]:
        return {
            "max_new_tokens": {"type": "int", "default": 512, "min": 32, "max": 4096, "step": 16, "label": "Max New Tokens"},
            "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "label": "Temperature"},
            "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Top-p"},
            "do_sample": {"type": "bool", "default": False, "label": "Enable Sampling"},
        }

    def update_generation_params(self, params: dict[str, object]) -> None:
        spec = self.generation_config_spec()
        for k, v in params.items():
            if k in spec:
                self.gen_params[k] = v
