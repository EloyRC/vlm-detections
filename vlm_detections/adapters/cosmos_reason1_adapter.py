from __future__ import annotations

"""
Cosmos-Reason1 VLM adapter

This adapter is derived from the provided inference script for Cosmos-Reason1.
It converts the script's CLI arguments into constructor parameters and exposes
the common PromptBasedVLM interface for single-image zero-shot detection. It
loads prompts/configs from `vlm_detections/config/cosmos` (historical path prior to migration: `src/config/cosmos`).

Notes:
- We build a chat conversation with a single image and an instruction that
    requests JSON detections. If a `prompt_yaml` is provided, we use its
    system/user prompts; otherwise we synthesize a detection-oriented prompt
    incorporating provided classes.
- Output text is parsed for JSON boxes; we accept either a JSON array or
    inline JSON objects. Boxes are absolute pixel coordinates [x1,y1,x2,y2].
"""

from typing import List, Tuple, Optional
import json
import re
import tempfile
import pathlib
import cv2

import numpy as np
from PIL import Image

from vlm_detections.core.adapter_base import PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.utils.image_utils import bgr_to_pil_rgb
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations
from vlm_detections.utils.qwen_message_utils import (
    build_qwen_messages,
    build_qwen_timed_messages,
    prepare_qwen_inputs_with_length_guard,
    validate_qwen_token_length,
    move_inputs_to_device,
    infer_video_qwen_style,
)

import logging
logger = logging.getLogger(__name__)


def _default_config_dir() -> pathlib.Path:
    # vlm_detections/adapters/ -> parents[1] == vlm_detections/
    return pathlib.Path(__file__).parents[1] / "config" / "cosmos"


def _parse_detections_from_text(text: str, threshold: float) -> List[Detection]:
    """Parse detections from model text output.

    Accepts either a JSON array of objects or multiple inline JSON objects.
    Each object should contain:
      - bbox_2d / bbox / bounding_box / box: [x1, y1, x2, y2]
      - label/class/name: string
      - score/confidence: optional 0..1
    """
    dets: List[Detection] = []

    # Try a top-level JSON array first
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("label") or obj.get("class") or obj.get("name") or "object"
                score = float(obj.get("score") or obj.get("confidence") or 1.0)
                box = obj.get("bbox_2d") or obj.get("bbox") or obj.get("bounding_box") or obj.get("box")
                if isinstance(box, (list, tuple)) and len(box) == 4 and score >= threshold:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    dets.append(Detection((x1, y1, x2, y2), score, str(label)))
            return dets
    except Exception:
        pass

    # Fallback: scan for inline JSON objects
    for m in re.finditer(r"\{[^{}]*?\}", text or ""):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        label = obj.get("label") or obj.get("class") or obj.get("name") or "object"
        score = float(obj.get("score") or obj.get("confidence") or 1.0)
        box = obj.get("bbox_2d") or obj.get("bbox") or obj.get("bounding_box") or obj.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4 and score >= threshold:
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append(Detection((x1, y1, x2, y2), score, str(label)))
    return dets


class CosmosReason1Adapter(PromptBasedVLM):
    def __init__(
        self,
        model: str = "nvidia/Cosmos-Reason1-7B",
        revision: Optional[str] = None,
        reasoning: bool = True,
        vision_config_path: Optional[str] = None,
        sampling_params_path: Optional[str] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Constructor mirrors the script CLI options as adapter parameters.

        - model: Hugging Face model id/path for Cosmos-Reason1
        - revision: optional model revision/tag/commit
    - prompt_yaml: path to a prompt YAML under vlm_detections/config/cosmos/prompts; if None, a detection prompt is synthesized
        - question: optional user prompt override (used when prompt_yaml is provided)
        - reasoning: include reasoning addon (<think>) if compatible with the prompt template
        - timestamp: overlay timestamp on video frames (ignored for single image)
    - vision_config_path: config yaml path (defaults to vlm_detections/config/cosmos/configs/vision_config.yaml)
    - sampling_params_path: sampling params yaml (defaults to vlm_detections/config/cosmos/configs/sampling_params.yaml)
        - verbose: enable verbose prints
        - output_dir: optional directory to dump intermediate tensors/images
        """
        self.model_id = model
        self.revision = revision
        base_dir = _default_config_dir()
        self.reasoning = reasoning
        self.verbose = verbose
        self.output_dir = output_dir

    # Default configs under vlm_detections/config/cosmos
        self.vision_config_path = vision_config_path or str(base_dir / "configs" / "vision_config.yaml")
        self.sampling_params_path = sampling_params_path or str(base_dir / "configs" / "sampling_params.yaml")
        self.prompts_dir = base_dir / "prompts"

        # Lazy-initialized runtime state
        self.model = None
        self.processor = None
        self.gen_params = {}
        self._gen_spec_cache: dict[str, dict] | None = None

    def name(self) -> str:
        return f"Cosmos-Reason1 ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        # cosmos_reason1_utils should be installed as per user note.
        import transformers
        import torch

        # The original script uses device_map="auto"; we stick to that for best compatibility.
        # torch_dtype="auto" lets HF pick fp16/bf16 when appropriate.
        self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype="auto",
            device_map="auto",
        )
        # Processor matches the model family
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_id, revision=self.revision)

        # Put model in eval mode
        try:
            self.model.eval()
        except Exception:
            pass
        # Informational only
        self.device = "auto"
        # Initialize generation params
        self._gen_spec_cache = None
        self._reset_generation_defaults()
        self._apply_sampling_defaults()

    def _load_sampling(self) -> dict:
        import yaml
        try:
            sampling = yaml.safe_load(open(self.sampling_params_path, "rb")) or {}
        except Exception:
            sampling = {}
        return sampling

    def _apply_sampling_defaults(self) -> None:
        sampling = self._load_sampling()
        if not sampling:
            return
        mapping = {
            "max_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "n": "num_return_sequences",
            "repetition_penalty": "repetition_penalty",
            "do_sample": "do_sample",
        }
        updates: dict[str, object] = {}
        for source_key, target_key in mapping.items():
            if source_key in sampling and sampling[source_key] is not None:
                updates[target_key] = sampling[source_key]
        if updates:
            self.update_generation_params(updates)

    def _gen_spec(self) -> dict[str, dict]:
        if self._gen_spec_cache is None:
            self._gen_spec_cache = self.generation_config_spec()
        return self._gen_spec_cache

    def _coerce_generation_value(self, name: str, value: object) -> object:
        meta = self._gen_spec().get(name)
        if meta is None:
            return value
        if value is None:
            raise ValueError(f"Generation parameter '{name}' cannot be None")
        ptype = meta.get("type")
        try:
            if ptype == "int":
                coerced = int(value)  # type: ignore[arg-type]
                if "min" in meta:
                    coerced = max(int(meta["min"]), coerced)
                if "max" in meta:
                    coerced = min(int(meta["max"]), coerced)
                return coerced
            if ptype == "float":
                coerced = float(value)  # type: ignore[arg-type]
                if "min" in meta:
                    coerced = max(float(meta["min"]), coerced)
                if "max" in meta:
                    coerced = min(float(meta["max"]), coerced)
                return coerced
            if ptype == "bool":
                if isinstance(value, str):
                    coerced = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    coerced = bool(value)
                return coerced
        except (TypeError, ValueError) as exc:
            raise exc
        return value

    def _get_generation_value(self, name: str) -> object:
        spec = self._gen_spec()
        if name not in spec:
            return self.gen_params.get(name)
        raw_value = self.gen_params.get(name, spec[name].get("default"))
        try:
            coerced = self._coerce_generation_value(name, raw_value)
        except (TypeError, ValueError):
            default_val = spec[name].get("default")
            try:
                coerced = self._coerce_generation_value(name, default_val)
            except (TypeError, ValueError):
                logger.warning("Falling back to unsanitized default for generation param %s", name)
                coerced = default_val
        self.gen_params[name] = coerced
        return coerced

    def _reset_generation_defaults(self) -> None:
        defaults: dict[str, object] = {}
        for key, meta in self._gen_spec().items():
            default_val = meta.get("default")
            try:
                defaults[key] = self._coerce_generation_value(key, default_val)
            except (TypeError, ValueError):
                logger.warning("Invalid default %r for generation param %s", default_val, key)
                defaults[key] = default_val
        self.gen_params = defaults

    def _build_generation_kwargs(self) -> dict[str, object]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        max_new_tokens = int(self._get_generation_value("max_new_tokens"))
        n_return = int(self._get_generation_value("num_return_sequences"))
        do_sample = bool(self._get_generation_value("do_sample"))
        temperature = float(self._get_generation_value("temperature"))
        top_p = float(self._get_generation_value("top_p"))
        top_k = int(self._get_generation_value("top_k"))
        rep_penalty = float(self._get_generation_value("repetition_penalty"))

        kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": n_return,
            "do_sample": do_sample,
            "use_cache": True,
        }

        if tokenizer is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
            if eos_id is not None:
                kwargs["eos_token_id"] = eos_id
            if pad_id is not None:
                kwargs["pad_token_id"] = pad_id

        if do_sample:
            if temperature <= 0.0:
                temperature = 1e-4
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            if top_k > 0:
                kwargs["top_k"] = top_k
        if rep_penalty != 1.0:
            kwargs["repetition_penalty"] = rep_penalty
        return kwargs

    def _generate(self, inputs) -> List[str]:  # type: ignore
        if self.model is None or self.processor is None:
            raise RuntimeError("Cosmos-Reason1 adapter not loaded. Call load() first.")
        from torch import no_grad
        import torch

        validate_qwen_token_length(self.model, self.processor, self.name(), inputs)
        try:
            target_device = self.model.device  # type: ignore[attr-defined]
            inputs = inputs.to(target_device)
        except Exception:
            fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = move_inputs_to_device(inputs, fallback_device)

        gen_kwargs = self._build_generation_kwargs()
        logger.info("Generating with kwargs: %s", gen_kwargs)
        with no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)

        try:
            input_len = inputs["input_ids"].shape[1]
            gen_only = generated[:, input_len:]
        except Exception:
            gen_only = generated
        return self.processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        classes: List[str],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> List[Detection]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Cosmos-Reason1 adapter not loaded. Call load() first.")
        h, w = image_bgr.shape[:2]
        # Convert to PIL and build Qwen-style messages using shared utils
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        messages = build_qwen_messages((system_prompt or "").strip(), [pil], (prompt or "").strip())
        inputs, _text = prepare_qwen_inputs_with_length_guard(
            self.processor,
            self.model,
            self.name(),
            messages,
            [pil],
        )
        texts = self._generate(inputs)
        logger.info(f"Generated texts: {texts}")
        first_text = (texts[0] if texts else "").rstrip()
        aggregated_text = "\n---\n".join(texts).rstrip() if len(texts) > 1 else first_text

        # Parse detections from the first sequence only
        dets = _parse_detections_from_text(first_text, threshold)
        # Clamp to image bounds
        clamped: List[Detection] = []
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            x1 = max(0.0, min(float(x1), float(w - 1)))
            y1 = max(0.0, min(float(y1), float(h - 1)))
            x2 = max(0.0, min(float(x2), float(w - 1)))
            y2 = max(0.0, min(float(y2), float(h - 1)))
            if x2 > x1 and y2 > y1 and d.score >= threshold:
                clamped.append(Detection((x1, y1, x2, y2), float(d.score), d.label))
        properties = parse_entity_properties(first_text, threshold=threshold)
        relations = parse_entity_relations(first_text, threshold=threshold)
        return clamped, properties, relations, aggregated_text

    def infer_from_batch(
        self,
        frames_with_ts: List[Tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Cosmos-Reason1 adapter not loaded. Call load() first.")
        if not frames_with_ts:
            return ""

        pil_with_ts = [
            (bgr_to_pil_rgb(im), float(ts))
            for im, ts in frames_with_ts
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
        texts = self._generate(inputs)
        logger.info(f"Generated texts for batch of size {len(frames_with_ts)}: {texts}")
        if len(texts) > 1:
            return "\n---\n".join([t.rstrip() for t in texts])
        return (texts[0].rstrip() if texts else "")

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Cosmos-Reason1 adapter not loaded. Call load() first.")

        # Delegate to shared Qwen-style video inference, using Cosmos model key and reasoning flag
        import torch
        return infer_video_qwen_style(
            system_prompt=(system_prompt or "").strip() or None,
            processor=self.processor,
            model=self.model,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            video_path=video_path,
            requested_fps=fps,
            prompt=prompt,
            max_new_tokens=int(self._get_generation_value("max_new_tokens")),
        )

    # --- Generation parameter support ---
    def generation_config_spec(self) -> dict[str, dict]:
        return {
            "max_new_tokens": {"type": "int", "default": 512, "min": 32, "max": 8192, "step": 32, "label": "Max New Tokens"},
            "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "label": "Temperature"},
            "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Top-p"},
            "top_k": {"type": "int", "default": 0, "min": 0, "max": 2048, "step": 1, "label": "Top-k (0=disabled)"},
            "do_sample": {"type": "bool", "default": False, "label": "Enable Sampling"},
            "num_return_sequences": {"type": "int", "default": 1, "min": 1, "max": 8, "step": 1, "label": "Return Seqs"},
            "repetition_penalty": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.01, "label": "Repetition Penalty"},
        }

    def update_generation_params(self, params: dict[str, object]) -> None:
        spec = self._gen_spec()
        for k, v in params.items():
            if k not in spec:
                continue
            try:
                self.gen_params[k] = self._coerce_generation_value(k, v)
            except (TypeError, ValueError):
                logger.warning("Invalid value %r for generation param %s", v, k)
