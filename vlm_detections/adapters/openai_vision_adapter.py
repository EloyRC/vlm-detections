from __future__ import annotations

from typing import List, Optional
import base64
import io
import os

import numpy as np
from PIL import Image

from vlm_detections.core.adapter_base import PromptBasedVLM, Detection, EntityProperty, EntityRelation
from vlm_detections.utils.bbox_parser import parse_bboxes_openai_style, clamp_detections_to_image
from vlm_detections.utils.entity_parser import parse_entity_properties, parse_entity_relations

import logging
logger = logging.getLogger(__name__)

DEFAULT_MIN_PIXELS = 64 * 32 * 32
DEFAULT_MAX_PIXELS = 9800 * 32 * 32
DEFAULT_VIDEO_MAX_FRAMES = 256
MAX_CHAT_MEDIA_PARTS = 8


class OpenAIVisionAdapter(PromptBasedVLM):
    """
    Generic adapter that calls an OpenAI-compatible Chat Completions endpoint (vision capability)
    with an image and textual prompt, expects the model to return JSON objects containing bboxes.

    Set your API key in the environment (OPENAI_API_KEY). Optionally provide a custom API base URL.
    """

    def __init__(self, model_id: str | None = None, base_url: Optional[str] = None):
        self.device = "api"
        self.client = None
        self.model_id = model_id or "gpt-4o-mini"
        self.base_url = base_url
        # Runtime generation params (initialized after load using spec defaults)
        self.gen_params: dict[str, object] = {}

    def name(self) -> str:
        return f"OpenAI Vision (API) ({self.model_id})"

    def load(self, device: str = "auto") -> None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "Install the OpenAI Python client: pip install openai"
            ) from e

        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        )
        api_base = self.base_url or os.getenv("OPENAI_BASE_HTTP_API_URL")

        # Initialize client; if base_url provided, use it (supports OpenAI-compatible endpoints)
        if api_base:
            self.client = OpenAI(base_url=api_base, api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key or None)
        # Initialize generation params with defaults
        spec = self.generation_config_spec()
        self.gen_params = {k: meta.get("default") for k, meta in spec.items()}

    def _image_to_data_url(self, image_bgr: np.ndarray) -> str:
        """Encode image as a base64 data URL."""
        image_rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _format_seconds(self, value: float) -> str:
        seconds = max(float(value), 0.0)
        text = f"{seconds:.3f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _image_message_item(self, image_bgr: np.ndarray) -> dict:
        min_pixels = int(self.gen_params.get("min_pixels", DEFAULT_MIN_PIXELS))
        max_pixels = int(self.gen_params.get("max_pixels", DEFAULT_MAX_PIXELS))
        return {
            "type": "image_url",
            "image_url": {"url": self._image_to_data_url(image_bgr)},
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }

    def _image_message_from_pil(self, pil_img: Image.Image | None) -> Optional[dict]:
        if pil_img is None:
            return None
        try:
            pil_rgb = pil_img.convert("RGB")
        except Exception:
            return None
        try:
            rgb = np.array(pil_rgb)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        bgr = rgb[:, :, ::-1]
        return self._image_message_item(bgr)

    @staticmethod
    def _rescale_detections(
        detections: List[Detection],
        width: int,
        height: int,
    ) -> List[Detection]:
        if not detections:
            return []
        sanitized: List[Detection] = []
        w = float(max(width, 1))
        h = float(max(height, 1))
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            coords = [abs(float(v)) for v in (x1, y1, x2, y2)]
            max_abs = max(coords) if coords else 0.0
            if max_abs <= 1.0005:
                x1 *= w
                x2 *= w
                y1 *= h
                y2 *= h
            elif max_abs <= 1000.0005:
                x1 = x1 / 1000.0 * w
                x2 = x2 / 1000.0 * w
                y1 = y1 / 1000.0 * h
                y2 = y2 / 1000.0 * h

            x1 = max(0.0, min(float(x1), w - 1))
            y1 = max(0.0, min(float(y1), h - 1))
            x2 = max(0.0, min(float(x2), w - 1))
            y2 = max(0.0, min(float(y2), h - 1))
            if x2 > x1 and y2 > y1:
                sanitized.append(Detection((x1, y1, x2, y2), det.score, det.label))
        return sanitized

    def _chat_completion(self, messages: List[dict]) -> List[str]:
        if self.client is None:
            raise RuntimeError("OpenAI Vision adapter not loaded. Call load() first.")
        temperature = float(self.gen_params.get("temperature", 0.0))
        top_p = float(self.gen_params.get("top_p", 1.0))
        top_k = int(self.gen_params.get("top_k", 0))
        repetition_penalty = float(self.gen_params.get("repetition_penalty", 1.0))
        n = int(self.gen_params.get("num_return_sequences", 1))
        do_sample = bool(self.gen_params.get("do_sample", False))
        max_tokens = int(self.gen_params.get("max_new_tokens", 400))
        create_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        }
        if do_sample:
            create_kwargs["top_p"] = top_p
        if top_k > 0:
            create_kwargs["top_k"] = top_k
        if repetition_penalty != 1.0:
            create_kwargs["repetition_penalty"] = repetition_penalty
        try:
            msgs = create_kwargs.pop("messages", [])
            logger.info(f"OpenAI Chat Completion request:\n{create_kwargs}")
            create_kwargs["messages"] = msgs
            resp = self.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e
        contents: List[str] = []
        if resp and resp.choices:
            for ch in resp.choices:
                if getattr(ch, "message", None) and getattr(ch.message, "content", None):
                    contents.append(ch.message.content)
        return contents

    # Prompts are provided by the caller; no in-code default instruction here.

    def infer_from_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ):
        height, width = image_bgr.shape[:2]
        prompt_text = (prompt or "").strip()

        messages = []
        sys_prompt = (system_prompt or "").strip()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        user_content: List[dict] = []
        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        user_content.append(self._image_message_item(image_bgr))
        messages.append({"role": "user", "content": user_content})
        contents = self._chat_completion(messages)
        primary = contents[0] if contents else ""
        detections = parse_bboxes_openai_style(primary or "", threshold)
        detections = self._rescale_detections(detections, width, height)
        properties = parse_entity_properties(primary or "", threshold=threshold)
        relations = parse_entity_relations(primary or "", threshold=threshold)
        # Aggregate raw output (all sequences separated by \n---\n)
        raw_text = "\n---\n".join(contents) if contents else primary
        return detections, properties, relations, raw_text

    def infer_from_batch(
        self,
        frames_with_ts: List[tuple[np.ndarray, float]],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI Vision adapter not loaded. Call load() first.")
        if not frames_with_ts:
            return ""

        prompt_text = (prompt or "").strip()

        messages = []
        sys_prompt = (system_prompt or "").strip()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        user_content: List[dict] = []
        for frame_bgr, ts in frames_with_ts:
            user_content.append({"type": "text", "text": f"<{self._format_seconds(ts)} seconds>"})
            user_content.append(self._image_message_item(frame_bgr))
        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})

        contents = self._chat_completion(messages)
        if not contents:
            return ""
        return "\n---\n".join(contents) if len(contents) > 1 else contents[0]

    def infer_from_video(
        self,
        video_path: str,
        fps: float,
        classes: List[str],
        prompt: str,
        system_prompt: Optional[str],
        threshold: float,
    ) -> str:
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")
        max_frames = int(self.gen_params.get("video_max_frames", DEFAULT_VIDEO_MAX_FRAMES))
        frames_with_ts: List[tuple[np.ndarray, float]] = []
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenCV (cv2) is required for video inference in OpenAIVisionAdapter."
            ) from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 0:
            native_fps = float(max(fps, 0.1))
        target_fps = float(fps if fps > 0 else native_fps)
        target_fps = max(0.1, target_fps)
        frame_period = 1.0 / target_fps

        next_capture_time = 0.0
        frame_index = 0
        while len(frames_with_ts) < max_frames:
            success, frame = cap.read()
            if not success:
                break
            current_time = frame_index / native_fps if native_fps > 0 else frame_index * frame_period
            if current_time + 1e-6 >= next_capture_time:
                frames_with_ts.append((frame.copy(), current_time))
                next_capture_time = current_time + frame_period
            frame_index += 1
        cap.release()

        if not frames_with_ts:
            return ""
        batch_result = self.infer_from_batch(frames_with_ts, prompt, system_prompt, threshold)
        if isinstance(batch_result, tuple):
            return batch_result[1]
        return batch_result

    # --- Chat support ---
    def chat_infer(self, conversation, threshold: float = 0.0) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI Vision adapter not loaded. Call load() first.")
        if isinstance(conversation, str):
            text = conversation.strip()
            if not text:
                return ""
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        else:
            messages = conversation
        contents = self._chat_completion(messages)
        return contents[0] if contents else ""

    def build_chat_messages(self, history: List[dict], system_prompt: Optional[str], include_media: bool = True):
        messages: List[dict] = []
        budget = MAX_CHAT_MEDIA_PARTS if include_media else 0
        sys_prompt = (system_prompt or "").strip()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        for entry in history:
            role = entry.get("role", "user")
            msg_type = entry.get("type", "text")
            content_text = str(entry.get("content", "") or "")
            meta = entry.get("meta") or {}

            if role == "user":
                parts: List[dict] = []
                if include_media and budget > 0:
                    if msg_type == "image":
                        pil_img = meta.get("pil_image") if isinstance(meta, dict) else None
                        image_item = self._image_message_from_pil(pil_img)
                        if image_item:
                            parts.append(image_item)
                            budget -= 1
                    elif msg_type == "video":
                        video_url = ""
                        if isinstance(meta, dict):
                            video_url = str(meta.get("video_path") or "").strip()
                        if video_url.startswith(("http://", "https://")) and budget > 0:
                            parts.append({"type": "video_url", "video_url": {"url": video_url}})
                            budget -= 1
                        else:
                            if isinstance(meta, dict):
                                frame_images = meta.get("frame_images") or []
                                frame_ts = meta.get("frame_timestamps") or []
                            else:
                                frame_images = []
                                frame_ts = []
                            for idx, frame in enumerate(frame_images):
                                if budget <= 0:
                                    break
                                image_item = None
                                if isinstance(frame, Image.Image):
                                    image_item = self._image_message_from_pil(frame)
                                elif isinstance(frame, np.ndarray):
                                    image_item = self._image_message_item(frame)
                                if not image_item:
                                    continue
                                if idx < len(frame_ts):
                                    try:
                                        ts_val = float(frame_ts[idx])
                                        parts.append({"type": "text", "text": f"<{self._format_seconds(ts_val)} seconds>"})
                                    except Exception:
                                        pass
                                parts.append(image_item)
                                budget -= 1
                                if budget <= 0:
                                    break
                if content_text:
                    parts.append({"type": "text", "text": content_text})
                if not parts:
                    placeholder = "[media omitted]" if msg_type in {"image", "video"} else "(empty)"
                    parts.append({"type": "text", "text": placeholder})
                messages.append({"role": "user", "content": parts})
            else:
                assistant_text = content_text or "(empty)"
                messages.append({"role": "assistant", "content": assistant_text})
        return messages

    # --- Generation parameter support ---
    def generation_config_spec(self) -> dict[str, dict]:
        return {
            "max_new_tokens": {"type": "int", "default": 400, "min": 16, "max": 4096, "step": 16, "label": "Max New Tokens"},
            "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "label": "Temperature"},
            "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Top-p"},
            "top_k": {"type": "int", "default": 0, "min": 0, "max": 2048, "step": 1, "label": "Top-k (0=auto)"},
            "do_sample": {"type": "bool", "default": False, "label": "Enable Sampling"},
            "num_return_sequences": {"type": "int", "default": 1, "min": 1, "max": 8, "step": 1, "label": "Return Seqs"},
            "repetition_penalty": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.01, "label": "Repetition Penalty"},
            "min_pixels": {"type": "int", "default": DEFAULT_MIN_PIXELS, "min": 1024, "max": 20480 * 32 * 32, "step": 1024, "label": "Min Pixels"},
            "max_pixels": {"type": "int", "default": DEFAULT_MAX_PIXELS, "min": DEFAULT_MIN_PIXELS, "max": 20480 * 32 * 32, "step": 1024, "label": "Max Pixels"},
            "video_max_frames": {"type": "int", "default": DEFAULT_VIDEO_MAX_FRAMES, "min": 8, "max": 1024, "step": 1, "label": "Video Max Frames"},
        }

    def update_generation_params(self, params: dict[str, object]) -> None:
        spec = self.generation_config_spec()
        for k, v in params.items():
            if k in spec:
                self.gen_params[k] = v
