from __future__ import annotations

"""Utility helpers for Qwen2.5-VL compatible multi-modal message construction.

This consolidates repeated logic in multiple adapters (e.g., TRex, Qwen2.5-VL)
for:
  * Building chat "messages" structures incorporating optional system prompt,
    one or more images, and an optional user text prompt.
  * Preparing processor inputs using the preferred qwen_vl_utils.process_vision_info
    path when available, with a graceful fallback to a simpler processor call.

The goal is to eliminate duplication and keep adapter code focused on
model-specific nuances (generation parameters, post-processing).

Future extension: unifying video message & input preparation (currently kept in
adapters because of additional budgeting / segmentation logic).
"""

import logging
from typing import List, Optional, Any, Dict, Tuple, Sequence

from PIL import Image
import json

logger = logging.getLogger(__name__)


_TOKEN_GUARD_LIMIT = 1_000_000_000


def _normalize_limit(value: Any) -> Optional[int]:
    try:
        limit = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    if limit >= _TOKEN_GUARD_LIMIT:
        return None
    return limit


def _infer_sequence_length(input_ids: Any) -> int:
    if input_ids is None:
        return 0
    if hasattr(input_ids, "shape"):
        shape = input_ids.shape
        if len(shape) >= 2:
            return int(shape[1])
        if len(shape) == 1:
            return int(shape[0])
    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)
    try:
        return len(input_ids)  # type: ignore[arg-type]
    except TypeError:
        return 0


def validate_qwen_token_length(
    model: Any,
    processor: Any,
    adapter_name: str,
    inputs: Dict[str, Any],
    *,
    original_limit: Optional[int] = None,
) -> None:
    """Ensure token sequence does not exceed model or tokenizer limits."""

    if inputs is None:
        return
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return

    seq_len = _infer_sequence_length(input_ids)
    limits: List[int] = []

    norm_orig = _normalize_limit(original_limit)
    if norm_orig is not None:
        limits.append(norm_orig)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tok_limit = _normalize_limit(getattr(tokenizer, "model_max_length", None))
        if tok_limit is not None:
            limits.append(tok_limit)

    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "max_sequence_length", "max_context_length", "max_length"):
            cfg_limit = _normalize_limit(getattr(config, attr, None))
            if cfg_limit is not None:
                limits.append(cfg_limit)

    # Hugging Face's GenerationConfig often defaults max_length to 20 even when
    # the underlying model supports far larger contexts. Using that value would
    # produce false positives (e.g., 343 tokens > 20) despite the model being
    # capable of handling the prompt. We therefore ignore generation_config
    # completely and rely on tokenizer/model limits that reflect the actual
    # architecture or any explicit overrides applied upstream.

    if not limits:
        return

    # Many Hugging Face tokenizers expose a placeholder limit (e.g., 20) while
    # the actual model context is much larger. When at least one plausible large
    # limit exists, ignore suspiciously tiny values to avoid false positives.
    plausible_limits = [val for val in limits if val >= 128]
    allowed_candidates = plausible_limits or limits
    allowed = min(allowed_candidates)
    if seq_len > allowed:
        message = (
            f"{adapter_name} prompt is too long ({seq_len} tokens > limit {allowed}). "
            "Reduce prompt length, chat history, or batch size and try again."
        )
        logger.error(message)
        raise RuntimeError(message)


def prepare_qwen_inputs_with_length_guard(
    processor: Any,
    model: Any,
    adapter_name: str,
    messages: List[dict],
    pil_images: List[Image.Image],
) -> Tuple[Dict[str, Any], str]:
    """Prepare inputs without truncation and validate final sequence length."""

    if processor is None:
        raise RuntimeError("Processor not provided")

    tokenizer = getattr(processor, "tokenizer", None)
    original_limit = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None

    try:
        if tokenizer is not None:
            norm_limit = _normalize_limit(original_limit)
            if norm_limit is not None:
                tokenizer.model_max_length = _TOKEN_GUARD_LIMIT
        inputs, text = prepare_qwen_inputs(processor, messages, pil_images)
    finally:
        if tokenizer is not None and original_limit is not None:
            try:
                tokenizer.model_max_length = original_limit
            except Exception:
                logger.debug("Failed to restore tokenizer max length", exc_info=True)

    validate_qwen_token_length(model, processor, adapter_name, inputs, original_limit=original_limit)
    return inputs, text


def build_qwen_messages(system_prompt: Optional[str], images: List[Image.Image], prompt: Optional[str]) -> List[dict]:
    """Create a Qwen-style messages list.

    Parameters
    ----------
    system_prompt : Optional[str]
        System instruction to include (first message) if non-empty.
    images : List[Image.Image]
        One or more PIL images to include. Each becomes a content item.
    prompt : Optional[str]
        Optional user text appended after images.

    Returns
    -------
    List[dict]
        Chat messages list, suitable for processor.apply_chat_template.
    """
    user_content = [
        {"type": "image", "image": im} for im in images
    ]
    if prompt:
        user_content.append({"type": "text", "text": prompt})

    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    logger.debug(f"Built Qwen messages:\n{json.dumps(messages, indent=2, default=str)}")
    return messages


def _format_seconds_label(seconds: float) -> str:
    value = max(float(seconds), 0.0)
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def build_qwen_timed_messages(
    system_prompt: Optional[str],
    images_with_ts: Sequence[Tuple[Image.Image, float]],
    prompt: Optional[str],
) -> List[dict]:
    """Create messages where each image is preceded by a timestamp text block."""
    user_content: List[Dict[str, Any]] = []
    for pil_img, ts in images_with_ts:
        ts_label = _format_seconds_label(ts)
        user_content.append({"type": "text", "text": f"<{ts_label} seconds>"})
        user_content.append({"type": "image", "image": pil_img})
    if prompt:
        user_content.append({"type": "text", "text": prompt})

    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    logger.debug(
        "Built Qwen timed messages:\n%s",
        json.dumps(messages, indent=2, default=str),
    )
    return messages


def build_qwen_conversation_messages(
    system_prompt: Optional[str],
    history: List[dict],
    max_turns: int = 30,
    max_media: int = 8,
    include_placeholder_for_empty: bool = True,
    include_media: bool = True,
) -> List[dict]:
    """Build a Qwen-style multi-turn messages list from internal chat history.

    history: list of entries with keys: role ('user'|'model'), type ('text'|'image'|'video'), content (str), meta (dict)
    We coalesce consecutive user or model messages into separate turns (do not merge).
    Images/videos appear only in the *user* messages they were originally introduced.
    For now, we re-emit media for each user turn to keep context (Qwen expects previous images repeated if needed).
    If media count exceeds max_media, older media are dropped (keep most recent).
    When include_media is False, media attachments are omitted but textual context is kept with
    placeholders to avoid empty turns.
    """
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Collect last max_turns entries (not pairs) but we will transform each entry into a message.
    recent = history[-max_turns:]
    media_seen = 0
    for entry in recent:
        role = entry.get("role")
        mtype = entry.get("type")
        content = entry.get("content") or ""
        meta = entry.get("meta") or {}
        if role == "user":
            user_content: List[dict] = []
            had_media = mtype in {"image", "video"}
            # Attach image if available and allowed
            if include_media and mtype == "image" and media_seen < max_media:
                pil_img = meta.get("pil_image")
                if pil_img is not None:
                    user_content.append({"type": "image", "image": pil_img})
                    media_seen += 1
            # Attach video if available and allowed
            if include_media and mtype == "video" and media_seen < max_media:
                frame_images = meta.get("frame_images") or meta.get("frames")
                frame_ts = meta.get("frame_timestamps") or meta.get("timestamps")
                if frame_images:
                    limit = max_media - media_seen
                    indices = list(range(len(frame_images)))[:limit]
                    for idx in indices:
                        pil_img = frame_images[idx]
                        if pil_img is None:
                            continue
                        ts_val = None
                        if frame_ts and idx < len(frame_ts):
                            try:
                                ts_val = float(frame_ts[idx])
                            except (TypeError, ValueError):
                                ts_val = None
                        if ts_val is not None:
                            label = _format_seconds_label(ts_val)
                            user_content.append({"type": "text", "text": f"<{label} seconds>"})
                        user_content.append({"type": "image", "image": pil_img})
                        media_seen += 1
                        if media_seen >= max_media:
                            break
                else:
                    vpath = meta.get("video_path")
                    if vpath:
                        fps_val = float(meta.get("fps", 5.0) or 5.0)
                        user_content.append({
                            "type": "video",
                            "video": vpath,
                            "video_fps": fps_val,
                            "fps": fps_val,
                        })
                        media_seen += 1
            # Text part
            text_added = False
            if content:
                user_content.append({"type": "text", "text": content})
                text_added = True
            elif include_placeholder_for_empty and not user_content:
                user_content.append({"type": "text", "text": "(no text)"})
            elif not include_media and had_media and not text_added:
                user_content.append({"type": "text", "text": "[media omitted]"})
            messages.append({"role": "user", "content": user_content})
        elif role in ("model", "assistant"):
            # Assistant textual reply
            if not content and include_placeholder_for_empty:
                content = "(empty)"
            messages.append({"role": "assistant", "content": content})
        # Ignore other roles silently (system handled separately)
    logger.debug("Built conversation messages (%d turns)" % len(messages))
    return messages


def gather_pil_images(messages: List[dict]) -> List[Image.Image]:
    """Extract all PIL images from user message content blocks for fallback encoding path."""
    imgs: List[Image.Image] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    im = item.get("image")
                    if isinstance(im, Image.Image):
                        imgs.append(im)
    return imgs


def prepare_qwen_inputs(processor: Any, messages: List[dict], pil_images: List[Image.Image]) -> Tuple[Dict[str, Any], str]:
    """Prepare model inputs for Qwen-style VL models.

    Tries the richer qwen_vl_utils.process_vision_info flow first (preferred,
    handles videos & complex layouts) and falls back to a simpler processor call.

    Parameters
    ----------
    processor : Any
        HuggingFace processor instance with apply_chat_template & __call__.
    messages : List[dict]
        Chat messages built by build_qwen_messages (or equivalent structure).
    pil_images : List[Image.Image]
        Images used ONLY for fallback path when qwen_vl_utils isn't available.

    Returns
    -------
    (inputs, text)
        inputs : dict of tensors / lists suitable for model.generate
        text   : rendered chat template text (for debugging / logging)
    """
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        return inputs, text
    except Exception:
        # Fallback: simpler path (works for images-only)
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if pil_images:
            processor_kwargs["images"] = pil_images
        inputs = processor(**processor_kwargs)
        return inputs, text


def move_inputs_to_device(inputs: Dict[str, Any], device: str):
    """In-place move of tensor values to target device.

    Returns the same dict for chaining.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch should be present in runtime
        return inputs
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def infer_video_qwen_style(
    system_prompt: Optional[str],
    processor: Any,
    model: Any,
    device: str,
    video_path: str,
    requested_fps: float,
    prompt: Optional[str],
    max_new_tokens: int = 800,
    image_patch_size: Optional[int] = None,
) -> str:
    """Unified single-pass video inference for Qwen2.5-VL style models.

    Parameters
    ----------
    system_prompt : Optional[str]
        System instruction to include in the conversation.
    processor, model : Any
        Loaded HF processor and model (already on correct device).
    device : str
        Target torch device string.
    video_path : str
        Input video file path.
    requested_fps : float
        Target output sampling fps (capped by source fps).
    prompt : Optional[str]
        Optional user text instruction.
    max_new_tokens : int
        Generation length.

    """
    import cv2  # type: ignore
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(max(0.1, requested_fps))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / in_fps) if in_fps > 0 else 0.0
    fps_effective = float(max(0.1, min(requested_fps if requested_fps > 0 else in_fps, in_fps)))
    frames_out = int(round(duration * fps_effective)) if duration > 0 else max(
        1, int(frame_count * (fps_effective / max(in_fps, 1e-6)))
    )
    cap.release()
    logger.info(f"Video info - input FPS: {in_fps}, requested FPS: {requested_fps}, effective FPS: {fps_effective}, duration: {duration:.2f}s, frames out: {frames_out}, width: {width}, height: {height}")


    if frames_out <= 0:
        frames_out = max(1, int(duration * fps_effective))
    total_pixels = int(max(28 * 28, round(fps_effective * 28 * 28)))
    logger.info(f"Total pixels from requested FPS ({fps_effective}): {total_pixels}")

    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "qwen_vl_utils is required for video inference with Qwen-style models."
        ) from e

    user_content = [
        {
            "type": "video",
            "video": video_path,
            # "video_fps": fps_effective,
            # "fps": fps_effective,
            "total_pixels": total_pixels,
            "max_frames": frames_out,
        }
    ]
    if prompt:
        user_content.append({"type": "text", "text": prompt})
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    pvi_kwargs: Dict[str, Any] = {
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }
    if image_patch_size is not None:
        pvi_kwargs["image_patch_size"] = image_patch_size
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], **pvi_kwargs
        )
    except TypeError:
        # Older qwen_vl_utils versions may not accept newer kwargs; retry without them.
        logger.warning("process_vision_info failed with TypeError; retrying without image_patch_size.")
        if "image_patch_size" in pvi_kwargs:
            pvi_kwargs.pop("image_patch_size", None)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages], **pvi_kwargs
            )
        else:
            raise
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    logger.info(f"video keyword arguments: {video_kwargs}")
    logger.info(f"Video metadata: {video_metadatas}")
    video_kwargs = video_kwargs or {}
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

