from __future__ import annotations

import os
import time
import logging
import json
from typing import List, Dict
from pathlib import Path

# Set config directory for standalone app before importing runtime
os.environ["VLM_CONFIG_DIR"] = str(Path(__file__).resolve().parent / "config")

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from vlm_detections.core.adapter_base import Detection, EntityProperty, EntityRelation
from vlm_detections.core.visualize import draw_detections
from prompt_manager import PromptManager
from state_persistence import load_state as load_session_state, save_state as save_session_state, merge as merge_session_state, SESSION_FILE as DEFAULT_SESSION_FILE
from vlm_detections.core.runtime import (
    DetectorRuntime,
    MODEL_VARIANTS,
    MODEL_REGISTRY,
    default_variant_for,
    ensure_valid_variant,
    parse_prompts,
    is_zero_shot_detector,
    is_prompt_based_vlm,
)

# Configure logging early
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)




class AppState(DetectorRuntime):
    def __init__(self):
        super().__init__()
        self.threshold: float = 0.25
        self.prompts: List[str] = []
        self.last_infer_ts: float = 0.0
        self.cached_frame_rgb = None
        self.cached_status = "Status: idle"
        self.run_enabled: bool = False
        self.last_input_frame_rgb: np.ndarray | None = None
        # Florence-2 task selection (OD, REF, CAPTION, OCR)
        self.florence_task: str = "OD"
        # Chat related state
        # Each message: {"role": "user"|"model", "type": "text"|"image"|"video", "content": str, "meta": {...}}
        self.chat_history: List[Dict[str, object]] = []
        # Models that support multi-turn chat (others reject text chat)
        self.chat_capable_models = {"OpenAI Vision (API)", "Qwen2.5-VL", "Qwen3-VL", "Cosmos-Reason1", "InternVL3.5", "TRex"}

    def is_chat_capable(self, model_name: str | None = None) -> bool:
        mn = model_name or self.current_model_name
        return bool(mn in self.chat_capable_models)

    def clear_chat(self):
        self.chat_history.clear()

    def append_chat(self, role: str, msg_type: str, content: str, **meta):
        message = {"role": role, "type": msg_type, "content": content, "meta": dict(meta)}
        self.chat_history.append(message)
        return message

    def export_chat_for_ui(self):
        # Convert to list of (user, assistant) pairs for gr.Chatbot
        pairs = []
        pending_user = None
        for m in self.chat_history:
            if m["role"] == "user":
                if pending_user is not None:
                    # previous user without model reply
                    pairs.append([pending_user, None])
                pending_user = self._render_message(m)
            else:  # model
                model_text = self._render_message(m)
                if pending_user is None:
                    pairs.append([None, model_text])
                else:
                    pairs.append([pending_user, model_text])
                    pending_user = None
        if pending_user is not None:
            pairs.append([pending_user, None])
        return pairs

    def _render_message(self, m: Dict[str, object]) -> str:
        t = m.get("type")
        content = str(m.get("content", ""))
        meta = m.get("meta") or {}
        if t == "text":
            return content
        if t == "image":
            return f"[Image]\n{content}" if content else "[Image]"
        if t == "video":
            frame_count = meta.get("frame_count", meta.get("frames")) if isinstance(meta, dict) else None
            try:
                fc = int(frame_count)
            except (TypeError, ValueError):
                fc = None
            if fc and fc > 0:
                label = f"Image[{fc}]"
                return f"{label}\n{content}" if content else label
            return f"[Video]\n{content}" if content else "[Video]"
        return content

    def build_plaintext_history(self, system_prompt: str | None) -> str:
        lines: List[str] = []
        if system_prompt:
            lines.append(f"System: {system_prompt}")
        for entry in self.chat_history:
            role = entry.get("role", "user")
            prefix = "User" if role == "user" else "Assistant"
            msg_type = entry.get("type")
            content = str(entry.get("content", "")).strip()
            meta = entry.get("meta") or {}
            if msg_type == "image":
                desc = content or "[Image]"
            elif msg_type == "video":
                frame_count = meta.get("frame_count") if isinstance(meta, dict) else None
                sampled = meta.get("frame_images") if isinstance(meta, dict) else None
                details = []
                if frame_count:
                    try:
                        details.append(f"frames={int(frame_count)}")
                    except (TypeError, ValueError):
                        pass
                if sampled:
                    details.append(f"samples={len(sampled)}")
                suffix = f" ({', '.join(details)})" if details else ""
                desc = (content or "[Video]") + suffix
            else:
                desc = content or "(empty)"
            lines.append(f"{prefix}: {desc}")
        return "\n".join(lines)

STATE = AppState()
_SESSION = load_session_state(DEFAULT_SESSION_FILE)
PromptManager.load_all()

# --- Session management helpers ---
def reload_session(path: str | None = None) -> dict:
    """Reload session JSON into module-level _SESSION.

    Parameters
    ----------
    path : str | None
        If provided, overrides current DEFAULT_SESSION_FILE for this reload only.
    Returns
    -------
    dict : The newly loaded session dictionary (may be empty on failure).
    """
    global _SESSION, DEFAULT_SESSION_FILE
    if path:
        try:
            # Update environment + DEFAULT_SESSION_FILE so subsequent loads are consistent
            os.environ["VLM_SESSION_FILE"] = path
            DEFAULT_SESSION_FILE = path  # type: ignore
        except Exception:
            pass
    new_state = load_session_state(path or DEFAULT_SESSION_FILE)
    if isinstance(new_state, dict):
        _SESSION = new_state
    else:
        _SESSION = {}
    return _SESSION

def build_ui_from_state(state: dict):
    """Temporarily substitute _SESSION with provided state dict then build UI."""
    global _SESSION
    prev = _SESSION
    try:
        if isinstance(state, dict):
            _SESSION = state
        return build_ui()
    finally:
        _SESSION = prev


def process_image(frame: np.ndarray, classes_text: str, prompt_text: str, model_name: str, model_variant: str, threshold: float, device_choice: str,
                  sys_choice: str | None, user_choice: str | None, override_user: bool,
                  resize_scale: int = 1):
    """Run a single-shot detection on the provided RGB frame and return visualization, status and raw detections."""
    if frame is None:
        return None, [], "No image provided"

    # Store last raw input for saving
    STATE.last_input_frame_rgb = frame.copy()

    # Ensure model is loaded
    STATE.ensure_model(model_name, model_variant, device_choice)

    classes = parse_prompts(classes_text)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = frame_bgr.shape[:2]
    try:
        s = int(resize_scale) if resize_scale is not None else 1
    except Exception:
        s = 1
    if s > 1:
        tw = max(1, orig_w // s)
        th = max(1, orig_h // s)
        if frame_bgr.shape[1] != tw or frame_bgr.shape[0] != th:
            frame_bgr = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

    t0 = time.perf_counter()
    try:
        # Prepare prompt conditioning
        h, w = frame_bgr.shape[:2]
        # Record selections for manager
        PromptManager.select(model_name, sys_choice, user_choice, override_user, prompt_text or "")
        # Apply per-adapter conditioning
        final_prompt = PromptManager.render_user_prompt(model_name, classes, prompt_text or "", width=w, height=h, threshold=threshold)
        system_prompt = PromptManager.get_system_prompt(model_name, reasoning=getattr(STATE.detector, "reasoning", None))

        # Call appropriate interface based on adapter protocol
        # Prefer PromptBasedVLM if we have a prompt (handles Florence-2 dual protocol)
        if is_prompt_based_vlm(STATE.detector) and final_prompt:
            detections, properties, relations, raw_text = STATE.detector.infer_from_image(frame_bgr, final_prompt, system_prompt, threshold)
        elif is_zero_shot_detector(STATE.detector):
            detections = STATE.detector.infer(frame_bgr, classes, threshold)
            properties, relations, raw_text = [], [], ""
        else:
            raise ValueError(f"Unknown adapter protocol for {STATE.current_model_name}")
        infer_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        logger.exception("Error during inference")
        img_bgr = frame_bgr.copy()
        label = f"Error: {type(e).__name__}: {e}"
        cv2.rectangle(img_bgr, (10, 10), (10 + 600, 40), (0, 0, 255), -1)
        cv2.putText(img_bgr, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status = f"Model: {STATE.current_model_name} | Variant: {STATE.current_model_id} | Device: {getattr(STATE.detector, 'device', 'unknown')} | ERROR: {type(e).__name__}: {e}"
        vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return vis_rgb, "", "", "", status

    # Visualize
    vis_bgr = draw_detections(frame_bgr, detections)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    device_str = getattr(STATE.detector, "device", "unknown")
    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
    cur_h, cur_w = frame_bgr.shape[:2]
    if s > 1:
        res_info = f"Res: {orig_w}x{orig_h} -> {cur_w}x{cur_h} (scale x{s})"
    else:
        res_info = f"Res: {cur_w}x{cur_h}"
    extra = ""
    if STATE.current_model_name == "Florence-2":
        try:
            extra = f" | Task: {STATE.florence_task}"
        except Exception:
            extra = ""
    status = (
        f"Model: {STATE.current_model_name} | Variant: {STATE.current_model_id} | Device: {device_str}{extra} | Classes: {len(classes)} | "
        f"Detections: {len(detections)} | Properties: {len(properties)} | Relations: {len(relations)} | Thr: {threshold:.2f} | Inference: {infer_ms:.1f} ms (~{fps:.1f} FPS) | {res_info}"
    )

    # Format properties and relations for display
    properties_text = "\n".join([f"{p.entity}.{p.property_name} = {p.property_value} (score: {p.score:.2f})" for p in properties]) if properties else "(none)"
    relations_text = "\n".join([f"{r.subject} --{r.predicate}--> {r.object} (score: {r.score:.2f})" for r in relations]) if relations else "(none)"
    
    return vis_rgb, properties_text, relations_text, (raw_text or ""), status


def _resolve_video_path(video_value):
    # Gradio Video may pass a file path (str) or a dict for recordings
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for k in ("name", "path", "video"):
            if k in video_value and isinstance(video_value[k], str):
                return video_value[k]
    return None


def _needs_video_transcode(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext not in {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"}


def process_video_file(video_value, classes_text: str, prompt_text: str, model_name: str, model_variant: str, threshold: float, device_choice: str, req_fps: float,
                       sys_choice: str | None = None, user_choice: str | None = None, override_user: bool = False,
                       resize_scale: int = 1):
    # Resolve path
    path = _resolve_video_path(video_value)
    if not path or not os.path.exists(path):
        return None, "Invalid video input", 0

    # Ensure model ready
    STATE.ensure_model(model_name, model_variant, device_choice)
    classes = parse_prompts(classes_text)

    cap = cv2.VideoCapture(path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_in <= 0:
        fps_in = max(1.0, float(req_fps) if req_fps > 0 else 10.0)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
    width, height = src_width, src_height
    try:
        s = int(resize_scale) if resize_scale is not None else 1
    except Exception:
        s = 1
    if s > 1:
        width = max(1, src_width // s)
        height = max(1, src_height // s)

    # Output path
    os.makedirs("assets/outputs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("assets/outputs", f"{base}_{STATE.current_model_id.replace('/', '_')}_{ts}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    interval = 1.0 / max(req_fps, 0.1)
    next_process_time = 0.0
    last_annotated = None

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # Resize to target dimensions if needed
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

        t = frame_idx / fps_in
        process_now = t >= next_process_time or last_annotated is None
        if process_now:
            try:
                # Build final prompt per current selections
                hh, ww = frame_bgr.shape[:2]
                PromptManager.select(model_name, sys_choice, user_choice, override_user, prompt_text or "")
                final_prompt = PromptManager.render_user_prompt(model_name, classes, prompt_text or "", width=ww, height=hh, threshold=threshold)
                system_prompt = PromptManager.get_system_prompt(model_name, reasoning=getattr(STATE.detector, "reasoning", None))
                # Call appropriate interface based on adapter protocol
                # Prefer PromptBasedVLM if we have a prompt (handles Florence-2 dual protocol)
                if is_prompt_based_vlm(STATE.detector) and final_prompt:
                    dets, _props, _rels, _raw = STATE.detector.infer_from_image(frame_bgr, final_prompt, system_prompt, threshold)
                elif is_zero_shot_detector(STATE.detector):
                    dets = STATE.detector.infer(frame_bgr, classes, threshold)
                    _props, _rels, _raw = [], [], ""
                else:
                    raise ValueError(f"Unknown adapter protocol for {STATE.current_model_name}")
            except Exception as e:
                # On error, overlay and keep going
                img_bgr = frame_bgr.copy()
                label = f"Error: {type(e).__name__}: {e}"
                cv2.rectangle(img_bgr, (10, 10), (10 + 800, 40), (0, 0, 255), -1)
                cv2.putText(img_bgr, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                last_annotated = img_bgr
            else:
                last_annotated = draw_detections(frame_bgr, dets)
            # Advance next process time relative to current timestamp
            next_process_time = t + interval
        # Write annotated (or previous) frame
        writer.write(last_annotated if last_annotated is not None else frame_bgr)
        frame_idx += 1

    cap.release()
    writer.release()

    if s > 1:
        res_info = f"Res: {src_width}x{src_height} -> {width}x{height} (scale x{s})"
    else:
        res_info = f"Res: {width}x{height}"
    status = f"Video saved: {out_path} | {res_info}"
    return out_path, status, frame_idx


def build_ui():
    """Build the full Gradio UI with all previously available features.

    Features:
      - Model + variant selection, device choice
      - Generation parameter dynamic controls (per-adapter spec)
      - Florence-2 task selection
      - Prompt preset management (system/user load, edit, save, refresh, override)
      - Cosmos-Reason1 reasoning toggle & warnings display
      - Image processing (annotated)
      - Video processing (annotated)
    - Batch & direct raw video analysis
    - Session state persistence (model, variant, threshold, prompts, etc.)
    - Frame & video saving utilities
    """
    logger.debug("Entering build_ui() full-featured")
    session_model = _SESSION.get("model") if isinstance(_SESSION, dict) else None
    default_model = session_model if session_model in MODEL_VARIANTS else list(MODEL_VARIANTS.keys())[0]
    session_variant = _SESSION.get("model_variant") if isinstance(_SESSION, dict) else None
    default_variant = ensure_valid_variant(default_model, session_variant or default_variant_for(default_model))
    classes_default = _SESSION.get("classes_text", "person, laptop, cup")
    prompt_default = _SESSION.get("prompt_text", "")
    thr_default = float(_SESSION.get("threshold", 0.25))
    resize_default = int(_SESSION.get("resize_scale", 1))
    try:
        video_fps_default = int(_SESSION.get("video_processing_fps", 5))
    except Exception:
        video_fps_default = 5
    if video_fps_default < 1:
        video_fps_default = 1
    elif video_fps_default > 60:
        video_fps_default = 60
    override_default = bool(_SESSION.get("override_user_prompt", False))

    with gr.Blocks() as demo:
        gr.Markdown("# VLM Object Detection & Prompt Playground")
        with gr.Row():
            model = gr.Dropdown(choices=list(MODEL_VARIANTS.keys()), value=default_model, label="Model")
            # Allow custom value so that during model switches the previous variant value doesn't trigger a validation error
            model_variant = gr.Dropdown(choices=MODEL_VARIANTS[default_model], value=default_variant, label="Variant", allow_custom_value=True)
            model_variant_state = gr.Textbox(value=default_variant, visible=False)
            device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value=_SESSION.get("device", "auto"), label="Device")

        # Generation parameters (initially hidden until spec indicates availability)
        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row():
                gp_max_new_tokens = gr.Slider(16, 8192, value=256, step=16, label="max_new_tokens", visible=False)
                gp_temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.01, label="temperature", visible=False)
                gp_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="top_p", visible=False)
                gp_repetition_penalty = gr.Slider(0.5, 3.0, value=1.0, step=0.01, label="repetition_penalty", visible=False)
            with gr.Row():
                gp_num_beams = gr.Slider(1, 16, value=1, step=1, label="num_beams", visible=False)
                gp_num_return_sequences = gr.Slider(1, 8, value=1, step=1, label="num_return_sequences", visible=False)
                gp_do_sample = gr.Checkbox(value=True, label="do_sample", visible=False)
                gp_top_k = gr.Slider(0, 2048, value=50, step=1, label="top_k", visible=False)
            apply_gen_btn = gr.Button("Apply Generation Parameters")
            gen_params_status = gr.Markdown("No tunable parameters.")

        florence_task_dd = gr.Dropdown(
            [
                "OD","REF","CAPTION","DETAILED_CAPTION","MORE_DETAILED_CAPTION","DENSE_REGION_CAPTION","REGION_PROPOSAL","REFERRING_EXPRESSION_SEGMENTATION","REGION_TO_SEGMENTATION","OCR","OCR_WITH_REGION","CAPTION_TO_PHRASE_GROUNDING"
            ],
            value=STATE.florence_task,
            label="Florence-2 task",
            visible=(default_model == "Florence-2"),
        )

        # Prompt selection + management
        with gr.Row():
            sys_prompt_dd = gr.Dropdown(choices=[], value=None, label="System prompt preset", allow_custom_value=True)
            user_prompt_dd = gr.Dropdown(choices=[], value=None, label="User prompt preset", allow_custom_value=True)
        cosmos_warnings_box = gr.Textbox(label="Cosmos Prompt Warnings", value="", lines=6, interactive=False, visible=False)
        refresh_prompts_btn = gr.Button("Refresh Prompts / Reload YAMLs")

        with gr.Accordion("Preset Prompts Show & Edit", open=False):
            with gr.Row():
                with gr.Column():
                    system_prompt_text = gr.Textbox(label="System prompt text (editable)", lines=6, value="")
                    reasoning_cb = gr.Checkbox(value=True, label="Cosmos: include reasoning addon (<think>)", visible=False)
                    new_sys_name = gr.Textbox(label="New system preset name", lines=1, value="")
                    with gr.Row():
                        update_sys_btn = gr.Button("Update System Preset")
                        save_new_sys_btn = gr.Button("Save as New System Preset")
                with gr.Column():
                    user_prompt_text = gr.Textbox(label="User prompt text (editable)", lines=6, value="")
                    new_user_name = gr.Textbox(label="New user preset name", lines=1, value="")
                    with gr.Row():
                        update_user_btn = gr.Button("Update User Preset")
                        save_new_user_btn = gr.Button("Save as New User Preset")

        # Core detection controls
        classes_tb = gr.Textbox(value=classes_default, lines=3, label="Classes (comma or newline separated)")
        prompt_tb = gr.Textbox(value=prompt_default, lines=2, label="Prompt (free-form instruction for API models)")
        override_cb = gr.Checkbox(value=override_default, label="Override user prompt with textbox content")
        with gr.Row():
            thr = gr.Slider(minimum=0.05, maximum=0.9, step=0.01, value=thr_default, label="Confidence threshold")
            resize_scale = gr.Number(minimum=1, maximum=10, step=1, value=resize_default, label="Resize scale (1 = no resize, 10 = 10x smaller)")

        # Image processing
        with gr.Accordion("Image Processing", open=False):
            with gr.Row():
                with gr.Column():
                    drop = gr.Image(sources=["upload", "webcam"], label="Image (upload/webcam)", interactive=True, height=720, width=1280, webcam_options=gr.WebcamOptions(mirror=False))
                with gr.Column():
                    out = gr.Image(label="Detections", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            properties_display = gr.Textbox(label="Entity Properties", lines=5, interactive=False)
                        with gr.Column():
                            relations_display = gr.Textbox(label="Entity Relations", lines=5, interactive=False)
                    btn_process_image = gr.Button("Process Image", variant="primary")
                    btn_save = gr.Button("Save Frame")
                    save_dir = gr.Textbox(value="assets/captures", label="Save directory")

        # Video processing
        with gr.Accordion("Video Processing", open=False):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(sources=["upload", "webcam"], label="Video (upload/webcam)", interactive=True)
                    vid_fps = gr.Slider(minimum=1, maximum=60, step=1, value=video_fps_default, label="Video processing FPS")
                    video_save_dir = gr.Textbox(value="assets/captures", label="Video save directory")
                    btn_save_video = gr.Button("Save Video")
                with gr.Column():
                    vid_out = gr.Video(label="Output Video")
                    btn_process_video = gr.Button("Process Video (annotate frames)", variant="primary")
                    btn_video_batch_infer = gr.Button("Analyze Video as Batch (raw only)", variant="primary")
                    btn_video_direct_infer = gr.Button("Analyze Video Direct (raw only)", variant="primary")

        status = gr.Markdown("Status: idle")
        chatbot = None
        chat_send_btn = None
        chat_input_tb = None
        raw_out = gr.Textbox(label="Raw Output (legacy)", lines=4, visible=False)
        with gr.Accordion("Chat", open=True):
            chatbot = gr.Chatbot(label="Chat History", height=300)
            with gr.Row():
                chat_input_tb = gr.Textbox(label="New message", lines=3, placeholder="Type a follow-up message...")
            with gr.Row():
                chat_send_btn = gr.Button("Send Message", variant="primary")

        # ---------------- Variant & Generation Params ----------------
        # ---------------- Chat Helper Functions ----------------
        def _chat_send(message_text: str, model_name: str, model_variant_value: str, classes_text: str, threshold: float, device_choice: str, sys_key: str | None, user_key: str | None, override_user: bool):
            msg = (message_text or "").strip()
            if not msg:
                return gr.update(), gr.update(), "Empty message ignored."
            if not STATE.is_chat_capable(model_name):
                note = f"Status: {model_name} is not chat-capable." if model_name else "Status: no chat-capable model selected."
                return gr.update(value=STATE.export_chat_for_ui()), gr.update(), note
            safe_variant = ensure_valid_variant(model_name, model_variant_value)
            if STATE.model_gen_params.get(model_name):
                STATE.apply_generation_params(STATE.model_gen_params.get(model_name, {}))
            STATE.append_chat("user", "text", msg)
            # Structured multi-turn messages
            STATE.ensure_model(model_name, safe_variant, device_choice)
            reply = "(no reply)"
            try:
                infer_fn = getattr(STATE.detector, 'chat_infer', None)
                system_prompt_full = PromptManager.get_system_prompt(model_name, reasoning=getattr(STATE.detector, "reasoning", None))

                def _build_chat_payload(include_media: bool):
                    formatter = getattr(STATE.detector, "build_chat_messages", None)
                    if callable(formatter):
                        try:
                            return formatter(STATE.chat_history, system_prompt_full, include_media=include_media)
                        except TypeError:
                            return formatter(STATE.chat_history, system_prompt_full)
                    return STATE.build_plaintext_history(system_prompt_full)

                if callable(infer_fn):
                    payload = _build_chat_payload(include_media=True)
                    try:
                        reply = infer_fn(payload, threshold=threshold)
                    except IndexError:
                        logger.warning("Chat inference failed with media attachments; retrying without media.", exc_info=True)
                        payload = _build_chat_payload(include_media=False)
                        reply = infer_fn(payload, threshold=threshold)
                else:
                    reply = f"(adapter lacks chat_infer)"
            except Exception as e:
                reply = f"Error: {type(e).__name__}: {e}"
            STATE.append_chat("model", "text", reply)
            return gr.update(value=STATE.export_chat_for_ui()), gr.update(value=""), "Message sent."

        if chat_send_btn and chatbot and chat_input_tb:
            chat_send_btn.click(
                fn=_chat_send,
                inputs=[chat_input_tb, model, model_variant_state, classes_tb, thr, device, sys_prompt_dd, user_prompt_dd, override_cb],
                outputs=[chatbot, chat_input_tb, status]
            )

        def update_variants(model_name: str):
            STATE.run_enabled = False
            variants = MODEL_VARIANTS.get(model_name)
            lst = list(variants) if variants else ["default"]
            # Attempt to restore last used variant for this model from session history (single mapping)
            sess = load_session_state()
            preferred = None
            if isinstance(sess, dict):
                hist_single = sess.get("model_variant_history", {})
                if isinstance(hist_single, dict):
                    cand = hist_single.get(model_name)
                    if cand in lst:
                        preferred = cand
            if preferred is None:
                preferred = lst[0]
            return (gr.update(choices=lst, value=preferred), preferred, "Model changed. Ready to process image.")

        model.change(fn=update_variants, inputs=model, outputs=[model_variant, model_variant_state, status])

        def sync_variant_state(variant_value: str):
            """Synchronize the hidden variant state when user changes the visible variant dropdown."""
            return variant_value

        model_variant.change(fn=sync_variant_state, inputs=[model_variant], outputs=[model_variant_state])

        if chatbot is not None:
            def _enforce_chat_capability(model_name: str, current_status: str):
                capable = STATE.is_chat_capable(model_name)
                if not capable:
                    STATE.clear_chat()
                    return gr.update(value=STATE.export_chat_for_ui()), f"{current_status}\nModel not chat-capable; chat cleared."
                return gr.update(value=STATE.export_chat_for_ui()), current_status
            model.change(fn=_enforce_chat_capability, inputs=[model, status], outputs=[chatbot, status])

        def _adapter_spec(model_name: str):
            factory = MODEL_REGISTRY.get(model_name)
            if not factory:
                return {}
            try:
                inst = factory(default_variant_for(model_name))
                fn = getattr(inst, "generation_config_spec", None)
                return fn() if callable(fn) else {}
            except Exception:
                return {}

        def refresh_generation_controls(model_name: str):
            spec = _adapter_spec(model_name)
            active = set(spec.keys())
            def vis(name): return name in active
            current_params = _SESSION.get("generation_params", {}).get(model_name, {}) if isinstance(_SESSION, dict) else {}
            sampling_enabled = bool(current_params.get("do_sample", False)) if "do_sample" in active else True
            disabled_sampling_deps = (not sampling_enabled)
            return (
                gr.update(visible=vis("max_new_tokens")),
                gr.update(visible=vis("temperature"), interactive=(sampling_enabled and vis("temperature"))),
                gr.update(visible=vis("top_p"), interactive=(sampling_enabled and vis("top_p"))),
                gr.update(visible=vis("repetition_penalty")),
                gr.update(visible=vis("num_beams")),
                gr.update(visible=vis("num_return_sequences")),
                gr.update(visible=vis("do_sample")),
                gr.update(visible=vis("top_k"), interactive=(sampling_enabled and vis("top_k"))),
                ("No tunable parameters." if not active else "Active: " + ", ".join(sorted(active)) + (" (sampling disabled)" if disabled_sampling_deps and sampling_enabled is False else ""))
            )

        model.change(
            fn=refresh_generation_controls,
            inputs=[model],
            outputs=[gp_max_new_tokens, gp_temperature, gp_top_p, gp_repetition_penalty, gp_num_beams, gp_num_return_sequences, gp_do_sample, gp_top_k, gen_params_status],
        )

        def apply_generation_params(model_name: str, max_new_tokens, temperature, top_p, repetition_penalty, num_beams, num_return_sequences, do_sample, top_k):
            spec = _adapter_spec(model_name)
            collected = {}
            def clamp(val, mn, mx):
                try:
                    return mn if val < mn else mx if val > mx else val
                except Exception:
                    return val
            if "max_new_tokens" in spec:
                meta = spec["max_new_tokens"]; collected["max_new_tokens"] = int(clamp(max_new_tokens, meta.get("min",16), meta.get("max",8192)))
            if "temperature" in spec:
                meta = spec["temperature"]; collected["temperature"] = float(clamp(temperature, meta.get("min",0.0), meta.get("max",2.0)))
            if "top_p" in spec:
                meta = spec["top_p"]; collected["top_p"] = float(clamp(top_p, meta.get("min",0.0), meta.get("max",1.0)))
            if "repetition_penalty" in spec:
                meta = spec["repetition_penalty"]; collected["repetition_penalty"] = float(clamp(repetition_penalty, meta.get("min",0.5), meta.get("max",3.0)))
            if "num_beams" in spec:
                meta = spec["num_beams"]; collected["num_beams"] = int(clamp(num_beams, meta.get("min",1), meta.get("max",16)))
            if "num_return_sequences" in spec:
                meta = spec["num_return_sequences"]; collected["num_return_sequences"] = int(clamp(num_return_sequences, meta.get("min",1), meta.get("max",8)))
            if "do_sample" in spec:
                collected["do_sample"] = bool(do_sample)
            if "top_k" in spec:
                meta = spec["top_k"]; collected["top_k"] = int(clamp(top_k, meta.get("min",0), meta.get("max",2048)))
            STATE.model_gen_params[model_name] = collected
            if STATE.current_model_name == model_name and STATE.detector is not None:
                STATE.apply_generation_params(collected)
            sess = load_session_state()
            gp = sess.get("generation_params", {})
            gp[model_name] = collected
            sess["generation_params"] = gp
            save_session_state(sess)
            return "Applied: " + ", ".join(f"{k}={v}" for k,v in collected.items())

        apply_gen_btn.click(
            fn=apply_generation_params,
            inputs=[model, gp_max_new_tokens, gp_temperature, gp_top_p, gp_repetition_penalty, gp_num_beams, gp_num_return_sequences, gp_do_sample, gp_top_k],
            outputs=[gen_params_status],
        )

        def on_do_sample_change(model_name: str, do_sample_val: bool, current_status: str):
            spec = _adapter_spec(model_name)
            if "do_sample" not in spec:
                return (gr.update(), gr.update(), gr.update(), current_status)
            def slider_update(name):
                if name not in spec: return gr.update()
                return gr.update(interactive=do_sample_val)
            status_suffix = " (sampling enabled)" if do_sample_val else " (sampling disabled)"
            return (slider_update("temperature"), slider_update("top_p"), slider_update("top_k"), (current_status.split(" (sampling")[0] + status_suffix) if current_status else ("Status" + status_suffix))

        gp_do_sample.change(
            fn=on_do_sample_change,
            inputs=[model, gp_do_sample, gen_params_status],
            outputs=[gp_temperature, gp_top_p, gp_top_k, gen_params_status]
        )

        # Florence visibility
        def toggle_florence_task(model_name: str):
            show = model_name == "Florence-2"
            return gr.update(visible=show, value=STATE.florence_task if show else "OD")
        model.change(fn=toggle_florence_task, inputs=[model], outputs=[florence_task_dd])

        def on_task_change(model_name: str, task: str, current_status: str):
            if model_name != "Florence-2":
                return current_status
            task = (task or "OD").upper()
            valid = {"OD","REF","CAPTION","DETAILED_CAPTION","MORE_DETAILED_CAPTION","DENSE_REGION_CAPTION","REGION_PROPOSAL","REFERRING_EXPRESSION_SEGMENTATION","REGION_TO_SEGMENTATION","OCR","OCR_WITH_REGION","CAPTION_TO_PHRASE_GROUNDING"}
            if task not in valid: task = "OD"
            STATE.florence_task = task
            if STATE.detector is not None and hasattr(STATE.detector, "set_task"):
                try: STATE.detector.set_task(task)
                except Exception: pass
            return f"Florence-2 task set to {task}."
        florence_task_dd.change(on_task_change, inputs=[model, florence_task_dd, status], outputs=[status])

        # Image / video core processing
        def on_drop_changed(img):
            STATE.last_input_frame_rgb = img.copy() if img is not None else None
            return "Image loaded. Click 'Process Image' to run detections." if img is not None else "No image loaded."
        drop.change(fn=on_drop_changed, inputs=[drop], outputs=[status])

        def save_current_frame(dirpath: str):
            frame = STATE.last_input_frame_rgb
            if frame is None: return gr.update(), "No frame to save"
            os.makedirs(dirpath, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(dirpath, f"frame_{ts}.png")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, bgr)
            return gr.update(), f"Saved frame to {out_path}"
        btn_save.click(save_current_frame, inputs=save_dir, outputs=[out, status])

        def on_process_image(image_value, classes_text, prompt_text, model_name, model_variant_value, threshold, device_choice, sys_key, user_key, override_user, resize_scale_val):
            if image_value is None:
                return None, "", "", gr.update(), "Please provide an image first.", gr.update()
            safe_variant = ensure_valid_variant(model_name, model_variant_value)
            if STATE.model_gen_params.get(model_name):
                STATE.apply_generation_params(STATE.model_gen_params.get(model_name, {}))
            STATE.clear_chat()
            desc = prompt_text.strip() if prompt_text else "Image inference"
            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(image_value)
            except Exception:
                pil_img = None
            STATE.append_chat("user", "image", desc, classes=classes_text, pil_image=pil_img)
            vis, props_text, rels_text, raw, stat = process_image(image_value, classes_text, prompt_text, model_name, safe_variant, threshold, device_choice, sys_key, user_key, override_user, resize_scale_val)
            STATE.append_chat("model", "text", raw if isinstance(raw, str) else str(raw))
            return vis, props_text, rels_text, gr.update(value=STATE.export_chat_for_ui()), stat, gr.update(value="")

        btn_process_image.click(
            fn=on_process_image,
            inputs=[drop, classes_tb, prompt_tb, model, model_variant_state, thr, device, sys_prompt_dd, user_prompt_dd, override_cb, resize_scale],
            outputs=[out, properties_display, relations_display, chatbot, status, chat_input_tb],
        )

        def on_process_video(video_value, classes_text, prompt_text, model_name, model_variant_value, threshold, device_choice, req_fps_value, sys_key, user_key, override_user, resize_scale_val):
            user_video_msg = None
            try:
                safe_variant = ensure_valid_variant(model_name, model_variant_value)
                if STATE.model_gen_params.get(model_name):
                    STATE.apply_generation_params(STATE.model_gen_params.get(model_name, {}))
                STATE.clear_chat()
                desc = (prompt_text.strip() if prompt_text else "Video inference")
                vid_path = _resolve_video_path(video_value)
                user_video_msg = STATE.append_chat("user", "video", desc, classes=classes_text, video_path=vid_path, fps=req_fps_value, frame_count=0)
                video_path, stat, frame_count = process_video_file(video_value, classes_text, prompt_text, model_name, safe_variant, threshold, device_choice, req_fps_value, sys_key, user_key, override_user, resize_scale_val)
                if user_video_msg and isinstance(user_video_msg.get("meta"), dict):
                    user_video_msg["meta"]["frame_count"] = frame_count
                STATE.append_chat("model", "text", stat)
                return video_path, gr.update(value=STATE.export_chat_for_ui()), stat
            except Exception as e:
                logger.exception("Process Video failed")
                if user_video_msg and isinstance(user_video_msg.get("meta"), dict):
                    user_video_msg["meta"].setdefault("frame_count", 0)
                STATE.append_chat("model", "text", f"Error: {type(e).__name__}: {e}")
                return None, gr.update(value=STATE.export_chat_for_ui()), f"Process Video failed: {type(e).__name__}: {e}"

        btn_process_video.click(
            fn=on_process_video,
            inputs=[vid_in, classes_tb, prompt_tb, model, model_variant_state, thr, device, vid_fps, sys_prompt_dd, user_prompt_dd, override_cb, resize_scale],
            outputs=[vid_out, chatbot, status],
        )

        def on_video_batch_raw(video_value, classes_text, prompt_text, model_name, model_variant_value, threshold, device_choice, req_fps_value, sys_key, user_key, override_user, resize_scale_val):
            safe_variant = ensure_valid_variant(model_name, model_variant_value)
            STATE.ensure_model(model_name, safe_variant, device_choice)
            if STATE.model_gen_params.get(model_name):
                STATE.apply_generation_params(STATE.model_gen_params.get(model_name, {}))
            STATE.clear_chat()
            path = _resolve_video_path(video_value)
            if not path or not os.path.exists(path):
                STATE.append_chat("model", "text", "Invalid video input")
                return gr.update(value=STATE.export_chat_for_ui()), "Invalid video input"
            cap = cv2.VideoCapture(path)
            in_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if in_fps <= 0: in_fps = req_fps_value if req_fps_value > 0 else 5.0
            step = max(1, int(round(in_fps / max(req_fps_value, 0.1))))
            frames_rgb = []
            timestamps = []
            idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_rgb.append(frame_rgb)
                    timestamps.append(idx / in_fps if in_fps > 0 else 0.0)
                idx += 1
            cap.release()
            if timestamps:
                base_ts = timestamps[0]
                timestamps = [max(0.0, ts - base_ts) for ts in timestamps]
            frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
            frame_count = len(frames_bgr)
            try: s = int(resize_scale_val) if resize_scale_val is not None else 1
            except Exception: s = 1
            if s > 1 and frames_bgr:
                h0, w0 = frames_bgr[0].shape[:2]
                tw = max(1, w0 // s); th = max(1, h0 // s)
                frames_bgr = [cv2.resize(im, (tw, th), interpolation=cv2.INTER_LINEAR) for im in frames_bgr]
            chat_frame_images: List[Image.Image] = []
            chat_frame_timestamps: List[float] = []
            if frames_bgr:
                sample_cap = min(8, len(frames_bgr))
                if sample_cap == len(frames_bgr):
                    indices = list(range(len(frames_bgr)))
                else:
                    stride = (len(frames_bgr) - 1) / max(sample_cap - 1, 1)
                    seen = set()
                    indices = []
                    for idx_raw in range(sample_cap):
                        approx_idx = int(round(idx_raw * stride))
                        approx_idx = max(0, min(approx_idx, len(frames_bgr) - 1))
                        if approx_idx in seen:
                            continue
                        seen.add(approx_idx)
                        indices.append(approx_idx)
                    indices.sort()
                for idx_sel in indices:
                    try:
                        rgb_frame = cv2.cvtColor(frames_bgr[idx_sel], cv2.COLOR_BGR2RGB)
                        chat_frame_images.append(Image.fromarray(rgb_frame))
                        if idx_sel < len(timestamps):
                            chat_frame_timestamps.append(float(timestamps[idx_sel]))
                        else:
                            chat_frame_timestamps.append(float(idx_sel))
                    except Exception:
                        logger.debug("Failed to capture chat frame for history", exc_info=True)
                        continue
            if len(timestamps) > len(frames_bgr):
                timestamps = timestamps[:len(frames_bgr)]
            elif len(timestamps) < len(frames_bgr):
                filler = timestamps[-1] if timestamps else 0.0
                timestamps.extend([filler] * (len(frames_bgr) - len(timestamps)))
            frames_with_ts = list(zip(frames_bgr, timestamps)) if frames_bgr else []
            classes = parse_prompts(classes_text)
            try:
                t0 = time.perf_counter()
                PromptManager.select(model_name, sys_key, user_key, override_user, prompt_text or "")
                hh, ww = frames_bgr[0].shape[:2] if frames_bgr else (0, 0)
                final_prompt = PromptManager.render_user_prompt(model_name, classes, prompt_text or "", width=ww, height=hh, threshold=float(threshold))
                system_prompt = PromptManager.get_system_prompt(model_name, reasoning=getattr(STATE.detector, "reasoning", None))
                raw_result = STATE.detector.infer_from_batch(frames_with_ts, classes, final_prompt or "", system_prompt, float(threshold))
                if isinstance(raw_result, tuple) and len(raw_result) == 2:
                    _, raw = raw_result
                else:
                    raw = raw_result  # type: ignore[assignment]
                dt = time.perf_counter() - t0
            except Exception as e:
                raw = f"Error: {type(e).__name__}: {e}"
                STATE.append_chat("model", "text", raw)
                return gr.update(value=STATE.export_chat_for_ui()), "Batch analysis failed."
            if s > 1 and frames_bgr:
                res_info = f"Res: {w0}x{h0} -> {tw}x{th} (scale x{s})"
            elif frames_bgr:
                res_info = f"Res: {frames_bgr[0].shape[1]}x{frames_bgr[0].shape[0]}"
            else:
                res_info = "Res: N/A"
            vid_path = _resolve_video_path(video_value)
            STATE.append_chat(
                "user",
                "video",
                (prompt_text.strip() if prompt_text else "Video batch analysis"),
                classes=classes_text,
                video_path=vid_path,
                fps=req_fps_value,
                frame_count=frame_count,
                frame_images=chat_frame_images,
                frame_timestamps=chat_frame_timestamps,
                sampled_frames=len(chat_frame_images),
            )
            STATE.append_chat("model", "text", raw if isinstance(raw, str) else str(raw))
            return gr.update(value=STATE.export_chat_for_ui()), f"Batch analysis complete in {dt:.2f}s. Frames: {len(frames_bgr)} | {res_info}"

        btn_video_batch_infer.click(
            fn=on_video_batch_raw,
            inputs=[vid_in, classes_tb, prompt_tb, model, model_variant_state, thr, device, vid_fps, sys_prompt_dd, user_prompt_dd, override_cb, resize_scale],
            outputs=[chatbot, status],
        )

        def on_video_direct_raw(video_value, classes_text, prompt_text, model_name, model_variant_value, threshold, device_choice, req_fps_value, sys_key, user_key, override_user, resize_scale_val):
            safe_variant = ensure_valid_variant(model_name, model_variant_value)
            STATE.ensure_model(model_name, safe_variant, device_choice)
            if STATE.model_gen_params.get(model_name):
                STATE.apply_generation_params(STATE.model_gen_params.get(model_name, {}))
            STATE.clear_chat()
            path = _resolve_video_path(video_value)
            if not path or not os.path.exists(path):
                STATE.append_chat("model", "text", "Invalid video input")
                return gr.update(value=STATE.export_chat_for_ui()), "Invalid video input"
            resized_path = path
            try:
                s = int(resize_scale_val) if resize_scale_val is not None else 1
            except Exception:
                s = 1
            src_w = src_h = None
            in_fps = None
            needs_transcode = _needs_video_transcode(path) or s > 1
            final_w = final_h = None
            try:
                cap_probe = cv2.VideoCapture(path)
                if cap_probe.isOpened():
                    in_fps = cap_probe.get(cv2.CAP_PROP_FPS) or None
                    src_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    src_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                else:
                    needs_transcode = True
                cap_probe.release()
            except Exception:
                needs_transcode = True
            if needs_transcode or src_w is None or src_h is None or src_w <= 0 or src_h <= 0:
                try:
                    os.makedirs("assets/outputs", exist_ok=True)
                    base = os.path.splitext(os.path.basename(path))[0]
                    suffix = f"x{s}" if s > 1 else "converted"
                    out_resized = os.path.join("assets/outputs", f"resized_{base}_{suffix}.mp4")
                    cap_r = cv2.VideoCapture(path)
                    if not cap_r.isOpened():
                        raise RuntimeError(f"Cannot open source video for transcode: {path}")
                    if in_fps is None or in_fps <= 0:
                        in_fps = cap_r.get(cv2.CAP_PROP_FPS) or float(req_fps_value) or 5.0
                    if src_w is None or src_w <= 0:
                        src_w = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
                    if src_h is None or src_h <= 0:
                        src_h = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
                    tw = max(1, src_w // s) if s > 1 else max(1, src_w)
                    th = max(1, src_h // s) if s > 1 else max(1, src_h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    wr = cv2.VideoWriter(out_resized, fourcc, float(in_fps), (tw, th))
                    while True:
                        ok, frm = cap_r.read()
                        if not ok:
                            break
                        if frm.shape[1] != tw or frm.shape[0] != th:
                            frm = cv2.resize(frm, (tw, th), interpolation=cv2.INTER_LINEAR)
                        wr.write(frm)
                    cap_r.release()
                    wr.release()
                    resized_path = out_resized
                    final_w, final_h = tw, th
                except Exception as trans_err:
                    logger.warning("Video transcode failed (%s); using original path", trans_err)
                    resized_path = path
            if final_w is None or final_h is None:
                if src_w is not None and src_h is not None and src_w > 0 and src_h > 0:
                    final_w, final_h = src_w, src_h
                else:
                    try:
                        cap_tmp = cv2.VideoCapture(resized_path)
                        if cap_tmp.isOpened():
                            final_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or final_w
                            final_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or final_h
                        cap_tmp.release()
                    except Exception:
                        final_w = final_w or None
                        final_h = final_h or None
            classes = parse_prompts(classes_text)
            try:
                t0 = time.perf_counter()
                PromptManager.select(model_name, sys_key, user_key, override_user, prompt_text or "")
                final_prompt = PromptManager.render_user_prompt(model_name, classes, prompt_text or "", width=None, height=None, threshold=float(threshold))
                system_prompt = PromptManager.get_system_prompt(model_name, reasoning=getattr(STATE.detector, "reasoning", None))
                raw = STATE.detector.infer_from_video(resized_path, float(req_fps_value), classes, final_prompt or "", system_prompt, float(threshold))
                dt = time.perf_counter() - t0
            except Exception as e:
                raw = f"Error: {type(e).__name__}: {e}"
                STATE.append_chat("model", "text", raw)
                return gr.update(value=STATE.export_chat_for_ui()), "Direct video analysis failed."
            if s > 1 and src_w and src_h and final_w and final_h:
                res_info = f"Res: {src_w}x{src_h} -> {final_w}x{final_h} (scale x{s})"
            elif final_w and final_h:
                res_info = f"Res: {final_w}x{final_h}"
            else:
                res_info = "Res: unknown"
            vid_path = path
            STATE.append_chat("user", "video", (prompt_text.strip() if prompt_text else "Video direct analysis"), classes=classes_text, video_path=vid_path, fps=req_fps_value)
            STATE.append_chat("model", "text", raw if isinstance(raw, str) else str(raw))
            return gr.update(value=STATE.export_chat_for_ui()), f"Direct video analysis complete in {dt:.2f}s. | {res_info}"

        btn_video_direct_infer.click(
            fn=on_video_direct_raw,
            inputs=[vid_in, classes_tb, prompt_tb, model, model_variant_state, thr, device, vid_fps, sys_prompt_dd, user_prompt_dd, override_cb, resize_scale],
            outputs=[chatbot, status],
        )

        # Frame/video saving
        def save_current_video(video_value, dirpath: str):
            path = _resolve_video_path(video_value)
            if not path or not os.path.exists(path): return "No valid video to save"
            os.makedirs(dirpath, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            ext = os.path.splitext(path)[1] or ".mp4"
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(dirpath, f"video_{base}_{ts}{ext}")
            try:
                import shutil; shutil.copy2(path, out_path); return f"Saved video to {out_path}"
            except Exception as e:
                return f"Failed to save video: {e}"
        btn_save_video.click(save_current_video, inputs=[vid_in, video_save_dir], outputs=[status])

        # -------- Prompt selection persistence --------
        def _load_prompt_selection(sess, model_name: str):
            sel = sess.get("prompt_selections")
            if not isinstance(sel, dict):
                base_override = bool(sess.get("override_user_prompt", False))
                base_text = sess.get("prompt_text") if isinstance(sess.get("prompt_text"), str) else ""
                return {"system": None, "user": None, "override": base_override, "override_text": base_text}
            entry = sel.get(model_name) or {}
            override_val = entry.get("override")
            if override_val is None:
                override_val = bool(sess.get("override_user_prompt", False))
            else:
                override_val = bool(override_val)
            override_text = entry.get("override_text")
            if not isinstance(override_text, str):
                override_text = sess.get("prompt_text") if isinstance(sess.get("prompt_text"), str) else ""
            return {"system": entry.get("system"), "user": entry.get("user"), "override": override_val, "override_text": override_text}

        def _save_prompt_selection(model_name: str, system_key: str | None, user_key: str | None, override_flag: bool, override_text: str):
            sess = load_session_state(); sel = sess.get("prompt_selections", {})
            if not isinstance(sel, dict): sel = {}
            sel[model_name] = {"system": system_key, "user": user_key, "override": bool(override_flag), "override_text": override_text or ""}
            sess["prompt_selections"] = sel; save_session_state(sess)

        def update_prompt_dropdowns(model_name: str):
            sys_opts = PromptManager.system_options(model_name)
            usr_opts = PromptManager.user_options(model_name)
            sel = _load_prompt_selection(load_session_state(), model_name)
            sys_val = sel.get("system") if sel.get("system") in sys_opts else (sys_opts[0] if sys_opts else None)
            usr_val = sel.get("user") if sel.get("user") in usr_opts else (usr_opts[0] if usr_opts else None)
            _save_prompt_selection(model_name, sys_val, usr_val, sel.get("override", False), sel.get("override_text", ""))
            PromptManager.select(model_name, sys_val, usr_val, sel.get("override", False), sel.get("override_text", ""))
            sys_text = PromptManager.get_raw_system_prompt(model_name, sys_val) if sys_val else ""
            user_text = PromptManager.get_raw_user_prompt(model_name, usr_val) if usr_val else ""
            return (gr.update(choices=sys_opts, value=sys_val), gr.update(choices=usr_opts, value=usr_val), gr.update(value=sys_text, interactive=(model_name != "Cosmos-Reason1")), gr.update(value=user_text, interactive=(model_name != "Cosmos-Reason1")))
        model.change(fn=update_prompt_dropdowns, inputs=[model], outputs=[sys_prompt_dd, user_prompt_dd, system_prompt_text, user_prompt_text])

        def update_cosmos_extras(model_name: str):
            if model_name == "Cosmos-Reason1":
                warnings = "\n".join(PromptManager.cosmos_warnings()) or "(no warnings)"
                return gr.update(visible=True), gr.update(value=warnings, visible=True)
            return gr.update(visible=False), gr.update(value="", visible=False)
        model.change(fn=update_cosmos_extras, inputs=[model], outputs=[reasoning_cb, cosmos_warnings_box])

        def toggle_edit_visibility(model_name: str):
            editable = model_name != "Cosmos-Reason1"
            # Components themselves don't expose an .update() method; return gr.update() objects for each output.
            return (
                gr.update(visible=editable),  # update_sys_btn
                gr.update(visible=editable),  # new_sys_name
                gr.update(visible=editable),  # save_new_sys_btn
                gr.update(visible=editable),  # update_user_btn
                gr.update(visible=editable),  # new_user_name
                gr.update(visible=editable),  # save_new_user_btn
            )
        model.change(fn=toggle_edit_visibility, inputs=[model], outputs=[update_sys_btn, new_sys_name, save_new_sys_btn, update_user_btn, new_user_name, save_new_user_btn])

        def on_prompt_select(model_name: str, sys_key: str | None, user_key: str | None, override_user: bool, override_text: str):
            if model_name == "Cosmos-Reason1":
                available = PromptManager.system_options(model_name)
                prev_sys, prev_user, _prev_override, _prev_text = PromptManager.get_selected(model_name)
                chosen = None
                if sys_key and sys_key in available and sys_key != prev_sys:
                    chosen = sys_key
                elif user_key and user_key in available and user_key != prev_user:
                    chosen = user_key
                elif sys_key and sys_key in available:
                    chosen = sys_key
                elif user_key and user_key in available:
                    chosen = user_key
                elif available:
                    chosen = available[0]
                sys_key = chosen
                user_key = chosen
            PromptManager.select(model_name, sys_key, user_key, override_user, override_text or "")
            _save_prompt_selection(model_name, sys_key, user_key, override_user, override_text or "")
            sys_text = PromptManager.get_raw_system_prompt(model_name, sys_key) if sys_key else ""
            user_text = PromptManager.get_raw_user_prompt(model_name, user_key) if user_key else ""
            if model_name == "Cosmos-Reason1":
                return ("Prompt selection updated (synced for Cosmos).", gr.update(value=sys_text), gr.update(value=user_text), gr.update(value=sys_key), gr.update(value=user_key))
            return ("Prompt selection updated.", gr.update(value=sys_text), gr.update(value=user_text), gr.update(), gr.update())
        sys_prompt_dd.change(on_prompt_select, inputs=[model, sys_prompt_dd, user_prompt_dd, override_cb, prompt_tb], outputs=[status, system_prompt_text, user_prompt_text, sys_prompt_dd, user_prompt_dd])
        user_prompt_dd.change(on_prompt_select, inputs=[model, sys_prompt_dd, user_prompt_dd, override_cb, prompt_tb], outputs=[status, system_prompt_text, user_prompt_text, sys_prompt_dd, user_prompt_dd])
        override_cb.change(on_prompt_select, inputs=[model, sys_prompt_dd, user_prompt_dd, override_cb, prompt_tb], outputs=[status, system_prompt_text, user_prompt_text, sys_prompt_dd, user_prompt_dd])
        prompt_tb.change(on_prompt_select, inputs=[model, sys_prompt_dd, user_prompt_dd, override_cb, prompt_tb], outputs=[status, system_prompt_text, user_prompt_text, sys_prompt_dd, user_prompt_dd])

        def on_reasoning_toggle(model_name: str, reasoning: bool, current_status: str):
            if model_name == "Cosmos-Reason1" and STATE.detector is not None and hasattr(STATE.detector, "reasoning"):
                STATE.detector.reasoning = bool(reasoning)
                return f"{current_status}\nReasoning set to {reasoning}."
            return current_status
        reasoning_cb.change(on_reasoning_toggle, inputs=[model, reasoning_cb, status], outputs=[status])

        # Update existing presets
        def do_update_system(model_name: str, sys_key: str | None, text: str):
            if not sys_key: return "No system preset selected.", gr.update(), gr.update()
            ok = PromptManager.update_system_prompt(model_name, sys_key, text)
            if ok: PromptManager.persist_model(model_name); return f"System preset '{sys_key}' updated.", gr.update(), gr.update()
            return "Failed to update system preset (maybe model not editable).", gr.update(), gr.update()
        update_sys_btn.click(fn=do_update_system, inputs=[model, sys_prompt_dd, system_prompt_text], outputs=[status, system_prompt_text, user_prompt_text])

        def do_update_user(model_name: str, user_key: str | None, text: str):
            if not user_key: return "No user preset selected.", gr.update(), gr.update()
            ok = PromptManager.update_user_prompt(model_name, user_key, text)
            if ok: PromptManager.persist_model(model_name); return f"User preset '{user_key}' updated.", gr.update(), gr.update()
            return "Failed to update user preset (maybe model not editable).", gr.update(), gr.update()
        update_user_btn.click(fn=do_update_user, inputs=[model, user_prompt_dd, user_prompt_text], outputs=[status, system_prompt_text, user_prompt_text])

        def do_save_new_system(model_name: str, new_name: str, text: str, sys_key: str | None, user_key: str | None):
            new_name = (new_name or "").strip();
            if not new_name: return ("Provide a new system preset name.", gr.update(), gr.update(), gr.update())
            ok = PromptManager.add_system_prompt(model_name, new_name, text)
            if ok:
                PromptManager.persist_model(model_name)
                sys_opts = PromptManager.system_options(model_name)
                PromptManager.select(model_name, new_name, user_key, False, "")
                return (f"System preset '{new_name}' added.", gr.update(choices=sys_opts, value=new_name), gr.update(), gr.update())
            return ("Failed to add system preset (name exists?).", gr.update(), gr.update(), gr.update())
        save_new_sys_btn.click(fn=do_save_new_system, inputs=[model, new_sys_name, system_prompt_text, sys_prompt_dd, user_prompt_dd], outputs=[status, sys_prompt_dd, system_prompt_text, user_prompt_text])

        def do_save_new_user(model_name: str, new_name: str, text: str, sys_key: str | None, user_key: str | None):
            new_name = (new_name or "").strip();
            if not new_name: return ("Provide a new user preset name.", gr.update(), gr.update(), gr.update())
            ok = PromptManager.add_user_prompt(model_name, new_name, text)
            if ok:
                PromptManager.persist_model(model_name)
                usr_opts = PromptManager.user_options(model_name)
                PromptManager.select(model_name, sys_key, new_name, False, "")
                return (f"User preset '{new_name}' added.", gr.update(choices=user_prompt_dd.choices if hasattr(user_prompt_dd,'choices') else usr_opts, value=new_name), gr.update(), gr.update())
            return ("Failed to add user preset (name exists?).", gr.update(), gr.update(), gr.update())
        save_new_user_btn.click(fn=do_save_new_user, inputs=[model, new_user_name, user_prompt_text, sys_prompt_dd, user_prompt_dd], outputs=[status, user_prompt_dd, system_prompt_text, user_prompt_text])

        def do_refresh_prompts(model_name: str, sys_key: str | None, user_key: str | None, override_user: bool, override_text: str, reasoning: bool):
            PromptManager.load_all()
            sys_opts = PromptManager.system_options(model_name); usr_opts = PromptManager.user_options(model_name)
            if sys_key not in sys_opts: sys_key = sys_opts[0] if sys_opts else None
            if user_key not in usr_opts: user_key = usr_opts[0] if usr_opts else None
            PromptManager.select(model_name, sys_key, user_key, override_user, override_text or "")
            sys_text = PromptManager.get_raw_system_prompt(model_name, sys_key) if sys_key else ""
            user_text = PromptManager.get_raw_user_prompt(model_name, user_key) if user_key else ""
            if model_name == "Cosmos-Reason1":
                sys_text = ""; user_text = ""; warnings = "\n".join(PromptManager.cosmos_warnings()) or "(no warnings)"
            else: warnings = ""
            if model_name == "Cosmos-Reason1" and STATE.detector is not None and hasattr(STATE.detector, "reasoning"):
                STATE.detector.reasoning = bool(reasoning)
            return (gr.update(choices=sys_opts, value=sys_key), gr.update(choices=usr_opts, value=user_key), gr.update(value=sys_text), gr.update(value=user_text), gr.update(value=warnings, visible=(model_name == "Cosmos-Reason1")), f"Prompts reloaded. {len(sys_opts)} system / {len(usr_opts)} user presets.")
        refresh_prompts_btn.click(fn=do_refresh_prompts, inputs=[model, sys_prompt_dd, user_prompt_dd, override_cb, prompt_tb, reasoning_cb], outputs=[sys_prompt_dd, user_prompt_dd, system_prompt_text, user_prompt_text, cosmos_warnings_box, status])

        # Session persistence
        session_debug = gr.Textbox(visible=False, label="Session Debug")
    # Removed variant_history_debug component (history list no longer tracked)
        def persist_state(model_name, variant, classes_text, prompt_text, threshold, resize_val, video_fps, florence_task, override_user, sys_key, user_key):
            sess = load_session_state()
            new = {
                "model": model_name,
                "model_variant": variant,
                "classes_text": classes_text,
                "prompt_text": prompt_text,
                "threshold": threshold,
                "resize_scale": resize_val,
                "video_processing_fps": video_fps,
                "florence_task": florence_task,
                "override_user_prompt": override_user,
                "device": device.value if hasattr(device,'value') else 'auto'
            }
            # Maintain per-model variant history mapping
            mvh = sess.get("model_variant_history", {})
            if not isinstance(mvh, dict):
                mvh = {}
            if model_name and variant:
                mvh[model_name] = variant
            new["model_variant_history"] = mvh
            merged = merge_session_state(sess, new)
            merged.pop("system_prompt_key", None)
            merged.pop("user_prompt_key", None)
            _save_prompt_selection(model_name, sys_key, user_key, override_user, prompt_text or "")
            merged["prompt_selections"] = load_session_state().get("prompt_selections", {})
            save_session_state(merged)
            return json.dumps({k: merged.get(k) for k in ("model","model_variant","threshold")})
        watch = [model, model_variant, classes_tb, prompt_tb, thr, resize_scale, vid_fps, florence_task_dd, override_cb, sys_prompt_dd, user_prompt_dd]
        for c in watch:
            c.change(persist_state, inputs=watch, outputs=[session_debug])

        # ------------- Per-client initial load (restore latest session & populate prompts/gen params) -------------
        def initialize_session_ui():
            # Always load fresh from disk for new browser clients
            sess = load_session_state()
            model_name = sess.get("model") if isinstance(sess, dict) else None
            if model_name not in MODEL_VARIANTS:
                model_name = model.value if hasattr(model, 'value') and model.value in MODEL_VARIANTS else list(MODEL_VARIANTS.keys())[0]
            # Variant handling
            variant_choices = MODEL_VARIANTS.get(model_name, ["default"])
            raw_variant = sess.get("model_variant") if isinstance(sess, dict) else None
            # Prefer per-model history entry if available
            mvh = sess.get("model_variant_history", {}) if isinstance(sess, dict) else {}
            preferred_variant = None
            if isinstance(mvh, dict):
                cand = mvh.get(model_name)
                if cand in variant_choices:
                    preferred_variant = cand
            if preferred_variant is None:
                preferred_variant = raw_variant
            safe_variant = ensure_valid_variant(model_name, preferred_variant or (variant_choices[0] if variant_choices else "default"))
            # Core simple fields
            classes_text = sess.get("classes_text", classes_tb.value if hasattr(classes_tb,'value') else "person, laptop, cup")
            prompt_text = sess.get("prompt_text", prompt_tb.value if hasattr(prompt_tb,'value') else "")
            threshold_val = float(sess.get("threshold", thr.value if hasattr(thr,'value') else 0.25))
            resize_val = int(sess.get("resize_scale", resize_scale.value if hasattr(resize_scale,'value') else 1))
            try:
                video_fps_val = int(sess.get("video_processing_fps", vid_fps.value if hasattr(vid_fps,'value') else 5))
            except Exception:
                video_fps_val = int(vid_fps.value if hasattr(vid_fps,'value') else 5)
            if video_fps_val < 1:
                video_fps_val = 1
            elif video_fps_val > 60:
                video_fps_val = 60
            override_flag = bool(sess.get("override_user_prompt", override_cb.value if hasattr(override_cb,'value') else False))
            device_choice = sess.get("device", device.value if hasattr(device,'value') else "auto")

            # Prompts
            PromptManager.load_all()  # ensure latest files
            sel_prompts = _load_prompt_selection(sess, model_name)
            sys_opts = PromptManager.system_options(model_name)
            usr_opts = PromptManager.user_options(model_name)
            sys_val = sel_prompts.get("system") if sel_prompts.get("system") in sys_opts else (sys_opts[0] if sys_opts else None)
            usr_val = sel_prompts.get("user") if sel_prompts.get("user") in usr_opts else (usr_opts[0] if usr_opts else None)
            override_flag = bool(sel_prompts.get("override", override_flag))
            if not isinstance(prompt_text, str):
                prompt_text = ""
            prompt_text = sel_prompts.get("override_text", prompt_text) or ""
            _save_prompt_selection(model_name, sys_val, usr_val, override_flag, prompt_text)
            PromptManager.select(model_name, sys_val, usr_val, override_flag, prompt_text)
            sys_text = PromptManager.get_raw_system_prompt(model_name, sys_val) if sys_val else ""
            user_text = PromptManager.get_raw_user_prompt(model_name, usr_val) if usr_val else ""

            # Generation params visibility + saved values
            spec = _adapter_spec(model_name)
            current_params = sess.get("generation_params", {}).get(model_name, {}) if isinstance(sess, dict) else {}
            active = set(spec.keys())
            def gp_update(name, base_control, *, interactive_override=None):
                visible = name in active
                val = current_params.get(name, getattr(base_control, 'value', None))
                if val is None:
                    return gr.update(visible=visible, interactive=interactive_override if interactive_override is not None else None)
                # Clamp into spec range if provided
                meta = spec.get(name, {})
                mn = meta.get('min'); mx = meta.get('max')
                try:
                    if mn is not None and val < mn: val = mn
                    if mx is not None and val > mx: val = mx
                except Exception:
                    pass
                upd_kwargs = {"visible": visible, "value": val}
                if interactive_override is not None:
                    upd_kwargs["interactive"] = interactive_override
                return gr.update(**upd_kwargs)
            sampling_enabled = bool(current_params.get("do_sample", True)) if "do_sample" in active else True
            # Compose updates
            model_upd = gr.update(value=model_name)
            variant_upd = gr.update(choices=variant_choices, value=safe_variant)
            variant_state_upd = gr.update(value=safe_variant)
            classes_upd = gr.update(value=classes_text)
            prompt_tb_upd = gr.update(value=prompt_text)
            thr_upd = gr.update(value=threshold_val)
            resize_upd = gr.update(value=resize_val)
            override_upd = gr.update(value=override_flag)
            vid_fps_upd = gr.update(value=video_fps_val)
            device_upd = gr.update(value=device_choice if device_choice in ["auto","cpu","cuda"] else "auto")
            sys_dd_upd = gr.update(choices=sys_opts, value=sys_val)
            usr_dd_upd = gr.update(choices=usr_opts, value=usr_val)
            sys_text_upd = gr.update(value=sys_text, interactive=(model_name != "Cosmos-Reason1"))
            user_text_upd = gr.update(value=user_text, interactive=(model_name != "Cosmos-Reason1"))
            gp_max_new_tokens_upd = gp_update("max_new_tokens", gp_max_new_tokens)
            gp_temperature_upd = gp_update("temperature", gp_temperature, interactive_override=sampling_enabled)
            gp_top_p_upd = gp_update("top_p", gp_top_p, interactive_override=sampling_enabled)
            gp_rep_upd = gp_update("repetition_penalty", gp_repetition_penalty)
            gp_num_beams_upd = gp_update("num_beams", gp_num_beams)
            gp_num_return_sequences_upd = gp_update("num_return_sequences", gp_num_return_sequences)
            gp_do_sample_upd = gr.update(visible=("do_sample" in active), value=current_params.get("do_sample", getattr(gp_do_sample,'value', True)))
            gp_top_k_upd = gp_update("top_k", gp_top_k, interactive_override=sampling_enabled)
            active_list = ", ".join(sorted(active))
            gen_status_text = ("No tunable parameters." if not active else f"Active: {active_list}" + (" (sampling disabled)" if ("temperature" in active or "top_p" in active or "top_k" in active) and not sampling_enabled else ""))
            gen_status_upd = gr.update(value=gen_status_text)
            reasoning_visible = (model_name == "Cosmos-Reason1")
            reasoning_upd = gr.update(visible=reasoning_visible)
            cosmos_warn_text = "\n".join(PromptManager.cosmos_warnings()) if reasoning_visible else ""
            cosmos_warn_upd = gr.update(visible=reasoning_visible, value=cosmos_warn_text)
            status_upd = gr.update(value="Status: restored from session")
            return (
                model_upd, variant_upd, variant_state_upd, classes_upd, prompt_tb_upd, thr_upd, resize_upd, vid_fps_upd, override_upd, device_upd,
                sys_dd_upd, usr_dd_upd, sys_text_upd, user_text_upd,
                gp_max_new_tokens_upd, gp_temperature_upd, gp_top_p_upd, gp_rep_upd, gp_num_beams_upd, gp_num_return_sequences_upd, gp_do_sample_upd, gp_top_k_upd, gen_status_upd,
                reasoning_upd, cosmos_warn_upd, status_upd
            )

        demo.load(
            fn=initialize_session_ui,
            inputs=None,
            outputs=[
                model, model_variant, model_variant_state, classes_tb, prompt_tb, thr, resize_scale, vid_fps, override_cb, device,
                sys_prompt_dd, user_prompt_dd, system_prompt_text, user_prompt_text,
                gp_max_new_tokens, gp_temperature, gp_top_p, gp_repetition_penalty, gp_num_beams, gp_num_return_sequences, gp_do_sample, gp_top_k, gen_params_status,
                reasoning_cb, cosmos_warnings_box, status
            ]
        )
        if chatbot is not None:
            demo.load(fn=lambda: gr.update(value=[]), inputs=None, outputs=[chatbot])

    logger.debug("Exiting build_ui() full-featured")
    # Expose chat components on demo for later reference (e.g., tests) if they exist
    setattr(demo, 'chatbot', chatbot)
    setattr(demo, 'chat_send_btn', chat_send_btn)
    setattr(demo, 'chat_input_tb', chat_input_tb)
    return demo


def main():
    logger.info("Starting application main()")
    demo = build_ui()

    # Add a simple health endpoint and request logging middleware (Gradio v5 compatibility)
    try:
        from fastapi import Request
        # Gradio 5 removed/renamed server_app; attempt multiple attribute names
        fastapi_app = getattr(demo, "server_app", None) or getattr(demo, "app", None)
        if fastapi_app is not None:
            @fastapi_app.get("/health")
            def health():  # type: ignore
                return {"status": "ok"}

            @fastapi_app.middleware("http")  # type: ignore
            async def log_requests(request: Request, call_next):  # type: ignore
                t0 = time.perf_counter()
                logger.info(f"--> {request.method} {request.url.path}")
                resp = await call_next(request)
                dt = (time.perf_counter() - t0) * 1000.0
                logger.info(f"<-- {resp.status_code} {request.url.path} {dt:.1f} ms")
                return resp
            logger.info("Registered /health endpoint and request logging middleware")
        else:
            logger.info("FastAPI app handle not exposed by this Gradio version; skipping /health endpoint registration")
    except Exception as e:
        logger.warning(f"Could not register FastAPI hooks: {e}")

    # Gradio 5: remove deprecated queue(default_enabled=...) usage
    try:
        # If queuing desired, simply call demo.queue(); otherwise skip entirely.
        # demo.queue()  # Uncomment if queueing is needed.
        pass
    except Exception as e:
        logger.warning(f"Queue initialization skipped: {e}")

    host = os.environ.get("HOST", "0.0.0.0")
    port_env = os.environ.get("PORT")
    server_port = None
    if port_env and port_env.isdigit():
        server_port = int(port_env)

    logger.info(f"Launching Gradio app on {host}:{server_port or '[auto]'} ...")
    demo.launch(quiet=False, show_error=True, server_name=host, server_port=server_port)
    logger.info("Gradio app stopped")


if __name__ == "__main__":
    main()
