from __future__ import annotations

import argparse
import os
import time
import json
from dataclasses import asdict
from typing import List, Dict, Callable, Any, Iterable, Tuple
from itertools import product
from pathlib import Path

# Set config directory for standalone app before importing runtime
os.environ["VLM_CONFIG_DIR"] = str(Path(__file__).resolve().parent / "config")

import cv2
import numpy as np

from vlm_detections.core.adapter_base import BaseVisionAdapter
from vlm_detections.core.runtime import is_zero_shot_detector, is_prompt_based_vlm
from vlm_detections.core.visualize import draw_detections
from vlm_detections.adapters.owlvit_adapter import OwlVitAdapter
from vlm_detections.adapters.groundingdino_adapter import GroundingDINOAdapter
from vlm_detections.adapters.florence2_adapter import Florence2Adapter
from vlm_detections.adapters.openai_vision_adapter import OpenAIVisionAdapter
from vlm_detections.adapters.internvl3_5_adapter import InternVL35Adapter
from vlm_detections.adapters.qwen3_vl_adapter import Qwen3VLAdapter
from prompt_manager import PromptManager


MODEL_REGISTRY: Dict[str, Callable[[str], BaseVisionAdapter]] = {
    "OWL-ViT": lambda model_id: OwlVitAdapter(model_id=model_id),
    "GroundingDINO": lambda model_id: GroundingDINOAdapter(model_id=model_id),
    "Florence-2": lambda model_id: Florence2Adapter(model_id=model_id),
    "OpenAI Vision (API)": lambda model_id: OpenAIVisionAdapter(model_id=model_id),
    "InternVL3.5": lambda model_id: InternVL35Adapter(model_id=model_id),
    "Qwen3-VL": lambda model_id: Qwen3VLAdapter(model_id=model_id),
}


def parse_prompts(text: str) -> List[str]:
    parts = [p.strip() for p in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _frange(min_v: float, max_v: float, step: float) -> Iterable[float]:
    cur = min_v
    # Avoid floating drift: iterate while cur <= max_v + tiny epsilon
    while cur <= max_v + 1e-9:
        yield round(cur, 10)
        cur += step


def _build_param_values(spec: Dict[str, Any]) -> List[Any]:
    if "values" in spec and isinstance(spec["values"], list):
        return spec["values"]
    if all(k in spec for k in ("min", "max", "step")):
        return list(_frange(float(spec["min"]), float(spec["max"]), float(spec["step"])))
    return []


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(folder: str, include_subdirs: bool = True) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                paths.append(os.path.join(root, f))
        if not include_subdirs:
            break
    return sorted(paths)


def extract_video_frames(video_path: str, sample_fps: float, max_frames: int | None = None, frame_stride: int = 1) -> List[Tuple[int, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[Tuple[int, Any]] = []
    frame_interval = max(int(round(native_fps / sample_fps))) if sample_fps > 0 else 1
    idx = 0
    kept = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            if (kept % frame_stride) == 0:
                frames.append((idx, frame))
                if max_frames and len(frames) >= max_frames:
                    break
            kept += 1
        idx += 1
    cap.release()
    return frames


def _adapter_supports_generation(detector: BaseVisionAdapter) -> bool:
    return all(hasattr(detector, attr) for attr in ("generation_config_spec", "update_generation_params"))


def run_config_experiment(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "OWL-ViT")
    model_id = model_cfg.get("model_id", "google/owlvit-base-patch32")
    device = model_cfg.get("device", "auto")
    threshold = float(model_cfg.get("threshold", 0.25))
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in registry: {list(MODEL_REGISTRY)}")
    detector = MODEL_REGISTRY[model_name](model_id)
    detector.load(device=device)
    reasoning_flag = getattr(detector, "reasoning", None)
    system_prompt_default = PromptManager.get_system_prompt(model_name, reasoning=reasoning_flag)

    # Prepare sweeps (generation params)
    gen_sweeps = cfg.get("generation_param_sweeps", {}) or {}
    sweep_param_names: List[str] = []
    sweep_values: List[List[Any]] = []
    if gen_sweeps and _adapter_supports_generation(detector):
        for pname, pspec in gen_sweeps.items():
            vals = _build_param_values(pspec if isinstance(pspec, dict) else {})
            if vals:
                sweep_param_names.append(pname)
                sweep_values.append(vals)
    else:
        # No sweeps (single empty combination)
        sweep_param_names = []
        sweep_values = []

    param_combos = list(product(*sweep_values)) if sweep_values else [tuple()]

    class_sets: List[List[str]] = cfg.get("class_sets", []) or [[]]
    instruction_prompts: List[str] = cfg.get("instruction_prompts", []) or [""]

    input_cfg = cfg.get("input", {})
    input_type = input_cfg.get("type", "folder")
    base_folder = input_cfg.get("folder", ".")
    include_subdirs = bool(input_cfg.get("include_subdirs", True))
    explicit_videos: List[str] = input_cfg.get("videos", []) or []
    video_sampling = input_cfg.get("video", {})
    sample_fps = float(video_sampling.get("sample_fps", 1.0))
    max_frames = video_sampling.get("max_frames")
    frame_stride = int(video_sampling.get("frame_stride", 1))

    images: List[str] = []
    videos: List[str] = []
    if input_type == "folder":
        for root, _, files in os.walk(base_folder):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                full = os.path.join(root, f)
                if ext in IMAGE_EXTS:
                    images.append(full)
                elif ext in VIDEO_EXTS:
                    videos.append(full)
            if not include_subdirs:
                break
    elif input_type == "video":
        videos = [os.path.join(base_folder, v) if not os.path.isabs(v) else v for v in explicit_videos]
    elif input_type == "list":
        # Provide explicit list of image paths (and optionally videos) under key 'paths'
        paths = input_cfg.get("paths", [])
        for p in paths:
            full = os.path.join(base_folder, p) if not os.path.isabs(p) else p
            ext = os.path.splitext(full)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(full)
            elif ext in VIDEO_EXTS:
                videos.append(full)
    else:
        raise ValueError(f"Unsupported input.type: {input_type}")

    output_cfg = cfg.get("output", {})
    out_base = output_cfg.get("base_dir", "outputs")
    os.makedirs(out_base, exist_ok=True)
    results_filename = output_cfg.get("results_filename", "results.json")
    save_visualizations = bool(output_cfg.get("save_visualizations", True))
    save_raw_text = bool(output_cfg.get("save_raw_text", True))
    separate_per_param_combo = bool(output_cfg.get("separate_per_param_combo", False))

    timestamp_dir = time.strftime(f"{model_id.replace('/', '_')}_%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(out_base, timestamp_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    def save_results_snapshot():
        # Always include original config + lightweight metadata
        payload = {
            "config": cfg,
            "model": detector.name(),
            "results": all_results,
            "num_results": len(all_results),
            "sweep_params": sweep_param_names,
            "timestamp": timestamp_dir,
        }
        with open(os.path.join(experiment_dir, results_filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # Iterate parameter combinations
    for combo_idx, combo_vals in enumerate(param_combos):
        active_params = dict(zip(sweep_param_names, combo_vals))
        if active_params and _adapter_supports_generation(detector):
            detector.update_generation_params(active_params)
        combo_tag = "_".join(f"{k}-{v}" for k, v in active_params.items()) if active_params else "default"
        combo_dir = os.path.join(experiment_dir, combo_tag)
        if separate_per_param_combo:
            os.makedirs(combo_dir, exist_ok=True)

        # Images
        for img_path in images:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue
            for cls_set in class_sets:
                for instr in instruction_prompts:
                    start = time.time()
                    # Call appropriate interface - prefer PromptBasedVLM if we have instruction
                    if is_prompt_based_vlm(detector) and instr:
                        detections, props, rels, raw_text = detector.infer_from_image(img_bgr, instr, system_prompt_default, threshold)
                    elif is_zero_shot_detector(detector):
                        detections = detector.infer(img_bgr, cls_set, threshold)
                        props, rels, raw_text = [], [], ""
                    else:
                        raise ValueError(f"Unknown adapter protocol for {model_name}")
                    elapsed = time.time() - start
                    record = {
                        "input_type": "image",
                        "path": img_path,
                        "classes": cls_set,
                        "prompt": instr,
                        "generation_params": active_params,
                        "detections": [
                            {
                                "xyxy": list(det.xyxy),
                                "score": det.score,
                                "label": det.label,
                                **({"polygon": det.polygon} if det.polygon else {}),
                                **({"text": det.text} if det.text else {}),
                            }
                            for det in detections
                        ],
                        "num_detections": len(detections),
                        "raw_output_text": raw_text if save_raw_text else None,
                        "elapsed_sec": elapsed,
                        "param_combo_index": combo_idx,
                    }
                    all_results.append(record)
                    if save_visualizations:
                        vis = draw_detections(img_bgr, detections)
                        rel = os.path.relpath(img_path, base_folder)
                        out_dir = combo_dir if separate_per_param_combo else experiment_dir
                        out_path = os.path.join(out_dir, "viz", rel)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cv2.imwrite(out_path, vis)
            if not separate_per_param_combo:
                save_results_snapshot()

        # Videos
        for video_path in videos:
            frames = extract_video_frames(video_path, sample_fps, max_frames=max_frames, frame_stride=frame_stride)
            if not frames:
                print(f"[WARN] No frames extracted or cannot open video: {video_path}")
                continue
            for cls_set in class_sets:
                for instr in instruction_prompts:
                    for frame_idx, frame_bgr in frames:
                        start = time.time()
                        # Call appropriate interface - prefer PromptBasedVLM if we have instruction
                        if is_prompt_based_vlm(detector) and instr:
                            detections, props, rels, raw_text = detector.infer_from_image(frame_bgr, instr, system_prompt_default, threshold)
                        elif is_zero_shot_detector(detector):
                            detections = detector.infer(frame_bgr, cls_set, threshold)
                            props, rels, raw_text = [], [], ""
                        else:
                            raise ValueError(f"Unknown adapter protocol for {model_name}")
                        elapsed = time.time() - start
                        record = {
                            "input_type": "video_frame",
                            "video_path": video_path,
                            "frame_index": frame_idx,
                            "classes": cls_set,
                            "prompt": instr,
                            "generation_params": active_params,
                            "detections": [
                                {
                                    "xyxy": list(det.xyxy),
                                    "score": det.score,
                                    "label": det.label,
                                    **({"polygon": det.polygon} if det.polygon else {}),
                                    **({"text": det.text} if det.text else {}),
                                }
                                for det in detections
                            ],
                            "num_detections": len(detections),
                            "raw_output_text": raw_text if save_raw_text else None,
                            "elapsed_sec": elapsed,
                            "param_combo_index": combo_idx,
                        }
                        all_results.append(record)
                        if save_visualizations:
                            vis = draw_detections(frame_bgr, detections)
                            base_name = os.path.splitext(os.path.basename(video_path))[0]
                            out_dir = combo_dir if separate_per_param_combo else experiment_dir
                            out_path = os.path.join(out_dir, "viz", base_name, f"frame_{frame_idx:06d}.jpg")
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            cv2.imwrite(out_path, vis)
                    if not separate_per_param_combo:
                        save_results_snapshot()

        # Write separate results for param combo if requested
        if separate_per_param_combo:
            combo_results = [r for r in all_results if r.get("param_combo_index") == combo_idx]
            payload = {
                "config": cfg,
                "model": detector.name(),
                "results": combo_results,
                "num_results": len(combo_results),
                "sweep_params": sweep_param_names,
                "param_combo": active_params,
                "timestamp": timestamp_dir,
            }
            with open(os.path.join(combo_dir, results_filename), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    # Final aggregate write
    if not separate_per_param_combo:
        save_results_snapshot()
    print(f"Experiment complete. Results written under: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Configurable batch zero-shot detection experiment runner")
    parser.add_argument("--config", type=str, help="Path to experiment JSON config. If provided, overrides legacy CLI mode.")
    # Legacy arguments (maintained for backward compatibility if --config not supplied)
    parser.add_argument("folder", nargs="?", type=str, help="(Legacy) Input folder with images")
    parser.add_argument("--prompts", type=str, help="(Legacy) Prompts (comma or newline separated)")
    parser.add_argument("--model", type=str, default="OWL-ViT", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model_id", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "api"])
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        run_config_experiment(cfg)
        return

    # Legacy path
    if not args.folder or not args.prompts:
        parser.error("Legacy mode requires folder positional argument and --prompts")
    prompts = parse_prompts(args.prompts)
    detector = MODEL_REGISTRY[args.model](args.model_id)
    detector.load(device=args.device)
    ts_dir = time.strftime(f"{args.model_id.replace('/', '_')}_%Y%m%d_%H%M%S")
    out_root = os.path.join(args.folder, ts_dir)
    os.makedirs(out_root, exist_ok=True)
    images = list_images(args.folder)
    if not images:
        print("No images found.")
        return
    for img_path in images:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Skipping unreadable file: {img_path}")
            continue
        legacy_reasoning = getattr(detector, "reasoning", None)
        system_prompt_legacy = PromptManager.get_system_prompt(args.model, reasoning=legacy_reasoning)
        # Call appropriate interface - in legacy mode prompts contain class names for zero-shot
        if is_zero_shot_detector(detector):
            detections = detector.infer(img_bgr, prompts, args.threshold)
        elif is_prompt_based_vlm(detector):
            # For VLMs in legacy mode, prompts should be treated as user prompt
            prompt_text = ", ".join(prompts) if isinstance(prompts, list) else str(prompts)
            detections, _, _, _ = detector.infer_from_image(img_bgr, prompt_text, system_prompt_legacy, args.threshold)
        else:
            raise ValueError(f"Unknown adapter protocol for {args.model}")
        vis = draw_detections(img_bgr, detections)
        rel = os.path.relpath(img_path, args.folder)
        out_path = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
