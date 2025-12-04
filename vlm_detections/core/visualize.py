from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple
from .adapter_base import Detection


def draw_detections(
    image_bgr,
    detections: List[Detection],
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.5,
    min_size: Tuple[int, int] | None = None,
    polygon_color=(255, 0, 0),
    polygon_alpha: float = 0.25,
    try_supervision: bool = True,
):
    """Draw detections including optional polygons and captions.

    - If det.polygon is provided, draw filled translucent polygon + outline.
    - If polygon absent, draw bbox rectangle.
    - If det.text present (e.g., OCR, region caption), append after label or replace label if label empty.
    """
    img = image_bgr.copy()
    overlay = img.copy()
    used_supervision = False
    sv = None
    if try_supervision:
        try:
            import supervision as sv  # type: ignore
        except Exception:
            sv = None
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        if min_size is not None:
            if (x2 - x1) < min_size[0] or (y2 - y1) < min_size[1]:
                continue
        label_core = det.label or "object"
        tail = ""
        if det.text and det.text.strip() and det.text.strip() != det.label:
            tail = f" | {det.text.strip()}"
        label = f"{label_core} {det.score:.2f}{tail}" if det.score is not None else f"{label_core}{tail}"
        if det.polygon and len(det.polygon) >= 3:
            if sv is not None and not used_supervision:
                # Defer drawing via supervision after loop by collecting
                pass
            # Expect list of (x,y). Filter invalid points
            pts = [(int(px), int(py)) for px, py in det.polygon if px is not None and py is not None]
            if len(pts) >= 3:
                pts_arr = cv2.convexHull(np.array(pts, dtype=np.int32)) if len(pts) > 3 else np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], polygon_color)
                cv2.polylines(img, [pts_arr], isClosed=True, color=polygon_color, thickness=thickness)
                # Use first point for label anchor (handle hull shape Nx1x2)
                first_pt = pts_arr[0]
                if isinstance(first_pt, (list, tuple, np.ndarray)) and len(first_pt) == 1:
                    first_pt = first_pt[0]
                if isinstance(first_pt, np.ndarray):
                    lx, ly = int(first_pt[0]), int(first_pt[1])
                else:
                    lx, ly = first_pt
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), polygon_color, thickness)
                lx, ly = x1, y1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            lx, ly = x1, y1
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        by = max(0, ly - th - 6)
        cv2.rectangle(img, (lx, by), (lx + tw + 4, by + th + 6), (0, 0, 0), -1)
        cv2.putText(img, label, (lx + 2, by + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    # Blend overlay for filled polygons
    if any(det.polygon for det in detections):
        cv2.addWeighted(overlay, polygon_alpha, img, 1 - polygon_alpha, 0, img)
    return img
