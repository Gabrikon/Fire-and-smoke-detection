"""YOLO26 fire/smoke detector.

Thin wrapper around Ultralytics YOLO. Runs on CPU (the trained weights are ~20 MB, so
per-frame inference is a few hundred ms, fast enough for sampled webcam frames and uploads).

`format_detections` keeps the same output shape the original GPT-4o pipeline used, so the
downstream advisory prompt stays compatible.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def load_model(model_path: str):
    """Load a YOLO model from a .pt path. Raises if the file is missing/invalid."""
    from ultralytics import YOLO

    return YOLO(model_path)


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes to a BGR ndarray (or None if undecodable)."""
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect(
    model: Any,
    image_bgr: np.ndarray,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
) -> list[dict]:
    """Run detection on a single BGR frame and return a list of detections.

    Each detection: {"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]} (xyxy, pixels).
    """
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    return _detections_from_results(results)


def _detections_from_results(results) -> list[dict]:
    detections: list[dict] = []
    if results and len(results) > 0:
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": result.names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(float(c), 1) for c in box.xyxy[0].tolist()],
            })
    return detections


def format_detections(detections: list[dict]) -> str:
    """Human-readable summary of detections, fed to the VLM/reasoner.

    Matches the format the original GPT-4o prompt expected.
    """
    if not detections:
        return "No fire or smoke detected."

    summary = f"Automated YOLO detections - {len(detections)} object(s):\n"
    for i, d in enumerate(detections):
        summary += (
            f"  [{i + 1}] Class: {d['class']}, "
            f"Confidence: {d['confidence']:.1%}, "
            f"BBox (xyxy): {d['bbox']}\n"
        )
    return summary


def alert_level(buffer: list[bool]) -> str:
    """Map a window of per-frame fire flags to an alert level (ported from the original)."""
    if not buffer:
        return "CLEAR"
    ratio = sum(buffer) / len(buffer)
    if ratio >= 0.8:
        return "HIGH"
    if ratio >= 0.5:
        return "MEDIUM"
    if ratio > 0:
        return "LOW"
    return "CLEAR"


def primary_box(detections: list[dict]) -> list[float] | None:
    """Return the highest-confidence detection box (the region to describe with DAM)."""
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])["bbox"]
