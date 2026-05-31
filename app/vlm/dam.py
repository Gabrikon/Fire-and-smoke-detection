"""NVIDIA DAM-3B-Video: localized region description (the "Describe Anything" model).

Given a frame (or a short clip) plus a region from YOLO, DAM produces a precise localized
description of exactly that region. We feed it the highest-confidence YOLO box, converted to a
binary mask. DAM-3B-Video is a joint image+video model, so the same `get_description` call handles
both: a single-frame list for an image, a multi-frame list for a clip.

The model is loaded once per Modal GPU container (see modal_app.py) and reused across requests.

Reference API (NVlabs/describe-anything):
    dam = DescribeAnythingModel(model_path=..., conv_mode="v1", prompt_mode="focal_prompt").to("cuda")
    for tok in dam.get_description([img], [mask], query, streaming=True, temperature=0.2,
                                   top_p=0.5, num_beams=1, max_new_tokens=512): ...
"""
from __future__ import annotations

import numpy as np

# Fire-focused description prompt. DAM is a captioner; we steer it toward the visual cues the
# safety reasoner needs, without asking it to make the safety decision itself.
DEFAULT_QUERY = (
    "<image>\nDescribe the highlighted region in precise detail. Focus on any fire, flame, or "
    "smoke: its size, color, and intensity; the equipment, surface, or structure involved; the "
    "amount and color of any smoke; and whether it looks like an active spreading fire, a smoke "
    "plume, steam or vapor, or a controlled flare. Note anything ambiguous or any sign it may be a "
    "false alarm."
)


def box_to_mask(height: int, width: int, box: list[float]) -> "object":
    """Build a binary PIL mask (L mode, 0/255) that is white inside the YOLO box.

    A rectangular mask is a good v1 region cue for DAM's focal prompt. (SAM2 could give a tighter
    mask and per-frame tracking for video; left as a future refinement.)
    """
    from PIL import Image

    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(c)) for c in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask)


def _bgr_to_pil(image_bgr: np.ndarray) -> "object":
    from PIL import Image

    # OpenCV frames are BGR; PIL expects RGB.
    return Image.fromarray(image_bgr[:, :, ::-1].copy()).convert("RGB")


class DAM:
    """Wrapper around DescribeAnythingModel, loaded once and reused."""

    def __init__(self, model_repo: str, conv_mode: str = "v1", prompt_mode: str = "focal_prompt"):
        import torch
        from huggingface_hub import snapshot_download
        from dam import DescribeAnythingModel

        local_dir = snapshot_download(model_repo)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DescribeAnythingModel(
            model_path=local_dir, conv_mode=conv_mode, prompt_mode=prompt_mode,
        ).to(self.device)

    def _describe(self, images: list, masks: list, query: str,
                  temperature: float, top_p: float, max_new_tokens: int) -> str:
        # streaming=True yields tokens; join them into the final description.
        tokens = self.model.get_description(
            images, masks, query,
            streaming=True, temperature=temperature, top_p=top_p,
            num_beams=1, max_new_tokens=max_new_tokens,
        )
        return "".join(tokens).strip()

    def describe_region(self, image_bgr: np.ndarray, box: list[float], query: str | None = None,
                        temperature: float = 0.2, top_p: float = 0.5,
                        max_new_tokens: int = 512) -> str:
        """Describe a single image region (single-frame path)."""
        h, w = image_bgr.shape[:2]
        img = _bgr_to_pil(image_bgr)
        mask = box_to_mask(h, w, box)
        return self._describe([img], [mask], query or DEFAULT_QUERY,
                              temperature, top_p, max_new_tokens)

    def describe_video(self, frames_bgr: list[np.ndarray], box: list[float], query: str | None = None,
                       temperature: float = 0.2, top_p: float = 0.5,
                       max_new_tokens: int = 512) -> str:
        """Describe a region across sampled frames of a clip (temporal path).

        The same box-derived mask is applied to each sampled frame as a v1 region cue (SAM2 mask
        propagation would track the region more tightly across frames).
        """
        if not frames_bgr:
            return ""
        h, w = frames_bgr[0].shape[:2]
        imgs = [_bgr_to_pil(f) for f in frames_bgr]
        masks = [box_to_mask(h, w, box) for _ in frames_bgr]
        return self._describe(imgs, masks, query or DEFAULT_QUERY,
                              temperature, top_p, max_new_tokens)
