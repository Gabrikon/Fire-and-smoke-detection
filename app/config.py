"""Central configuration for the fire/smoke detection service.

All tunables live here and are sourced from environment variables (see .env.example),
so the same code runs locally, in CI, and on Modal.
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


APP_NAME = "fire-vlm"

# Class names produced by the trained YOLO26 model.
CLASS_NAMES = ["fire", "smoke"]


class Settings:
    """Resolved settings (read once at import)."""

    # --- Detector (YOLO26) ---
    model_path: str = os.environ.get("MODEL_PATH", "weights/fire_yolo_best.pt")
    # Per-class confidence: smoke is diffuse/low-contrast and scores lower, so it gets a lower
    # threshold (high recall); the VLM + reasoner then filter steam/vapor false alarms.
    conf_threshold: float = _f("CONF_THRESHOLD", 0.35)          # fire
    conf_threshold_smoke: float = _f("CONF_THRESHOLD_SMOKE", 0.18)
    iou_threshold: float = _f("IOU_THRESHOLD", 0.45)
    input_size: int = _i("INPUT_SIZE", 960)                     # higher res helps diffuse smoke

    def conf_by_class(self) -> dict:
        return {"fire": self.conf_threshold, "smoke": self.conf_threshold_smoke}

    # --- Alert gating (when to escalate to the VLM) ---
    consecutive_frames: int = _i("CONSECUTIVE_FRAMES", 5)
    vlm_cooldown_seconds: int = _i("VLM_COOLDOWN_SECONDS", 30)

    # --- Localized description model (NVIDIA DAM-3B, self-hosted on Modal GPU) ---
    dam_model: str = os.environ.get("DAM_MODEL", "nvidia/DAM-3B-Video")
    dam_image_fallback: str = os.environ.get("DAM_IMAGE_FALLBACK", "nvidia/DAM-3B-Self-Contained")
    dam_gpu: str = os.environ.get("DAM_GPU", "L4")
    dam_max_new_tokens: int = _i("DAM_MAX_NEW_TOKENS", 512)
    dam_video_frames: int = _i("DAM_VIDEO_FRAMES", 8)  # frames sampled from a clip

    # --- Reasoning model (NVIDIA NIM, OpenAI-compatible cloud API) ---
    nvidia_api_key: str = os.environ.get("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    reasoner_model: str = os.environ.get("NIM_REASONER_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
    reasoner_temperature: float = _f("REASONER_TEMPERATURE", 0.2)
    reasoner_max_tokens: int = _i("REASONER_MAX_TOKENS", 1024)

    # --- Database (self-hosted Supabase / Postgres on Railway) ---
    db_url: str = os.environ.get("SUPABASE_DB_URL", "")
    db_schema: str = os.environ.get("DB_SCHEMA", "fire_detection")

    # --- Access model (open UI; mirrors naija-petro) ---
    admin_token: str = os.environ.get("ADMIN_TOKEN", "")
    free_daily_limit: int = _i("FREE_DAILY_LIMIT", 20)


settings = Settings()
