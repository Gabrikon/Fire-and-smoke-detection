"""Modal deployment for the fire/smoke detection app.

    modal serve  modal_app.py     # local dev (hot reload, ephemeral URL)
    modal deploy modal_app.py     # production

Components:
  * YOLOService  - YOLO26 fire/smoke detector (CPU; weights baked into the image)
  * DAMService   - NVIDIA DAM-3B-Video localized description (one GPU, scale-to-zero)
  * fastapi_app  - the web front door (ASGI), with GPU work injected as deps
  * The NIM reasoning model runs on NVIDIA's cloud (called from the web layer), no GPU here.

Note: the DAM image installs the NVlabs describe-anything package and a CUDA torch build; if the
first deploy hits a dependency conflict, pin transformers/torch here (same iteration the
naija-petro vLLM image needed). Force fresh code after redeploy with `modal app stop fire-vlm --yes`.
"""
from __future__ import annotations

import modal

from app.config import APP_NAME, settings

app = modal.App(APP_NAME)

# Persisted HF cache so the DAM weights download once across cold starts.
hf_cache = modal.Volume.from_name("fire-vlm-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"
FRONTEND_REMOTE = "/assets/frontend"
WEIGHTS_REMOTE = "/assets/weights"

secrets = [modal.Secret.from_name("fire-vlm-secrets")]

# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV runtime libs
    .pip_install("ultralytics>=8.3.0", "opencv-python-headless>=4.8.0", "numpy>=1.26")
    .env({"MODEL_PATH": f"{WEIGHTS_REMOTE}/fire_yolo_best.pt"})
    .add_local_python_source("app")
    .add_local_dir("weights", remote_path=WEIGHTS_REMOTE)
)

dam_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision")  # CUDA build by default on linux
    .pip_install(
        "transformers>=4.40", "accelerate", "einops", "sentencepiece", "protobuf",
        "huggingface_hub>=0.27", "hf_transfer", "pillow",
        # opencv 4.10 is the last line that supports numpy 1.x; describe-anything pins numpy 1.26,
        # so newer opencv (which requires numpy>=2) would fail to import in this container.
        "opencv-python-headless==4.10.0.84", "numpy>=1.26,<2", "decord",
    )
    .pip_install("git+https://github.com/NVlabs/describe-anything")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("app")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]>=0.115", "asyncpg>=0.29", "httpx>=0.27", "openai>=1.55",
        "opencv-python-headless>=4.8.0", "numpy>=1.26", "pillow",
    )
    .env({"FRONTEND_DIR": FRONTEND_REMOTE})
    .add_local_python_source("app")
    .add_local_dir("app/frontend", remote_path=FRONTEND_REMOTE)
)


# --------------------------------------------------------------------------- #
# YOLO detector (CPU)
# --------------------------------------------------------------------------- #
@app.cls(image=yolo_image, secrets=secrets, scaledown_window=120, timeout=120)
@modal.concurrent(max_inputs=8)
class YOLOService:
    @modal.enter()
    def start(self):
        from app.detect import yolo

        self._yolo = yolo
        # Load from the baked-in image path directly. We do NOT read MODEL_PATH from the
        # environment here: the Modal secret (built from .env) carries the local relative
        # MODEL_PATH and would override the image env, breaking the lookup in-container.
        self.model = yolo.load_model(f"{WEIGHTS_REMOTE}/fire_yolo_best.pt")

    @modal.method()
    def detect_bytes(self, image_bytes: bytes) -> list[dict]:
        img = self._yolo.decode_image(image_bytes)
        if img is None:
            return []
        return self._yolo.detect(
            self.model, img,
            conf=settings.conf_threshold, iou=settings.iou_threshold, imgsz=settings.input_size,
        )


# --------------------------------------------------------------------------- #
# DAM-3B-Video (GPU, scale-to-zero)
# --------------------------------------------------------------------------- #
@app.cls(
    image=dam_image,
    gpu=settings.dam_gpu,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=secrets,
    scaledown_window=120,        # idle GPU shuts down quickly
    timeout=20 * 60,
    max_containers=1,            # one GPU at most
)
@modal.concurrent(max_inputs=4)
class DAMService:
    @modal.enter()
    def start(self):
        from app.vlm.dam import DAM

        self.dam = DAM(settings.dam_model)

    @modal.method()
    def describe_region_bytes(self, image_bytes: bytes, box: list) -> str:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None or not box:
            return ""
        return self.dam.describe_region(img, box, max_new_tokens=settings.dam_max_new_tokens)

    @modal.method()
    def describe_video_bytes(self, frames_bytes: list, box: list) -> str:
        import cv2
        import numpy as np

        frames = []
        for b in frames_bytes:
            fr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if fr is not None:
                frames.append(fr)
        if not frames or not box:
            return ""
        return self.dam.describe_video(frames, box, max_new_tokens=settings.dam_max_new_tokens)


# --------------------------------------------------------------------------- #
# Dependency wiring (GPU/CPU services -> async callables for the web layer)
# --------------------------------------------------------------------------- #
async def _detect_bytes(image_bytes: bytes) -> list[dict]:
    return await YOLOService().detect_bytes.remote.aio(image_bytes)


async def _describe_region_bytes(image_bytes: bytes, box: list) -> str:
    return await DAMService().describe_region_bytes.remote.aio(image_bytes, box)


async def _describe_video_bytes(frames_bytes: list, box: list) -> str:
    return await DAMService().describe_video_bytes.remote.aio(frames_bytes, box)


# --------------------------------------------------------------------------- #
# Web front door
# --------------------------------------------------------------------------- #
@app.function(image=web_image, secrets=secrets, scaledown_window=300, timeout=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="fire-vlm")
def fastapi_app():
    from app.api.server import Deps, create_app

    deps = Deps(
        detect_bytes=_detect_bytes,
        describe_region_bytes=_describe_region_bytes,
        describe_video_bytes=_describe_video_bytes,
    )
    return create_app(deps)
