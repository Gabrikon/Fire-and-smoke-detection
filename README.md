# Hybrid CNN + VLM Fire & Smoke Detection (NVIDIA, on Modal)

Real-time fire/smoke detection for oil & gas facilities. A **YOLO26** CNN localizes fire/smoke in
each frame; when a detection is sustained, **NVIDIA DAM-3B-Video** (the "Describe Anything" 3B
model) produces a precise localized description of the detected region (image or short video), and
an **NVIDIA NIM** reasoning model turns that into a structured safety advisory. Everything is served
from **Modal** behind a single beautiful dashboard, with users and events stored in **Supabase**.

This is a rebuild of the original Streamlit + GPT-4o app (kept under `legacy/`) to a fully NVIDIA,
Modal-hosted stack.

## Pipeline

```
Browser (webcam @ ~2-4 fps  OR  image/video upload)
   -> YOLO26 (Modal, CPU)            fire/smoke boxes + confidences
   -> on sustained detection:
        DAM-3B-Video (Modal GPU)     localized description of the YOLO region (image or clip)
        NVIDIA NIM reasoner (cloud)  -> structured JSON safety advisory
   -> Supabase (schema `fire`)       users, detection_events, advisories, alerts
```

- **YOLO26** runs on CPU (weights are 20 MB) - cheap, never blocks.
- **DAM-3B-Video** (3B) runs on one Modal GPU, scales to zero, and only fires on a sustained alert.
- **Reasoning** runs on NVIDIA's cloud (your `nvapi-` key), so no second GPU.

## Models

| Role | Model | Where |
|------|-------|-------|
| Detector | YOLO26 (`weights/fire_yolo_best.pt`) | Modal CPU |
| Localized description (image + video) | `nvidia/DAM-3B-Video` | Modal GPU |
| Safety-advisory reasoning | `nvidia/llama-3.3-nemotron-super-49b-v1` (configurable) | NVIDIA NIM cloud |

DAM-3B is under NVIDIA's non-commercial license (research/academic use).

## Quick start (local dev)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in NVIDIA_API_KEY, HF_TOKEN, SUPABASE_DB_URL

# Run on Modal (recommended; GPU + scale-to-zero handled for you)
pip install modal && modal token new
modal serve modal_app.py    # ephemeral dev URL with hot reload
modal deploy modal_app.py   # production
```

Apply the database schema once: run `supabase/schema.sql` against your Supabase/Postgres instance.

## Repo layout

```
modal_app.py          Modal app: YOLOService (CPU), DAMService (GPU), FastAPI asgi
app/config.py         settings (models, thresholds, DB, NIM)
app/detect/yolo.py    YOLO load + predict + detection formatting
app/vlm/dam.py        DAM-3B-Video: describe_region / describe_video, box->mask
app/vlm/reasoner.py   NVIDIA NIM client + advisory schema + JSON parsing
app/db.py             asyncpg layer (users, events, advisories, tokens)
app/api/server.py     FastAPI: /detect, /advise, /advise_video, admin, logging
app/frontend/         Tailwind + Alpine dashboard + admin panel
supabase/schema.sql   `fire` schema
legacy/               original Streamlit + GPT-4o app (reference)
```

## Credits

Original detection app by Gabrikon. NVIDIA/Modal rebuild and UI by Ashinze Emmanuel
([GitHub](https://github.com/Mystique1337), [Hugging Face](https://huggingface.co/Shinzmann)).
