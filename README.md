<div align="center">

# 🔥 Emberwatch

### Hybrid CNN + NVIDIA VLM fire &amp; smoke detection for oil &amp; gas facilities

Real time **YOLO26** detection, **NVIDIA DAM-3B** localized vision, and an **NVIDIA NIM** reasoning model,
turning any camera feed into a structured fire and smoke safety advisory. Served on **Modal**, stored in **Supabase**.

<br>

[![Live demo](https://img.shields.io/badge/demo-live-ef4444?style=for-the-badge&logo=rocket&logoColor=white)](https://chidi-ashinze--fire-vlm.modal.run)
[![CI](https://img.shields.io/github/actions/workflow/status/Gabrikon/Fire-and-smoke-detection/ci.yml?branch=main&style=for-the-badge&label=CI&logo=githubactions&logoColor=white)](https://github.com/Gabrikon/Fire-and-smoke-detection/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-serverless%20GPU-7B3FE4?logo=modal&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-DAM--3B%20%2B%20NIM-76B900?logo=nvidia&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLO26-Ultralytics-111F68?logo=yolo&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Postgres-3ECF8E?logo=supabase&logoColor=white)
![Tailwind](https://img.shields.io/badge/UI-Tailwind%20%2B%20Alpine-38BDF8?logo=tailwindcss&logoColor=white)
![Code style](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)

[Live app](https://chidi-ashinze--fire-vlm.modal.run) ·
[How it works](#how-it-works) ·
[Quickstart](#quickstart) ·
[API](#api-reference) ·
[Roadmap](#roadmap)

</div>

---

## Overview

Fire is easy to spot; the hard parts are **smoke** (translucent, easily confused with steam or vapor) and
**context** (is that flare stack an incident or normal operation?). Emberwatch splits the problem across three
models, each doing what it is best at:

1. **Detect.** A YOLO26 CNN scans every sampled frame on CPU and marks fire and smoke regions in milliseconds.
2. **Describe.** When a detection is sustained, **NVIDIA DAM-3B** ("Describe Anything", a 3B vision model) looks
   at the exact detected region and describes it in detail, including telling real smoke from steam, vapor, fog,
   or a controlled flare.
3. **Advise.** An **NVIDIA NIM** reasoning model turns that description plus the detection data into a strict JSON
   safety advisory: severity, threat type, affected zone, recommended actions, and an escalation level.

This is a full rebuild of the original Streamlit + GPT-4o app (preserved under `legacy/`) into a fully NVIDIA,
Modal-hosted stack with a custom dashboard, an admin panel, persistence, and CI/CD.

> [!NOTE]
> The DAM-3B models are released under NVIDIA's **non-commercial** license (research and academic use). This
> project is a research/portfolio build; commercial facility deployment is not permitted under that license.

## How it works

```
Browser (webcam @ ~3 fps  OR  image / video upload)
   │  JPEG frame / clip
   ▼
FastAPI (Modal, ASGI)
   ├─ /detect   YOLO26 (Modal, CPU)        ──►  fire / smoke boxes + confidences   [fast, gated]
   │
   └─ on sustained detection (5 frames) or upload:
        ├─ DAM-3B-Video (Modal, 1 GPU)     ──►  localized description per class (image or clip)
        ├─ NVIDIA NIM reasoner (cloud)     ──►  structured JSON safety advisory
        └─ Supabase (schema fire_detection) ──►  users, detection_events, advisories, usage
   ◄─ Server-Sent Events stream: detecting → describing → reasoning → advisory
```

- **YOLO26** runs on **CPU** (weights are ~20 MB), so it is cheap and never blocks the video.
- **DAM-3B-Video** runs on **one** Modal GPU, **scales to zero** when idle, and only fires on a sustained alert.
- **Reasoning** runs on **NVIDIA's cloud** via your `nvapi-` key, so there is no second GPU to pay for.

## Models

| Role | Model | Runs on |
|------|-------|---------|
| Detector | YOLO26 (`weights/fire_yolo_best.pt`) | Modal, CPU |
| Localized description (image + video) | [`nvidia/DAM-3B-Video`](https://huggingface.co/nvidia/DAM-3B-Video) | Modal, GPU (L4) |
| Safety-advisory reasoning | `nvidia/llama-3.3-nemotron-super-49b-v1` (configurable) | NVIDIA NIM cloud |

## Smoke detection

Smoke is the hard class, so it gets special handling (no retraining required):

- **Per-class confidence thresholds.** Smoke uses a lower bar (`0.18`) than fire (`0.35`), so diffuse smoke is
  not silently filtered out. The detector casts a wide net and the VLM makes the real call.
- **Higher inference resolution** (`960`), which helps low-contrast, diffuse smoke.
- **Describe every class.** DAM describes a fire region **and** a smoke region, instead of only the single
  highest-confidence box (which would always be the brighter fire).
- **Smoke-aware prompt.** DAM is explicitly asked to judge genuine combustion smoke versus steam, vapor, fog,
  dust, or flare exhaust, and the reasoner can flag a false alarm.
- **Temporal persistence** on video: each class is counted across sampled frames, so a one-frame artifact is
  downweighted and sustained smoke is trusted.

## Features

- 🎥 Live webcam detection with client-side bounding boxes, plus image and short-clip upload
- 🧠 Two-stage NVIDIA vision + reasoning pipeline with streamed (SSE) status and advisory
- 🛢️ Oil and gas domain prompt (flare stacks, steam, equipment risk, false-alarm logic)
- 🗄️ Supabase persistence (users, detection events, advisories, usage)
- 🔑 Open access with a per-IP daily free limit, bypassed by access tokens
- 🛠️ Admin panel at `/admin` for token management and usage stats
- 🎨 Polished "Emberwatch" dashboard (Tailwind + Alpine + Lucide), responsive, dark
- ⚙️ GitHub Actions CI (lint + smoke tests) with optional auto-deploy to Modal

## Quickstart

```bash
git clone https://github.com/Gabrikon/Fire-and-smoke-detection.git
cd Fire-and-smoke-detection

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # fill in the keys below
```

Fill `.env` with:

| Variable | What it is |
|----------|------------|
| `NVIDIA_API_KEY` | `nvapi-...` key from [build.nvidia.com](https://build.nvidia.com) (reasoning model) |
| `HF_TOKEN` | Hugging Face token; accept the `nvidia/DAM-3B-Video` license once |
| `SUPABASE_DB_URL` | Postgres connection string (self-hosted Supabase / Railway) |
| `ADMIN_TOKEN` | any secret string; unlocks `/admin` |

Then deploy to Modal (GPU + scale-to-zero handled for you):

```bash
pip install modal && modal token new          # one-time Modal login
bash scripts/setup_modal_secret.sh            # build the Modal secret from .env
psql "$SUPABASE_DB_URL" -f supabase/schema.sql # create the fire_detection schema (once)
python scripts/seed_tokens.py                 # optional: seed access tokens

modal serve  modal_app.py                     # ephemeral dev URL (hot reload)
modal deploy modal_app.py                     # production URL
```

> [!TIP]
> Values in `.env` flow into the Modal secret and **override** image defaults. After changing a detection
> tunable (`INPUT_SIZE`, `CONF_THRESHOLD*`), re-run `bash scripts/setup_modal_secret.sh` before redeploying.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONF_THRESHOLD` | `0.35` | Fire confidence threshold |
| `CONF_THRESHOLD_SMOKE` | `0.18` | Smoke confidence threshold (lower; VLM filters false alarms) |
| `INPUT_SIZE` | `960` | YOLO inference resolution |
| `CONSECUTIVE_FRAMES` | `5` | Sustained fire frames before the VLM is triggered (live) |
| `VLM_COOLDOWN_SECONDS` | `30` | Minimum gap between advisories (live) |
| `NIM_REASONER_MODEL` | `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA NIM reasoning model |
| `DAM_MODEL` | `nvidia/DAM-3B-Video` | Localized description model |
| `DAM_GPU` | `L4` | Modal GPU type for DAM |
| `FREE_DAILY_LIMIT` | `20` | Anonymous advisories per IP per day |

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Dashboard UI |
| `GET`  | `/admin` | Admin panel |
| `GET`  | `/healthz` | Status and model config |
| `POST` | `/detect` | Run YOLO on one frame, returns detections (cheap, CPU) |
| `POST` | `/advise` | Frame to advisory, streamed via SSE (GPU) |
| `POST` | `/advise_video` | Short clip to advisory, streamed via SSE (GPU) |
| `POST` | `/user` | Capture an email |
| `GET`  | `/token/check` | Validate an access token |
| `*`    | `/admin/api/*` | Token management and usage stats (admin token required) |

Send `X-Access-Token: <token>` to bypass the daily free limit; `X-Admin-Token: <token>` for admin routes.

## Access model

The interface is **open** to try. Anonymous visitors get `FREE_DAILY_LIMIT` advisories per IP per UTC day.
A valid access token removes the limit. Tokens (3 primary + 7 secondary) are seeded by
`scripts/seed_tokens.py` and managed in the `/admin` panel (protected by `ADMIN_TOKEN`).

## Project structure

```
modal_app.py            Modal app: YOLOService (CPU), DAMService (GPU), FastAPI asgi
app/
  config.py             settings (models, per-class thresholds, DB, NIM)
  detect/yolo.py        YOLO load + predict + per-class filtering + helpers
  vlm/dam.py            DAM-3B-Video: describe_region / describe_video, box->mask, class prompts
  vlm/reasoner.py       NVIDIA NIM client + advisory schema + robust JSON parsing
  db.py                 asyncpg layer (users, events, advisories, tokens, usage)
  api/server.py         FastAPI: /detect, /advise, /advise_video, admin, SSE, logging
  frontend/             Emberwatch dashboard (index.html) + admin panel (admin.html)
supabase/schema.sql     fire_detection schema
scripts/                seed_tokens.py, setup_modal_secret.sh
.github/workflows/ci.yml  lint + smoke tests + optional modal deploy
vlm_eval.py             evaluate the deployed /advise endpoint over a folder of images
legacy/                 original Streamlit + GPT-4o app (reference)
```

## CI/CD

`.github/workflows/ci.yml` runs on every push and PR: `compileall`, `ruff` (F and E9), and an import smoke
test. On push to `main`, a deploy job runs `modal deploy` if `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repo
secrets are set (it skips cleanly if they are not).

## Roadmap

- [ ] Retrain YOLO with more smoke data and hard negatives (steam, fog, dust) for higher smoke precision
- [ ] Feedback loop: mine the Supabase event log (and false-alarm verdicts) to build a retraining set
- [ ] SAM2 mask tracking for tighter region masks and true per-frame video tracking
- [ ] Multi-camera dashboard and alert notifications

## Tech stack

`Python 3.11` · `FastAPI` (SSE) · `Modal` (serverless GPU) · `Ultralytics YOLO26` ·
`NVIDIA DAM-3B-Video` · `NVIDIA NIM` · `Supabase / Postgres` (`asyncpg`) ·
`Tailwind CSS` + `Alpine.js` + `Lucide`

## License

Code is released under the [MIT License](LICENSE). The NVIDIA DAM-3B models are under NVIDIA's
**non-commercial** license (research and academic use only).

## Credits

Original detection app by **Gabrikon**. NVIDIA + Modal rebuild, smoke pipeline, and Emberwatch UI by
**Ashinze Emmanuel** ([GitHub](https://github.com/Mystique1337) · [Hugging Face](https://huggingface.co/Shinzmann)).
