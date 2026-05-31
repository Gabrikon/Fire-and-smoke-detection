"""FastAPI front door for the fire/smoke detection app.

Endpoints
  GET  /                  dashboard UI
  GET  /admin             admin panel
  GET  /healthz           status + config
  POST /detect            run YOLO on one frame  -> detections + alert level   (cheap, CPU)
  POST /advise            frame -> DAM region description -> NIM advisory (SSE) (gated, GPU)
  POST /advise_video      clip  -> sampled YOLO + DAM video -> NIM advisory (SSE)
  POST /user              capture an email (store the user)
  GET  /token/check       validate an access token
  admin: /admin/api/{auth,tokens,tokens/toggle,tokens/create,stats}

GPU-bound work is injected as async callables via `Deps`, so this module has no torch/YOLO/DAM
import and the same code runs locally and on Modal. Mirrors the naija-petro create_app(deps) pattern.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app import db
from app.config import CLASS_NAMES, settings
from app.detect import yolo

FRONTEND_DIR = os.environ.get("FRONTEND_DIR", os.path.join(os.path.dirname(__file__), "..", "frontend"))


@dataclass
class Deps:
    """GPU/inference callables injected by the Modal layer (or a local stub)."""
    detect_bytes: Callable[[bytes], Awaitable[list[dict]]]
    describe_region_bytes: Callable[[bytes, list, str], Awaitable[str]]
    describe_video_bytes: Callable[[list, list, str], Awaitable[str]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _ip_hash(request: Request) -> str:
    return hashlib.sha256(_client_ip(request).encode()).hexdigest()[:32]


def _req_token(request: Request) -> str | None:
    return request.headers.get("x-access-token") or request.query_params.get("token")


def _admin_ok(request: Request) -> bool:
    tok = request.headers.get("x-admin-token", "")
    return bool(settings.admin_token) and tok == settings.admin_token


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _read_html(name: str) -> str:
    path = os.path.join(FRONTEND_DIR, name)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>{name} not found</h1>"


def _sample_video_frames(video_bytes: bytes, n: int) -> list:
    """Decode a clip to up to n evenly-spaced BGR frames."""
    import tempfile

    import cv2

    frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # Fallback: read sequentially.
            while len(frames) < n:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            return frames
        idxs = [int(i * (total - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
    return frames


def _encode_jpeg(frame_bgr) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(deps: Deps) -> FastAPI:
    app = FastAPI(title="Fire-Smoke VLM")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_read_html("index.html"))

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page():
        return HTMLResponse(_read_html("admin.html"))

    @app.get("/healthz")
    async def healthz():
        return {
            "open": True,
            "daily_limit": settings.free_daily_limit,
            "admin": bool(settings.admin_token),
            "detector": "YOLO26",
            "vision_model": settings.dam_model,
            "reasoner_model": settings.reasoner_model,
            "classes": CLASS_NAMES,
        }

    @app.post("/user")
    async def capture_user(request: Request):
        body = await request.json()
        email = (body.get("email") or "").strip() or None
        uid = await db.upsert_user(email, _ip_hash(request))
        return {"ok": True, "user_id": uid}

    @app.get("/token/check")
    async def token_check(request: Request):
        return {"valid": await db.token_active(_req_token(request))}

    # ---- Detection (cheap, per-frame) ----
    @app.post("/detect")
    async def detect(request: Request, file: UploadFile = File(...)):
        image_bytes = await file.read()
        detections = await deps.detect_bytes(image_bytes)
        has_fire = len(detections) > 0
        level = "HIGH" if has_fire else "CLEAR"
        return {"detections": detections, "alert_level": level, "has_fire": has_fire}

    async def _limit_ok(request: Request) -> tuple[bool, str | None, bool]:
        """(allowed, token_if_valid, has_token). Daily limit applies only to anonymous users."""
        token = _req_token(request)
        has_token = await db.token_active(token)
        if has_token:
            return True, token, True
        count = await db.daily_ip_count(_ip_hash(request))
        return (count < settings.free_daily_limit), None, False

    # ---- Advisory on a single frame (gated, GPU) ----
    @app.post("/advise")
    async def advise(request: Request, file: UploadFile = File(...)):
        image_bytes = await file.read()
        allowed, token, has_token = await _limit_ok(request)
        if not allowed:
            return JSONResponse(
                {"limit": "daily", "daily_limit": settings.free_daily_limit,
                 "message": "Daily free limit reached. Add an access token to continue."},
                status_code=429,
            )

        async def gen():
            t0 = time.time()
            yield _sse({"event": "status", "stage": "detecting"})
            detections = await deps.detect_bytes(image_bytes)
            if not detections:
                yield _sse({"event": "advisory", "advisory": _clear_advisory(), "detections": []})
                return
            # Describe one region per detected class (so smoke is never dropped in favor of fire).
            regions = yolo.boxes_per_class(detections)
            summary = yolo.format_detections(detections)

            yield _sse({"event": "status", "stage": "describing", "detections": detections})
            parts = []
            for cls, det in regions.items():
                desc = await deps.describe_region_bytes(image_bytes, det["bbox"], cls)
                if desc:
                    parts.append(f"{cls.capitalize()} region: {desc}")
            description = "\n\n".join(parts)

            yield _sse({"event": "status", "stage": "reasoning", "description": description})
            advisory, _raw = await _reason(summary, description)

            latency = int((time.time() - t0) * 1000)
            await _persist(request, token, has_token, "image", "advise",
                           detections, advisory, description, latency)
            yield _sse({"event": "advisory", "advisory": advisory,
                        "detections": detections, "description": description,
                        "latency_ms": latency})

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- Advisory on a video clip (gated, GPU) ----
    @app.post("/advise_video")
    async def advise_video(request: Request, file: UploadFile = File(...)):
        video_bytes = await file.read()
        allowed, token, has_token = await _limit_ok(request)
        if not allowed:
            return JSONResponse(
                {"limit": "daily", "daily_limit": settings.free_daily_limit,
                 "message": "Daily free limit reached. Add an access token to continue."},
                status_code=429,
            )

        async def gen():
            t0 = time.time()
            yield _sse({"event": "status", "stage": "sampling"})
            frames = _sample_video_frames(video_bytes, settings.dam_video_frames)
            if not frames:
                yield _sse({"event": "error", "message": "Could not read the video."})
                return

            yield _sse({"event": "status", "stage": "detecting"})
            # Detect on each sampled frame; aggregate per-class peak box + how many frames it
            # appears in (temporal persistence: smoke seen across many frames is more likely real
            # than a one-frame artifact).
            n = len(frames)
            det_frames = []                # jpegs that contained any detection
            per_class: dict = {}           # cls -> {"conf","box","frames"}
            peak_dets, peak_conf = [], -1.0
            for fr in frames:
                jpeg = _encode_jpeg(fr)
                dets = await deps.detect_bytes(jpeg)
                if not dets:
                    continue
                det_frames.append(jpeg)
                for d in dets:
                    e = per_class.setdefault(d["class"], {"conf": -1.0, "box": None, "frames": 0})
                    e["frames"] += 1
                    if d["confidence"] > e["conf"]:
                        e["conf"], e["box"] = d["confidence"], d["bbox"]
                top = max(dets, key=lambda d: d["confidence"])
                if top["confidence"] > peak_conf:
                    peak_conf, peak_dets = top["confidence"], dets

            if not per_class:
                yield _sse({"event": "advisory", "advisory": _clear_advisory(), "detections": []})
                return

            summary = yolo.format_detections(peak_dets)
            persist = "; ".join(f"{c} present in {e['frames']}/{n} sampled frames"
                                for c, e in per_class.items())
            yield _sse({"event": "status", "stage": "describing", "detections": peak_dets})
            parts = []
            for cls, e in per_class.items():
                desc = await deps.describe_video_bytes(det_frames, e["box"], cls)
                if desc:
                    parts.append(f"{cls.capitalize()} region: {desc}")
            description = "\n\n".join(parts)

            yield _sse({"event": "status", "stage": "reasoning", "description": description})
            advisory, _raw = await _reason(
                summary, description,
                extra=f"Temporal persistence across the clip: {persist}. Consider whether the fire "
                      "or smoke appears to be growing, shrinking, or stable, and that something seen "
                      "in only one frame is more likely a false detection.",
            )

            latency = int((time.time() - t0) * 1000)
            await _persist(request, token, has_token, "video", "advise_video",
                           peak_dets, advisory, description, latency)
            yield _sse({"event": "advisory", "advisory": advisory,
                        "detections": peak_dets, "description": description,
                        "latency_ms": latency})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/history")
    async def history():
        return {"advisories": await db.recent_advisories(20)}

    # ---- Admin ----
    @app.post("/admin/api/auth")
    async def admin_auth(request: Request):
        return {"ok": _admin_ok(request)}

    @app.get("/admin/api/tokens")
    async def admin_tokens(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"tokens": await db.list_tokens()}

    @app.post("/admin/api/tokens/toggle")
    async def admin_toggle(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        await db.set_token_active(int(body["id"]), bool(body["active"]))
        return {"ok": True}

    @app.post("/admin/api/tokens/create")
    async def admin_create(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        kind = body.get("kind", "secondary")
        cap = 3 if kind == "primary" else 7
        if await db.count_tokens_by_kind(kind) >= cap:
            return JSONResponse({"error": f"cap reached ({cap} {kind})"}, status_code=400)
        token = body.get("token") or f"fire-{kind[:3]}-{os.urandom(8).hex()}"
        await db.create_token(token, body.get("label", ""), kind)
        return {"ok": True, "token": token}

    @app.get("/admin/api/stats")
    async def admin_stats(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await db.usage_overview(14)

    # ---- shared internals ----
    _reasoner_holder: dict = {}

    async def _reason(summary: str, description: str, extra: str = "") -> tuple[dict, str]:
        if "r" not in _reasoner_holder:
            from app.vlm.reasoner import default_reasoner
            _reasoner_holder["r"] = default_reasoner()
        return await _reasoner_holder["r"].advise(summary, description, extra)

    async def _persist(request, token, has_token, source, kind,
                       detections, advisory, description, latency):
        uid = await db.upsert_user(None, _ip_hash(request))
        event_id = await db.log_event(
            user_id=uid, ip_hash=_ip_hash(request), source=source,
            detections=detections, alert_level="HIGH",
        )
        await db.log_advisory(
            event_id=event_id, advisory=advisory, dam_description=description,
            model_used=settings.reasoner_model, latency_ms=latency,
        )
        await db.log_usage(ip_hash=_ip_hash(request), kind=kind, user_id=uid,
                           token_used=token if has_token else None)

    return app


def _clear_advisory() -> dict:
    return {
        "severity": "LOW", "threat_type": "No fire or smoke detected", "is_false_alarm": False,
        "false_alarm_reason": None, "affected_zone": "N/A", "estimated_scale": "small",
        "recommended_actions": ["Continue monitoring"], "escalation_level": "MONITOR",
        "reasoning": "The detector found no fire or smoke in the analyzed input.", "confidence": 0.9,
    }
