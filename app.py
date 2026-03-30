"""
Live Fire Detection Dashboard
Hybrid CNN-VLM Framework — YOLO26 + GPT-4o via streamlit-webrtc

Run:  streamlit run app.py
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Auto-download weights if not present ───
WEIGHTS_PATH = "weights/fire_yolo26s_best.pt"
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs("weights", exist_ok=True)
    logger.warning("Model weights not found at %s.", WEIGHTS_PATH)
    # ── UNCOMMENT to auto-download from Google Drive: ──
    # import subprocess
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    # import gdown
    # gdown.download(
    #     "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE",
    #     WEIGHTS_PATH, quiet=False,
    # )

import streamlit as st
import av
import cv2
import time
import numpy as np
import threading
from collections import deque
from datetime import datetime

from config import (
    MODEL_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    ALERT_BUFFER_SIZE,
    VLM_COOLDOWN_SECONDS,
    WEBRTC_VIDEO_CONSTRAINTS,
    ICE_SERVERS,
)
from vlm_reasoner import (
    format_detections,
    encode_image_bytes,
    query_gpt4o,
)

try:
    from ultralytics import YOLO
except ImportError:
    st.error("ultralytics not installed. Run: `pip install ultralytics`")
    st.stop()

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        VideoProcessorBase,
        RTCConfiguration,
    )
except ImportError:
    st.error("streamlit-webrtc not installed. Run: `pip install streamlit-webrtc`")
    st.stop()


# ══════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════

st.set_page_config(
    page_title="Fire Detection — Oil & Gas",
    page_icon="\U0001F525",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@keyframes pulse-border{
    0%,100%{border-color:#dc2626;box-shadow:0 0 0 0 rgba(220,38,38,.4)}
    50%{border-color:#f87171;box-shadow:0 0 15px 5px rgba(220,38,38,.2)}
}
.alert-banner-critical{animation:pulse-border 1.5s infinite;border:2px solid #dc2626;border-radius:10px;padding:16px 20px;background:linear-gradient(135deg,#1a0000,#2d0a0a);color:#fca5a5;margin-bottom:12px}
.alert-banner-high{border:2px solid #ea580c;border-radius:10px;padding:16px 20px;background:linear-gradient(135deg,#1a0f00,#2d1a0a);color:#fdba74;margin-bottom:12px}
.alert-banner-medium{border:2px solid #d97706;border-radius:10px;padding:16px 20px;background:linear-gradient(135deg,#1a1400,#2d220a);color:#fcd34d;margin-bottom:12px}
.alert-banner-low{border:2px solid #65a30d;border-radius:10px;padding:16px 20px;background:linear-gradient(135deg,#0a1a00,#152d0a);color:#bef264;margin-bottom:12px}
.status-badge{display:inline-block;padding:6px 16px;border-radius:20px;font-weight:600;font-size:14px;letter-spacing:.5px}
.status-clear{background:#166534;color:#bbf7d0}
.status-low{background:#854d0e;color:#fef08a}
.status-medium{background:#9a3412;color:#fed7aa}
.status-high{background:#991b1b;color:#fecaca}
.metric-row{display:flex;gap:12px;margin:8px 0}
.metric-card{flex:1;background:#1e1e2e;border:1px solid #333;border-radius:8px;padding:12px 16px;text-align:center}
.metric-card .label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}
.metric-card .value{font-size:22px;font-weight:700;color:#e0e0e0;margin-top:4px}
.action-item{background:#1a1a2e;border-left:3px solid #dc2626;padding:8px 12px;margin:4px 0;border-radius:0 6px 6px 0;font-size:14px}
.det-log{font-family:'Courier New',monospace;font-size:12px;background:#0d0d1a;border:1px solid #222;border-radius:6px;padding:10px;max-height:200px;overflow-y:auto}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════
#  LOAD YOLO26 MODEL
# ══════════════════════════════════════════

@st.cache_resource
def load_yolo_model(path: str):
    if not os.path.exists(path):
        return None
    model = YOLO(path)
    logger.info("YOLO26 model loaded: %s  classes=%s", path, model.names)
    return model


model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error(
        f"**Model weights not found** at `{MODEL_PATH}`.  \n"
        "Download your trained YOLO26 `.pt` file from Google Drive and place it in `weights/`.  \n"
        "See `README.md` for instructions."
    )
    st.stop()


# ══════════════════════════════════════════
#  THREAD-SAFE SHARED STATE
# ══════════════════════════════════════════

_lock = threading.Lock()

if "shared_state" not in st.session_state:
    st.session_state.shared_state = {
        "detection_count": 0,
        "latest_detections": [],
        "latest_summary": "",
        "fps": 0.0,
        "inference_ms": 0.0,
        "frame_count": 0,
        "fire_buffer": deque(maxlen=ALERT_BUFFER_SIZE),
        "alert_level": "CLEAR",
        "last_vlm_time": 0.0,
        "latest_advisory": None,
        "vlm_in_progress": False,
        "vlm_call_count": 0,
        "alert_history": [],
    }

shared = st.session_state.shared_state


# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════

with st.sidebar:
    st.markdown("## Configuration")

    st.markdown("**OpenAI API Key**")
    api_key = st.text_input(
        "API Key",
        type="password",
        key="k_oai",
        value=os.environ.get("OPENAI_API_KEY", ""),
        label_visibility="collapsed",
    )

    st.markdown("**Detection**")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, DEFAULT_CONF_THRESHOLD, 0.05)
    frame_skip = st.slider("Process every N frames", 1, 10, DEFAULT_FRAME_SKIP)

    st.markdown("**Alert Rules**")
    vlm_cooldown = st.slider("VLM cooldown (sec)", 10, 120, VLM_COOLDOWN_SECONDS)
    consecutive_frames = st.slider("Consecutive fire frames", 2, 15, ALERT_BUFFER_SIZE)

    with _lock:
        shared["fire_buffer"] = deque(shared["fire_buffer"], maxlen=consecutive_frames)

    st.divider()
    if st.button("Reset Alerts"):
        with _lock:
            shared["alert_history"] = []
            shared["latest_advisory"] = None
            shared["fire_buffer"] = deque(maxlen=consecutive_frames)
            shared["alert_level"] = "CLEAR"
            shared["vlm_call_count"] = 0

    st.caption(f"Model: `{MODEL_PATH}`")
    st.caption(f"Classes: {', '.join(model.names.values())}")
    st.caption("VLM: OpenAI GPT-4o")


# ══════════════════════════════════════════
#  VIDEO PROCESSOR (background thread)
# ══════════════════════════════════════════

class FireDetectionProcessor(VideoProcessorBase):
    """Receives webcam frames via WebRTC, runs YOLO26, triggers GPT-4o."""

    def __init__(self):
        self._count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._count += 1
        img = frame.to_ndarray(format="bgr24")

        if self._count % frame_skip != 0:
            return av.VideoFrame.from_ndarray(self._draw_overlay(img), format="bgr24")

        # YOLO26 inference
        t0 = time.time()
        results = model.predict(
            img, conf=conf_threshold, iou=DEFAULT_IOU_THRESHOLD,
            imgsz=640, verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000.0

        summary, detections = format_detections(results)
        annotated = results[0].plot()
        has_fire = len(detections) > 0

        with _lock:
            shared["frame_count"] = self._count
            shared["detection_count"] = len(detections)
            shared["latest_detections"] = detections
            shared["latest_summary"] = summary
            shared["inference_ms"] = inference_ms
            shared["fps"] = 1000.0 / max(inference_ms, 1.0)

            buf = shared["fire_buffer"]
            buf.append(has_fire)

            if len(buf) == 0:
                shared["alert_level"] = "CLEAR"
            else:
                ratio = sum(buf) / len(buf)
                if ratio >= 0.8:
                    shared["alert_level"] = "HIGH"
                elif ratio >= 0.5:
                    shared["alert_level"] = "MEDIUM"
                elif ratio > 0:
                    shared["alert_level"] = "LOW"
                else:
                    shared["alert_level"] = "CLEAR"

            # VLM trigger check
            recent = list(buf)[-consecutive_frames:] if len(buf) >= consecutive_frames else []
            should_vlm = (
                api_key
                and len(recent) == consecutive_frames
                and all(recent)
                and not shared["vlm_in_progress"]
                and (time.time() - shared["last_vlm_time"]) > vlm_cooldown
            )

            if should_vlm:
                shared["vlm_in_progress"] = True
                shared["last_vlm_time"] = time.time()
                ok, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    b64 = encode_image_bytes(jpeg_buf.tobytes())
                    threading.Thread(
                        target=self._vlm_worker, args=(b64, summary), daemon=True,
                    ).start()
                else:
                    shared["vlm_in_progress"] = False

        return av.VideoFrame.from_ndarray(self._draw_overlay(annotated), format="bgr24")

    def _vlm_worker(self, b64_image: str, summary: str):
        """Background GPT-4o call."""
        try:
            advisory = query_gpt4o(b64_image, summary, api_key)
            advisory["_timestamp"] = datetime.now().strftime("%H:%M:%S")

            with _lock:
                shared["latest_advisory"] = advisory
                shared["vlm_call_count"] += 1
                shared["alert_history"].append(advisory)
                if len(shared["alert_history"]) > 50:
                    shared["alert_history"] = shared["alert_history"][-50:]

        except Exception as exc:
            logger.exception("VLM worker error")
            with _lock:
                shared["latest_advisory"] = {
                    "error": str(exc),
                    "_timestamp": datetime.now().strftime("%H:%M:%S"),
                }
        finally:
            with _lock:
                shared["vlm_in_progress"] = False

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw HUD status bar on frame."""
        h, w = frame.shape[:2]

        with _lock:
            level = shared["alert_level"]
            n_det = shared["detection_count"]
            fps = shared["fps"]
            ms = shared["inference_ms"]
            vlm_busy = shared["vlm_in_progress"]

        # Semi-transparent top bar
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

        if level in ("HIGH", "MEDIUM"):
            color = (0, 0, 255) if level == "HIGH" else (0, 165, 255)
            label = f"FIRE DETECTED  ({n_det})"
            thick = 4 if int(time.time() * 3) % 2 == 0 else 2
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thick)
        elif level == "LOW":
            color = (0, 200, 255)
            label = "POSSIBLE DETECTION"
        else:
            color = (0, 220, 0)
            label = "ALL CLEAR"

        cv2.putText(frame, label, (12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        info = f"FPS:{fps:.0f} | {ms:.0f}ms"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
        cv2.putText(frame, info, (w - tw - 12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        if vlm_busy:
            cv2.putText(frame, "GPT-4o ANALYZING...", (12, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)

        return frame


# ══════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════

st.markdown("# Live Fire Detection Dashboard")
st.markdown(
    "*Hybrid CNN-VLM framework — YOLO26 + GPT-4o — "
    "real-time fire detection for oil & gas facilities*"
)

# Status bar
s1, s2, s3, s4 = st.columns(4)
with _lock:
    _level = shared["alert_level"]
    _ndet = shared["detection_count"]
    _fps = shared["fps"]
    _vcalls = shared["vlm_call_count"]

with s1:
    cls = f"status-{_level.lower()}"
    st.markdown(f'<span class="status-badge {cls}">{_level}</span>', unsafe_allow_html=True)
with s2:
    st.metric("Detections", _ndet)
with s3:
    st.metric("YOLO FPS", f"{_fps:.0f}")
with s4:
    st.metric("VLM Calls", _vcalls)

# Two-column layout
col_vid, col_adv = st.columns([3, 2])

# ─── VIDEO COLUMN ───
with col_vid:
    st.markdown("### Live Feed")
    ctx = webrtc_streamer(
        key="fire-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": ICE_SERVERS}),
        video_processor_factory=FireDetectionProcessor,
        media_stream_constraints=WEBRTC_VIDEO_CONSTRAINTS,
        async_processing=True,
    )

    if not ctx.state.playing:
        st.info("Click **START** above to open your webcam. Allow camera access when prompted.")

    with st.expander("Detection Log", expanded=False):
        with _lock:
            dets = shared["latest_detections"]
        if dets:
            lines = [
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{d['class']:>5s}  conf={d['confidence']:.2f}  bbox={d['bbox']}"
                for d in dets
            ]
            st.markdown('<div class="det-log">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        else:
            st.caption("No active detections.")

# ─── ADVISORY COLUMN ───
with col_adv:
    st.markdown("### GPT-4o Safety Advisory")

    with _lock:
        advisory = shared["latest_advisory"]
        vlm_busy = shared["vlm_in_progress"]

    if vlm_busy:
        st.warning("GPT-4o is analysing the scene...")

    if advisory is None:
        st.info(
            f"No advisory yet. GPT-4o activates after **{consecutive_frames}** "
            "consecutive fire frames. Enter your OpenAI API key in the sidebar."
        )
    elif "error" in advisory:
        st.error(f"GPT-4o Error: {advisory['error']}")
    else:
        sev = advisory.get("severity", "UNKNOWN").upper()
        esc = advisory.get("escalation_level", "UNKNOWN")
        banner = {
            "CRITICAL": "alert-banner-critical", "HIGH": "alert-banner-high",
            "MEDIUM": "alert-banner-medium",
        }.get(sev, "alert-banner-low")

        st.markdown(
            f'<div class="{banner}">'
            f"<strong>SEVERITY: {sev}</strong><br>Escalation: {esc}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="metric-row">
            <div class="metric-card"><div class="label">Threat</div>
                <div class="value" style="font-size:13px">{advisory.get("threat_type","—")}</div></div>
            <div class="metric-card"><div class="label">Scale</div>
                <div class="value" style="font-size:15px">{advisory.get("estimated_scale","—")}</div></div>
            <div class="metric-card"><div class="label">Confidence</div>
                <div class="value">{advisory.get("confidence",0):.0%}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        if advisory.get("is_false_alarm"):
            st.warning(f"Possible false alarm: {advisory.get('false_alarm_reason','N/A')}")

        st.markdown(f"**Affected Zone:** {advisory.get('affected_zone', '—')}")

        actions = advisory.get("recommended_actions", [])
        if actions:
            st.markdown("**Recommended Actions:**")
            for a in actions:
                st.markdown(f'<div class="action-item">{a}</div>', unsafe_allow_html=True)

        with st.expander("GPT-4o Reasoning"):
            st.write(advisory.get("reasoning", "—"))
            st.caption(f"Generated at {advisory.get('_timestamp','—')}")

        with st.expander("Raw JSON"):
            st.json(advisory)


# Alert history
st.divider()
with st.expander("Alert History", expanded=False):
    with _lock:
        hist = list(shared["alert_history"])
    if not hist:
        st.caption("No alerts yet.")
    else:
        for alert in reversed(hist):
            if "error" in alert:
                st.error(f"[{alert.get('_timestamp','?')}] {alert['error']}")
            else:
                st.markdown(
                    f"**[{alert.get('_timestamp','?')}]** "
                    f"{alert.get('severity','?')} — "
                    f"{alert.get('threat_type','?')} "
                    f"→ *{alert.get('escalation_level','?')}*"
                )

# Static image test
st.divider()
with st.expander("Test with Static Image", expanded=False):
    st.markdown("Upload an image to test the pipeline without a webcam.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img_up")

    if uploaded is not None:
        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Could not decode image.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(img[..., ::-1], caption="Original", use_container_width=True)

            results = model.predict(img, conf=conf_threshold, verbose=False)
            summary, dets = format_detections(results)
            annotated_img = results[0].plot()

            with c2:
                st.image(annotated_img[..., ::-1], caption=f"{len(dets)} detection(s)", use_container_width=True)

            if dets and api_key:
                if st.button("Run GPT-4o Advisory on This Image"):
                    with st.spinner("Calling GPT-4o..."):
                        ok, jbuf = cv2.imencode(".jpg", img)
                        if ok:
                            b64 = encode_image_bytes(jbuf.tobytes())
                            try:
                                result = query_gpt4o(b64, summary, api_key)
                                st.json(result)
                            except Exception as exc:
                                st.error(f"GPT-4o error: {exc}")
            elif dets and not api_key:
                st.info("Enter your OpenAI API key in the sidebar to get a VLM advisory.")

# Footer
st.divider()
st.caption(
    "Hybrid CNN-VLM Fire Detection  |  "
    "YOLO26 + GPT-4o  |  "
    "Oil & Gas Facility Safety"
)
