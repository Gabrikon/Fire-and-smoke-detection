"""
Central configuration — all tunable constants in one place.
"""
import os

# ─── YOLO26 Settings ───
MODEL_PATH = os.environ.get("MODEL_PATH", "weights/fire_yolo_best.pt")
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# ─── Live Processing ───
DEFAULT_FRAME_SKIP = 3          # Process every Nth frame
ALERT_BUFFER_SIZE = 5           # Consecutive fire frames to trigger VLM
VLM_COOLDOWN_SECONDS = 30       # Min gap between VLM API calls

# ─── OpenAI VLM Settings ───
OPENAI_MODEL = "gpt-4o"
VLM_MAX_TOKENS = 1024
VLM_TEMPERATURE = 0.2

# ─── Class Names (must match data.yaml) ───
CLASS_NAMES = ["fire", "smoke"]

# ─── WebRTC ───
WEBRTC_VIDEO_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
        "frameRate": {"ideal": 30},
    },
    "audio": False,
}

# STUN servers for WebRTC NAT traversal (needed for cloud deploy)
ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]
