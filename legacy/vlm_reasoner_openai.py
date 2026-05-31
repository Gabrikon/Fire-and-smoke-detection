"""
VLM Reasoning Layer — OpenAI GPT-4o Vision
Takes a frame + YOLO detection summary → structured safety advisory JSON.
"""
import base64
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert fire safety analyst for oil and gas facilities.

You receive an image from a facility camera along with automated fire/smoke
detection results from a YOLO-based computer vision system.

Analyze the scene and provide a structured safety advisory.

ALWAYS respond with ONLY valid JSON (no markdown fences, no preamble) matching
this exact schema:

{
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "threat_type": "string — what is detected (flare stack fire, process fire, smoke plume, electrical fire, etc.)",
  "is_false_alarm": false,
  "false_alarm_reason": null,
  "affected_zone": "string — description of the area/equipment visible",
  "estimated_scale": "small | medium | large | catastrophic",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "escalation_level": "MONITOR | ALERT_OPERATOR | EVACUATE_ZONE | FULL_EMERGENCY",
  "reasoning": "string — 2-3 sentence explanation of your analysis",
  "confidence": 0.85
}

Oil and gas context to consider:
- Flare stacks produce EXPECTED fire — do not classify as incidents
- Steam from cooling towers or vents resembles smoke — check carefully
- Gas leaks may show heat shimmer before visible ignition
- Equipment type matters: compressors, separators, storage tanks carry different risk
- Small contained fires near wellheads may be routine vs fire spreading along a pipe rack
- Time of day and lighting conditions affect visual interpretation
- If the image quality is poor or the scene is ambiguous, lower your confidence score
"""

USER_PROMPT_TEMPLATE = """Analyze this facility camera image.

Automated detection results:
{detection_summary}

Provide your structured safety advisory as JSON only."""


def encode_image_bytes(image_bytes: bytes) -> str:
    """Base64-encode raw image bytes (e.g. from cv2.imencode)."""
    return base64.b64encode(image_bytes).decode("utf-8")


def encode_image_file(image_path: str) -> str:
    """Base64-encode an image file from disk."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_detections(yolo_results) -> tuple:
    """
    Convert YOLO results to (summary_text, detections_list).

    Args:
        yolo_results: list returned by model.predict()

    Returns:
        (summary: str, detections: list[dict])
    """
    detections = []
    if yolo_results and len(yolo_results) > 0:
        result = yolo_results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": result.names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(c, 1) for c in box.xyxy[0].tolist()],
            })

    if not detections:
        return "No fire or smoke detected.", []

    summary = f"Automated YOLO detections — {len(detections)} object(s):\n"
    for i, d in enumerate(detections):
        summary += (
            f"  [{i + 1}] Class: {d['class']}, "
            f"Confidence: {d['confidence']:.1%}, "
            f"BBox (xyxy): {d['bbox']}\n"
        )
    return summary, detections


def _parse_vlm_response(text: str) -> dict:
    """Clean markdown fences and parse JSON from GPT-4o output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse VLM JSON: %s\nRaw: %s", e, text[:500])
        return {
            "severity": "UNKNOWN",
            "threat_type": "Parse error — GPT-4o returned invalid JSON",
            "is_false_alarm": False,
            "false_alarm_reason": None,
            "affected_zone": "Unknown",
            "estimated_scale": "unknown",
            "recommended_actions": ["Manual review required"],
            "escalation_level": "ALERT_OPERATOR",
            "reasoning": f"Response could not be parsed: {str(e)[:200]}",
            "confidence": 0.0,
        }


def query_gpt4o(
    image_b64: str,
    detection_summary: str,
    api_key: str,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Query GPT-4o Vision for contextual fire safety advisory.

    Args:
        image_b64: base64-encoded image string
        detection_summary: text summary from format_detections()
        api_key: OpenAI API key
        media_type: MIME type of the encoded image

    Returns:
        Parsed advisory dict
    """
    client = OpenAI(api_key=api_key)
    user_text = USER_PROMPT_TEMPLATE.format(detection_summary=detection_summary)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    return _parse_vlm_response(response.choices[0].message.content)
