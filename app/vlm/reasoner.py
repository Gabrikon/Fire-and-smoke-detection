"""Safety-advisory reasoner: NVIDIA NIM (OpenAI-compatible cloud API).

Stage 2 of the two-stage VLM. DAM-3B already "saw" the scene and produced a precise localized
description of the fire/smoke region; this stage is a strong instruction-following LLM that turns
that description plus the YOLO detection summary into a strict, schema-locked safety advisory for
oil & gas facilities. Running this on NVIDIA's cloud (your nvapi key) keeps it off a second GPU.

The advisory schema and the oil & gas domain guidance are ported from the original GPT-4o prompt
(legacy/vlm_reasoner_openai.py), adapted so the model reasons over text (the DAM description)
rather than an image.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert fire safety analyst for oil and gas facilities.

You receive: (1) automated fire/smoke detection results from a YOLO computer-vision system, and
(2) a detailed localized description of the detected region produced by a specialized vision model
(NVIDIA DAM-3B). Use both to produce a structured safety advisory.

ALWAYS respond with ONLY valid JSON (no markdown fences, no preamble, no reasoning text) matching
this exact schema:

{
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "threat_type": "string - what is detected (flare stack fire, process fire, smoke plume, electrical fire, etc.)",
  "is_false_alarm": false,
  "false_alarm_reason": null,
  "affected_zone": "string - description of the area/equipment involved",
  "estimated_scale": "small | medium | large | catastrophic",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "escalation_level": "MONITOR | ALERT_OPERATOR | EVACUATE_ZONE | FULL_EMERGENCY",
  "reasoning": "string - 2-3 sentence explanation of your analysis",
  "confidence": 0.85
}

Oil and gas context to consider:
- Flare stacks produce EXPECTED fire - do not classify as incidents.
- Steam from cooling towers or vents resembles smoke - check the description carefully.
- Gas leaks may show heat shimmer before visible ignition.
- Equipment type matters: compressors, separators, storage tanks carry different risk.
- Small contained fires near wellheads may be routine vs fire spreading along a pipe rack.
- If the localized description is ambiguous or suggests steam/vapor or a controlled flare, lower
  confidence and consider is_false_alarm.

Do not use em-dashes or en-dashes in any field. Use commas, colons, or hyphens.
"""

USER_PROMPT_TEMPLATE = """Automated YOLO detection results:
{detection_summary}

Localized region description from the vision model (NVIDIA DAM-3B):
{region_description}

{extra}Provide your structured safety advisory as JSON only."""


def _strip_dashes(text: str) -> str:
    return text.replace("—", "-").replace("–", "-")


def _parse_advisory(text: str) -> dict:
    """Extract the advisory JSON from the model output.

    Tolerates: <think>...</think> reasoning traces, ```json fences, and leading prose. Falls back
    to a safe ALERT_OPERATOR advisory if nothing parses.
    """
    text = (text or "").strip()
    # Drop any chain-of-thought trace some reasoning models emit.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # If there is surrounding prose, grab the outermost JSON object.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse advisory JSON: %s\nRaw: %s", e, text[:500])
        return {
            "severity": "UNKNOWN",
            "threat_type": "Parse error - reasoner returned invalid JSON",
            "is_false_alarm": False,
            "false_alarm_reason": None,
            "affected_zone": "Unknown",
            "estimated_scale": "unknown",
            "recommended_actions": ["Manual review required"],
            "escalation_level": "ALERT_OPERATOR",
            "reasoning": f"Response could not be parsed: {str(e)[:200]}",
            "confidence": 0.0,
        }


class Reasoner:
    """Async NVIDIA NIM client that produces the structured advisory."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 0.2, max_tokens: int = 1024):
        from openai import AsyncOpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def advise(self, detection_summary: str, region_description: str,
                     extra_context: str = "") -> tuple[dict, str]:
        """Return (parsed_advisory, raw_text)."""
        extra = f"{extra_context.strip()}\n\n" if extra_context.strip() else ""
        user = USER_PROMPT_TEMPLATE.format(
            detection_summary=detection_summary,
            region_description=region_description or "(no description available)",
            extra=extra,
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = _strip_dashes(resp.choices[0].message.content or "")
        return _parse_advisory(raw), raw


def default_reasoner() -> "Reasoner":
    """Build a Reasoner from environment settings."""
    from app.config import settings

    return Reasoner(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        model=settings.reasoner_model,
        temperature=settings.reasoner_temperature,
        max_tokens=settings.reasoner_max_tokens,
    )
