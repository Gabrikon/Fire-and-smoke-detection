"""Evaluate the deployed fire/smoke pipeline (YOLO + NVIDIA DAM-3B + NIM) over a folder of images.

Instead of calling a VLM directly (the original called GPT-4o), this hits the deployed app's
/advise SSE endpoint, so it measures the real end-to-end pipeline.

Usage:
    python vlm_eval.py --images test_images/ --base-url https://<your>--fire-vlm.modal.run
    # optional: --token fire-sec-...   (to bypass the daily free limit)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import httpx


def _parse_sse_advisory(text: str) -> dict | None:
    """Pull the final advisory object out of an SSE response body."""
    advisory = None
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        try:
            msg = json.loads(block[5:].strip())
        except json.JSONDecodeError:
            continue
        if msg.get("event") == "advisory":
            advisory = msg
    return advisory


def evaluate(images_dir: str, base_url: str, token: str, output_path: str) -> None:
    url = base_url.rstrip("/") + "/advise"
    headers = {"X-Access-Token": token} if token else {}

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(images_dir, ext))
    )
    print(f"Found {len(image_paths)} test images. Endpoint: {url}\n")

    results = []
    with httpx.Client(timeout=900) as client:
        for idx, img_path in enumerate(image_paths):
            name = os.path.basename(img_path)
            print(f"[{idx+1}/{len(image_paths)}] {name}")
            with open(img_path, "rb") as f:
                files = {"file": (name, f.read(), "image/jpeg")}
            t0 = time.time()
            try:
                resp = client.post(url, headers=headers, files=files)
                elapsed = round(time.time() - t0, 2)
                if resp.status_code != 200:
                    entry = {"image": name, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
                else:
                    msg = _parse_sse_advisory(resp.text) or {}
                    adv = msg.get("advisory", {})
                    entry = {
                        "image": name,
                        "n_detections": len(msg.get("detections", [])),
                        "severity": adv.get("severity"),
                        "escalation_level": adv.get("escalation_level"),
                        "is_false_alarm": adv.get("is_false_alarm"),
                        "description": (msg.get("description") or "")[:300],
                        "response_time_s": elapsed,
                    }
                    print(f"  {elapsed}s  severity={adv.get('severity')}  dets={entry['n_detections']}")
            except Exception as exc:  # noqa: BLE001
                entry = {"image": name, "error": str(exc)}
                print(f"  ERROR: {exc}")
            results.append(entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    times = [r["response_time_s"] for r in results if "response_time_s" in r]
    if times:
        print(f"Avg response: {sum(times)/len(times):.2f}s  (min {min(times):.2f}, max {max(times):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--base-url", required=True, help="Deployed app base URL")
    parser.add_argument("--token", default=os.environ.get("FIRE_ACCESS_TOKEN", ""))
    parser.add_argument("--output", default="vlm_eval_results.json")
    args = parser.parse_args()
    evaluate(args.images, args.base_url, args.token, args.output)
