"""
GPT-4o Fire Detection Evaluation
Usage:
    export OPENAI_API_KEY="sk-..."
    python vlm_eval.py --images test_images/ --model weights/fire_yolo_best.pt
"""
import os
import sys
import json
import time
import glob
import argparse
from ultralytics import YOLO
from vlm_reasoner import format_detections, encode_image_file, query_gpt4o


def evaluate(images_dir, model_path, output_path):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    model = YOLO(model_path)
    print(f"Model: {model_path}  classes={model.names}")

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(images_dir, ext))
    )
    print(f"Found {len(image_paths)} test images\n")

    results = []
    for idx, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_paths)}] {name}")

        yolo_results = model.predict(img_path, conf=0.3, verbose=False)
        summary, dets = format_detections(yolo_results)
        print(f"  YOLO: {len(dets)} detection(s)")

        b64 = encode_image_file(img_path)
        entry = {"image": name, "yolo_detections": len(dets),
                 "classes": [d["class"] for d in dets]}

        try:
            t0 = time.time()
            adv = query_gpt4o(b64, summary, api_key)
            elapsed = round(time.time() - t0, 2)
            adv["_response_time_s"] = elapsed
            entry["advisory"] = adv
            print(f"  GPT-4o: {elapsed}s  severity={adv.get('severity')}")
        except Exception as e:
            entry["advisory"] = {"error": str(e)}
            print(f"  GPT-4o ERROR: {e}")

        results.append(entry)
        time.sleep(1.0)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary
    times = [r["advisory"]["_response_time_s"] for r in results
             if "error" not in r.get("advisory", {})]
    if times:
        print(f"\nAvg response: {sum(times)/len(times):.2f}s")
        print(f"Min/Max: {min(times):.2f}s / {max(times):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default="weights/fire_yolo_best.pt")
    parser.add_argument("--output", default="vlm_eval_results.json")
    args = parser.parse_args()
    evaluate(args.images, args.model, args.output)
