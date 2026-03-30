"""
Dataset Preparation Utilities
Usage:
    python prepare_dataset.py consolidate --output consolidated_dataset
    python prepare_dataset.py remap labels_dir --map 0:0,1:1
    python prepare_dataset.py subset input_dir 20000 --output subset_dataset
"""
import os
import glob
import random
import shutil
import argparse


def remap_labels(labels_dir, class_map):
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    changed = 0
    for txt in txt_files:
        lines = open(txt).readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_cls = int(parts[0])
                if old_cls in class_map:
                    parts[0] = str(class_map[old_cls])
                    new_lines.append(" ".join(parts))
        with open(txt, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        changed += 1
    print(f"Remapped {changed} label files in {labels_dir}")


def consolidate(sources, output_dir, split_ratios=(0.80, 0.15, 0.05)):
    random.seed(42)
    pairs = []
    for img_dir, lbl_dir in sources:
        if not os.path.isdir(img_dir):
            print(f"  WARNING: {img_dir} not found, skipping")
            continue
        for img_path in glob.glob(os.path.join(img_dir, "*.*")):
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))

    print(f"Total image-label pairs: {len(pairs)}")
    random.shuffle(pairs)

    n = len(pairs)
    t1 = int(n * split_ratios[0])
    t2 = int(n * (split_ratios[0] + split_ratios[1]))
    splits = {"train": pairs[:t1], "val": pairs[t1:t2], "test": pairs[t2:]}

    for name, sp in splits.items():
        img_out = os.path.join(output_dir, "images", name)
        lbl_out = os.path.join(output_dir, "labels", name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for i, (ip, lp) in enumerate(sp):
            ext = os.path.splitext(ip)[1]
            new = f"{name}_{i:05d}"
            shutil.copy(ip, os.path.join(img_out, new + ext))
            shutil.copy(lp, os.path.join(lbl_out, new + ".txt"))
        print(f"  {name}: {len(sp)} images")

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write("train: ./images/train\nval: ./images/val\ntest: ./images/test\n\nnc: 2\nnames: ['fire', 'smoke']\n")
    print(f"Dataset ready at: {output_dir}")


def subset(input_dir, count, output_dir):
    random.seed(42)
    img_dir = os.path.join(input_dir, "images", "train")
    lbl_dir = os.path.join(input_dir, "labels", "train")
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sampled = random.sample(all_imgs, min(count, len(all_imgs)))

    out_img = os.path.join(output_dir, "images", "train")
    out_lbl = os.path.join(output_dir, "labels", "train")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    copied = 0
    for img_name in sampled:
        base = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if os.path.exists(lbl_path):
            shutil.copy(os.path.join(img_dir, img_name), os.path.join(out_img, img_name))
            shutil.copy(lbl_path, os.path.join(out_lbl, base + ".txt"))
            copied += 1
    print(f"Subset: {copied} images → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("consolidate")
    p.add_argument("--output", default="consolidated_dataset")

    p = sub.add_parser("remap")
    p.add_argument("labels_dir")
    p.add_argument("--map", default="0:0,1:1")

    p = sub.add_parser("subset")
    p.add_argument("input_dir")
    p.add_argument("count", type=int)
    p.add_argument("--output", default="subset_dataset")

    args = parser.parse_args()

    if args.command == "consolidate":
        sources = [
            ("DFireDataset/images/train", "DFireDataset/labels/train"),
            ("FASDD/images", "FASDD/labels"),
        ]
        consolidate(sources, args.output)
    elif args.command == "remap":
        cmap = {int(k): int(v) for k, v in (p.split(":") for p in args.map.split(","))}
        remap_labels(args.labels_dir, cmap)
    elif args.command == "subset":
        subset(args.input_dir, args.count, args.output)
    else:
        parser.print_help()
