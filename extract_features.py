"""Extract frozen DINOv2 and CLIP image features for PlantVillage and PlantDoc,
and build the SET1 text anchors. Everything later runs from these files.

    python extract_features.py --pv-root data/plantvillage --pd-root data/plantdoc \
        --plantclef-ckpt weights/model_best.pth.tar --out features
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from frond.common import list_images
from frond.descriptions import SET1
from frond.encoders import (clip_image_features, dinov2_features, load_clip, load_dinov2,
                            text_anchors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-root", required=True, help="PlantVillage, YOLO format (split folders with images/ and labels/)")
    ap.add_argument("--pd-root", required=True, help="PlantDoc, YOLO format (images/ and labels/)")
    ap.add_argument("--plantclef-ckpt", required=True, help="PlantCLEF 2024 ViT-B/14 checkpoint (model_best.pth.tar)")
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dino, dino_tf = load_dinov2(args.plantclef_ckpt, device)
    clip, proc = load_clip(device)

    for name, root in [("pv", args.pv_root), ("pd", args.pd_root)]:
        paths, y = list_images(root)
        print(f"{name}: {len(paths)} images, per class {np.bincount(y).tolist()}")
        feats = {
            "dino": dinov2_features(dino, dino_tf, paths, device, args.batch_size),
            "clip": clip_image_features(clip, proc, paths, device, args.batch_size),
            "y": y,
            "paths": [str(p) for p in paths],
        }
        torch.save(feats, out / f"{name}.pt")

    torch.save(text_anchors(clip, proc, SET1, device), out / "anchors_set1.pt")
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
