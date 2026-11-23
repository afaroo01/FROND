"""Qualitative figures: per-class macro-F1 radar and fusion rescue examples.

Uses the probabilities saved by run_frond.py (first partition).

    python figures.py --results results --out figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from frond.common import CLASSES, NUM_CLASSES, fuse

plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "savefig.dpi": 300, "savefig.bbox": "tight"})
COLOURS = {"visual": "#1f6fb4", "language": "#e07b00", "fused": "#2a9d4b"}


def per_class_f1(y, pred):
    f1 = []
    for c in range(NUM_CLASSES):
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        f1.append(0.0 if denom == 0 else 200 * tp / denom)
    return np.array(f1)


def radar(y, preds, path):
    angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False)
    closed = np.append(angles, angles[0])
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for name, pred in preds.items():
        f = per_class_f1(y, pred)
        ax.plot(closed, np.append(f, f[0]), lw=3, color=COLOURS[name], label=name)
        ax.fill(closed, np.append(f, f[0]), color=COLOURS[name], alpha=0.08)
    ax.set_xticks(angles)
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in CLASSES], fontsize=18)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.tick_params(axis="x", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=18)
    fig.savefig(path)
    plt.close(fig)


def rescue_grid(paths, y, pv, pt, fused, loser, path, k=4):
    """Images the fused prediction got right while `loser` ("visual" or "language")
    got them wrong, choosing cases where the losing channel was confident."""
    lose_p, win_p = (pv, pt) if loser == "visual" else (pt, pv)
    ok = (lose_p.argmax(1) != y) & (win_p.argmax(1) == y) & (fused.argmax(1) == y)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-lose_p[idx].max(1))][:k]
    if len(idx) == 0:
        return
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 5.2))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.axis("off")
        ax.set_title(CLASSES[y[i]], fontsize=15, fontweight="bold")
        ax.text(0.5, -0.06, f"{loser}: {CLASSES[lose_p[i].argmax()]} ({lose_p[i].max():.2f})\n"
                            f"fused: {CLASSES[fused[i].argmax()]}",
                transform=ax.transAxes, ha="center", va="top", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(args.results) / "frond_probs.npz")
    y, pv, pt, alpha, paths = z["y_true"], z["p_vis"], z["p_txt"], float(z["alpha"]), z["paths"]
    fused = fuse(pv, pt, alpha)

    vis_ok, txt_ok, fus_ok = pv.argmax(1) == y, pt.argmax(1) == y, fused.argmax(1) == y
    print(f"both channels right {np.sum(vis_ok & txt_ok)} | visual only {np.sum(vis_ok & ~txt_ok)} | "
          f"language only {np.sum(~vis_ok & txt_ok)} | neither {np.sum(~vis_ok & ~txt_ok)}")
    print(f"fusion corrected {np.sum(~vis_ok & fus_ok)} of the {np.sum(~vis_ok)} visual errors "
          f"and broke {np.sum(vis_ok & ~fus_ok)} correct visual predictions")

    radar(y, {"visual": pv.argmax(1), "language": pt.argmax(1), "fused": fused.argmax(1)},
          out / "perclass_radar.png")
    rescue_grid(paths, y, pv, pt, fused, "visual", out / "rescue_visual_errors.png")
    rescue_grid(paths, y, pv, pt, fused, "language", out / "rescue_language_errors.png")
    print("figures saved to", out.resolve())


if __name__ == "__main__":
    main()
