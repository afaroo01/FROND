"""Constants, dataset listing, metrics and the fusion rule used throughout."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CLASSES = [
    "Apple healthy", "Apple rust", "Bell pepper bacterial spot", "Bell pepper healthy",
    "Grape black rot", "Grape healthy", "Tomato healthy", "Tomato septoria leaf spot",
]
NUM_CLASSES = len(CLASSES)

# YOLO class id in the PlantVillage / PlantDoc label files -> FROND class index
LABEL_MAP = {0: 0, 1: 1, 3: 2, 4: 3, 10: 4, 11: 5, 22: 6, 27: 7}

TAU_V = 0.5    # visual channel temperature
TAU_T = 0.02   # language channel temperature

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")


def list_images(root):
    """List images in a YOLO-format dataset that belong to the eight shared classes.

    `root` holds either images/ and labels/ directly (PlantDoc) or one
    sub-folder per split, each with images/ and labels/ (PlantVillage).
    An image with boxes from several of the eight classes takes the lowest
    class index. Images are grouped by class, then sorted by file name, so the
    order is stable across runs.
    """
    root = Path(root)
    label_dirs = [d for d in [root / "labels", *sorted(root.glob("*/labels"))] if d.is_dir()]
    by_class = defaultdict(list)
    for ldir in label_dirs:
        idir = ldir.parent / "images"
        for lp in sorted(ldir.glob("*.txt")):
            classes = set()
            for line in lp.read_text().splitlines():
                q = line.split()
                if len(q) == 5 and int(float(q[0])) in LABEL_MAP:
                    classes.add(LABEL_MAP[int(float(q[0]))])
            if not classes:
                continue
            for ext in IMG_EXTS:
                ip = idir / (lp.stem + ext)
                if ip.exists():
                    by_class[min(classes)].append(ip)
                    break
    paths = [p for c in range(NUM_CLASSES) for p in by_class[c]]
    labels = np.array([c for c in range(NUM_CLASSES) for _ in by_class[c]])
    return paths, labels


def stratified_split(y, fracs=(0.7, 0.1, 0.2), seed=0):
    """Per-class random split into disjoint index sets with the given fractions."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fracs]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        cuts = np.round(np.cumsum(fracs)[:-1] * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, cuts)):
            part.extend(chunk.tolist())
    return [np.array(sorted(p)) for p in parts]


def balanced_subset(idx, y, per_class, seed=0):
    """At most `per_class` indices per class, drawn from `idx`."""
    rng = np.random.default_rng(seed)
    keep = [rng.permutation(idx[y[idx] == c])[:per_class] for c in np.unique(y[idx])]
    return np.sort(np.concatenate(keep))


def macro_f1(y, pred, num_classes=NUM_CLASSES):
    """Macro-F1 in %, every class counted even if never predicted."""
    f1 = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        f1.append(0.0 if denom == 0 else 2 * tp / denom)
    return 100 * float(np.mean(f1))


def class_means(feats, y, num_classes=NUM_CLASSES):
    """l2-normalised mean of l2-normalised features per class (Eqs. 1 and 3)."""
    feats = F.normalize(torch.as_tensor(feats).float(), dim=1)
    y = torch.as_tensor(y)
    means = torch.stack([feats[y == c].mean(0) for c in range(num_classes)])
    return F.normalize(means, dim=1)


def channel_probs(feats, refs, tau):
    """Temperature-scaled softmax over cosine similarities (Eqs. 2 and 4)."""
    feats = F.normalize(torch.as_tensor(feats).float(), dim=1)
    return F.softmax(feats @ refs.T / tau, dim=1).numpy()


def fuse(p_vis, p_txt, alpha):
    """F(alpha) = alpha * p_v + (1 - alpha) * p_t (Eq. 5). alpha may be per image."""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    return alpha * p_vis + (1 - alpha) * p_txt


def select_alpha(p_vis, p_txt, y, step=0.01):
    """Grid sweep of alpha in [0, 1] maximising source macro-F1 (Eq. 6).

    With ties, the first (smallest) alpha on the grid is kept.
    """
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    scores = [macro_f1(y, fuse(p_vis, p_txt, a).argmax(1)) for a in grid]
    return float(grid[int(np.argmax(scores))]), scores
