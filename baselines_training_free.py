"""Training-free baselines (Table 7): CLIP prompting variants, MVPDR and ProKeR.

All use the same frozen CLIP ViT-L/14 as FROND's language channel. MVPDR and
ProKeR also use labelled source images: a class-balanced sample of PV CLIP
features from the 70% training split of the first partition. PlantDoc labels
are never used to choose a setting.

    python baselines_training_free.py --features features --out results
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from frond.common import (NUM_CLASSES, TAU_T, balanced_subset, macro_f1, stratified_split)
from frond.descriptions import PROMPT_NAMES, SET1
from frond.encoders import clip_text_features, load_clip


def zero_shot(x, anchors):
    return (x @ anchors.T.numpy()).argmax(1)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------- CLIP prompting families ----------------

TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a photograph of a {}.", "an image of a {}.",
             "a cropped photo of a {}.", "a picture of a {}.", "a leaf image of a {}."]

# DCLIP-style: "a photo of a {class}, which has {attribute}"
DCLIP_ATTRIBUTES = {
    0: ["smooth glossy green surface", "no spots or lesions", "even green pigmentation"],
    1: ["bright orange-yellow pustules", "circular rust spots", "raised fungal lesions"],
    2: ["small dark water-soaked spots", "angular brown lesions", "yellow halos around spots"],
    3: ["deep green waxy surface", "broad glossy leaf", "no blemishes"],
    4: ["tan-brown circular lesions", "dark purple-black margins", "tiny black specks"],
    5: ["bright green lobed leaf", "serrated margins", "clear venation"],
    6: ["medium green serrated leaflets", "slightly fuzzy texture", "no spots"],
    7: ["small grey-white circular spots", "dark brown margins", "many tiny lesions"],
}

# WaffleCLIP-style: random descriptor words instead of real attributes
WAFFLE_WORDS = ["spotted", "glossy", "serrated", "lobed", "fuzzy", "veined", "pale", "dark", "circular", "angular",
                "raised", "smooth", "mottled", "ringed", "necrotic", "waxy", "broad", "narrow", "textured", "speckled"]


def prompt_anchors(clip, proc, device):
    def anchor(texts):
        return clip_text_features(clip, proc, texts, device).mean(0)

    def stack(rows):
        return F.normalize(torch.stack(rows), dim=1)

    rng = random.Random(0)
    waffle = []
    for name in PROMPT_NAMES:
        texts = [f"a photo of a {name}, which is {rng.choice(WAFFLE_WORDS)} and {rng.choice(WAFFLE_WORDS)}."
                 for _ in range(8)]
        waffle.append(anchor(texts))
    return {
        "Vanilla CLIP": stack([anchor([f"a photo of a {n}."]) for n in PROMPT_NAMES]),
        "CLIP template ensemble": stack([anchor([t.format(n) for t in TEMPLATES]) for n in PROMPT_NAMES]),
        "DCLIP-style": stack([anchor([f"a photo of a {PROMPT_NAMES[c]}, which has {a}." for a in DCLIP_ATTRIBUTES[c]])
                              for c in range(NUM_CLASSES)]),
        "WaffleCLIP-style": stack(waffle),
    }


# ---------------- MVPDR (Wei et al., ACM MM 2024), training-free form ----------------

def mvpdr(xs, ys, xt, desc_emb, k=4, tau=0.01, weights=(0.3, 0.5, 0.5)):
    """K-means visual prototypes per class plus per-description text prototypes.

    Visual score sums cosine similarities to a class's centroids (Eq. 1); text
    scores take the best and the mean description (Eqs. 2 and 3); the three are
    combined with the paper's weights (Eq. 8). tau and k are not given in the
    paper: tau is CLIP's logit scale and k sits in the middle of its ablation.
    """
    centroids = []
    for c in range(NUM_CLASSES):
        km = KMeans(n_clusters=min(k, int((ys == c).sum())), n_init=10, random_state=0).fit(xs[ys == c])
        cen = km.cluster_centers_
        centroids.append(cen / np.linalg.norm(cen, axis=1, keepdims=True))
    s_v = np.stack([(xt @ centroids[c].T).sum(1) for c in range(NUM_CLASSES)], 1)
    s_max = np.stack([(xt @ desc_emb[c].T).max(1) for c in range(NUM_CLASSES)], 1)
    s_avg = np.stack([(xt @ desc_emb[c].T).mean(1) for c in range(NUM_CLASSES)], 1)
    a, b, g = weights
    return (a * softmax(s_v / tau) + b * softmax(s_max / tau) + g * softmax(s_avg / tau)).argmax(1)


# ---------------- ProKeR (Bendou et al., CVPR 2025) ----------------

def rbf(x, y, beta):
    return np.exp(-0.5 * beta * np.clip(2 - 2 * x @ y.T, 0, None))   # unit vectors: ||x-y||^2 = 2 - 2<x,y>


def proker_fit(xs, ys, w, lam, beta):
    """Closed-form kernel ridge correction to the zero-shot classifier w (Eqs. 13 and 14)."""
    target = np.eye(NUM_CLASSES)[ys] - xs @ w
    gamma = np.linalg.solve(np.eye(len(xs)) + rbf(xs, xs, beta) / lam, target)
    return lambda x: x @ w + rbf(x, xs, beta) @ gamma


def proker(xs, ys, xt, w, lambdas=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100), betas=(1, 2, 5, 10, 20)):
    """lambda and beta are not reported in the paper; they are picked on a 30% PV
    hold-out, then the model is refitted on all source features."""
    fit_idx, val_idx = stratified_split(ys, (0.7, 0.3), seed=0)
    best = None
    for lam in lambdas:
        for beta in betas:
            z = proker_fit(xs[fit_idx], ys[fit_idx], w, lam, beta)(xs[val_idx])
            f1 = macro_f1(ys[val_idx], z.argmax(1))
            nll = -np.log(np.clip(softmax(z)[np.arange(len(val_idx)), ys[val_idx]], 1e-12, None)).mean()
            key = (f1, -nll, lam)   # best F1, then lower NLL, then larger lambda
            if best is None or key > best[0]:
                best = (key, lam, beta)
    _, lam, beta = best
    print(f"  ProKeR selected on PV: lambda={lam}, beta={beta}")
    return proker_fit(xs, ys, w, lam, beta)(xt).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-class", type=int, default=199, help="labelled PV images per class for MVPDR and ProKeR")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = torch.load(feat / "pv.pt", weights_only=False)
    pd = torch.load(feat / "pd.pt", weights_only=False)
    anchors = torch.load(feat / "anchors_set1.pt")
    clip, proc = load_clip(device)

    xt = F.normalize(pd["clip"].float(), dim=1).numpy()
    y_pd = pd["y"]
    train, _, _ = stratified_split(pv["y"], (0.7, 0.1, 0.2), seed=0)
    src = balanced_subset(train, pv["y"], args.per_class)
    xs = F.normalize(pv["clip"][src].float(), dim=1).numpy()
    ys = pv["y"][src]

    preds = {name: zero_shot(xt, a) for name, a in prompt_anchors(clip, proc, device).items()}
    preds["Language anchors"] = zero_shot(xt, anchors)   # sanity check: FROND's language channel

    desc_emb = [clip_text_features(clip, proc, SET1[c], device).numpy() for c in range(NUM_CLASSES)]
    preds["MVPDR"] = mvpdr(xs, ys, xt, desc_emb)
    preds["ProKeR"] = proker(xs, ys, xt, anchors.numpy().T)

    for name, p in preds.items():
        print(f"{name:24s} {macro_f1(y_pd, p):.1f}")
    preds.pop("Language anchors")
    np.savez(out / "training_free_preds.npz", y_true=y_pd, **preds)


if __name__ == "__main__":
    main()
