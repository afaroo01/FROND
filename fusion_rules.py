"""Alternative ways of setting the fusion weight (Table 6).

Every rule sees the same two channels and the same source splits as FROND; only
the way alpha is obtained changes. Source statistics come from the 10% PV
validation split, never from PlantDoc.

    python fusion_rules.py --features features --out results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from frond.common import (NUM_CLASSES, TAU_T, TAU_V, channel_probs, class_means, fuse,
                          macro_f1, select_alpha, stratified_split)


def norm_entropy(p):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(1) / np.log(p.shape[1])


def true_class_var(p, y):
    return float(p[np.arange(len(y)), y].var())


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def gate_features(pv, pt):
    return np.stack([1 - norm_entropy(pv), 1 - norm_entropy(pt), pv.max(1), pt.max(1)], 1)


def apply_rules(pv_pd, pt_pd, pv_val, pt_val, y_val):
    """Predicted PlantDoc labels for every rule."""
    preds = {"Visual prototype": pv_pd.argmax(1), "Language anchors": pt_pd.argmax(1)}

    # inverse-variance weight from true-class score variances (Eq. 8 with the covariance dropped)
    iv, it = 1 / (true_class_var(pv_val, y_val) + 1e-9), 1 / (true_class_var(pt_val, y_val) + 1e-9)
    preds["Minimum-variance"] = fuse(pv_pd, pt_pd, iv / (iv + it)).argmax(1)

    # per-image gates from each channel's own confidence on the target image (Eq. 7)
    cv, ct = 1 - norm_entropy(pv_pd), 1 - norm_entropy(pt_pd)
    preds["Entropy based"] = fuse(pv_pd, pt_pd, cv / (cv + ct + 1e-9)).argmax(1)
    mv, mt = pv_pd.max(1), pt_pd.max(1)
    preds["Confidence based"] = fuse(pv_pd, pt_pd, mv / (mv + mt + 1e-9)).argmax(1)

    # CAWPE: weight from source accuracy, sharpened with k = 4
    av, at = accuracy(pv_val, y_val) ** 4, accuracy(pt_val, y_val) ** 4
    preds["CAWPE (k=4)"] = fuse(pv_pd, pt_pd, av / (av + at)).argmax(1)

    # trained scalar gate: logistic regression on images where exactly one channel is right
    vis_right = pv_val.argmax(1) == y_val
    mask = vis_right ^ (pt_val.argmax(1) == y_val)
    if mask.sum() >= 10 and len(np.unique(vis_right[mask])) > 1:
        clf = LogisticRegression(max_iter=500).fit(gate_features(pv_val, pt_val)[mask], vis_right[mask])
        preds["Trained scalar"] = fuse(pv_pd, pt_pd, clf.predict_proba(gate_features(pv_pd, pt_pd))[:, 1]).argmax(1)

    # trained per-class gate: one alpha per class, picked by the visual prediction
    a_class = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        m = y_val == c
        if m.sum() > 3:
            a_class[c], _ = select_alpha(pv_val[m], pt_val[m], y_val[m])
    preds["Trained per-class"] = fuse(pv_pd, pt_pd, a_class[pv_pd.argmax(1)]).argmax(1)

    alpha, _ = select_alpha(pv_val, pt_val, y_val)
    preds["FROND"] = fuse(pv_pd, pt_pd, alpha).argmax(1)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = torch.load(feat / "pv.pt", weights_only=False)
    pd = torch.load(feat / "pd.pt", weights_only=False)
    anchors = torch.load(feat / "anchors_set1.pt")
    pt_pd = channel_probs(pd["clip"], anchors, TAU_T)

    scores = {}
    for seed in range(args.runs):
        train, val, _ = stratified_split(pv["y"], (0.7, 0.1, 0.2), seed)
        protos = class_means(pv["dino"][train], pv["y"][train])
        preds = apply_rules(channel_probs(pd["dino"], protos, TAU_V), pt_pd,
                            channel_probs(pv["dino"][val], protos, TAU_V),
                            channel_probs(pv["clip"][val], anchors, TAU_T), pv["y"][val])
        for k, p in preds.items():
            scores.setdefault(k, []).append(macro_f1(pd["y"], p))
        if seed == 0:
            np.savez(out / "fusion_rules_preds.npz", y_true=pd["y"], **preds)

    for k, v in scores.items():
        print(f"{k:20s} {np.mean(v):5.1f} +/- {np.std(v, ddof=1):.1f}")
    (out / "fusion_rules.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
