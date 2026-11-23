"""FROND on PlantVillage -> PlantDoc.

PlantVillage is split per class into 70% / 10% / 20%. The visual prototypes are
built from the 70% split, alpha is chosen on the 10% split, and the 20% split is
held out and not used. PlantDoc is only used for the final evaluation. This is
repeated over ten random partitions.

    python run_frond.py --features features --out results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frond.common import (TAU_T, TAU_V, channel_probs, class_means, fuse, macro_f1,
                          select_alpha, stratified_split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    feat = Path(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = torch.load(feat / "pv.pt", weights_only=False)
    pd = torch.load(feat / "pd.pt", weights_only=False)
    anchors = torch.load(feat / "anchors_set1.pt")
    y_pd = pd["y"]

    # the language channel does not depend on the source split
    pt_pd = channel_probs(pd["clip"], anchors, TAU_T)

    runs = []
    for seed in range(args.runs):
        train, val, _ = stratified_split(pv["y"], (0.7, 0.1, 0.2), seed)
        protos = class_means(pv["dino"][train], pv["y"][train])

        # choose alpha on the source validation split only
        pv_val = channel_probs(pv["dino"][val], protos, TAU_V)
        pt_val = channel_probs(pv["clip"][val], anchors, TAU_T)
        alpha, _ = select_alpha(pv_val, pt_val, pv["y"][val])

        pv_pd = channel_probs(pd["dino"], protos, TAU_V)
        fused = fuse(pv_pd, pt_pd, alpha)
        runs.append({
            "seed": seed,
            "alpha": alpha,
            "visual": macro_f1(y_pd, pv_pd.argmax(1)),
            "language": macro_f1(y_pd, pt_pd.argmax(1)),
            "frond": macro_f1(y_pd, fused.argmax(1)),
        })
        print("seed {seed}: alpha*={alpha:.2f}  visual {visual:.1f}  language {language:.1f}  "
              "FROND {frond:.1f}".format(**runs[-1]))

        if seed == 0:
            # predictions and probabilities of the first partition, for bootstrap.py and figures.py
            np.savez(out / "frond_preds.npz", y_true=y_pd, **{
                "Visual prototype": pv_pd.argmax(1),
                "Language anchors": pt_pd.argmax(1),
                "FROND": fused.argmax(1)})
            np.savez(out / "frond_probs.npz", y_true=y_pd, p_vis=pv_pd, p_txt=pt_pd, alpha=alpha,
                     paths=np.array(pd["paths"]))

    for key in ["visual", "language", "frond"]:
        v = [r[key] for r in runs]
        print(f"{key:9s} {np.mean(v):.1f} +/- {np.std(v, ddof=1):.1f}")
    print("alpha* per run:", [r["alpha"] for r in runs])
    (out / "frond_runs.json").write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
