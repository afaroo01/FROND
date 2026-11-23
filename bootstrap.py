"""Paired bootstrap over the PlantDoc test images (Tables 6 and 7).

Every method is scored on the same resampled image sets, so differences from
FROND are paired. Reports macro-F1, bootstrap std, 95% CI, the difference from
FROND with its 95% CI, and a two-sided p-value with Holm correction.

    python bootstrap.py results/frond_preds.npz results/training_free_preds.npz ...
"""
import argparse

import numpy as np

from frond.common import NUM_CLASSES


def macro_f1_resamples(y, pred, idx, num_classes=NUM_CLASSES):
    """Macro-F1 (%) for each row of resampled indices `idx` (b, n)."""
    t, p = y[idx], pred[idx]
    b = len(idx)
    flat = (np.arange(b)[:, None] * num_classes * num_classes + t * num_classes + p).ravel()
    cm = np.bincount(flat, minlength=b * num_classes ** 2).reshape(b, num_classes, num_classes)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(2) + cm.sum(1)                  # 2tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return 100 * f1.mean(1)


def paired_bootstrap(y, preds, ref="FROND", n_boot=10_000, seed=0, chunk=1_000):
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {m: [] for m in preds}
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, (min(chunk, n_boot - s), n))   # shared by all methods
        for m, p in preds.items():
            boot[m].append(macro_f1_resamples(y, p, idx))
    boot = {m: np.concatenate(v) for m, v in boot.items()}
    full = np.arange(n)[None]

    rows = []
    for m, p in preds.items():
        r = {"method": m, "f1": macro_f1_resamples(y, p, full)[0], "std": boot[m].std(ddof=1),
             "ci": np.percentile(boot[m], [2.5, 97.5])}
        if m != ref:
            d = boot[ref] - boot[m]
            r["delta"] = macro_f1_resamples(y, preds[ref], full)[0] - r["f1"]
            r["delta_ci"] = np.percentile(d, [2.5, 97.5])
            r["p"] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            r["wins"] = (d > 0).mean()
        rows.append(r)

    # Holm step-down correction over the comparisons with FROND
    comp = sorted([r for r in rows if "p" in r], key=lambda r: r["p"])
    running = 0.0
    for j, r in enumerate(comp):
        running = max(running, min(1.0, (len(comp) - j) * r["p"]))
        r["p_holm"] = running
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".npz files with y_true and one prediction array per method")
    ap.add_argument("--ref", default="FROND")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    y, preds = None, {}
    for f in args.files:
        z = np.load(f)
        if y is None:
            y = z["y_true"]
        elif not np.array_equal(y, z["y_true"]):
            raise ValueError(f"{f}: PlantDoc image order differs from {args.files[0]}")
        for k in z.files:
            if k != "y_true":
                preds.setdefault(k, z[k])
    if args.ref not in preds:
        raise KeyError(f"'{args.ref}' not found; methods are {list(preds)}")

    print(f"{'method':22s} {'macro-F1':>14s} {'95% CI':>15s} {'diff vs ' + args.ref:>24s} {'p (Holm)':>9s} {'wins':>6s}")
    for r in paired_bootstrap(y, preds, args.ref, args.n_boot):
        line = f"{r['method']:22s} {r['f1']:6.1f} +/- {r['std']:3.1f}  [{r['ci'][0]:4.1f}, {r['ci'][1]:4.1f}]"
        if "p" in r:
            mark = "*" if r["p_holm"] < args.alpha else " "
            line += (f"  {r['delta']:+5.1f} [{r['delta_ci'][0]:+5.1f}, {r['delta_ci'][1]:+5.1f}]{mark}"
                     f"  {r['p_holm']:7.4f}  {100 * r['wins']:5.1f}%")
        print(line)


if __name__ == "__main__":
    main()
