"""Source-trained baselines (Table 7): CoOp and an AdvST-style trained head.

Both are trained on labelled PV data only and evaluated once on PlantDoc.
Five seeds are run; the median seed is exported for the bootstrap.

    python baselines_source_trained.py --features features --out results
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from frond.common import NUM_CLASSES, TAU_T, balanced_subset, macro_f1, stratified_split
from frond.encoders import load_clip

COOP_NAMES = ["apple healthy", "apple rust", "bell pepper bacterial spot", "bell pepper healthy",
              "grape black rot", "grape healthy", "tomato healthy", "tomato septoria"]


class CoOpPrompt(nn.Module):
    """CoOp (Zhou et al., IJCV 2022): learnable context tokens in front of each
    class name, passed through the frozen CLIP text encoder."""

    def __init__(self, clip, tokenizer, names, n_ctx=4):
        super().__init__()
        self.clip = clip
        tm = clip.text_model
        ids = tokenizer([("X " * n_ctx) + n + "." for n in names], return_tensors="pt", padding=True)["input_ids"]
        ids = ids.to(clip.device)
        with torch.no_grad():
            self.register_buffer("base", tm.embeddings.token_embedding(ids))
            self.register_buffer("pos", tm.embeddings.position_embedding(torch.arange(ids.shape[1], device=ids.device))[None])
        L = ids.shape[1]
        self.register_buffer("causal", torch.full((L, L), float("-inf"), device=ids.device).triu(1)[None, None])
        self.eos = ids.argmax(-1)                     # end-of-text has the largest token id
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(0.02 * torch.randn(n_ctx, self.base.shape[-1], device=ids.device))

    def forward(self):
        tm = self.clip.text_model
        x = self.base.clone()
        x[:, 1:1 + self.n_ctx] = self.ctx            # position 0 is the start token
        x = x + self.pos
        try:
            h = tm.encoder(inputs_embeds=x, causal_attention_mask=self.causal)
        except TypeError:                             # argument name differs across transformers versions
            h = tm.encoder(inputs_embeds=x, attention_mask=self.causal)
        h = tm.final_layer_norm(h[0])
        pooled = h[torch.arange(len(h)), self.eos]
        return F.normalize(self.clip.text_projection(pooled), dim=1)


def train_coop(clip, tokenizer, xs, ys, seed, epochs=200, lr=2e-3):
    torch.manual_seed(seed)
    prompt = CoOpPrompt(clip, tokenizer, COOP_NAMES)
    opt = torch.optim.Adam([prompt.ctx], lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(xs @ prompt().T / TAU_T, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return prompt()


def train_head(xs, ys, seed, augment, epochs=300):
    """Linear head on frozen DINOv2 features. With `augment`, each step perturbs
    feature scale and adds noise, a feature-space stand-in for AdvST's style
    augmentation."""
    torch.manual_seed(seed)
    head = nn.Linear(xs.shape[1], NUM_CLASSES).to(xs.device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        x = xs
        if augment:
            scale = 1 + 0.3 * torch.randn(len(x), 1, device=x.device)
            x = F.normalize(x * scale + 0.05 * torch.randn_like(x), dim=1)
        loss = F.cross_entropy(head(x), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-class", type=int, default=199, help="labelled PV images per class for CoOp")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = torch.load(feat / "pv.pt", weights_only=False)
    pd = torch.load(feat / "pd.pt", weights_only=False)
    y_pd = pd["y"]
    train, _, test = stratified_split(pv["y"], (0.7, 0.1, 0.2), seed=0)

    def norm(x):
        return F.normalize(x.float(), dim=1).to(device)

    clip, proc = load_clip(device)
    coop_idx = balanced_subset(train, pv["y"], args.per_class)
    xs_clip, ys_clip = norm(pv["clip"][coop_idx]), torch.as_tensor(pv["y"][coop_idx], device=device)
    xs_dino, ys_dino = norm(pv["dino"][train]), torch.as_tensor(pv["y"][train], device=device)
    xt_clip, xt_dino = norm(pd["clip"]), norm(pd["dino"])

    preds = {"CoOp": [], "AdvST-style head": []}
    for s in args.seeds:
        text = train_coop(clip, proc.tokenizer, xs_clip, ys_clip, s)
        preds["CoOp"].append((xt_clip @ text.T).argmax(1).cpu().numpy())
        head = train_head(xs_dino, ys_dino, s, augment=True)
        with torch.no_grad():
            preds["AdvST-style head"].append(head(xt_dino).argmax(1).cpu().numpy())
        print(f"seed {s}: CoOp {macro_f1(y_pd, preds['CoOp'][-1]):.1f} | "
              f"AdvST-style {macro_f1(y_pd, preds['AdvST-style head'][-1]):.1f}")

    median = {}
    for k, runs in preds.items():
        f = [macro_f1(y_pd, p) for p in runs]
        median[k] = runs[int(np.argsort(f)[len(f) // 2])]
        print(f"{k:18s} {np.mean(f):.1f} +/- {np.std(f, ddof=1):.1f}")
    np.savez(out / "source_trained_preds.npz", y_true=y_pd, **median)

    # in-domain vs field accuracy of a plain trained head, with and without augmentation
    x_test, y_test = norm(pv["dino"][test]), pv["y"][test]
    for aug in (False, True):
        head = train_head(xs_dino, ys_dino, 0, augment=aug)
        with torch.no_grad():
            in_domain = macro_f1(y_test, head(x_test).argmax(1).cpu().numpy())
            field = macro_f1(y_pd, head(xt_dino).argmax(1).cpu().numpy())
        print(f"trained head{' + aug' if aug else ''}: PV held-out {in_domain:.1f}, PlantDoc {field:.1f}")


if __name__ == "__main__":
    main()
