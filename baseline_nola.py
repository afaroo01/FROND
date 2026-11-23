"""NoLA (Imam et al., BMVC 2025), source-only, for Table 7.

NoLA adapts CLIP with unlabelled images. In the single-source setting it only
sees unlabelled PV images; PV labels are used for diagnostics, never for training.

  1. CLIP pseudo-labels PV images with the SET1 description classifier and keeps
     the most confident 20% per class (at least 16).
  2. A linear alignment module on frozen DINOv2 features is trained on those
     pseudo-labels and becomes the teacher.
  3. Learnable visual prompts are prepended to the frozen CLIP vision encoder.
     Prompts and the description classifier are trained so that CLIP's
     prediction on a strongly augmented view matches the teacher on the original.

Settings follow the paper where it gives them (top-k, stage 2 optimiser, 16
prompts, strong augmentation, stage 3 lr 2e-3 and batch 512). Not given and
fixed here: label smoothing 0.1, shallow prompts, 30 stage 3 epochs. CLIP
ViT-L/14 and the PlantCLEF DINOv2 replace the paper's backbones so that the
encoders match FROND. Needs a GPU; the effective batch is reached by
gradient accumulation.

    python baseline_nola.py --features features --out results
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from frond.common import NUM_CLASSES, TAU_T, balanced_subset, macro_f1, stratified_split
from frond.encoders import CLIP_NAME

SMOOTH = 0.1
S2_EPOCHS, S2_LR, S2_BS = 50, 1e-3, 32
N_PROMPTS = 16
S3_EPOCHS, S3_LR, S3_BATCH = 30, 2e-3, 512
LOGIT_SCALE = 100.0

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
STRONG = T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
                    T.RandomPerspective(0.2, p=0.5), T.ToTensor(), T.Normalize(MEAN, STD)])
PLAIN = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                   T.ToTensor(), T.Normalize(MEAN, STD)])


class PromptedCLIP:
    """CLIP vision encoder with shallow visual prompts inserted after the class token."""

    def __init__(self, device):
        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).to(device).requires_grad_(False)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.prompts = None
        self.model.vision_model.embeddings.register_forward_hook(self._insert)

    def _insert(self, module, inputs, out):
        if self.prompts is None:
            return out
        p = self.prompts.to(out.dtype).expand(out.shape[0], -1, -1)
        return torch.cat([out[:, :1], p, out[:, 1:]], 1)

    def encode(self, images, transform):
        x = torch.stack([transform(im) for im in images]).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            return F.normalize(self.model(pixel_values=x).image_embeds.float(), dim=1)


def train_teacher(dino_feats, keep, pseudo, seed, device):
    """Stage 2: linear alignment module on DINOv2 features of the pseudo-labelled images."""
    torch.manual_seed(seed)
    x, y = dino_feats[keep].to(device), torch.as_tensor(pseudo, device=device)
    head = nn.Linear(x.shape[1], NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=S2_LR)
    for _ in range(S2_EPOCHS):
        for b in torch.randperm(len(x), device=device).split(S2_BS):
            loss = F.cross_entropy(head(x[b]), y[b], label_smoothing=SMOOTH)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def run_seed(seed, clip, pv_images, pv_dino, keep, pseudo, anchors, pd_images, micro_bs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = clip.device
    head = train_teacher(pv_dino, keep, pseudo, seed, dev)
    with torch.no_grad():
        teacher = head(pv_dino.to(dev)).argmax(1)      # teacher label for every PV image

    clip.prompts = nn.Parameter(0.02 * torch.randn(1, N_PROMPTS, clip.model.config.hidden_size, device=dev))
    classifier = nn.Parameter(anchors.clone().to(dev))
    opt = torch.optim.AdamW([clip.prompts, classifier], lr=S3_LR)
    accum = max(1, S3_BATCH // micro_bs)
    n = len(pv_images)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=S3_EPOCHS * math.ceil(n / (micro_bs * accum)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    # stage 3: CLIP on a strong view learns to match the teacher on the original view
    clip.model.train()
    for _ in range(S3_EPOCHS):
        order = np.random.permutation(n)
        for step, i in enumerate(range(0, n, micro_bs), 1):
            b = order[i:i + micro_bs]
            f = clip.encode([pv_images[j] for j in b], STRONG)
            logits = LOGIT_SCALE * f @ F.normalize(classifier, dim=1).T
            loss = F.cross_entropy(logits, teacher[b], label_smoothing=SMOOTH) / accum
            scaler.scale(loss).backward()
            if step % accum == 0 or i + micro_bs >= n:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()

    clip.model.eval()
    with torch.no_grad():
        w = F.normalize(classifier, dim=1)
        pred = torch.cat([(clip.encode(pd_images[i:i + 32], PLAIN) @ w.T).argmax(1).cpu()
                          for i in range(0, len(pd_images), 32)]).numpy()
    clip.prompts = None
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-class", type=int, default=200, help="unlabelled PV images sampled per class")
    ap.add_argument("--micro-batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = torch.load(feat / "pv.pt", weights_only=False)
    pd = torch.load(feat / "pd.pt", weights_only=False)
    anchors = torch.load(feat / "anchors_set1.pt")
    y_pd = pd["y"]

    train, _, _ = stratified_split(pv["y"], (0.7, 0.1, 0.2), seed=0)
    idx = balanced_subset(train, pv["y"], args.per_class)
    pv_clip = F.normalize(pv["clip"][idx].float(), dim=1)
    pv_dino = F.normalize(pv["dino"][idx].float(), dim=1)
    pv_true = pv["y"][idx]                                 # diagnostics only

    # stage 1: pseudo-labels from the description classifier, top 20% per class
    topk = max(16, round(0.2 * len(idx) / NUM_CLASSES))
    probs = F.softmax(pv_clip @ anchors.T / TAU_T, dim=1)
    conf, lab = (t.numpy() for t in probs.max(1))
    keep = np.concatenate([np.where(lab == c)[0][np.argsort(-conf[lab == c])[:topk]] for c in range(NUM_CLASSES)])
    pseudo = lab[keep]
    print(f"stage 1: kept {len(keep)} images, pseudo-label accuracy {100 * (pseudo == pv_true[keep]).mean():.1f}%")

    pv_images = [Image.open(pv["paths"][i]).convert("RGB").resize((256, 256)) for i in idx]
    pd_images = [Image.open(p).convert("RGB") for p in pd["paths"]]
    clip = PromptedCLIP(device)

    preds, scores = [], []
    for s in args.seeds:
        preds.append(run_seed(s, clip, pv_images, pv_dino, keep, pseudo, anchors, pd_images, args.micro_batch))
        scores.append(macro_f1(y_pd, preds[-1]))
        print(f"seed {s}: NoLA {scores[-1]:.1f}")
    print(f"NoLA (source-only): {np.mean(scores):.1f} +/- {np.std(scores, ddof=1):.1f}")
    np.savez(out / "nola_preds.npz", y_true=y_pd, NoLA=preds[int(np.argsort(scores)[len(scores) // 2])])


if __name__ == "__main__":
    main()
