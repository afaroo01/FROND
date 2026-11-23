"""Frozen encoders: PlantCLEF DINOv2 ViT-B/14 (visual) and CLIP ViT-L/14 (language)."""
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-large-patch14"
DINO_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"


def load_dinov2(checkpoint, device):
    """DINOv2 trained ViT-B/14 with PlantCLEF 2024 weights, classifier removed.

    timm's eval transform for this model resizes to 518 (bicubic), centre crops
    518 x 518 and applies ImageNet normalisation: 37 x 37 patches of size 14.
    The output is the 768-d class token of the final layer.
    """
    model = timm.create_model(DINO_NAME, pretrained=False, num_classes=7806, checkpoint_path=str(checkpoint))
    model.reset_classifier(0)
    model = model.to(device).eval().requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    return model, timm.data.create_transform(**cfg, is_training=False)


def load_clip(device):
    """CLIP ViT-L/14 (OpenAI weights). Images go in at 224 x 224; embeddings are 768-d."""
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval().requires_grad_(False)
    return model, CLIPProcessor.from_pretrained(CLIP_NAME)


def _batches(paths, batch_size):
    for i in range(0, len(paths), batch_size):
        yield [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]


@torch.no_grad()
def dinov2_features(model, transform, paths, device, batch_size=32):
    out = []
    for ims in _batches(paths, batch_size):
        x = torch.stack([transform(im) for im in ims]).to(device)
        out.append(model(x).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def clip_image_features(model, processor, paths, device, batch_size=32):
    out = []
    for ims in _batches(paths, batch_size):
        inp = processor(images=ims, return_tensors="pt").to(device)
        out.append(model.get_image_features(**inp).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def clip_text_features(model, processor, texts, device):
    """l2-normalised CLIP text embeddings, one row per text."""
    inp = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
    return F.normalize(model.get_text_features(**inp).float(), dim=1).cpu()


def text_anchors(model, processor, descriptions, device):
    """One anchor per class: l2-normalised mean of its description embeddings (Eq. 3)."""
    anchors = [clip_text_features(model, processor, descriptions[c], device).mean(0)
               for c in sorted(descriptions)]
    return F.normalize(torch.stack(anchors), dim=1)
