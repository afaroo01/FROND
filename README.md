# FROND

Code for *A Training-Free Vision and Language Fusion Framework for Domain Generalisation in Plant Disease Classification* (Adil Farooq and Rudi Villing, Maynooth University), WILD VISION Workshop at BMVC 2026.

FROND classifies field images of plant disease using only laboratory images as the source domain. It fuses two frozen channels at the probability level:

- **Visual:** a DINOv2 trained ViT-B/14 with PlantCLEF 2024 weights. Each image is compared with one prototype per class, the mean embedding of PlantVillage training images.
- **Language:** CLIP ViT-L/14. Each image is compared with one text anchor per class, the mean embedding of class-level symptom descriptions written offline.

The fused prediction is `alpha * p_v + (1 - alpha) * p_t`, with a single `alpha` chosen on PlantVillage and applied unchanged to PlantDoc. Nothing is trained, and no PlantDoc image is used for any choice.

## Setup

```
pip install -r requirements.txt
```

The CLIP weights download automatically from Hugging Face. The PlantCLEF 2024 ViT-B/14 checkpoint (`model_best.pth.tar`) is available from [Zenodo](https://zenodo.org/records/10848263).

## Data

- **PlantVillage** (source): https://github.com/spMohanty/PlantVillage-Dataset
- **PlantDoc** (target): https://github.com/pratikkayal/PlantDoc-Dataset

Both datasets are used in YOLO format. Only the eight classes shared by both datasets are used:

| YOLO id | Class | PlantVillage | PlantDoc |
|---|---|---|---|
| 0 | Apple healthy | 1645 | 68 |
| 1 | Apple rust | 275 | 77 |
| 3 | Bell pepper bacterial spot | 997 | 58 |
| 4 | Bell pepper healthy | 1478 | 55 |
| 10 | Grape black rot | 1180 | 47 |
| 11 | Grape healthy | 423 | 64 |
| 22 | Tomato healthy | 1591 | 68 |
| 27 | Tomato septoria leaf spot | 1771 | 128 |

Expected layout:

```
data/plantvillage/   train/images, train/labels, test/images, test/labels, ...
data/plantdoc/       images/, labels/
```

## Running

```
# 1. features from both frozen encoders, and the text anchors
python extract_features.py --pv-root data/plantvillage --pd-root data/plantdoc \
    --plantclef-ckpt weights/model_best.pth.tar --out features

# 2. FROND over ten random 70/10/20 partitions of PlantVillage
python run_frond.py

# 3. other ways of setting the fusion weight (Table 6)
python fusion_rules.py

# 4. baselines (Table 7)
python baselines_training_free.py      # CLIP prompting, DCLIP, WaffleCLIP, MVPDR, ProKeR
python baselines_source_trained.py     # CoOp, AdvST-style head
python baseline_nola.py                # NoLA, source-only (GPU)

# 5. paired bootstrap against FROND
python bootstrap.py results/frond_preds.npz results/training_free_preds.npz \
    results/source_trained_preds.npz results/nola_preds.npz
python bootstrap.py results/fusion_rules_preds.npz

# 6. per-class radar and fusion rescue figures
python figures.py
```

Images are resized to 518 x 518 for DINOv2 and 224 x 224 for CLIP; both give 768-d embeddings. The temperatures are fixed at 0.5 (visual) and 0.02 (language). Experiments were run on a single RTX 3060 laptop GPU (6 GB).

The baselines are reimplementations. Where the original paper does not give a setting, the value used is stated at the top of each script and was fixed before evaluation.

## Citation

```bibtex
@inproceedings{farooq2026frond,
  title     = {A Training-Free Vision and Language Fusion Framework for Domain Generalisation in Plant Disease Classification},
  author    = {Farooq, Adil and Villing, Rudi},
  booktitle = {WILD VISION Workshop, British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## Acknowledgement

This publication has emanated from research conducted with the financial support of Research Ireland under Grant number 18/CRT/6222.
