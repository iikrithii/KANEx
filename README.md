# KANEx — KAN-based Explainability for Chest X-ray Classification

KANEx replaces the standard linear classification head on pretrained CNN and ViT
backbones with Kolmogorov-Arnold Network (KAN) heads, and evaluates the resulting
spatial interpretability via three heatmap methods compared against Grad-CAM.

---

## Project structure

```
kanex/
├── run_all.py                    # Master pipeline (edit CONFIG block, then run)
├── requirements.txt
├── src/
│   ├── train.py                  # Train any model variant
│   ├── evaluate.py               # AUROC + F1 on test set
│   ├── generate_heatmaps.py      # Heatmap generation + IoU evaluation
│   ├── evaluate_faithfulness.py  # Insertion / deletion AUC
│   ├── evaluate_lext.py          # LExT radiology report quality (needs LLaVA)
│   ├── models/
│   │   └── backbones.py          # All 9 model variants + MODEL_REGISTRY
│   ├── heatmaps/
│   │   ├── gradcam.py            # GradCAM (ResNet) + ViTGradCAM
│   │   ├── attention_rollout.py  # Gradient-Weighted Attention Rollout (ViT only)
│   │   └── kan_cam.py            # KAN-CAM (KAN head models only)
│   ├── layers/
│   │   ├── efficient_kan.py      # KANLinear — B-spline KAN (inline)
│   │   ├── rational_kan.py       # RationalKANLinear — Jacobi polynomial KAN
│   │   ├── group_kan.py          # GroupKANHead — grouped B-spline KAN
│   ├── data/
│   │   └── dataset.py            # MIMIC-CXR dataset + dataloaders
│   └── utils/
│       ├── metrics.py            # AUROC, F1, IoU
│       └── visualization.py      # Heatmap overlays, training curves
```

---

## Models

| Registry key | Backbone | Head | Grad-CAM | KAN-CAM | Rollout |
|---|---|---|---|---|---|
| `resnet_baseline` | ResNet-50 | FC | ✅ | ❌ | ❌ |
| `resnet_vanilla_kan` | ResNet-50 | KANLinear ×2 | ✅ | ✅ | ❌ |
| `resnet_rational_kan` | ResNet-50 | RationalKAN | ✅ | ✅ | ❌ |
| `resnet_group_kan` | ResNet-50 | GroupKAN | ✅ | ✅ | ❌ |
| `resnet_conv_kan` | ResNet-50 | KAN conv + FC | ✅ | ❌ | ❌ |
| `vit_baseline` | ViT-S/16 | FC | ✅ | ❌ | ✅ |
| `vit_vanilla_kan` | ViT-S/16 | KANLinear ×2 | ✅ | ✅ | ✅ |
| `vit_group_kan` | ViT-S/16 | GroupKAN | ✅ | ✅ | ✅ |
| `vit_rational_kan` | ViT-S/16 | RationalKAN | ✅ | ✅ | ✅ |

All backbones are frozen during training. Only the head trains.

---

## Installation

```bash
git clone <repo>
cd kanex
pip install -r requirements.txt
```

If you plan to run `evaluate_lext.py`, also install:
```bash
pip install accelerate>=0.20.0
```
LLaVA-1.5-7b weights (~14 GB) download automatically on first run.

---

## Data

Expected CSV columns (same for train and test):

```
img_name, path, subject_id, study_id,
Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum,
Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion,
Pleural Other, Pneumonia, Pneumothorax,
x, y, w, h, image_width, image_height       ← bounding box (for IoU)
```

For LExT evaluation only, the test CSV must additionally have:
```
radiologist_explanation    ← free-text radiology report
category_name              ← bbox annotation category (fallback if no label=1)
```

The `path` column is a relative path appended to `--image_root`.

---

## Step-by-step usage

### 0 · Quick start (run everything at once)

Edit the `CONFIG` block in `run_all.py`, then:

```bash
python run_all.py
```

This runs train → evaluate → generate_heatmaps in sequence. For faithfulness and
LExT you still need to run the separate scripts below.

---

### 1 · Train

Trains one or more model variants. Saves `best_model.pth` per model.

```bash
python src/train.py \
    --train_csv    /path/to/train.csv \
    --image_root   /path/to/files/ \
    --metadata_csv /path/to/mimic-cxr-2.0.0-metadata.csv \
    --output_dir   ./outputs \
    --epochs       20 \
    --batch_size   32 \
    --lr           1e-3 \
    --models       all \
    --device       cuda
```

Train specific models only:
```bash
--models resnet_baseline vit_baseline vit_vanilla_kan
```

**ViT note:** if the ViT head is not converging, lower the learning rate:
```bash
--lr 3e-4
```
The backbone is frozen so only ~5K parameters train — it converges in a few epochs.

Key defaults: `--val_split 0.1` (10% of train CSV held out for validation),
`--freeze_backbone` (backbone always frozen; pass `--no_freeze_backbone` to fine-tune).

Outputs per model:
```
outputs/{model_name}/
    best_model.pth         ← best checkpoint by val AUROC
    training_curves.png    ← loss + AUROC curves
```

---

### 2 · Evaluate classification performance

Loads best checkpoints, runs inference on the test set, reports AUROC and F1.

```bash
python src/evaluate.py \
    --test_csv    /path/to/test.csv \
    --image_root  /path/to/files/ \
    --output_dir  ./outputs \
    --batch_size  32 \
    --models      all \
    --device      cuda
```

Outputs:
```
outputs/{model_name}/test_results.csv     ← per-label AUC + F1
outputs/evaluation_comparison.csv         ← macro AUC + F1 for all models
```

---

### 3 · Generate heatmaps + IoU

Reads trained checkpoints, generates heatmaps, saves as `.npy`, computes IoU
against ground-truth bounding boxes. All downstream evaluation scripts read
from these `.npy` files — run this before steps 4 and 5.

```bash
python src/generate_heatmaps.py \
    --test_csv    /path/to/test.csv \
    --image_root  /path/to/files/ \
    --output_dir  ./outputs \
    --models      all \
    --methods     gradcam kancam rollout \
    --device      cuda
```

Inapplicable method/model combinations (e.g. rollout on ResNet) are silently
skipped and show as N/A in the output table.

Optional flags:
```
--save_viz            save PNG overlays alongside .npy files (default: on)
--iou_percentile 75   percentile threshold for binarising heatmaps for IoU
--discard_ratio 0.9   rollout only: fraction of low-attention values to zero out
```

Outputs:
```
outputs/{model_name}/heatmaps/{method}/{dicom_id}.npy   ← raw heatmaps
outputs/{model_name}/heatmaps/{method}/viz/*.png        ← overlays
outputs/{model_name}/iou_results.csv                    ← per-method IoU
outputs/iou_comparison.csv                              ← all models × methods
```

Load a heatmap later:
```python
import numpy as np
heatmap = np.load("outputs/vit_vanilla_kan/heatmaps/kancam/675d792f-....npy")
# shape: (14, 14) for ViT methods, (7, 7) for ResNet methods
```

---

### 4 · Faithfulness evaluation (insertion / deletion AUC)

Reads saved `.npy` heatmaps — does not regenerate them. Requires step 3 first.

```bash
python src/evaluate_faithfulness.py \
    --test_csv    /path/to/test.csv \
    --image_root  /path/to/files/ \
    --output_dir  ./outputs \
    --models      all \
    --methods     gradcam kancam rollout \
    --steps       50 \
    --device      cuda
```

`--steps` controls the number of insertion/deletion steps per image (more = smoother
curve, slower). 50 is a good default; use 20 for a quick run.

Outputs:
```
outputs/{model_name}/faithfulness_{method}.csv    ← per-image insertion/deletion AUC
outputs/faithfulness_comparison.csv               ← aggregate table
```

---

### 5 · LExT evaluation (radiology report quality)

The most GPU-heavy step. Runs LLaVA-1.5-7b twice per image (original +
heatmap-enhanced), scores generated reports against radiologist ground truth
using SAP-BERT + Clinical NER. Requires step 3 first.

Only works on the test CSV that has `radiologist_explanation` column.

```bash
python src/evaluate_lext.py \
    --test_csv    /path/to/test_with_gt.csv \
    --image_root  /path/to/files/ \
    --output_dir  ./outputs \
    --models      all \
    --methods     gradcam kancam rollout \
    --device      cuda
```

Optional flags:
```
--importance_threshold 0.4   heatmap value above which a pixel is "important"
--blur_ksize 51              Gaussian kernel size for unimportant regions
--bright_gamma 0.5           gamma for brightening important regions (<1 = brighter)
--max_new_tokens 400         max tokens LLaVA generates per report
--max_samples N              cap samples per model/method (for debugging)
```

Outputs:
```
outputs/lext_eval/{model}_{method}_lext_scores.csv   ← per-image scores + reports
outputs/lext_eval/lext_comparison_table.csv          ← aggregate table
```

---

### 6 · Significance testing

Tests whether score distributions between model/method pairs are statistically
different. Uses Wilcoxon signed-rank test (non-parametric paired test).

**IoU and LExT** — per-image scores are already saved; run immediately after steps 3/5:

```bash
python src/significance_test.py \
    --output_dir ./outputs \
    --metric     iou        # or lext
```

**Faithfulness AUC** — re-run step 4 with `--save_per_image` to get per-image AUCs
first, then pass `--metric faithfulness`.

**Classification AUROC** — uses bootstrap resampling (1000 resamples) since AUROC
is a ranking metric across images and doesn't have a per-image value:

```bash
python src/significance_test.py \
    --output_dir  ./outputs \
    --metric      auroc \
    --test_csv    /path/to/test.csv \
    --image_root  /path/to/files/ \
    --device      cuda
```

Outputs:
```
outputs/significance_{metric}.csv    ← p-values for all model/method pairs
```

---

## Output directory layout (after all steps)

```
outputs/
├── evaluation_comparison.csv         ← macro AUC + F1 for all models
├── iou_comparison.csv                ← mean IoU for all models × methods
├── faithfulness_comparison.csv       ← insertion/deletion AUC table
├── significance_iou.csv              ← Wilcoxon p-values for IoU
├── significance_lext.csv             ← Wilcoxon p-values for LExT
├── significance_auroc.csv            ← bootstrap p-values for AUROC
├── lext_eval/
│   ├── lext_comparison_table.csv
│   └── {model}_{method}_lext_scores.csv
└── {model_name}/                     ← one directory per model
    ├── best_model.pth
    ├── training_curves.png
    ├── test_results.csv
    ├── iou_results.csv
    ├── faithfulness_{method}.csv
    └── heatmaps/
        └── {method}/
            ├── {dicom_id}.npy        ← raw heatmap, loadable with np.load()
            └── viz/
                └── {dicom_id}.png    ← overlay visualization
```

---

## Recommended run order

```
1  train.py                   ~2–6h per model depending on dataset size
2  evaluate.py                ~10 min for all models
3  generate_heatmaps.py       ~1–2h for all models × methods
4  evaluate_faithfulness.py   ~1–2h (reads .npy files, runs model inference)
5  evaluate_lext.py           ~4–8h (runs LLaVA twice per image)
6  significance_test.py       <5 min for IoU + LExT; ~15 min for AUROC bootstrap
```

Steps 4, 5, and 6 are all independent of each other and can be run in parallel
on separate GPUs once step 3 is complete.

---

## Heatmap method reference

| Method | Models | How it works |
|---|---|---|
| **Grad-CAM** | All | Gradients at last conv (ResNet) or last transformer block (ViT) pooled to channel weights, weighted feature sum |
| **KAN-CAM** | KAN-head models only | KAN splines evaluated pointwise at each spatial location before pooling — no backward pass |
| **Rollout** | ViT models only | Attention matrices multiplied across all layers, gradient-weighted per class, CLS-to-patch row extracted |

---


KAN references:
- Liu et al. (2024) KAN: Kolmogorov-Arnold Networks
- Blealtan (2024) efficient-kan
- Chefer et al. (2021) Transformer Interpretability Beyond Attention Visualization
- Selvaraju et al. (2017) Grad-CAM
