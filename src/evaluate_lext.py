"""
LExT Evaluation: Heatmap-Enhanced Radiology Report Quality

For each model variant × heatmap method:
  1. Load saved heatmap (.npy) for each test image
  2. Build two versions of the image:
       - Original: unmodified RGB image
       - Enhanced: important regions brightened, unimportant regions blurred
  3. Pass both through LLaVA-1.5-7b with the diagnosis prompt
  4. Score both generated reports against the radiologist ground-truth
     explanation using LExT weighted accuracy (SAP-BERT cosine sim × NER overlap)
  5. Save per-image scores and aggregate comparison table

LExT modifications from original:
  - SAP-BERT (cambridgeltl/SapBERT-from-PubMedBERT-fullscale) replaces plain BERT
    for medically grounded semantic similarity
  - NER overlap returns 1e-8 (not 0) when there is zero overlap, to preserve
    the geometric mean structure without zeroing the whole score
  - Only weighted_accuracy is used (no context_relevancy)

Output files:
  outputs/lext_eval/{model_name}_{method}_lext_scores.csv
      columns: img_name, diagnosis, lext_original, lext_enhanced, lext_diff

  outputs/lext_eval/lext_comparison_table.csv
      columns: model, method, mean_lext_original, mean_lext_enhanced, mean_lext_diff

Usage:
  python src/evaluate_lext.py \\
      --test_csv    /path/to/test.csv \\
      --image_root  /path/to/files/ \\
      --output_dir  ./outputs \\
      --models      all \\
      --methods     gradcam swam kancam \\
      --device      cuda
"""

import os
import sys
import argparse
import glob
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer, AutoModel,
    pipeline,
    LlavaForConditionalGeneration, LlavaProcessor,
    logging as hf_logging,
)

# Suppress noisy transformer logs
hf_logging.set_verbosity_error()
for _name in logging.root.manager.loggerDict:
    if "transformers" in _name.lower():
        logging.getLogger(_name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.backbones import MODEL_REGISTRY
from src.data.dataset import CHEXPERT_LABELS

# All 14 label columns in the test CSV
LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
]


def get_diagnosis(row: pd.Series) -> str:
    """
    Build diagnosis string from all label columns that equal 1.
    e.g. "Pneumonia, Lung Opacity"
    Falls back to category_name if no label columns are found/positive.
    """
    active = [col for col in LABEL_COLS if col in row.index and row[col] == 1]
    if active:
        return ", ".join(active)
    fallback = str(row.get("category_name", "Unknown pathology"))
    return fallback if fallback not in ("nan", "") else "Unknown pathology"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LExT radiology explanation evaluation with heatmap-enhanced images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test_csv",    required=True,
                   help="Test CSV with radiologist_explanation, category_name, path, img_name columns")
    p.add_argument("--image_root",  required=True,
                   help="Root directory for image paths (prepended to CSV 'path' column)")
    p.add_argument("--output_dir",  default="./outputs",
                   help="Root output dir — heatmaps expected at <output_dir>/<model>/heatmaps/<method>/<dicom_id>.npy")
    p.add_argument("--lext_dir",    default=None,
                   help="Where to save LExT CSVs. Defaults to <output_dir>/lext_eval/")
    p.add_argument("--models", nargs="+", default=["all"],
                   choices=list(MODEL_REGISTRY.keys()) + ["all"])
    p.add_argument("--methods", nargs="+", default=["gradcam", "kancam", "rollout"],
                   choices=["gradcam", "kancam", "rollout"])
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image_size",  type=int, default=224,
                   help="Image size to use (must match what heatmaps were generated at)")
    p.add_argument("--blur_ksize",  type=int, default=51,
                   help="Gaussian blur kernel size for unimportant regions (must be odd)")
    p.add_argument("--bright_gamma", type=float, default=0.5,
                   help="Gamma for brightening important regions (<1 = brighter). "
                        "Applied via power transform: pixel^gamma")
    p.add_argument("--importance_threshold", type=float, default=0.4,
                   help="Heatmap value above which a pixel is 'important'")
    p.add_argument("--max_new_tokens", type=int, default=400,
                   help="Max tokens for LLaVA to generate per report")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per model/method (for debugging)")
    p.add_argument("--sapbert_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="HuggingFace model name for SAP-BERT embeddings")
    p.add_argument("--ner_model",     default="Clinical-AI-Apollo/Medical-NER",
                   help="HuggingFace model for clinical NER")
    p.add_argument("--llava_model",   default="llava-hf/llava-1.5-7b-hf",
                   help="HuggingFace model for LLaVA")
    return p.parse_args()


# ── SAP-BERT embeddings ────────────────────────────────────────────────────────

class SAPBertScorer:
    """
    Computes semantic similarity using SAP-BERT
    (PubMedBERT fine-tuned for biomedical entity alignment).
    Uses CLS-token embedding (mean pooling for SAP-BERT is also fine).
    """

    def __init__(self, model_name: str, device: str):
        print(f"[SAP-BERT] Loading {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        print("[SAP-BERT] Ready.")

    def embed(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            text = str(text) if text else ""
        # SAP-BERT was trained with CLS token pooling
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # CLS token
        cls_emb = out.last_hidden_state[:, 0, :]   # (1, hidden)
        return cls_emb.squeeze(0).cpu().float().numpy()

    def similarity(self, text1: str, text2: str) -> float:
        e1 = self.embed(text1)
        e2 = self.embed(text2)
        return float(cosine_similarity([e1], [e2])[0][0])


# ── LExT weighted accuracy (SAP-BERT + NER) ───────────────────────────────────

class LExTScorer:
    """
    LExT Weighted Accuracy using SAP-BERT + Clinical NER.

    Score = (ner_overlap_fraction ^ 0.2) * sapbert_cosine_similarity

    The 0.2 exponent (same as original LExT) softens the impact of NER
    overlap so that even partial entity matches contribute meaningfully.
    When NER overlap is zero, returns 1e-8 instead of 0 to avoid
    collapsing the geometric structure.
    """

    def __init__(self, sapbert: SAPBertScorer, ner_model: str, device: str):
        self.sapbert = sapbert
        print(f"[NER] Loading {ner_model} ...")
        self.ner = pipeline(
            "token-classification",
            model=ner_model,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1,
        )
        print("[NER] Ready.")

    def _ner_words(self, text: str) -> set:
        """Extract unique entity words from clinical NER."""
        try:
            entities = self.ner(text)
            return {e["word"].lower().strip() for e in entities}
        except Exception:
            return set()

    def score(self, ground_truth: str, predicted: str) -> float:
        """
        Compute LExT weighted accuracy between ground truth and prediction.

        Args:
            ground_truth: Radiologist explanation (reference)
            predicted:    Model-generated report (hypothesis)

        Returns:
            float in (0, 1]
        """
        if not ground_truth or not predicted:
            return 1e-8

        # SAP-BERT semantic similarity
        base_sim = self.sapbert.similarity(ground_truth, predicted)
        base_sim = max(base_sim, 1e-8)   # cosine can be negative for bad outputs

        # Clinical NER overlap
        gt_words   = self._ner_words(ground_truth)
        pred_words = self._ner_words(predicted)

        if pred_words:
            overlap = len(gt_words & pred_words) / len(pred_words)
        else:
            overlap = 1e-8   # no entities extracted → very small, not zero

        overlap = max(overlap, 1e-8)

        return float((overlap ** 0.2) * base_sim)


# ── Image enhancement ──────────────────────────────────────────────────────────

def build_enhanced_image(
    original_rgb: np.ndarray,     # (H, W, 3) uint8
    heatmap: np.ndarray,          # (h, w) float32 in [0, 1]
    importance_threshold: float,
    blur_ksize: int,
    gamma: float,
) -> np.ndarray:
    """
    Create heatmap-enhanced image:
      - Pixels where heatmap >= threshold: brighten via gamma correction
      - Pixels where heatmap <  threshold: replace with Gaussian-blurred version

    Args:
        original_rgb:         (H, W, 3) uint8 original image
        heatmap:              (h, w) float32 importance map, normalized to [0, 1]
        importance_threshold: cutoff separating important / unimportant regions
        blur_ksize:           Gaussian kernel size for blurring (must be odd, >= 1)
        gamma:                Power for brightening (<1 = brighter, >1 = darker)

    Returns:
        (H, W, 3) uint8 enhanced image
    """
    H, W = original_rgb.shape[:2]

    # Ensure blur kernel size is odd and at least 1
    ksize = max(1, blur_ksize | 1)   # bitwise OR with 1 forces odd

    # Upsample heatmap to full image size
    hm = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    hm = np.clip(hm, 0.0, 1.0)

    # Binary mask: 1 = important, 0 = unimportant
    mask_important = (hm >= importance_threshold).astype(np.float32)   # (H, W)

    # Soft mask — blend at edges to avoid harsh boundaries
    # Smooth the mask itself with a small blur
    mask_soft = cv2.GaussianBlur(mask_important, (21, 21), 0)           # (H, W) in [0,1]

    # ── Brightened version: gamma correction on [0,1] float ──────────────────
    img_float = original_rgb.astype(np.float32) / 255.0
    brightened = np.power(np.clip(img_float, 1e-6, 1.0), gamma)        # (H, W, 3)

    # ── Blurred version for unimportant regions ────────────────────────────────
    blurred = cv2.GaussianBlur(original_rgb, (ksize, ksize), 0).astype(np.float32) / 255.0

    # ── Blend: important → brightened, unimportant → blurred ──────────────────
    mask3 = mask_soft[:, :, np.newaxis]     # (H, W, 1) for broadcast
    enhanced = mask3 * brightened + (1.0 - mask3) * blurred
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

    return enhanced


# ── LLaVA inference ────────────────────────────────────────────────────────────

LLAVA_PROMPT_TEMPLATE = (
    "USER: <image> A chest X-ray of a hospitalized patient is provided. "
    "Analyze the provided chest X-ray and generate a structured radiology report "
    "using radiologic descriptors appropriate for a physician. "
    "The diagnosis for this case is {diagnosis}.\n"
    "Your response should follow this format and use language appropriate for a physician:\n\n"
    "Findings: Describe the radiographic abnormalities that support the given diagnosis "
    "or state if the findings are subtle.\n"
    "Explanation: Explain to a physician and expert in medical imaging how the findings "
    "support the given diagnosis and discuss any relevant differential considerations "
    "if applicable.\n\n"
    "Example Report:\n\n"
    "Findings: There is a right lower lobe consolidation with air bronchograms, consistent "
    "with pneumonia. A small right pleural effusion is noted. The cardiac silhouette is "
    "within normal limits. No evidence of pneumothorax or acute osseous abnormalities.\n"
    "Explanation: The presence of right lower lobe consolidation with air bronchograms is "
    "characteristic of pneumonia, suggesting alveolar filling with inflammatory exudate. "
    "The small pleural effusion is a common associated finding. The absence of pneumothorax "
    "or acute osseous abnormalities helps rule out alternative causes of respiratory distress.\n\n"
    "Now generate a report for the given chest X-ray using the same format.\n\nASSISTANT:"
)


class LLaVAReporter:
    """Generates radiology reports from images using LLaVA-1.5-7b."""

    def __init__(self, model_name: str, device: str, max_new_tokens: int = 400):
        print(f"[LLaVA] Loading {model_name} ...")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.model.eval()
        print("[LLaVA] Ready.")

    @torch.no_grad()
    def generate(self, pil_image: Image.Image, diagnosis: str) -> str:
        """
        Generate a radiology report for the given PIL image and diagnosis.

        Args:
            pil_image: RGB PIL image
            diagnosis: Diagnosis string (e.g. "Pneumonia")

        Returns:
            Generated report text (str)
        """
        prompt = LLAVA_PROMPT_TEMPLATE.format(diagnosis=diagnosis)
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,   # greedy for reproducibility
        )

        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


# ── Heatmap loading ────────────────────────────────────────────────────────────

def load_heatmap(npy_path: str, target_h: int, target_w: int) -> np.ndarray:
    """Load and resize a saved .npy heatmap to (target_h, target_w)."""
    hm = np.load(npy_path).astype(np.float32)
    if hm.shape[0] != target_h or hm.shape[1] != target_w:
        hm = cv2.resize(hm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mn, mx = hm.min(), hm.max()
    if mx - mn > 1e-8:
        hm = (hm - mn) / (mx - mn)
    return hm


# ── Per-model/method evaluation ────────────────────────────────────────────────

def evaluate_model_method(
    model_name:   str,
    method:       str,
    test_df:      pd.DataFrame,
    args,
    llava:        LLaVAReporter,
    lext:         LExTScorer,
    lext_dir:     str,
) -> pd.DataFrame | None:
    """
    Run full evaluation for one model × method pair.

    Returns a DataFrame with per-image results, or None if no heatmaps found.
    """
    heatmap_dir = os.path.join(args.output_dir, model_name, "heatmaps", method)
    if not os.path.isdir(heatmap_dir):
        print(f"    [SKIP] {model_name}/{method}: no heatmap directory")
        return None

    rows = []
    count = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc=f"  {model_name}/{method}", leave=False):

        img_name  = str(row["img_name"])
        dicom_id  = img_name   # the .npy files are saved as {dicom_id}.npy
        diagnosis = get_diagnosis(row)   # all label columns that equal 1, comma-joined
        gt_explanation = str(row.get("radiologist_explanation", ""))
        rel_path  = str(row.get("path", ""))

        if not gt_explanation or gt_explanation in ("nan", ""):
            continue

        # ── Find heatmap ──────────────────────────────────────────────────────
        npy_path = os.path.join(heatmap_dir, f"{dicom_id}.npy")
        if not os.path.exists(npy_path):
            # Some dicom_ids have the full filename as img_name; try stem
            stem     = os.path.splitext(img_name)[0]
            npy_path = os.path.join(heatmap_dir, f"{stem}.npy")
            if not os.path.exists(npy_path):
                continue

        # ── Find image ────────────────────────────────────────────────────────
        img_path = os.path.join(args.image_root, rel_path)
        if not os.path.exists(img_path):
            # Fallback: try the img_name directly
            img_path = os.path.join(args.image_root, img_name)
            if not os.path.exists(img_path):
                continue

        # ── Load image ────────────────────────────────────────────────────────
        try:
            pil_orig = Image.open(img_path).convert("RGB").resize(
                (args.image_size, args.image_size)
            )
            orig_np = np.array(pil_orig)   # (H, W, 3) uint8
        except Exception as e:
            print(f"    [WARN] Cannot open {img_path}: {e}")
            continue

        # ── Load heatmap ──────────────────────────────────────────────────────
        try:
            heatmap = load_heatmap(npy_path, args.image_size, args.image_size)
        except Exception as e:
            print(f"    [WARN] Cannot load heatmap {npy_path}: {e}")
            continue

        # ── Build enhanced image ──────────────────────────────────────────────
        enhanced_np  = build_enhanced_image(
            original_rgb=orig_np,
            heatmap=heatmap,
            importance_threshold=args.importance_threshold,
            blur_ksize=args.blur_ksize,
            gamma=args.bright_gamma,
        )
        pil_enhanced = Image.fromarray(enhanced_np)

        # ── LLaVA inference ───────────────────────────────────────────────────
        try:
            report_orig     = llava.generate(pil_orig,     diagnosis)
            report_enhanced = llava.generate(pil_enhanced, diagnosis)
        except Exception as e:
            print(f"    [WARN] LLaVA failed on {img_name}: {e}")
            continue

        # ── LExT scoring ──────────────────────────────────────────────────────
        lext_orig     = lext.score(gt_explanation, report_orig)
        lext_enhanced = lext.score(gt_explanation, report_enhanced)
        lext_diff     = lext_enhanced - lext_orig

        rows.append({
            "img_name":      img_name,
            "diagnosis":     diagnosis,
            "lext_original":  lext_orig,
            "lext_enhanced":  lext_enhanced,
            "lext_diff":      lext_diff,
            "report_original":  report_orig,
            "report_enhanced":  report_enhanced,
        })

        count += 1
        if args.max_samples is not None and count >= args.max_samples:
            break

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Save per-model/method CSV
    out_csv = os.path.join(lext_dir, f"{model_name}_{method}_lext_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"    Saved {len(df)} rows → {out_csv}")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    lext_dir = args.lext_dir or os.path.join(args.output_dir, "lext_eval")
    os.makedirs(lext_dir, exist_ok=True)

    model_names = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    print(f"\n{'='*70}")
    print(f"  LExT Evaluation — Heatmap-Enhanced Radiology Report Quality")
    print(f"{'='*70}")
    print(f"  Models:  {model_names}")
    print(f"  Methods: {args.methods}")
    print(f"  Device:  {args.device}")
    print(f"  Output:  {lext_dir}")
    print(f"{'='*70}\n")

    # ── Load test CSV ─────────────────────────────────────────────────────────
    test_df = pd.read_csv(args.test_csv)
    # Keep only rows that have a radiologist explanation
    test_df = test_df[
        test_df["radiologist_explanation"].notna() &
        (test_df["radiologist_explanation"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"[Test CSV] {len(test_df)} samples with radiologist explanations\n")

    # ── Load models (once, shared across all model variants) ──────────────────
    print("[Init] Loading SAP-BERT ...")
    sapbert = SAPBertScorer(args.sapbert_model, args.device)

    print("[Init] Loading LExT scorer (NER) ...")
    lext = LExTScorer(sapbert, args.ner_model, args.device)

    print("[Init] Loading LLaVA ...")
    llava = LLaVAReporter(args.llava_model, args.device, args.max_new_tokens)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    summary_rows = []   # for final comparison table

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        for method in args.methods:
            df = evaluate_model_method(
                model_name=model_name,
                method=method,
                test_df=test_df,
                args=args,
                llava=llava,
                lext=lext,
                lext_dir=lext_dir,
            )

            if df is None or df.empty:
                continue

            mean_orig     = float(df["lext_original"].mean())
            mean_enhanced = float(df["lext_enhanced"].mean())
            mean_diff     = float(df["lext_diff"].mean())

            print(
                f"  {method.upper():<10}  "
                f"LExT-Orig: {mean_orig:.4f}  "
                f"LExT-Enh: {mean_enhanced:.4f}  "
                f"Diff: {mean_diff:+.4f}  "
                f"(n={len(df)})"
            )

            summary_rows.append({
                "model":               model_name,
                "method":              method,
                "n":                   len(df),
                "mean_lext_original":  mean_orig,
                "mean_lext_enhanced":  mean_enhanced,
                "mean_lext_diff":      mean_diff,
            })

    # ── Final comparison table ────────────────────────────────────────────────
    if not summary_rows:
        print("\n[WARNING] No results collected. Check heatmap paths and test CSV.")
        return

    summary_df = pd.DataFrame(summary_rows)
    comp_path  = os.path.join(lext_dir, "lext_comparison_table.csv")
    summary_df.to_csv(comp_path, index=False)

    print(f"\n{'='*70}")
    print(f"  FINAL LExT COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"  {'Model':<28} {'Method':<12}  {'Orig':>8}  {'Enh':>8}  {'Diff':>8}  N"
    print(header)
    print(f"  {'─'*65}")
    for _, r in summary_df.iterrows():
        diff_str = f"{r['mean_lext_diff']:+.4f}"
        print(
            f"  {r['model']:<28} {r['method']:<12}  "
            f"{r['mean_lext_original']:>8.4f}  "
            f"{r['mean_lext_enhanced']:>8.4f}  "
            f"{diff_str:>8}  "
            f"{int(r['n'])}"
        )
    print(f"{'='*70}")
    print(f"\nComparison table saved: {comp_path}\n")

    # ── Per-method aggregate (across all model variants) ──────────────────────
    print(f"{'─'*50}")
    print(f"  PER-METHOD AVERAGE (across all model variants)")
    print(f"{'─'*50}")
    for method in args.methods:
        sub = summary_df[summary_df["method"] == method]
        if sub.empty:
            continue
        print(
            f"  {method.upper():<12}  "
            f"Orig: {sub['mean_lext_original'].mean():.4f}  "
            f"Enh: {sub['mean_lext_enhanced'].mean():.4f}  "
            f"Diff: {sub['mean_lext_diff'].mean():+.4f}"
        )
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()