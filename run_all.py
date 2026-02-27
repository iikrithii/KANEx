"""
run_all.py — KANEx Master Pipeline

Runs the full KANEx pipeline in order:
  1. Train all models
  2. Evaluate on test set
  3. Generate heatmaps + compute IoU

Edit the CONFIG block at the top with your paths, then run:
    python run_all.py

Or run each step separately:
    python src/train.py --help
    python src/evaluate.py --help
    python src/generate_heatmaps.py --help
"""

import os
import subprocess
import sys


# ═══════════════════════════════════════════════════════════════════
#  CONFIG — Edit these paths before running
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # CSV files
    "train_csv":    "/path/to/train.csv",
    "val_csv":      "/path/to/val.csv",       # split from your 20k
    "test_csv":     "/path/to/test.csv",       # the annotated test set

    # MIMIC-CXR image root — the directory containing p10/, p11/, etc.
    # Example: /home/gokul/vlm_xray/physionet.org/files/big_mimic/
    #          physionet.org/files/mimic-cxr-jpg/2.1.0/files/
    "image_root":   "/home/gokul/vlm_xray/physionet.org/files/big_mimic/"
                    "physionet.org/files/mimic-cxr-jpg/2.1.0/files/",

    # MIMIC-CXR metadata CSV — used for PA/AP view selection (highly recommended)
    # Usually found at: .../mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv
    "metadata_csv": "/home/gokul/vlm_xray/physionet.org/files/big_mimic/"
                    "physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv",

    # Training hyperparameters
    "epochs":       20,
    "batch_size":   32,
    "lr":           1e-3,
    "num_workers":  4,
    "image_size":   224,

    # Models to train ("all" or a list like ["resnet_baseline", "resnet_vanilla_kan"])
    "models":       "all",

    # Heatmap methods ("gradcam", "swam", or both)
    "methods":      "gradcam swam",

    # GPU or CPU
    "device":       "cuda",
}

# ═══════════════════════════════════════════════════════════════════


def run(cmd: str):
    """Run a shell command, printing it first."""
    print(f"\n{'─'*60}")
    print(f"$ {cmd}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    c = CONFIG
    models_str = c["models"] if c["models"] != "all" else "all"

    # ── Step 1: Train ─────────────────────────────────────────────
    run(f"""python src/train.py \
        --train_csv   "{c['train_csv']}" \
        --val_csv     "{c['val_csv']}" \
        --image_root  "{c['image_root']}" \
        --metadata_csv "{c['metadata_csv']}" \
        --output_dir  "{c['output_dir']}" \
        --epochs      {c['epochs']} \
        --batch_size  {c['batch_size']} \
        --lr          {c['lr']} \
        --num_workers {c['num_workers']} \
        --image_size  {c['image_size']} \
        --device      {c['device']} \
        --models      {models_str}
    """)

    # ── Step 2: Evaluate on test set ──────────────────────────────
    run(f"""python src/evaluate.py \
        --test_csv    "{c['test_csv']}" \
        --image_root  "{c['image_root']}" \
        --metadata_csv "{c['metadata_csv']}" \
        --output_dir  "{c['output_dir']}" \
        --batch_size  {c['batch_size']} \
        --num_workers {c['num_workers']} \
        --image_size  {c['image_size']} \
        --device      {c['device']} \
        --models      {models_str}
    """)

    # ── Step 3: Generate heatmaps + IoU ───────────────────────────
    run(f"""python src/generate_heatmaps.py \
        --test_csv    "{c['test_csv']}" \
        --image_root  "{c['image_root']}" \
        --metadata_csv "{c['metadata_csv']}" \
        --output_dir  "{c['output_dir']}" \
        --num_workers {c['num_workers']} \
        --image_size  {c['image_size']} \
        --device      {c['device']} \
        --models      {models_str} \
        --methods     {c['methods']} \
        --save_viz
    """)

    print(f"\n{'='*60}")
    print(f"  ✓ All done! Results in: {c['output_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()