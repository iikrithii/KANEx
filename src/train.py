"""
Training script for KANEx models.

Trains all 5 model variants on MIMIC-CXR.
For each model:
  - Prints estimated time to completion
  - Saves best checkpoint (by val AUROC)
  - Saves train/val loss and AUC curves
  - Logs per-epoch metrics

Usage:
    python src/train.py \
        --train_csv /path/to/train.csv \
        --val_csv /path/to/val.csv \
        --image_root /home/gokul/vlm_xray/.../files/ \
        --output_dir ./outputs \
        --epochs 20 \
        --batch_size 32 \
        --models all

    # Or train a specific model:
    python src/train.py ... --models resnet_baseline resnet_vanilla_kan
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import build_model, MODEL_REGISTRY
from src.data.dataset import build_dataloaders, CHEXPERT_LABELS, NUM_CLASSES
from src.utils.metrics import compute_auroc, compute_f1, print_results_table
from src.utils.visualisation import plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train KANEx models on MIMIC-CXR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--train_csv", required=True,
        help="Path to the training CSV (split into train+val internally)",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of the train CSV to hold out as validation",
    )
    parser.add_argument(
        "--val_seed", type=int, default=42,
        help="Random seed for the train/val split",
    )
    parser.add_argument(
        "--image_root", required=True,
        help="MIMIC-CXR image root directory (contains p10/, p11/, ... subdirs)",
    )
    parser.add_argument(
        "--metadata_csv", default=None,
        help=(
            "Path to mimic-cxr-2.0.0-metadata.csv for PA/AP view selection. "
            "Highly recommended. Without it the first jpg in each study dir is used."
        ),
    )
    parser.add_argument(
        "--output_dir", default="./outputs",
        help="Root output directory. Per-model subdirectories are created inside.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=20,
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for the KAN head (backbone is frozen and not optimized)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
    )

    # Model selection
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help=(
            "Which models to train. Use 'all' for every variant, "
            "or list names e.g. --models resnet_baseline resnet_vanilla_kan"
        ),
    )

    # Backbone freezing — BooleanOptionalAction lets you pass
    # --freeze_backbone or --no_freeze_backbone at the command line
    parser.add_argument(
        "--freeze_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the ResNet-50 backbone (recommended). "
            "Use --no_freeze_backbone to fine-tune the full network."
        ),
    )

    # Hardware
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )

    return parser.parse_args()


# ── Training functions ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """
    One full training epoch.
    Returns: (avg_loss: float, macro_auroc: float)
    """
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:3d}/{total_epochs} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — helps stability with KAN spline layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    avg_loss   = total_loss / len(loader)
    macro_auc  = compute_auroc(all_labels, all_probs, CHEXPERT_LABELS)["macro"]

    return avg_loss, macro_auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    One full validation pass.
    Returns: (avg_loss: float, macro_auroc: float)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for batch in tqdm(loader, desc="           [Val]  ", leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    avg_loss   = total_loss / len(loader)
    macro_auc  = compute_auroc(all_labels, all_probs, CHEXPERT_LABELS)["macro"]

    return avg_loss, macro_auc


def estimate_epoch_time(model, loader, device, n_warmup_batches=5):
    """
    Estimate seconds per epoch by timing a few forward+backward passes.
    Gradients are zeroed afterward so they do not contaminate real training.
    Returns: estimated seconds per epoch (float)
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    times = []

    for i, batch in enumerate(loader):
        if i >= n_warmup_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        t0 = time.perf_counter()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Critical: zero grads so timing doesn't bleed into real training
        model.zero_grad()
        times.append(t1 - t0)

    # Drop first batch (GPU warm-up), average the rest
    valid_times    = times[1:] if len(times) > 1 else times
    avg_batch_secs = float(np.mean(valid_times))
    return avg_batch_secs * len(loader)


# ── Per-model training loop ───────────────────────────────────────────────────

def train_model(model_name, args, train_loader, val_loader):
    """
    Full training loop for one model variant.
    Returns: (checkpoint_path: str, best_val_auc: float)
    """
    print(f"\n{'#'*70}")
    print(f"# Training: {model_name}")
    print(f"{'#'*70}")

    # Build model
    model = build_model(
        model_name,
        num_classes=NUM_CLASSES,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(args.device)

    # Optimizer only covers trainable parameters (i.e. the KAN head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print(f"  [WARNING] No trainable parameters found for {model_name}.")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing: decays LR smoothly to 1% of start by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.BCEWithLogitsLoss()

    # Estimate and print training time before starting
    print(f"\n[Timing] Running warmup batches to estimate training time...")
    est_epoch_secs = estimate_epoch_time(model, train_loader, args.device)
    est_total_mins = (est_epoch_secs * args.epochs) / 60
    print(
        f"[Timing] ~{est_epoch_secs:.0f}s per epoch  |  "
        f"~{est_total_mins:.0f} min total for {args.epochs} epochs\n"
    )

    # Output paths for this model
    model_dir   = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path   = os.path.join(model_dir, "best_model.pth")
    curves_path = os.path.join(model_dir, "training_curves.png")
    log_path    = os.path.join(model_dir, "training_log.txt")

    # State tracking
    best_val_auc = 0.0
    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    log_lines = []
    t_start   = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            args.device, epoch, args.epochs,
        )
        val_loss, val_auc = validate(
            model, val_loader, criterion, args.device,
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        epoch_secs     = time.time() - t_epoch
        remaining_secs = epoch_secs * (args.epochs - epoch)

        line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val AUC: {val_auc:.4f} | "
            f"Time: {epoch_secs:.0f}s  ETA: {remaining_secs/60:.1f}min"
        )
        print(line)
        log_lines.append(line)

        # Save checkpoint whenever val AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch":                epoch,
                    "model_name":           model_name,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc":              val_auc,
                    "val_loss":             val_loss,
                    "args":                 vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved  (Val AUC: {best_val_auc:.4f})")

    # Save training curves
    plot_training_curves(
        train_losses, val_losses, train_aucs, val_aucs,
        save_path=curves_path, model_name=model_name,
    )

    # Write per-epoch log
    with open(log_path, "w") as f:
        f.write(f"Model:        {model_name}\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        f.write(f"Total time:   {(time.time() - t_start)/60:.1f} min\n\n")
        f.write("\n".join(log_lines))

    total_mins = (time.time() - t_start) / 60
    print(f"\n[Done] {model_name}  |  Best Val AUC: {best_val_auc:.4f}  |  "
          f"Total: {total_mins:.1f} min")
    print(f"       Checkpoint:      {ckpt_path}")
    print(f"       Training curves: {curves_path}")

    return ckpt_path, best_val_auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_names = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    print(f"\n{'='*70}")
    print(f"  KANEx Training")
    print(f"{'='*70}")
    print(f"  Models:          {model_names}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  LR:              {args.lr}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Val split:       {args.val_split}  (seed={args.val_seed})")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Device:          {args.device}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"{'='*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Train / val split ─────────────────────────────────────────────────────
    full_df = pd.read_csv(args.train_csv)
    full_df = full_df.sample(frac=1, random_state=args.val_seed).reset_index(drop=True)

    n_val    = int(len(full_df) * args.val_split)
    val_df   = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)

    print(
        f"[Split] Total: {len(full_df):,}  |  "
        f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  "
        f"(split={args.val_split}, seed={args.val_seed})"
    )

    # Save splits for reproducibility / debugging
    tmp_train_csv = os.path.join(args.output_dir, "_tmp_train_split.csv")
    tmp_val_csv   = os.path.join(args.output_dir, "_tmp_val_split.csv")
    train_df.to_csv(tmp_train_csv, index=False)
    val_df.to_csv(tmp_val_csv,     index=False)
    print(f"[Split] Saved to {tmp_train_csv} and {tmp_val_csv}\n")

    # ── Dataloaders (built once, reused across all model runs) ────────────────
    train_loader, val_loader = build_dataloaders(
        train_csv=tmp_train_csv,
        val_csv=tmp_val_csv,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        metadata_csv=args.metadata_csv,
    )

    # ── Train each model ───────────────────────────────────────────────────────
    results = {}
    for model_name in model_names:
        ckpt_path, best_auc = train_model(
            model_name=model_name,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        results[model_name] = {"best_val_auc": best_auc, "checkpoint": ckpt_path}

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    for name, res in results.items():
        print(f"  {name:<30}  Val AUC = {res['best_val_auc']:.4f}")
    print(f"{'='*70}\n")

    summary_path = os.path.join(args.output_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        for name, res in results.items():
            f.write(f"{name}: {res['best_val_auc']:.4f} -> {res['checkpoint']}\n")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
