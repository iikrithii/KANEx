"""
MIMIC-CXR Dataset for multi-label chest X-ray classification.

Handles:
  - Train CSV: subject_id, study_id, 14 CheXpert labels
  - Test CSV:  dicom_id, bounding box annotations (x, y, w, h), labels
  - Image path resolution: finds frontal (PA) view from study directory
  - Multi-label binary targets (14 CheXpert conditions)

Why two images per study?
  MIMIC-CXR often has both a PA (posterior-anterior, frontal) and
  a lateral view. We always pick the PA view for classification,
  since that's the standard view for CheXpert labels.
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# The 14 CheXpert labels used throughout the project
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(CHEXPERT_LABELS)


def load_metadata_index(metadata_csv_path: str) -> dict:
    """
    Load the MIMIC-CXR metadata CSV and build a lookup dict:
        dicom_id -> ViewPosition (e.g. 'PA', 'AP', 'LATERAL', 'LL')

    The metadata CSV has columns including:
        dicom_id, subject_id, study_id, ViewPosition, ...

    Args:
        metadata_csv_path: Path to mimic-cxr metadata.csv
    Returns:
        dict: {dicom_id: view_position_string}
    """
    df = pd.read_csv(metadata_csv_path, usecols=["dicom_id", "ViewPosition"])
    df["ViewPosition"] = df["ViewPosition"].fillna("UNKNOWN").str.upper().str.strip()
    return dict(zip(df["dicom_id"], df["ViewPosition"]))


# Module-level metadata index — populated once on first call
_METADATA_INDEX = None
_METADATA_PATH = None


def set_metadata_path(path: str):
    """
    Call this once at startup with the path to metadata.csv.
    Subsequent calls to find_frontal_image will use it automatically.
    """
    global _METADATA_INDEX, _METADATA_PATH
    _METADATA_PATH = path
    _METADATA_INDEX = load_metadata_index(path)
    print(f"[Metadata] Loaded {len(_METADATA_INDEX):,} DICOM view entries from {path}")


def find_frontal_image(study_dir: str, view_stats: dict = None):
    """
    Given a study directory, pick the best image using the metadata CSV
    ViewPosition field. Priority order: PA > AP > anything else.

    If metadata is not loaded (set_metadata_path not called), falls back
    to returning the first .jpg found.

    Args:
        study_dir:  Path to the study directory (contains .jpg files)
        view_stats: Optional dict to accumulate view selection counts.
                    Pass the same dict across all calls to get totals.
                    Keys: 'PA', 'AP', 'LATERAL', 'LL', 'OTHER', 'SINGLE'

    Returns:
        path (str) or None if no images found
    """
    global _METADATA_INDEX

    all_jpgs = glob.glob(os.path.join(study_dir, "*.jpg"))
    if not all_jpgs:
        return None

    # Only one image — just use it regardless of view
    if len(all_jpgs) == 1:
        if view_stats is not None:
            view_stats["SINGLE"] = view_stats.get("SINGLE", 0) + 1
        return all_jpgs[0]

    # Multiple images — use metadata to pick the best view
    if _METADATA_INDEX is not None:
        # Build a map: {dicom_id: full_path} for images in this study dir
        dicom_to_path = {}
        for jpg in all_jpgs:
            dicom_id = os.path.splitext(os.path.basename(jpg))[0]
            dicom_to_path[dicom_id] = jpg

        # Categorize each image by ViewPosition
        categorized = {"PA": [], "AP": [], "OTHER": []}
        for dicom_id, path in dicom_to_path.items():
            view = _METADATA_INDEX.get(dicom_id, "UNKNOWN")
            if view == "PA":
                categorized["PA"].append((path, view))
            elif view == "AP":
                categorized["AP"].append((path, view))
            else:
                categorized["OTHER"].append((path, view))

        # Select in priority order: PA > AP > other
        if categorized["PA"]:
            chosen_path, chosen_view = categorized["PA"][0]
        elif categorized["AP"]:
            chosen_path, chosen_view = categorized["AP"][0]
        else:
            chosen_path, chosen_view = categorized["OTHER"][0]

        # Accumulate stats
        if view_stats is not None:
            view_stats[chosen_view] = view_stats.get(chosen_view, 0) + 1

        return chosen_path

    # Fallback if no metadata: return first file
    if view_stats is not None:
        view_stats["UNKNOWN"] = view_stats.get("UNKNOWN", 0) + 1
    return all_jpgs[0]


class MIMICCXRDataset(Dataset):
    """
    Multi-label chest X-ray dataset for training/validation.

    Args:
        csv_path:    Path to the train or val CSV
        image_root:  Root directory of MIMIC-CXR images
                     e.g. /home/gokul/vlm_xray/.../files/
        transform:   torchvision transform pipeline
        labels:      List of label column names (default: CHEXPERT_LABELS)
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        labels: list = None,
        metadata_csv: str = None,   # path to mimic-cxr metadata.csv for PA/AP selection
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.labels = labels or CHEXPERT_LABELS

        # Load metadata for view-based image selection if provided
        if metadata_csv is not None:
            set_metadata_path(metadata_csv)

        # Fill NaN labels with 0 (treat uncertain as negative, common practice)
        self.df[self.labels] = self.df[self.labels].fillna(0).clip(0, 1)

        # Resolve image paths once at init (faster than per-item)
        self.image_paths = self._resolve_paths()
        valid_mask = [p is not None for p in self.image_paths]
        n_total = len(self.df)
        n_valid = sum(valid_mask)
        print(f"[Dataset] {csv_path}: {n_valid}/{n_total} studies with valid images")

        # Filter to valid rows only
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.image_paths = [p for p, v in zip(self.image_paths, valid_mask) if v]

    def _resolve_paths(self):
        """Build image paths for all rows. Returns list of str or None."""
        paths = []
        view_stats = {}  # accumulate PA/AP/LATERAL/etc counts

        for _, row in self.df.iterrows():
            subject_id = str(row["subject_id"])
            study_id = str(row["study_id"])

            # MIMIC-CXR directory structure: files/p{first2}/p{subject}/s{study}/
            p_prefix = f"p{subject_id[:2]}"
            study_dir = os.path.join(
                self.image_root, p_prefix, f"p{subject_id}", f"s{study_id}"
            )
            img_path = find_frontal_image(study_dir, view_stats=view_stats)
            paths.append(img_path)

        # Print view selection summary
        total = sum(view_stats.values())
        print(f"[Dataset] View selection summary ({total} studies):")
        for view, count in sorted(view_stats.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f"          {view:<10}: {count:>5} ({pct:.1f}%)")

        return paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        row = self.df.iloc[idx]

        # Load image (MIMIC-CXR is grayscale JPEG → convert to RGB for ResNet)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Labels: float tensor (multi-label binary)
        label = torch.tensor(
            row[self.labels].values.astype(np.float32),
            dtype=torch.float32
        )

        # Return identifier for later matching
        pair_id = str(row.get("pair", f"({row['subject_id']},{row['study_id']})"))

        return {
            "image": img,
            "label": label,
            "pair_id": pair_id,
            "subject_id": str(row["subject_id"]),
            "study_id": str(row["study_id"]),
            "image_path": img_path,
        }


class MIMICCXRTestDataset(Dataset):
    """
    Test dataset with bounding box annotations for IoU evaluation.

    The test CSV includes:
      - dicom_id: unique identifier
      - category_name: which label has a bounding box
      - x, y, w, h: bounding box in pixel coordinates
      - image_width, image_height: original image size
      - path: relative path from some base (we prepend image_root)
      - 14 label columns

    Args:
        csv_path:    Path to the test CSV
        image_root:  Root directory for resolving relative paths
        transform:   torchvision transform
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        labels: list = None,
        metadata_csv: str = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.labels = labels or CHEXPERT_LABELS

        if metadata_csv is not None:
            set_metadata_path(metadata_csv)

        # Fill NaN labels
        self.df[self.labels] = self.df[self.labels].fillna(0).clip(0, 1)

        # Resolve absolute paths from relative 'path' column
        self.image_paths = self._resolve_paths()
        valid_mask = [os.path.exists(p) for p in self.image_paths]
        n_valid = sum(valid_mask)
        print(f"[TestDataset] {csv_path}: {n_valid}/{len(self.df)} images found")

        self.df = self.df[valid_mask].reset_index(drop=True)
        self.image_paths = [p for p, v in zip(self.image_paths, valid_mask) if v]

    def _resolve_paths(self):
        """
        The test CSV has a relative 'path' column like:
          files/p10/p10233088/s54276838/675d792f-....jpg
        We prepend image_root to get the absolute path.
        Note: test CSV already identifies specific DICOMs, so no view
        selection is needed here — the dicom_id directly identifies the file.
        """
        paths = []
        for _, row in self.df.iterrows():
            rel_path = str(row["path"])
            abs_path = os.path.join(self.image_root, rel_path)
            paths.append(abs_path)
        return paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # for bbox scaling later

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        label = torch.tensor(
            row[self.labels].values.astype(np.float32),
            dtype=torch.float32
        )

        # Bounding box (in original image coordinates)
        bbox = {
            "x": float(row.get("x", 0)),
            "y": float(row.get("y", 0)),
            "w": float(row.get("w", 0)),
            "h": float(row.get("h", 0)),
            "img_w": float(row.get("image_width", orig_w)),
            "img_h": float(row.get("image_height", orig_h)),
            "category": str(row.get("category_name", "")),
        }

        return {
            "image": img_tensor,
            "label": label,
            "bbox": bbox,
            "dicom_id": str(row["dicom_id"]),
            "pair_id": str(row.get("pair_chex", row.get("pair", ""))),
            "image_path": img_path,
        }


# ── Transform pipelines ───────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224):
    """
    Training transform: resize, random flips, color jitter (mild for medical),
    normalize with ImageNet stats (since we use pretrained ResNet).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Mild color jitter only — X-rays are grayscale so brightness/contrast matter
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size: int = 224):
    """
    Validation / test transform: just resize and normalize (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── DataLoader builders ───────────────────────────────────────────────────────

def build_dataloaders(
    train_csv: str,
    val_csv: str,
    image_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    metadata_csv: str = None,   # path to mimic-cxr metadata.csv
):
    """
    Build train and validation DataLoaders.

    Args:
        train_csv:     Path to training CSV
        val_csv:       Path to validation CSV
        image_root:    MIMIC-CXR image root directory
        batch_size:    Batch size
        num_workers:   DataLoader workers
        image_size:    Image resize target
        metadata_csv:  Path to MIMIC-CXR metadata.csv for PA/AP view selection.
                       Highly recommended — without it view selection is random.

    Returns:
        train_loader, val_loader
    """
    train_ds = MIMICCXRDataset(
        csv_path=train_csv,
        image_root=image_root,
        transform=get_train_transform(image_size),
        metadata_csv=metadata_csv,
    )
    # metadata already loaded by train_ds; val_ds reuses the global index
    val_ds = MIMICCXRDataset(
        csv_path=val_csv,
        image_root=image_root,
        transform=get_val_transform(image_size),
        metadata_csv=None,   # already loaded above
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"[DataLoader] Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    return train_loader, val_loader


def build_test_dataloader(
    test_csv: str,
    image_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224,
    metadata_csv: str = None,
):
    """
    Build test DataLoader (with bounding box annotations).
    metadata_csv is optional here since the test CSV already specifies exact dicom paths.
    """
    test_ds = MIMICCXRTestDataset(
        csv_path=test_csv,
        image_root=image_root,
        transform=get_val_transform(image_size),
        metadata_csv=metadata_csv,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"[DataLoader] Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    return test_loader, test_ds