"""
AffectNet YOLO loader — 25,262 expression-labelled face images (96x96 PNG).

Source: Kaggle affectnet-yolo-format dataset.
Each image has one face bounding box and one of 8 expression class labels.

YOLO label file format (one line per image):
    class_id  cx  cy  w  h     (all normalised to [0, 1])

Expression classes:
    0 Anger  1 Contempt  2 Disgust  3 Fear
    4 Happy  5 Neutral   6 Sad      7 Surprise

Project label mapping:
    Neutral (5)  → 0  used as additional Normal-class training data
    All others   → expression class preserved as-is (0-7)
    Use neutral_only=True to load only Neutral images mapped to label 0.

Splits: "train" | "valid" | "test"
  ("valid" not "val" — matches the on-disk folder name)

Torch is optional at import time.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from datasets.config import (
    YOLO_CLASS_NAMES,
    YOLO_NEUTRAL_ID,
    YOLO_ROOT,
    YOLO_TEST_IMGS,
    YOLO_TEST_LBLS,
    YOLO_TRAIN_IMGS,
    YOLO_TRAIN_LBLS,
    YOLO_VALID_IMGS,
    YOLO_VALID_LBLS,
)

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False

# ── Types ─────────────────────────────────────────────────────────────────────

Split = Literal["train", "valid", "test"]

_SPLIT_DIRS: dict[str, tuple[Path, Path]] = {
    "train": (YOLO_TRAIN_IMGS, YOLO_TRAIN_LBLS),
    "valid": (YOLO_VALID_IMGS, YOLO_VALID_LBLS),
    "test":  (YOLO_TEST_IMGS,  YOLO_TEST_LBLS),
}


# ── YOLO label parser ─────────────────────────────────────────────────────────

def _parse_label(lbl_path: Path) -> tuple[int, np.ndarray]:
    """
    Parse a single-line YOLO label file.
    Returns (class_id, bbox) where bbox is float32 [cx, cy, w, h] normalised.
    """
    line = lbl_path.read_text().strip()
    parts = line.split()
    class_id = int(parts[0])
    bbox = np.array(parts[1:5], dtype=np.float32)
    return class_id, bbox


# ── Standalone helpers ────────────────────────────────────────────────────────

def count_by_class(split: Split = "train") -> dict[str, int]:
    """Return {class_name: count} for a split without loading images."""
    _, lbl_dir = _SPLIT_DIRS[split]
    counts: Counter[int] = Counter()
    for lbl in lbl_dir.glob("*.txt"):
        line = lbl.read_text().strip()
        if line:
            counts[int(line.split()[0])] += 1
    return {YOLO_CLASS_NAMES[k]: counts[k] for k in sorted(counts)}


def count_all_splits() -> dict[str, dict[str, int]]:
    """Return {split: {class_name: count}} for all three splits."""
    return {split: count_by_class(split) for split in ("train", "valid", "test")}  # type: ignore[arg-type]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class AffectNetYOLODataset(_Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset wrapping the AffectNet YOLO-format image set.

    Parameters
    ----------
    split         : "train" | "valid" | "test"
    neutral_only  : if True, load only Neutral (class 5) images and map label → 0
    expression_ids: optional list of class IDs to include (e.g. [5] == neutral_only shortcut)
    transform     : optional transform applied to PIL image

    Each item dict:
      image        : PIL.Image or transformed tensor
      class_id     : int  0-7  original AffectNet expression class
      expression   : str  human-readable expression name
      label        : int  0 if Neutral else class_id  (use neutral_only for clean normal set)
      bbox         : float32 tensor (4,)  [cx, cy, w, h] normalised
      filename     : str
    """

    def __init__(
        self,
        split: Split = "train",
        neutral_only: bool = False,
        expression_ids: list[int] | None = None,
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("pip install torch  — required to use AffectNetYOLODataset")

        self.imgs_dir, lbl_dir = _SPLIT_DIRS[split]
        self.transform = transform

        # Collect all samples, optionally filtered by class
        keep_ids: set[int] | None = None
        if neutral_only:
            keep_ids = {YOLO_NEUTRAL_ID}
        elif expression_ids is not None:
            keep_ids = set(expression_ids)

        self.samples: list[tuple[Path, int, np.ndarray]] = []
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            class_id, bbox = _parse_label(lbl_path)
            if keep_ids is None or class_id in keep_ids:
                img_path = self.imgs_dir / (lbl_path.stem + ".png")
                self.samples.append((img_path, class_id, bbox))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, class_id, bbox = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Neutral → label 0 (Normal); all others keep their expression class ID
        label = 0 if class_id == YOLO_NEUTRAL_ID else class_id

        return {
            "image":      image,
            "class_id":   class_id,
            "expression": YOLO_CLASS_NAMES[class_id],
            "label":      label,
            "bbox":       torch.from_numpy(bbox),
            "filename":   img_path.name,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    print("AffectNet YOLO — class distribution")
    print(f"{'Class':12s}  {'train':>6}  {'valid':>6}  {'test':>6}  {'total':>7}")
    print("-" * 46)

    all_counts = count_all_splits()
    totals: Counter[str] = Counter()
    for name in YOLO_CLASS_NAMES:
        tr = all_counts["train"].get(name, 0)
        va = all_counts["valid"].get(name, 0)
        te = all_counts["test"].get(name, 0)
        totals[name] = tr + va + te
        marker = "  <- Normal class" if name == "Neutral" else ""
        print(f"{name:12s}  {tr:>6}  {va:>6}  {te:>6}  {totals[name]:>7}{marker}")

    print("-" * 46)
    grand = sum(totals.values())
    print(f"{'TOTAL':12s}  "
          f"{sum(all_counts['train'].values()):>6}  "
          f"{sum(all_counts['valid'].values()):>6}  "
          f"{sum(all_counts['test'].values()):>6}  "
          f"{grand:>7}")

    # Sample verification
    print()
    img_path, class_id, bbox = AffectNetYOLODataset.__new__(AffectNetYOLODataset).samples[0] \
        if False else (YOLO_TRAIN_IMGS / "ffhq_0.png", None, None)
    # Use _parse_label directly — no torch needed for inspection
    lbl = YOLO_TRAIN_LBLS / "ffhq_0.txt"
    cid, bb = _parse_label(lbl)
    img = Image.open(YOLO_TRAIN_IMGS / "ffhq_0.png")
    print(f"Sample ffhq_0:  size={img.size}  class={cid} ({YOLO_CLASS_NAMES[cid]})  bbox={bb.tolist()}")
