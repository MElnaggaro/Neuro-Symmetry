"""
CelebA dataset loader — 202,599 aligned & cropped face images.

Used as the "normal" class (label 0) for pretraining the symmetry classifier.
Also exposes 5-point landmarks and 40 binary attribute annotations.

Split logic follows the official eval partition file:
  0 = train  |  1 = val  |  2 = test

Torch is an optional dependency at import time; it is only required when
CelebADataset.__getitem__ is called (i.e. during actual training in Phase 4).
The __main__ block works with stdlib + numpy only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from datasets.config import CELEBA_ANNO, CELEBA_EVAL, CELEBA_IMGS

# ── Optional torch import ─────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False

# ── Types ─────────────────────────────────────────────────────────────────────

Split = Literal["train", "val", "test", "all"]
_SPLIT_ID: dict[str, int | None] = {"train": 0, "val": 1, "test": 2, "all": None}

LANDMARK_COLS = [
    "lefteye_x", "lefteye_y",
    "righteye_x", "righteye_y",
    "nose_x",     "nose_y",
    "leftmouth_x","leftmouth_y",
    "rightmouth_x","rightmouth_y",
]

# ── Standalone helpers (no torch required) ────────────────────────────────────

def _load_partition(eval_file: Path) -> dict[str, int]:
    """Return {filename: partition_id} from list_eval_partition.txt."""
    result: dict[str, int] = {}
    with eval_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                result[parts[0]] = int(parts[1])
    return result


def _load_landmarks(landmark_file: Path) -> dict[str, np.ndarray]:
    """Return {filename: float32 array shape (5, 2)} from aligned landmarks file."""
    result: dict[str, np.ndarray] = {}
    with landmark_file.open() as f:
        next(f)  # total count line
        next(f)  # header line
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 11:
                result[tokens[0]] = np.array(tokens[1:], dtype=np.float32).reshape(5, 2)
    return result


def _load_attributes(attr_file: Path) -> dict[str, list[int]]:
    """Return {filename: list[int]} — 40 attributes each {-1, 1}."""
    result: dict[str, list[int]] = {}
    with attr_file.open() as f:
        next(f)  # total count line
        next(f)  # attribute names header
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 41:
                result[tokens[0]] = [int(v) for v in tokens[1:]]
    return result


def count_by_split() -> dict[str, int]:
    """Return sample counts per split without loading any images."""
    partition = _load_partition(CELEBA_EVAL / "list_eval_partition.txt")
    raw = Counter(partition.values())
    return {"train": raw[0], "val": raw[1], "test": raw[2]}


# ── PyTorch Dataset (requires torch) ─────────────────────────────────────────

class CelebADataset(_Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset wrapping CelebA aligned & cropped images.

    Each item is a dict:
      image      : PIL.Image or transformed tensor
      landmarks  : float32 tensor (5, 2) — pixel coords
      attributes : int8 tensor   (40,)   — {-1, 1}
      label      : int — always 0 (normal class)
      filename   : str
    """

    def __init__(
        self,
        split: Split = "train",
        transform: Callable | None = None,
        load_images: bool = True,
    ) -> None:
        if not _TORCH:
            raise RuntimeError(
                "torch is required to instantiate CelebADataset. "
                "Install it with: pip install torch"
            )
        self.transform   = transform
        self.load_images = load_images

        partition   = _load_partition(CELEBA_EVAL / "list_eval_partition.txt")
        self._lm    = _load_landmarks(CELEBA_ANNO / "list_landmarks_align_celeba.txt")
        self._attrs = _load_attributes(CELEBA_ANNO / "list_attr_celeba.txt")

        split_id = _SPLIT_ID[split]
        self.filenames = sorted(
            k for k, v in partition.items()
            if split_id is None or v == split_id
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]

        if self.load_images:
            # Annotation files use .jpg names; on-disk images are .png
            img_name = Path(fname).stem + ".png"
            img: Image.Image | object = Image.open(CELEBA_IMGS / img_name).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.zeros(3, 218, 178)

        lm_np   = self._lm.get(fname, np.zeros((5, 2), np.float32))
        attr_np = self._attrs.get(fname, [0] * 40)

        return {
            "image":      img,
            "landmarks":  torch.from_numpy(lm_np),
            "attributes": torch.tensor(attr_np, dtype=torch.int8),
            "label":      0,
            "filename":   fname,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    counts = count_by_split()
    for split, n in counts.items():
        print(f"CelebA {split:5s}: {n:>7,} samples")
    print(f"CelebA total: {sum(counts.values()):>7,} samples")

    # Sanity-check one landmark record (no torch required)
    lm = _load_landmarks(CELEBA_ANNO / "list_landmarks_align_celeba.txt")
    first_key = next(iter(lm))
    print(f"\nFirst landmark entry  : {first_key}  ->  {lm[first_key].tolist()}")

    # Verify image file exists (annotation keys are .jpg; files on disk are .png)
    sample_img = CELEBA_IMGS / (Path(first_key).stem + ".png")
    if sample_img.exists():
        img = Image.open(sample_img)
        print(f"Sample image size     : {img.size}  mode={img.mode}")
    else:
        print(f"WARNING: image not found at {sample_img}")
