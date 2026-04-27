"""
CelebA dataset loader — 202,599 face images in three variants.

Image sets (select via image_set parameter):
  "aligned"     JPG  178x218  eye-aligned & cropped  (default, best for training)
  "aligned_png" PNG  178x218  same alignment, lossless (best for texture analysis)
  "wild"        JPG  varies   original in-the-wild images (use when pose variety needed)

Split logic follows the official eval partition file:
  0 = train  |  1 = val  |  2 = test

Torch is optional at import time — only required when CelebADataset.__getitem__
is called (Phase 4 training). The __main__ block runs on stdlib + numpy only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from datasets.config import (
    CELEBA_ANNO,
    CELEBA_EVAL,
    CELEBA_IMGS_ALIGNED,
    CELEBA_IMGS_ALIGNED_PNG,
    CELEBA_IMGS_WILD,
    CELEBA_LM_ALIGNED,
    CELEBA_LM_WILD,
)

# ── Optional torch import ─────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False

# ── Types ─────────────────────────────────────────────────────────────────────

Split     = Literal["train", "val", "test", "all"]
ImageSet  = Literal["aligned", "aligned_png", "wild"]

_SPLIT_ID: dict[str, int | None] = {"train": 0, "val": 1, "test": 2, "all": None}

# Maps image_set name → (image directory, landmark annotation file)
_IMAGE_SET_MAP: dict[str, tuple[Path, Path]] = {
    "aligned":     (CELEBA_IMGS_ALIGNED,     CELEBA_LM_ALIGNED),
    "aligned_png": (CELEBA_IMGS_ALIGNED_PNG, CELEBA_LM_ALIGNED),
    "wild":        (CELEBA_IMGS_WILD,        CELEBA_LM_WILD),
}

LANDMARK_COLS = [
    "lefteye_x", "lefteye_y",
    "righteye_x", "righteye_y",
    "nose_x",     "nose_y",
    "leftmouth_x", "leftmouth_y",
    "rightmouth_x", "rightmouth_y",
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
    """Return {filename: float32 array shape (5, 2)}."""
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
    PyTorch Dataset wrapping CelebA images.

    Parameters
    ----------
    split      : "train" | "val" | "test" | "all"
    image_set  : "aligned" (default) | "aligned_png" | "wild"
                 Selects which image folder and landmark file to use.
    transform  : optional torchvision transform applied to each PIL image
    load_images: set False to skip disk I/O (useful for counting / dry-runs)

    Each item dict:
      image      : PIL.Image or transformed tensor
      landmarks  : float32 tensor (5, 2) — pixel coords
      attributes : int8 tensor   (40,)   — {-1, 1}
      label      : int  — always 0 (normal class)
      image_set  : str  — which folder was used
      filename   : str
    """

    def __init__(
        self,
        split: Split = "train",
        image_set: ImageSet = "aligned",
        transform: Callable | None = None,
        load_images: bool = True,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("pip install torch  — required to use CelebADataset")
        if image_set not in _IMAGE_SET_MAP:
            raise ValueError(f"image_set must be one of {list(_IMAGE_SET_MAP)}, got {image_set!r}")

        self.imgs_dir, lm_file = _IMAGE_SET_MAP[image_set]
        self.image_set   = image_set
        self.transform   = transform
        self.load_images = load_images

        partition   = _load_partition(CELEBA_EVAL / "list_eval_partition.txt")
        self._lm    = _load_landmarks(lm_file)
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
            # aligned_png images are .png on disk but annotation keys are .jpg
            disk_name = Path(fname).stem + ".png" if self.image_set == "aligned_png" else fname
            img: Image.Image | object = Image.open(self.imgs_dir / disk_name).convert("RGB")
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
            "image_set":  self.image_set,
            "filename":   fname,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    # ── Split counts ──────────────────────────────────────────────────────────
    counts = count_by_split()
    for split, n in counts.items():
        print(f"CelebA {split:5s}: {n:>7,} samples")
    print(f"CelebA total: {sum(counts.values()):>7,} samples")

    # ── Per image-set sample verification ─────────────────────────────────────
    print()
    lm_aligned = _load_landmarks(CELEBA_LM_ALIGNED)
    lm_wild    = _load_landmarks(CELEBA_LM_WILD)
    first_key  = next(iter(lm_aligned))

    image_sets = [
        ("aligned",     CELEBA_IMGS_ALIGNED,     first_key),
        ("aligned_png", CELEBA_IMGS_ALIGNED_PNG, Path(first_key).stem + ".png"),
        ("wild",        CELEBA_IMGS_WILD,         first_key),
    ]

    for name, imgs_dir, disk_name in image_sets:
        img_path = imgs_dir / disk_name
        if img_path.exists():
            img = Image.open(img_path)
            print(f"{name:15s}  size={str(img.size):12s}  mode={img.mode}  format={img.format}")
        else:
            print(f"{name:15s}  WARNING: not found at {img_path}")
