"""
300-W (AFW subset) dataset loader — 200 annotated face images.

Each image has a paired .pts file with 68 facial landmark coordinates.
Used to validate landmark accuracy in the landmark engine.

.pts format:
    version: 1
    n_points:  68
    {
    x1 y1
    ...
    x68 y68
    }

Torch is optional at import time; only required when W300Dataset.__getitem__
is called during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from datasets.config import W300_AFW

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False


def _parse_pts(pts_path: Path) -> np.ndarray:
    """Parse a .pts landmark file → float32 array of shape (68, 2)."""
    points: list[list[float]] = []
    with pts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("{", "}", "version", "n_points")):
                continue
            x, y = map(float, line.split())
            points.append([x, y])
    return np.array(points, dtype=np.float32)


class W300Dataset(_Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset wrapping the AFW subset of 300-W.

    Each item:
      image     : PIL.Image or tensor
      landmarks : float32 tensor (68, 2) — pixel coords
      label     : int — always 0 (used for landmark accuracy eval only)
      filename  : str
    """

    def __init__(
        self,
        root: Path = W300_AFW,
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("torch required — pip install torch")
        self.root      = root
        self.transform = transform
        self.samples   = sorted(root.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        pts_path = img_path.with_suffix(".pts")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        lm_np = _parse_pts(pts_path) if pts_path.exists() else np.zeros((68, 2), np.float32)

        return {
            "image":     image,
            "landmarks": torch.from_numpy(lm_np),
            "label":     0,
            "filename":  img_path.name,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    samples = sorted(W300_AFW.glob("*.jpg"))
    pts_files = sorted(W300_AFW.glob("*.pts"))
    print(f"300-W AFW: {len(samples):>4} images  |  {len(pts_files):>4} .pts files")

    if samples:
        first_pts = samples[0].with_suffix(".pts")
        lm = _parse_pts(first_pts)
        print(f"\nFirst sample  : {samples[0].name}")
        print(f"Landmarks     : shape {lm.shape},  first point = {lm[0].tolist()}")
        img = Image.open(samples[0])
        print(f"Image size    : {img.size}  mode={img.mode}")
