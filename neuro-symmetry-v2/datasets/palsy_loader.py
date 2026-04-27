"""
YFP (Yale Facial Palsy) dataset loader.

Directory structure on disk:
    YFP_Dataset/
        Eye/
            Mild eye/              *.bmp
            Moderate eye/          *.bmp
            Moderate severe eye/   *.bmp
            Severe eye/            *.bmp
        Eyebrow/
            Mild eyebrow/ ...
        Mouth/
            Mild mouth/ ...

Severity → class label (aligns with the 3-class classifier):
    Mild             → 1  (Mild)
    Moderate         → 1  (Mild — borderline)
    Moderate severe  → 2  (Severe)
    Severe           → 2  (Severe)

Images are cropped region patches, not full-face images.
Used for texture / region-level asymmetry training (Phase 4).

Torch is optional at import time.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image

from datasets.config import YFP_ROOT

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False

# ── Severity mapping ──────────────────────────────────────────────────────────

_SEVERITY_LABEL: dict[str, int] = {
    "mild":            1,
    "moderate":        1,
    "moderate severe": 2,
    "severe":          2,
}

_REGIONS = ("Eye", "Eyebrow", "Mouth")


def _infer_label(folder_name: str) -> int:
    lower = folder_name.lower()
    # Match longest prefix first so "moderate severe" beats "moderate"
    for key in sorted(_SEVERITY_LABEL, key=len, reverse=True):
        if lower.startswith(key):
            return _SEVERITY_LABEL[key]
    return 1


def _collect_samples(root: Path) -> list[tuple[Path, int, str]]:
    """Return flat list of (image_path, label, region)."""
    samples: list[tuple[Path, int, str]] = []
    for region in _REGIONS:
        region_dir = root / region
        if not region_dir.exists():
            continue
        for sev_dir in sorted(region_dir.iterdir()):
            if not sev_dir.is_dir():
                continue
            label = _infer_label(sev_dir.name)
            for img_path in sorted(sev_dir.glob("*.bmp")):
                samples.append((img_path, label, region))
    return samples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PalsyDataset(_Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset wrapping the YFP facial palsy image set.

    Each item:
      image    : PIL.Image or tensor
      label    : int — 1 (Mild) or 2 (Severe)
      region   : str — "Eye" | "Eyebrow" | "Mouth"
      filename : str
    """

    def __init__(
        self,
        root: Path = YFP_ROOT,
        regions: list[str] | None = None,
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("torch required — pip install torch")
        self.transform = transform
        all_samples    = _collect_samples(root)
        self.samples   = (
            [(p, l, r) for p, l, r in all_samples if r in regions]
            if regions else all_samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label, region = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image":    image,
            "label":    label,
            "region":   region,
            "filename": img_path.name,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    samples = _collect_samples(YFP_ROOT)
    label_counts  = Counter(l for _, l, _ in samples)
    region_counts = Counter(r for _, _, r in samples)

    print(f"YFP total : {len(samples)} samples")
    print(f"  by label  : { {k: label_counts[k] for k in sorted(label_counts)} }")
    print(f"  by region : { {r: region_counts[r] for r in sorted(region_counts)} }")

    if samples:
        img_path, label, region = samples[0]
        img = Image.open(img_path)
        print(f"\nFirst sample  : {img_path.name}")
        print(f"Image size    : {img.size}  mode={img.mode}")
        print(f"Label         : {label}  region={region}")
