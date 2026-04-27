"""
YFP (Yale Facial Palsy) dataset loader.

Directory structure:
    YFP_Dataset/
        Eye/
            Mild eye/              *.bmp   (611)
            Moderate eye/          *.bmp   (2,246)
            Moderate severe eye/   *.bmp   (1,022)
            Severe eye/            *.bmp   (334)
        Eyebrow/   (721 / 2,832 / 2,284 / 438)
        Mouth/     (603 / 576 / 2,308 / 414)

Two label modes (set fine_grained=True to switch):

  Default — 3-class (aligns with main classifier):
    Mild + Moderate         → 1  (Mild)
    Moderate severe + Severe → 2  (Severe)

  Fine-grained — 4-class (use for severity sub-classifier):
    Mild             → 0
    Moderate         → 1
    Moderate severe  → 2
    Severe           → 3

Images are cropped facial-region patches, not full-face images.
Torch is optional at import time.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image

from datasets.config import YFP_REGIONS, YFP_ROOT, YFP_SEVERITIES

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except ImportError:
    _Dataset = object  # type: ignore[assignment,misc]
    _TORCH = False

# ── Label maps ────────────────────────────────────────────────────────────────

# 3-class: Mild/Moderate → 1 (Mild),  Moderate severe/Severe → 2 (Severe)
_LABEL_3CLASS: dict[str, int] = {
    "mild":            1,
    "moderate":        1,
    "moderate severe": 2,
    "severe":          2,
}

# 4-class: each severity gets its own index (0–3)
_LABEL_4CLASS: dict[str, int] = {
    "mild":            0,
    "moderate":        1,
    "moderate severe": 2,
    "severe":          3,
}


def _infer_label(folder_name: str, fine_grained: bool) -> int:
    lower = folder_name.lower()
    label_map = _LABEL_4CLASS if fine_grained else _LABEL_3CLASS
    # Match longest prefix first so "moderate severe" beats "moderate"
    for key in sorted(label_map, key=len, reverse=True):
        if lower.startswith(key):
            return label_map[key]
    return 0 if fine_grained else 1  # fallback


def _collect_samples(
    root: Path,
    fine_grained: bool,
) -> list[tuple[Path, int, str, str]]:
    """Return flat list of (image_path, label, region, severity)."""
    samples: list[tuple[Path, int, str, str]] = []
    for region in YFP_REGIONS:
        region_dir = root / region
        if not region_dir.exists():
            continue
        for sev_dir in sorted(region_dir.iterdir()):
            if not sev_dir.is_dir():
                continue
            label    = _infer_label(sev_dir.name, fine_grained)
            severity = sev_dir.name  # e.g. "Mild eye"
            for img_path in sorted(sev_dir.glob("*.bmp")):
                samples.append((img_path, label, region, severity))
    return samples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PalsyDataset(_Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset wrapping the YFP facial palsy image set.

    Parameters
    ----------
    root         : dataset root (defaults to config.YFP_ROOT)
    regions      : filter to specific regions, e.g. ["Eye", "Mouth"]
    fine_grained : False (default) → 2-class labels {1, 2}
                   True            → 4-class labels {0, 1, 2, 3}
    transform    : optional transform applied to PIL image

    Each item dict:
      image    : PIL.Image or tensor
      label    : int
      region   : str  — "Eye" | "Eyebrow" | "Mouth"
      severity : str  — original folder name, e.g. "Moderate severe eye"
      filename : str
    """

    def __init__(
        self,
        root: Path = YFP_ROOT,
        regions: list[str] | None = None,
        fine_grained: bool = False,
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("pip install torch  — required to use PalsyDataset")
        self.transform = transform
        all_samples    = _collect_samples(root, fine_grained)
        self.samples   = (
            [(p, l, r, s) for p, l, r, s in all_samples if r in regions]
            if regions else all_samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label, region, severity = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image":    image,
            "label":    label,
            "region":   region,
            "severity": severity,
            "filename": img_path.name,
        }


if __name__ == "__main__":
    from datasets.config import verify_paths

    verify_paths()

    for mode, fg in (("3-class (default)", False), ("4-class (fine_grained)", True)):
        samples = _collect_samples(YFP_ROOT, fg)
        label_counts  = Counter(l for _, l, _, _ in samples)
        region_counts = Counter(r for _, _, r, _ in samples)
        sev_counts    = Counter(s for _, _, _, s in samples)

        print(f"\nYFP {mode} — {len(samples)} samples")
        print(f"  labels  : { {k: label_counts[k] for k in sorted(label_counts)} }")
        print(f"  regions : { {r: region_counts[r] for r in sorted(region_counts)} }")

    print()
    samples_3 = _collect_samples(YFP_ROOT, fine_grained=False)
    img_path, label, region, severity = samples_3[0]
    img = Image.open(img_path)
    print(f"Sample : {img_path.name}")
    print(f"  size={img.size}  mode={img.mode}  label={label}  region={region}  severity={severity!r}")
