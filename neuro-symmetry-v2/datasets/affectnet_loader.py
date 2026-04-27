"""
AffectNet loader — redirect module.

The AffectNet subset used in this project is stored in YOLO detection format
under Datasets/YOLO_format/ (sourced from Kaggle affectnet-yolo-format).

Use AffectNetYOLODataset from yolo_affectnet_loader instead:

    from datasets.yolo_affectnet_loader import AffectNetYOLODataset

    # All expressions — 25,262 images
    ds = AffectNetYOLODataset(split="train")

    # Neutral only — maps to label 0 (Normal class)
    ds_normal = AffectNetYOLODataset(split="train", neutral_only=True)

This stub is kept so that any code referencing affectnet_loader raises a clear
error rather than a silent import failure.
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name == "AffectNetDataset":
        raise ImportError(
            "AffectNetDataset is not implemented here. "
            "Use: from datasets.yolo_affectnet_loader import AffectNetYOLODataset"
        )
    raise AttributeError(f"module 'datasets.affectnet_loader' has no attribute {name!r}")


if __name__ == "__main__":
    print("AffectNet data is available via datasets.yolo_affectnet_loader.")
    print("Run:  python -m datasets.yolo_affectnet_loader")
