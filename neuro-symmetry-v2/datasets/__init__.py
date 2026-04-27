from __future__ import annotations


def __getattr__(name: str) -> object:
    if name == "CelebADataset":
        from datasets.celeba_loader import CelebADataset
        return CelebADataset
    if name == "PalsyDataset":
        from datasets.palsy_loader import PalsyDataset
        return PalsyDataset
    if name == "W300Dataset":
        from datasets.w300_loader import W300Dataset
        return W300Dataset
    raise AttributeError(f"module 'datasets' has no attribute {name!r}")


__all__ = ["CelebADataset", "PalsyDataset", "W300Dataset"]
