"""
AffectNet dataset loader — stub.

AffectNet (~450k images) is not yet downloaded.
This module exists as a placeholder so import paths don't break during
Phase 1. Full implementation belongs to Phase 4 (AI Training).

Expected on-disk layout when downloaded:
    Datasets/AffectNet/
        Manually_Annotated_Images/   (train images)
        Manually_Annotated_file_list/
            training.csv
            validation.csv
        Automatically_Annotated_Images/

Expression labels used in this project (subset):
    0 = Neutral  ← kept (maps to normal class)
    6 = Fear     ← filtered out (not neurologically relevant)
    7 = Contempt ← filtered out

See: http://mohammadmahoor.com/affectnet/
"""

from __future__ import annotations

from pathlib import Path

from datasets.config import AFFECTNET_ROOT


def is_available() -> bool:
    return AFFECTNET_ROOT.exists() and any(AFFECTNET_ROOT.iterdir())


class AffectNetDataset:
    def __init__(self, *args: object, **kwargs: object) -> None:
        if not is_available():
            raise RuntimeError(
                f"AffectNet dataset not found at {AFFECTNET_ROOT}. "
                "Download it from http://mohammadmahoor.com/affectnet/ "
                "and place it under Datasets/AffectNet/."
            )
        raise NotImplementedError("AffectNet loader — full implementation in Phase 4.")


if __name__ == "__main__":
    if is_available():
        print(f"AffectNet found at {AFFECTNET_ROOT}")
    else:
        print(f"AffectNet NOT downloaded (expected: {AFFECTNET_ROOT})")
        print("Stub only — full loader implemented in Phase 4.")
