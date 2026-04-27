from pathlib import Path

# Root of the Neuro-Symmetry repository (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared datasets folder sitting next to the project root
DATASETS_ROOT = PROJECT_ROOT.parent / "Datasets"

# ── Per-dataset paths ─────────────────────────────────────────────────────────

CELEBA_ROOT    = DATASETS_ROOT / "Celeba"
CELEBA_IMGS    = CELEBA_ROOT / "img" / "img_align_celeba_png"
CELEBA_ANNO    = CELEBA_ROOT / "Anno"
CELEBA_EVAL    = CELEBA_ROOT / "Eval"

W300_ROOT      = DATASETS_ROOT / "ibug_300W_large_face_landmark_dataset"
W300_AFW       = W300_ROOT / "afw"

YFP_ROOT       = DATASETS_ROOT / "YFP_Dataset"

# AffectNet is not downloaded yet — stub path for future use
AFFECTNET_ROOT = DATASETS_ROOT / "AffectNet"


def verify_paths() -> None:
    """Raise FileNotFoundError for any dataset root that does not exist."""
    required = {
        "CelebA":    CELEBA_ROOT,
        "300-W AFW": W300_AFW,
        "YFP":       YFP_ROOT,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found at {path}")
