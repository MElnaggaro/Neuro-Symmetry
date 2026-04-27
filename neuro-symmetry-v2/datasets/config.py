from pathlib import Path

# Root of the Neuro-Symmetry repository (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared datasets folder sitting next to the project root
DATASETS_ROOT = PROJECT_ROOT.parent / "Datasets"

# ── CelebA ────────────────────────────────────────────────────────────────────
#
# Three image sets — all 202,599 images, different formats / alignment:
#
#   img_align_celeba     JPG  178x218  eye-aligned & cropped  (default for training)
#   img_align_celeba_png PNG  178x218  same alignment, lossless (texture analysis)
#   img_celeba           JPG  varies   original in-the-wild images (pose variety)
#
# Landmark annotation files differ by image set:
#   list_landmarks_align_celeba.txt  →  img_align_celeba / img_align_celeba_png
#   list_landmarks_celeba.txt        →  img_celeba (in-the-wild coords)

CELEBA_ROOT         = DATASETS_ROOT / "Celeba"
CELEBA_ANNO         = CELEBA_ROOT / "Anno"
CELEBA_EVAL         = CELEBA_ROOT / "Eval"

CELEBA_IMGS_ALIGNED     = CELEBA_ROOT / "img" / "img_align_celeba"      # JPG, aligned
CELEBA_IMGS_ALIGNED_PNG = CELEBA_ROOT / "img" / "img_align_celeba_png"  # PNG, aligned
CELEBA_IMGS_WILD        = CELEBA_ROOT / "img" / "img_celeba"             # JPG, in-the-wild

CELEBA_LM_ALIGNED   = CELEBA_ANNO / "list_landmarks_align_celeba.txt"
CELEBA_LM_WILD      = CELEBA_ANNO / "list_landmarks_celeba.txt"

CELEBA_IMGS         = CELEBA_IMGS_ALIGNED   # default

# ── 300-W ─────────────────────────────────────────────────────────────────────
#
# AFW subset — 200 images with 68-point landmark annotations (.pts files).
# Used to validate landmark engine accuracy.

W300_ROOT = DATASETS_ROOT / "ibug_300W_large_face_landmark_dataset"
W300_AFW  = W300_ROOT / "afw"

# ── YFP Palsy ─────────────────────────────────────────────────────────────────
#
# Yale Facial Palsy — 14,389 cropped region images across 3 regions and
# 4 severity grades:
#
#   Region   │ Mild   Moderate  Mod.Severe  Severe   Total
#   ─────────┼────────────────────────────────────────────
#   Eye      │  611     2,246      1,022      334     4,213
#   Eyebrow  │  721     2,832      2,284      438     6,275
#   Mouth    │  603       576      2,308      414     3,901
#   Total    │ 1,935    5,654      5,614      1,186  14,389
#
# Default 3-class mapping for our classifier:
#   Mild + Moderate         → label 1  (Mild)
#   Moderate severe + Severe → label 2  (Severe)
#
# fine_grained=True exposes all 4 severity levels (labels 0-3).

YFP_ROOT = DATASETS_ROOT / "YFP_Dataset"

YFP_REGIONS    = ("Eye", "Eyebrow", "Mouth")
YFP_SEVERITIES = ("Mild", "Moderate", "Moderate severe", "Severe")

# ── AffectNet — YOLO format ───────────────────────────────────────────────────
#
# AffectNet subset in YOLO detection format (source: Kaggle affectnet-yolo-format).
# 25,262 images total — 96x96 PNG, one face bounding box per image.
#
#   Split   Images
#   train   17,101
#   valid    5,406
#   test     2,755
#
# 8 expression classes (nc=8):
#   ID  Name       Count(train)   Project use
#   0   Anger        2,339        expression variation
#   1   Contempt     1,996        expression variation
#   2   Disgust      2,242        expression variation
#   3   Fear         2,021        expression variation
#   4   Happy        2,154        expression variation
#   5   Neutral      1,616        → label 0 (Normal) for classifier training
#   6   Sad          1,914        expression variation
#   7   Surprise     2,819        expression variation
#
# Label file format:  class_id  cx  cy  w  h   (all normalised to [0,1])
# Note: uses "valid" (not "val") for the validation split.

YOLO_ROOT        = DATASETS_ROOT / "YOLO_format"
YOLO_DATA_YAML   = YOLO_ROOT / "data.yaml"

YOLO_TRAIN_IMGS  = YOLO_ROOT / "train" / "images"
YOLO_TRAIN_LBLS  = YOLO_ROOT / "train" / "labels"
YOLO_VALID_IMGS  = YOLO_ROOT / "valid" / "images"
YOLO_VALID_LBLS  = YOLO_ROOT / "valid" / "labels"
YOLO_TEST_IMGS   = YOLO_ROOT / "test"  / "images"
YOLO_TEST_LBLS   = YOLO_ROOT / "test"  / "labels"

YOLO_CLASS_NAMES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Neutral",  "Sad",     "Surprise",
]
YOLO_NEUTRAL_ID  = 5   # the "normal face" class in our classifier


# ── Path verification ─────────────────────────────────────────────────────────

def verify_paths() -> None:
    """Raise FileNotFoundError for any required dataset root that does not exist."""
    required = {
        "CelebA":       CELEBA_ROOT,
        "300-W AFW":    W300_AFW,
        "YFP":          YFP_ROOT,
        "AffectNet YOLO": YOLO_ROOT,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found at {path}")
