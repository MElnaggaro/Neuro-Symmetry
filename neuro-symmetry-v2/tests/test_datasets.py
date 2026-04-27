"""Phase 1 smoke tests — dataset loaders (no torch required)."""

from datasets.celeba_loader import count_by_split, _load_landmarks
from datasets.config import CELEBA_ANNO, CELEBA_EVAL, verify_paths
from datasets.palsy_loader import _collect_samples
from datasets.w300_loader import _parse_pts
from datasets.yolo_affectnet_loader import count_by_class
from datasets.config import YFP_ROOT, W300_AFW, YOLO_ROOT


def test_paths_exist() -> None:
    verify_paths()


def test_celeba_split_counts() -> None:
    counts = count_by_split()
    assert counts["train"] == 162_770
    assert counts["val"]   ==  19_867
    assert counts["test"]  ==  19_962
    assert sum(counts.values()) == 202_599


def test_celeba_landmark_shape() -> None:
    lm = _load_landmarks(CELEBA_ANNO / "list_landmarks_align_celeba.txt")
    assert len(lm) == 202_599
    first = next(iter(lm.values()))
    assert first.shape == (5, 2)


def test_w300_pts_parsing() -> None:
    pts_files = sorted(W300_AFW.glob("*.pts"))
    assert len(pts_files) == 100
    lm = _parse_pts(pts_files[0])
    assert lm.shape == (68, 2)


def test_palsy_3class_totals() -> None:
    samples = _collect_samples(YFP_ROOT, fine_grained=False)
    assert len(samples) == 14_389
    labels = {s[1] for s in samples}
    assert labels == {1, 2}


def test_palsy_4class_totals() -> None:
    samples = _collect_samples(YFP_ROOT, fine_grained=True)
    assert len(samples) == 14_389
    labels = {s[1] for s in samples}
    assert labels == {0, 1, 2, 3}


def test_yolo_train_total() -> None:
    counts = count_by_class("train")
    assert sum(counts.values()) == 17_101
    assert "Neutral" in counts
