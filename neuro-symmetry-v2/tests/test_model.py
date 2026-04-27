"""Phase 4 — model architecture, calibration, and baseline tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.calibrate_baseline import BaselineCalibrator, UserBaseline, _MIN_SAMPLES
from ai.calibrate_temperature import compute_ece
from ai.model import CalibratedModel, N_CLASSES, N_FEATURES, NeuroSymmetryNet
from ai.synthetic_augment import (
    build_dataset,
    generate_mild,
    generate_normal,
    generate_severe,
)


# ── NeuroSymmetryNet ──────────────────────────────────────────────────────────

class TestNeuroSymmetryNet:
    def test_forward_shape(self) -> None:
        model = NeuroSymmetryNet()
        x = torch.randn(8, N_FEATURES)
        out = model(x)
        assert out.shape == (8, N_CLASSES)

    def test_single_sample(self) -> None:
        # BatchNorm requires eval() for batch_size=1 — matches production usage
        model = NeuroSymmetryNet()
        model.eval()
        x = torch.randn(1, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_param_count_reasonable(self) -> None:
        model = NeuroSymmetryNet()
        # 50→128→64→32→3 with BatchNorm: roughly 15K–25K params
        assert 10_000 < model.n_params < 50_000

    def test_output_finite(self) -> None:
        model = NeuroSymmetryNet()
        x = torch.randn(16, N_FEATURES)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_eval_deterministic(self) -> None:
        model = NeuroSymmetryNet()
        model.eval()
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)

    def test_train_mode_nondeterministic_with_dropout(self) -> None:
        torch.manual_seed(0)
        model = NeuroSymmetryNet(dropout=0.9)  # heavy dropout
        model.train()
        x = torch.randn(64, N_FEATURES)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert not torch.allclose(o1, o2)


# ── CalibratedModel ───────────────────────────────────────────────────────────

class TestCalibratedModel:
    def _model(self) -> CalibratedModel:
        return CalibratedModel(NeuroSymmetryNet())

    def test_forward_shape(self) -> None:
        model = self._model()
        x = torch.randn(8, N_FEATURES)
        assert model(x).shape == (8, N_CLASSES)

    def test_default_temperature_one(self) -> None:
        model = self._model()
        assert abs(float(model.temperature.item()) - 1.0) < 1e-6

    def test_temperature_scales_logits(self) -> None:
        model = self._model()
        model.eval()
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            raw = model.base(x)
            cal = model(x)
        model.temperature.data.fill_(2.0)
        with torch.no_grad():
            cal2 = model(x)
        assert torch.allclose(cal2, raw / 2.0, atol=1e-5)

    def test_save_load_roundtrip(self) -> None:
        model = self._model()
        model.eval()
        model.temperature.data.fill_(1.23)
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            out_before = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            model.save(path)
            loaded = CalibratedModel.load(path)
            loaded.eval()
            with torch.no_grad():
                out_after = loaded(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)
        assert abs(float(loaded.temperature.item()) - 1.23) < 1e-4

    def test_calibrate_returns_positive_temperature(self) -> None:
        model = self._model()
        rng = np.random.default_rng(0)
        Xv = torch.randn(200, N_FEATURES)
        yv = torch.tensor(rng.integers(0, N_CLASSES, 200), dtype=torch.long)
        T = model.calibrate(Xv, yv)
        assert T > 0

    def test_onnx_export(self) -> None:
        model = self._model()
        model.eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.onnx"
            model.to_onnx(path)
            assert path.exists()
            assert path.stat().st_size > 0


# ── Synthetic augmentation ────────────────────────────────────────────────────

class TestSyntheticAugment:
    RNG = np.random.default_rng(99)

    def test_normal_shape(self) -> None:
        X = generate_normal(100, self.RNG)
        assert X.shape == (100, 50)
        assert X.dtype == np.float32

    def test_mild_shape(self) -> None:
        assert generate_mild(50, self.RNG).shape == (50, 50)

    def test_severe_shape(self) -> None:
        assert generate_severe(50, self.RNG).shape == (50, 50)

    def test_symmetry_error_ordering(self) -> None:
        """Normal < Mild < Severe on average symmetry_error."""
        rng = np.random.default_rng(7)
        e_norm   = generate_normal(500, rng)[:, 49].mean()
        e_mild   = generate_mild(500,   rng)[:, 49].mean()
        e_severe = generate_severe(500, rng)[:, 49].mean()
        assert e_norm < e_mild < e_severe, (
            f"expected normal<mild<severe: {e_norm:.3f} {e_mild:.3f} {e_severe:.3f}"
        )

    def test_bilateral_distances_nonnegative(self) -> None:
        rng = np.random.default_rng(1)
        for gen in [generate_normal, generate_mild, generate_severe]:
            X = gen(100, rng)
            assert (X[:, :40] >= 0).all(), f"{gen.__name__} has negative bilateral distances"

    def test_texture_score_in_range(self) -> None:
        rng = np.random.default_rng(2)
        for gen in [generate_normal, generate_mild, generate_severe]:
            X = gen(100, rng)
            assert ((X[:, 48] >= 0) & (X[:, 48] <= 1)).all()

    def test_build_dataset_balanced(self) -> None:
        X, y = build_dataset(n_per_class=200, seed=0)
        assert X.shape == (600, 50)
        assert y.shape == (600,)
        for cls in range(3):
            assert (y == cls).sum() == 200

    def test_build_dataset_shuffled(self) -> None:
        _, y = build_dataset(n_per_class=300, seed=0)
        # If not shuffled, first 300 would all be class 0
        assert not all(y[:300] == 0)


# ── ECE ───────────────────────────────────────────────────────────────────────

class TestComputeECE:
    def test_perfect_calibration_near_zero(self) -> None:
        n = 1000
        probs = np.zeros((n, 3), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        # Perfect confidence in the correct class
        probs[:, 0] = 1.0
        ece = compute_ece(probs, labels)
        assert ece < 0.01

    def test_overconfident_gives_positive_ece(self) -> None:
        n = 500
        probs = np.zeros((n, 3), dtype=np.float32)
        labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
        # Always predict class 0 with 100% confidence
        probs[:, 0] = 1.0
        ece = compute_ece(probs, labels)
        assert ece > 0.1

    def test_output_in_range(self) -> None:
        rng = np.random.default_rng(3)
        probs = rng.dirichlet([1, 1, 1], size=200).astype(np.float32)
        labels = rng.integers(0, 3, 200)
        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


# ── Personalized baseline ─────────────────────────────────────────────────────

class TestBaselineCalibrator:
    def _calibrated(self, n: int = _MIN_SAMPLES + 10) -> tuple:
        rng = np.random.default_rng(0)
        cal = BaselineCalibrator("u1")
        frames = rng.normal(0.0, 0.05, (n, 50)).astype(np.float32)
        for f in frames:
            cal.record(f)
        return cal, frames

    def test_compute_returns_baseline(self) -> None:
        cal, _ = self._calibrated()
        bl = cal.compute()
        assert isinstance(bl, UserBaseline)
        assert bl.mean.shape == (50,)
        assert bl.std.shape  == (50,)

    def test_too_few_samples_raises(self) -> None:
        cal = BaselineCalibrator("u2")
        for _ in range(_MIN_SAMPLES - 1):
            cal.record(np.zeros(50, dtype=np.float32))
        with pytest.raises(RuntimeError, match="calibration frames"):
            cal.compute()

    def test_anomaly_score_neutral_near_zero(self) -> None:
        cal, frames = self._calibrated()
        bl = cal.compute()
        # A frame drawn from the same distribution should have low anomaly score
        rng = np.random.default_rng(1)
        typical = rng.normal(0.0, 0.05, 50).astype(np.float32)
        score = bl.anomaly_score(typical)
        assert score < 3.0, f"typical frame should not be anomalous, got {score:.2f}"

    def test_anomaly_score_deviant_high(self) -> None:
        cal, _ = self._calibrated()
        bl = cal.compute()
        deviant = np.full(50, 10.0, dtype=np.float32)  # far from calibration mean
        score = bl.anomaly_score(deviant)
        assert score > 5.0, f"deviant frame should be highly anomalous, got {score:.2f}"

    def test_save_load_roundtrip(self) -> None:
        cal, _ = self._calibrated()
        bl = cal.compute()
        with tempfile.TemporaryDirectory() as d:
            bl.save(Path(d), "u1")
            bl2 = UserBaseline.load(Path(d), "u1")
        assert np.allclose(bl.mean, bl2.mean, atol=1e-6)
        assert np.allclose(bl.std,  bl2.std,  atol=1e-6)
        assert bl2.user_id == "u1"

    def test_n_samples_counter(self) -> None:
        cal = BaselineCalibrator("u3")
        assert cal.n_samples == 0
        cal.record(np.zeros(50, dtype=np.float32))
        assert cal.n_samples == 1

    def test_reset_clears_frames(self) -> None:
        cal, _ = self._calibrated()
        cal.reset()
        assert cal.n_samples == 0
