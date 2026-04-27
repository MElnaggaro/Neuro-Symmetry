"""
Phase 5 tests — change detection, decision engine, confirmation layer, API.

Runs without a landmark model or ONNX model so CI remains fast.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# ChangeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeDetector:
    from backend.api.change_detection import ChangeDetector

    def _make(self, **kw):
        from backend.api.change_detection import ChangeDetector
        return ChangeDetector(**kw)

    def test_not_ready_before_warmup(self):
        cd = self._make(warmup=10, window=20)
        for i in range(9):
            result = cd.update(0.9)
            assert result is None
        assert not cd.ready

    def test_ready_after_warmup(self):
        cd = self._make(warmup=5, window=20)
        for _ in range(5):
            cd.update(0.9)
        assert cd.ready

    def test_no_event_on_stable_scores(self):
        cd = self._make(warmup=10, window=30, z_threshold=2.5)
        for _ in range(30):
            r = cd.update(0.9)
            assert r is None

    def test_event_on_sharp_drop(self):
        cd = self._make(warmup=10, window=30, z_threshold=2.0)
        for _ in range(29):
            cd.update(0.9)
        event = cd.update(0.1)   # sudden drop
        assert event is not None
        assert event.score == pytest.approx(0.1)
        assert event.z_score < -2.0

    def test_no_event_when_all_scores_identical(self):
        cd = self._make(warmup=5, window=10, z_threshold=1.0)
        for _ in range(10):
            r = cd.update(0.5)
        assert r is None   # std=0 → guard prevents division

    def test_reset_clears_buffer(self):
        cd = self._make(warmup=5, window=10)
        for _ in range(10):
            cd.update(0.9)
        assert cd.ready
        cd.reset()
        assert not cd.ready
        assert cd.n_frames == 0

    def test_n_frames_tracks_count(self):
        cd = self._make(window=20)
        for i in range(15):
            cd.update(0.8)
        assert cd.n_frames == 15

    def test_window_caps_buffer(self):
        cd = self._make(window=10, warmup=5)
        for i in range(50):
            cd.update(0.8)
        assert cd.n_frames == 10


# ─────────────────────────────────────────────────────────────────────────────
# DecisionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionEngine:

    def _make_features(self, symmetry_error: float = 0.1) -> np.ndarray:
        """Minimal 50-dim feature vector with controlled symmetry_error [49]."""
        f = np.zeros(50, dtype=np.float32)
        f[49] = symmetry_error
        return f

    def test_threshold_normal(self):
        from backend.api.decision_engine import _threshold_decision
        f = self._make_features(symmetry_error=0.05)   # score ≈ 0.95
        r = _threshold_decision(f)
        assert r.class_id == 0
        assert r.class_label == "Normal"
        assert r.source == "threshold"

    def test_threshold_mild(self):
        from backend.api.decision_engine import _threshold_decision
        f = self._make_features(symmetry_error=0.35)   # score ≈ 0.70 → Mild
        r = _threshold_decision(f)
        assert r.class_id == 1
        assert r.class_label == "Mild"

    def test_threshold_severe(self):
        from backend.api.decision_engine import _threshold_decision
        f = self._make_features(symmetry_error=0.75)   # score ≈ 0.47 → Severe
        r = _threshold_decision(f)
        assert r.class_id == 2
        assert r.class_label == "Severe"

    def test_probabilities_sum_to_one(self):
        from backend.api.decision_engine import _threshold_decision
        for err in [0.05, 0.35, 0.75]:
            f = self._make_features(symmetry_error=err)
            r = _threshold_decision(f)
            assert abs(sum(r.probabilities) - 1.0) < 1e-6

    def test_engine_uses_fallback_when_no_model(self, tmp_path):
        from backend.api.decision_engine import DecisionEngine
        engine = DecisionEngine(model_dir=tmp_path)
        assert not engine.has_model
        f = self._make_features(0.05)
        r = engine.infer(f)
        assert r.source == "threshold"

    def test_softmax(self):
        from backend.api.decision_engine import _softmax
        logits = np.array([1.0, 2.0, 3.0])
        p = _softmax(logits)
        assert abs(p.sum() - 1.0) < 1e-6
        assert p[2] > p[1] > p[0]

    def test_infer_accepts_flat_or_batched(self, tmp_path):
        from backend.api.decision_engine import DecisionEngine
        engine = DecisionEngine(model_dir=tmp_path)
        f_flat    = self._make_features(0.05)
        f_batched = f_flat.reshape(1, 50)
        r1 = engine.infer(f_flat)
        r2 = engine.infer(f_batched)
        assert r1.class_id == r2.class_id


# ─────────────────────────────────────────────────────────────────────────────
# ConfirmationLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmationLayer:

    def _make(self, **kw):
        from backend.api.confirmation_layer import ConfirmationLayer
        return ConfirmationLayer(**kw)

    def test_initial_state_is_normal(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make()
        assert cl.state == ConfirmationState.NORMAL

    def test_transitions_to_pending_on_first_alert(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make()
        ev = cl.update(1)
        assert cl.state == ConfirmationState.PENDING
        assert ev is not None
        assert ev.class_id == 1

    def test_ok_in_pending_resets_to_normal(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make(confirm_frames=5)
        cl.update(1)   # → PENDING
        ev = cl.update(0)   # single OK → back to NORMAL
        assert cl.state == ConfirmationState.NORMAL
        assert ev is not None
        assert ev.state == ConfirmationState.NORMAL

    def test_confirmed_after_enough_alerts(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make(confirm_frames=3)
        cl.update(2)  # → PENDING
        cl.update(2)
        ev = cl.update(2)   # 3rd alert → CONFIRMED
        assert cl.state == ConfirmationState.CONFIRMED
        assert ev is not None
        assert ev.state == ConfirmationState.CONFIRMED

    def test_clears_after_enough_ok_frames(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make(confirm_frames=2, clear_frames=3)
        cl.update(1); cl.update(1)   # → CONFIRMED
        cl.update(0); cl.update(0)
        ev = cl.update(0)            # 3rd OK → NORMAL
        assert cl.state == ConfirmationState.NORMAL
        assert ev is not None

    def test_alert_in_confirmed_resets_clear_countdown(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make(confirm_frames=2, clear_frames=3)
        cl.update(1); cl.update(1)   # CONFIRMED
        cl.update(0); cl.update(0)   # 2 OKs
        cl.update(1)                 # alert resets countdown
        cl.update(0); cl.update(0)   # only 2 OKs — not enough to clear
        assert cl.state == ConfirmationState.CONFIRMED

    def test_reset_returns_to_normal(self):
        from backend.api.confirmation_layer import ConfirmationState
        cl = self._make(confirm_frames=2)
        cl.update(1); cl.update(1)
        cl.reset()
        assert cl.state == ConfirmationState.NORMAL

    def test_no_event_when_staying_in_state(self):
        cl = self._make(confirm_frames=5)
        cl.update(1)   # → PENDING
        ev = cl.update(1)   # still PENDING, no transition yet
        assert ev is None

    def test_no_event_for_ok_in_normal(self):
        cl = self._make()
        ev = cl.update(0)   # OK in NORMAL → no change
        assert ev is None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI endpoints (no landmark model needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    """
    These tests patch out the LandmarkEngine and DecisionEngine so
    the test suite passes even without the MediaPipe model file.
    """

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from backend.api import main as m

        # Provide a stub landmark engine that always returns None (no face)
        class _StubEngine:
            def process(self, _frame): return None
            def close(self): pass

        monkeypatch.setattr(m._state, "landmark_engine", _StubEngine())
        monkeypatch.setattr(
            m._state,
            "decision_engine",
            m.DecisionEngine.__new__(m.DecisionEngine),
        )
        # Patch decision engine has_model to False (threshold mode)
        from backend.api.decision_engine import DecisionEngine
        de = DecisionEngine.__new__(DecisionEngine)
        de._session = None
        de._input_name = "features"
        monkeypatch.setattr(m._state, "decision_engine", de)

        with TestClient(m.app, raise_server_exceptions=False) as c:
            yield c

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_analyze_returns_unreliable_for_no_face(self, client):
        """A 1×1 white pixel — no face — should return UNRELIABLE."""
        import io as _io
        import struct
        import zlib

        # Minimal 1×1 white PNG
        def make_png_1x1():
            sig = b'\x89PNG\r\n\x1a\n'
            def chunk(name, data):
                c = struct.pack('>I', len(data)) + name + data
                return c + struct.pack('>I', zlib.crc32(name + data) & 0xFFFFFFFF)
            ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            idat = chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff'))
            iend = chunk(b'IEND', b'')
            return sig + ihdr + idat + iend

        png = make_png_1x1()
        resp = client.post(
            "/analyze",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["quality_code"] == "UNRELIABLE"

    def test_analyze_rejects_bad_file(self, client):
        resp = client.post(
            "/analyze",
            files={"file": ("bad.bin", b"not an image", "application/octet-stream")},
        )
        assert resp.status_code == 400
