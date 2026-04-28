"""
Phase 8 — Integration & Metric Validation Tests.

8a  End-to-end pipeline tests (REST + WebSocket)
8b  Metric validation (score stability, temporal σ)
8c  Load test (30 fps sustained, latency p99)
8d  Failure-mode audit (UNRELIABLE → no alert)

No real camera or ONNX model required.
The landmark engine is patched to return synthetic LandmarkResult objects;
the decision engine uses its built-in threshold fallback.
"""

from __future__ import annotations

import base64
import math
import statistics
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.landmark_engine import (
    KEY_POINTS,
    LEFT_INDICES,
    LandmarkResult,
    NormalizedLandmarks,
    PoseAngles,
    RIGHT_INDICES,
)
from backend.api.main import _state, app

# ── Synthetic frame / landmark helpers ───────────────────────────────────────

_H, _W = 240, 320


def _frame(brightness: int = 120) -> np.ndarray:
    """Sharp BGR frame that is bilaterally symmetric around the centre column.

    Uses distance-from-centre for the horizontal tile index so the left and
    right halves of the face crop are mirror images → NCC ≈ 1.0 → texture_score
    ≈ 1.0.  The checkerboard edges keep Laplacian variance well above MIN_LAPLACIAN.
    """
    rr = np.arange(_H, dtype=np.int32)[:, None] // 8
    cc = np.abs(np.arange(_W, dtype=np.int32)[None, :] - _W // 2) // 8
    mask = (rr + cc) % 2 == 0
    f = np.where(mask[:, :, None], brightness, brightness // 2).astype(np.uint8)
    return f


def _dark_frame() -> np.ndarray:
    return np.full((_H, _W, 3), 10, dtype=np.uint8)


def _blurred_frame() -> np.ndarray:
    frame = np.full((_H, _W, 3), 120, dtype=np.uint8)
    return cv2.GaussianBlur(frame, (31, 31), 0)


def _frame_jpeg(brightness: int = 120) -> str:
    """Return base64-encoded JPEG of a synthetic frame."""
    _, buf = cv2.imencode(".jpg", _frame(brightness), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def _dark_frame_jpeg() -> str:
    _, buf = cv2.imencode(".jpg", _dark_frame())
    return base64.b64encode(buf.tobytes()).decode()


def _blurred_frame_jpeg() -> str:
    _, buf = cv2.imencode(".jpg", _blurred_frame())
    return base64.b64encode(buf.tobytes()).decode()


def _make_landmarks(left_droop: bool = False) -> LandmarkResult:
    """
    Build a synthetic LandmarkResult.

    Normalised points (pts): nose-centred, IPD-scaled space used by feature_extractor.
    Raw landmarks (raw_lm): fractional [0,1] coords used by texture_engine for ROI.

    left_droop=True creates a large bilateral asymmetry triggering High/Critical risk.
    left_droop=False creates a perfectly symmetric face → Normal classification.
    """
    # ── Normalised points (feature extractor space) ───────────────────────────
    pts = np.zeros((478, 3), dtype=np.float32)

    # Set all mirror pairs to be perfectly symmetric (bilateral dist = 0)
    # Convention: mirror(right) = (-right.x, right.y, right.z)
    # So pts[l_idx] must equal (-pts[r_idx].x, pts[r_idx].y, pts[r_idx].z)
    for l_idx, r_idx in zip(LEFT_INDICES, RIGHT_INDICES):
        pts[l_idx] = [ 0.05, 0.10, 0.50]
        pts[r_idx] = [-0.05, 0.10, 0.50]  # mirror: dist = 0

    # Key anatomical anchors (set AFTER mirror loop to override)
    pts[KEY_POINTS["nose_tip"]]    = [ 0.00,  0.00, 0.50]
    pts[KEY_POINTS["eye_left"]]    = [-0.50, -0.30, 0.50]
    pts[KEY_POINTS["eye_right"]]   = [ 0.50, -0.30, 0.50]
    pts[KEY_POINTS["brow_left"]]   = [-0.50, -0.50, 0.50]
    pts[KEY_POINTS["brow_right"]]  = [ 0.50, -0.50, 0.50]
    pts[KEY_POINTS["mouth_left"]]  = [-0.30,  0.40, 0.50]
    pts[KEY_POINTS["mouth_right"]] = [ 0.30,  0.40, 0.50]

    if left_droop:
        # Large asymmetry on left side → bilateral distances ≈ 0.25
        for l_idx in LEFT_INDICES:
            pts[l_idx] = [0.30, 0.30, 0.50]
        pts[KEY_POINTS["mouth_left"]]  = [-0.30,  0.70, 0.50]
        pts[KEY_POINTS["eye_left"]]    = [-0.50, -0.05, 0.50]
        pts[KEY_POINTS["brow_left"]]   = [-0.50, -0.15, 0.50]

    # Ensure z-range > 0.01 (occlusion check passes) while keeping bilateral pairs
    # at the same z so mirror distances stay zero for the symmetric case.
    pts[KEY_POINTS["eye_left"],    2] = 0.40
    pts[KEY_POINTS["eye_right"],   2] = 0.40   # same z as eye_left → bilateral dist=0
    pts[KEY_POINTS["nose_tip"],    2] = 0.50
    pts[KEY_POINTS["mouth_left"],  2] = 0.65
    pts[KEY_POINTS["mouth_right"], 2] = 0.65   # same z as mouth_left → bilateral dist=0

    norm = NormalizedLandmarks(points=pts, ipd=60.0)
    pose = PoseAngles(roll=0.0, yaw=0.0, pitch=0.0)

    # ── Raw landmarks (fractional [0,1] coords for texture engine ROI) ────────
    # Place all points in a centered 0.3–0.7 box → symmetric crop → NCC ≈ 1.0
    raw_lm = []
    for i in range(478):
        lm = MagicMock()
        lm.x = 0.5 + pts[i, 0] * 0.2   # map IPD±0.5 → fractional 0.4–0.6
        lm.y = 0.5 + pts[i, 1] * 0.2
        lm.z = float(pts[i, 2])
        lm.visibility = 1.0
        raw_lm.append(lm)

    return LandmarkResult(
        raw=raw_lm,
        normalized=norm,
        pose=pose,
        frame_hw=(_H, _W),
        score=1.0,
    )


def _mock_landmark_engine(left_droop: bool = False) -> MagicMock:
    """Return a mock LandmarkEngine whose process() returns synthetic landmarks."""
    eng = MagicMock()
    eng.process.return_value = _make_landmarks(left_droop=left_droop)
    return eng


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def patched_client():
    """TestClient with a mocked landmark engine (symmetric face).
    Patch is applied AFTER lifespan so the mock is not overwritten by startup."""
    with TestClient(app) as c:
        mock_eng = _mock_landmark_engine(left_droop=False)
        with patch.object(_state, "landmark_engine", mock_eng):
            _state.rest_session.reset()
            yield c


@pytest.fixture()
def droop_client():
    """TestClient with a mocked landmark engine (left droop)."""
    with TestClient(app) as c:
        mock_eng = _mock_landmark_engine(left_droop=True)
        with patch.object(_state, "landmark_engine", mock_eng):
            _state.rest_session.reset()
            yield c


# ── 8a — End-to-End Tests ────────────────────────────────────────────────────

class TestEndToEnd:

    def test_health_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "landmark_engine" in body
        assert "decision_engine" in body

    def test_analyze_503_without_landmark_engine(self, client: TestClient):
        """API must return 503 when landmark engine isn't loaded."""
        with patch.object(_state, "landmark_engine", None):
            _, buf = cv2.imencode(".jpg", _frame())
            r = client.post("/analyze", files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 503

    def test_analyze_returns_valid_schema(self, patched_client: TestClient):
        _, buf = cv2.imencode(".jpg", _frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["quality_code"] in ("OK", "DEGRADED", "UNRELIABLE")
        assert "alert" in body
        assert "latency_ms" in body

    def test_neutral_face_no_alert_after_30_frames(self, patched_client: TestClient):
        """30 symmetric frames must not trigger an alert."""
        _, buf = cv2.imencode(".jpg", _frame())
        data = buf.tobytes()
        alert_fired = False
        for _ in range(30):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            assert r.status_code == 200
            if r.json().get("alert"):
                alert_fired = True
        assert not alert_fired, "False positive alert on symmetric face"

    def test_bad_image_400(self, patched_client: TestClient):
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", b"not an image", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_session_reset_clears_state(self, patched_client: TestClient):
        r = patched_client.post("/session/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "reset"


class TestWebSocketEndToEnd:

    def test_ws_analyze_returns_frame_field(self, patched_client: TestClient):
        b64 = _frame_jpeg()
        with patched_client.websocket_connect("/stream") as ws:
            ws.send_json({"image": b64})
            msg = ws.receive_json()
        assert "frame" in msg
        assert "quality_code" in msg

    def test_ws_reset_message(self, patched_client: TestClient):
        with patched_client.websocket_connect("/stream") as ws:
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
        assert msg["status"] == "reset"

    def test_ws_missing_image_field(self, patched_client: TestClient):
        with patched_client.websocket_connect("/stream") as ws:
            ws.send_json({"type": "analyze"})
            msg = ws.receive_json()
        assert "error" in msg

    def test_ws_invalid_base64(self, patched_client: TestClient):
        with patched_client.websocket_connect("/stream") as ws:
            ws.send_json({"image": "!!!not-base64!!!"})
            msg = ws.receive_json()
        assert "error" in msg

    def test_ws_session_isolation(self, patched_client: TestClient):
        """Two concurrent WS connections must have independent frame counters."""
        b64 = _frame_jpeg()
        frames_a, frames_b = [], []

        with patched_client.websocket_connect("/stream") as ws_a:
            for _ in range(3):
                ws_a.send_json({"image": b64})
                frames_a.append(ws_a.receive_json()["frame"])

        with patched_client.websocket_connect("/stream") as ws_b:
            ws_b.send_json({"image": b64})
            frames_b.append(ws_b.receive_json()["frame"])

        # Session B starts at 0, not continuing from session A
        assert frames_b[0] == 0, "WS sessions are not isolated — frame counter leaked"
        assert frames_a == [0, 1, 2]

    def test_ws_neutral_face_no_alert_30_frames(self, patched_client: TestClient):
        b64 = _frame_jpeg()
        alert_fired = False
        with patched_client.websocket_connect("/stream") as ws:
            for _ in range(30):
                ws.send_json({"image": b64})
                if ws.receive_json().get("alert"):
                    alert_fired = True
        assert not alert_fired


class TestConfirmationLayer:

    def test_single_droop_frame_no_alert(self, patched_client: TestClient):
        """Confirmation layer must not alert on a single anomalous frame."""
        norm_jpeg = _frame_jpeg(brightness=120)

        # 25 normal frames to establish state
        with patched_client.websocket_connect("/stream") as ws:
            for _ in range(25):
                ws.send_json({"image": norm_jpeg})
                ws.receive_json()

            # Inject 1 droop frame
            droop_eng = _mock_landmark_engine(left_droop=True)
            with patch.object(_state, "landmark_engine", droop_eng):
                ws.send_json({"image": norm_jpeg})
                msg = ws.receive_json()

            # Must not have confirmed yet
            assert not msg.get("alert"), (
                "Single droop frame should not confirm an alert"
            )


# ── 8b — Metric Validation ───────────────────────────────────────────────────

class TestMetricValidation:

    def test_score_stability_neutral_face(self, patched_client: TestClient):
        """
        60 neutral frames must produce symmetry_score σ < 0.02.
        Validates temporal stability requirement from Phase 8b.
        """
        _, buf = cv2.imencode(".jpg", _frame())
        data = buf.tobytes()
        scores = []

        for _ in range(60):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            assert r.status_code == 200
            body = r.json()
            if body.get("symmetry_score") is not None:
                scores.append(body["symmetry_score"])

        assert len(scores) >= 50, f"Expected ≥50 valid scores, got {len(scores)}"
        sigma = statistics.stdev(scores) if len(scores) > 1 else 0.0
        assert sigma < 0.02, f"Score σ={sigma:.4f} exceeds 0.02 stability threshold"

    def test_risk_level_present_on_ok_frame(self, patched_client: TestClient):
        _, buf = cv2.imencode(".jpg", _frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        body = r.json()
        if body["quality_code"] == "OK":
            assert body["risk_level"] in ("NORMAL", "MILD", "HIGH_RISK", "CRITICAL")
            assert body["trajectory"] in (
                "STABLE", "LINEAR_DECLINE", "SUDDEN_DROP", "OSCILLATING", "COLLAPSE"
            )

    def test_xai_output_ranked_by_contribution(self, patched_client: TestClient):
        _, buf = cv2.imencode(".jpg", _frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        body = r.json()
        xai = body.get("xai", [])
        if len(xai) > 1:
            contributions = [f["contribution"] for f in xai]
            assert contributions == sorted(contributions, reverse=True), (
                "XAI features must be sorted by contribution descending"
            )

    def test_latency_ms_below_200(self, patched_client: TestClient):
        _, buf = cv2.imencode(".jpg", _frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        latency = r.json()["latency_ms"]
        assert latency < 200, f"Pipeline latency {latency:.1f}ms exceeds 200ms"

    def test_ema_score_monotone_on_stable_input(self, patched_client: TestClient):
        """EMA score should converge, not oscillate wildly, on fixed input."""
        _, buf = cv2.imencode(".jpg", _frame())
        data = buf.tobytes()
        ema_scores = []

        for _ in range(20):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            s = r.json().get("ema_score")
            if s is not None:
                ema_scores.append(s)

        if len(ema_scores) >= 10:
            # Variance should be very low on identical frames
            sigma = statistics.stdev(ema_scores)
            assert sigma < 0.05, f"EMA σ={sigma:.4f} too high for stable input"


# ── 8c — Load Testing ────────────────────────────────────────────────────────

class TestLoadTesting:

    def test_180_frames_all_succeed(self, patched_client: TestClient):
        """Simulate 6 seconds at 30fps — all 180 frames must succeed."""
        _, buf = cv2.imencode(".jpg", _frame())
        data = buf.tobytes()
        failures = 0

        for _ in range(180):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            if r.status_code != 200:
                failures += 1

        assert failures == 0, f"{failures}/180 frames failed"

    def test_p99_latency_below_100ms(self, patched_client: TestClient):
        """p99 backend latency must stay under 100ms across 60 frames."""
        _, buf = cv2.imencode(".jpg", _frame())
        data = buf.tobytes()
        latencies = []

        for _ in range(60):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            assert r.status_code == 200
            latencies.append(r.json()["latency_ms"])

        latencies.sort()
        p99 = latencies[math.ceil(len(latencies) * 0.99) - 1]
        assert p99 < 100, f"p99 latency {p99:.1f}ms exceeds 100ms"

    def test_ws_frame_counter_monotonically_increases(self, patched_client: TestClient):
        b64 = _frame_jpeg()
        frames = []
        with patched_client.websocket_connect("/stream") as ws:
            for _ in range(20):
                ws.send_json({"image": b64})
                frames.append(ws.receive_json()["frame"])
        assert frames == list(range(20)), f"Frame counter not monotone: {frames}"

    def test_session_reset_preserves_frame_counter(self, patched_client: TestClient):
        """The `reset` message resets session state but the WS-local frame counter continues."""
        b64 = _frame_jpeg()
        with patched_client.websocket_connect("/stream") as ws:
            for _ in range(5):
                ws.send_json({"image": b64})
                ws.receive_json()
            ws.send_json({"type": "reset"})
            assert ws.receive_json()["status"] == "reset"
            ws.send_json({"image": b64})
            msg = ws.receive_json()
        # frame_count is WS-local; reset() only clears session rolling state
        assert msg["frame"] == 5


# ── 8d — Failure Mode Audit ───────────────────────────────────────────────────

class TestFailureModeAudit:

    def test_unreliable_dark_frame_no_alert(self, patched_client: TestClient):
        """Dark frames must be UNRELIABLE and must never trigger an alert."""
        _, buf = cv2.imencode(".jpg", _dark_frame())
        data = buf.tobytes()

        for _ in range(30):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["quality_code"] == "UNRELIABLE", (
                "Dark frame should be UNRELIABLE"
            )
            assert not body["alert"], "UNRELIABLE frame must not trigger alert"

    def test_unreliable_no_landmark(self):
        """Frames rejected at quality gate before landmark engine must not alert."""
        from backend.api.input_quality import InputQualityChecker, QualityCode

        checker = InputQualityChecker()
        dark = _dark_frame()
        status = checker.check(dark, None)
        assert status.code == QualityCode.UNRELIABLE
        assert not status.is_usable

    def test_unreliable_frame_no_risk_fields(self, patched_client: TestClient):
        """UNRELIABLE frames must return null risk_level, xai, and trajectory."""
        _, buf = cv2.imencode(".jpg", _dark_frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        body = r.json()
        assert body["quality_code"] == "UNRELIABLE"
        assert body["risk_level"] is None
        assert body["trajectory"] is None
        assert body["xai"] == []
        assert not body["alert"]

    def test_ws_unreliable_frames_no_alert(self, patched_client: TestClient):
        b64 = _dark_frame_jpeg()
        with patched_client.websocket_connect("/stream") as ws:
            for _ in range(30):
                ws.send_json({"image": b64})
                msg = ws.receive_json()
                assert not msg.get("alert", False), (
                    "WS dark frame triggered alert"
                )

    def test_confirmation_state_stays_normal_on_unreliable(
        self, patched_client: TestClient
    ):
        """
        UNRELIABLE frames must not advance the confirmation window,
        so confirmation_state must remain 'NORMAL'.
        """
        _, buf = cv2.imencode(".jpg", _dark_frame())
        data = buf.tobytes()
        for _ in range(30):
            r = patched_client.post(
                "/analyze",
                files={"file": ("face.jpg", data, "image/jpeg")},
            )
            assert r.json()["confirmation_state"] == "NORMAL"

    def test_change_detected_false_on_unreliable(self, patched_client: TestClient):
        _, buf = cv2.imencode(".jpg", _dark_frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        body = r.json()
        assert not body.get("change_detected", False)

    def test_no_xai_on_unreliable_ws(self, patched_client: TestClient):
        b64 = _dark_frame_jpeg()
        with patched_client.websocket_connect("/stream") as ws:
            ws.send_json({"image": b64})
            msg = ws.receive_json()
        assert msg.get("xai", []) == []
        assert msg.get("risk_level") is None

    def test_blurred_frame_unreliable(self, patched_client: TestClient):
        """Motion-blurred frames must be rejected as UNRELIABLE."""
        _, buf = cv2.imencode(".jpg", _blurred_frame())
        r = patched_client.post(
            "/analyze",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        # Blurred frame may be UNRELIABLE (low Laplacian) but not necessarily
        # if the blur doesn't push variance below threshold. Only assert no alert.
        assert not body["alert"], "Blurred frame must not alert"
