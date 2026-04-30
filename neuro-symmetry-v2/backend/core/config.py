"""
Centralised, strictly-typed application settings.

Single source of truth for every tuneable constant in the backend.  Loaded once
at process start via ``get_settings()`` (LRU-cached) so all modules see the same
immutable object.

Override mechanisms (highest precedence first):
  1. Explicit kwargs to ``Settings(...)``  — used by tests
  2. Environment variables                 — production / Docker
  3. ``.env`` file in CWD                  — local development
  4. Defaults in this module               — fallback

Environment-variable naming uses a flat prefix per group, so a deployment can
override only the bits it cares about without rewriting the whole config:

    NS_LOG_LEVEL=DEBUG
    NS_CORS_ORIGINS=https://app.example.com,https://staging.example.com
    NS_ONNX_MODEL_PATH=/models/model_v2.onnx
    NS_QUALITY__MIN_LUMINANCE=35
    NS_TRAJECTORY__COLLAPSE_MEAN=0.25

Nested-group overrides use the ``__`` (double-underscore) delimiter (per
pydantic-settings convention).  See the ``model_config`` block below.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Filesystem anchor ─────────────────────────────────────────────────────────

# Project root = .../neuro-symmetry-v2/  (parents[2] from this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Nested setting groups ─────────────────────────────────────────────────────


class MediaPipeSettings(BaseModel):
    """MediaPipe FaceLandmarker tuning.  All values must be in [0, 1]."""
    min_face_detection_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_face_presence_confidence:  float = Field(0.5, ge=0.0, le=1.0)
    num_faces:                     int   = Field(1, ge=1, le=4)


class QualitySettings(BaseModel):
    """Frame-level acceptance gate (see ``backend.api.input_quality``)."""
    min_luminance: float = Field(40.0, gt=0.0)   # mean V of HSV
    # 40 works for typical USB/webcam streams; raise via NS_QUALITY__MIN_LAPLACIAN
    # if running in a controlled studio environment.  80 was too aggressive for
    # real-world cameras where natural motion reduces Laplacian variance.
    min_laplacian: float = Field(40.0, gt=0.0)   # blur proxy
    min_z_range:   float = Field(0.01, gt=0.0)   # occlusion proxy
    max_roll_deg:  float = Field(15.0, gt=0.0)
    max_yaw_deg:   float = Field(25.0, gt=0.0)
    max_pitch_deg: float = Field(20.0, gt=0.0)


class DecisionSettings(BaseModel):
    """ONNX inference + threshold-fallback classifier."""
    mild_threshold:   float = Field(0.75, gt=0.0, lt=1.0)
    severe_threshold: float = Field(0.55, gt=0.0, lt=1.0)
    # Search order — first match wins
    onnx_candidates: tuple[str, ...] = (
        "model_v2.onnx",
    )
    scaler_filename: str = "feature_scaler.pkl"

    @field_validator("severe_threshold")
    @classmethod
    def _severe_below_mild(cls, v: float, info) -> float:
        mild = info.data.get("mild_threshold")
        if mild is not None and v >= mild:
            raise ValueError("severe_threshold must be < mild_threshold")
        return v


class TemporalSettings(BaseModel):
    """EMA + slope tracking."""
    ema_alpha:    float = Field(0.10, gt=0.0, le=1.0)
    slope_window: int   = Field(30, gt=0)
    min_slope_n:  int   = Field(5, gt=0)


class TrajectorySettings(BaseModel):
    """Trend-classification heuristics."""
    window_size:    int   = Field(60, gt=0)
    min_frames:     int   = Field(10, gt=0)
    drop_threshold: float = Field(0.25, gt=0.0)
    drop_window:    int   = Field(30, gt=0)
    collapse_mean:  float = Field(0.30, gt=0.0)
    slope_thresh:   float = Field(-0.003)
    r2_thresh:      float = Field(0.50, ge=0.0, le=1.0)
    osc_var_thresh: float = Field(0.05, gt=0.0)
    osc_slope_abs:  float = Field(0.001, gt=0.0)


class ChangeDetectionSettings(BaseModel):
    """Rolling-window z-score anomaly detector."""
    window:      int   = Field(90, gt=0)
    warmup:      int   = Field(30, gt=0)
    z_threshold: float = Field(2.5, gt=0.0)


class ConfirmationSettings(BaseModel):
    """Multi-frame state machine to suppress false positives."""
    confirm_frames: int = Field(5,  gt=0)
    clear_frames:   int = Field(10, gt=0)


class ProdromalSettings(BaseModel):
    """Browser-side temporal-asymmetry assessment thresholds."""
    asym_alert_ms:    float = Field(200.0, gt=0.0)   # YELLOW / RED gate
    asym_report_ms:   float = Field(50.0,  gt=0.0)   # below → "normal range"
    history_max:      int   = Field(500,   gt=0)
    confidence_base:  float = Field(0.40, ge=0.0, le=1.0)
    confidence_step:  float = Field(0.30, ge=0.0, le=1.0)


class SessionSettings(BaseModel):
    """Streaming pipeline / session bookkeeping."""
    assumed_fps:        float = Field(30.0, gt=0.0)
    min_calib_frames:   int   = Field(30,   gt=0)


# ── Top-level Settings singleton ──────────────────────────────────────────────


CommaList = Annotated[list[str], Field(default_factory=list)]


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Access via ``get_settings()`` — never instantiate directly except in tests.
    """

    # ── Process / runtime ────────────────────────────────────────────────────
    log_level:    str  = Field("INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    api_title:    str  = "Neuro-Symmetry API"
    api_version:  str  = "2.0.0"
    debug:        bool = False

    # ── CORS ─────────────────────────────────────────────────────────────────
    # CSV string in env (NS_CORS_ORIGINS=https://a,https://b) → list[str]
    cors_origins:    CommaList
    cors_allow_methods: CommaList
    cors_allow_headers: CommaList
    cors_allow_credentials: bool = False

    # ── Filesystem paths ─────────────────────────────────────────────────────
    project_root:   Path = PROJECT_ROOT
    ai_dir:         Path = PROJECT_ROOT / "ai"
    checkpoints_dir: Path = PROJECT_ROOT / "ai" / "checkpoints"
    models_dir:     Path = PROJECT_ROOT / "backend" / "models"
    metrics_path:   Path = PROJECT_ROOT / "ai" / "checkpoints" / "metrics_v2.json"
    landmarker_model_path: Path = (
        PROJECT_ROOT / "backend" / "models" / "face_landmarker.task"
    )

    # ── Nested groups ────────────────────────────────────────────────────────
    mediapipe:    MediaPipeSettings        = Field(default_factory=MediaPipeSettings)
    quality:      QualitySettings          = Field(default_factory=QualitySettings)
    decision:     DecisionSettings         = Field(default_factory=DecisionSettings)
    temporal:     TemporalSettings         = Field(default_factory=TemporalSettings)
    trajectory:   TrajectorySettings       = Field(default_factory=TrajectorySettings)
    change:       ChangeDetectionSettings  = Field(default_factory=ChangeDetectionSettings)
    confirmation: ConfirmationSettings     = Field(default_factory=ConfirmationSettings)
    prodromal:    ProdromalSettings        = Field(default_factory=ProdromalSettings)
    session:      SessionSettings          = Field(default_factory=SessionSettings)

    # ── pydantic-settings configuration ─────────────────────────────────────
    model_config = SettingsConfigDict(
        env_prefix="NS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Validators / coercion ────────────────────────────────────────────────
    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> list[str]:
        """Allow ``NS_CORS_ORIGINS=a,b,c`` env-var syntax."""
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return [str(s).strip() for s in v if str(s).strip()]
        raise TypeError(f"Cannot parse CSV list from {type(v).__name__}")

    def model_post_init(self, __context: object) -> None:
        # Sensible CORS defaults if nothing was provided via env
        if not self.cors_origins:
            object.__setattr__(self, "cors_origins", ["*"])
        if not self.cors_allow_methods:
            object.__setattr__(self, "cors_allow_methods", ["*"])
        if not self.cors_allow_headers:
            object.__setattr__(self, "cors_allow_headers", ["*"])


# ── Public accessor ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton (cached after first call).

    Tests may override individual fields by calling ``get_settings.cache_clear()``
    and then constructing ``Settings(...)`` manually with kwargs.
    """
    return Settings()  # type: ignore[call-arg]
