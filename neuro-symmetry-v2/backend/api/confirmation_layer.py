"""
Confirmation Layer — multi-frame confirmation to suppress false positives.

State machine
-------------
  NORMAL    → any alert (class_id ≠ 0)             → PENDING
  PENDING   → confirm_frames consecutive alerts    → CONFIRMED
  PENDING   → one OK frame (class_id = 0)          → NORMAL  (resets)
  CONFIRMED → clear_frames consecutive OK frames   → NORMAL  (clears)
  CONFIRMED → any alert frame                      → stays CONFIRMED, resets clear countdown

Frame counts come from ``backend.core.config.ConfirmationSettings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.core import ConfirmationSettings, get_settings


_CLASS_LABELS: tuple[str, str, str] = ("Normal", "Mild", "Severe")


class ConfirmationState(str, Enum):
    NORMAL    = "NORMAL"
    PENDING   = "PENDING"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class ConfirmationEvent:
    state:           ConfirmationState
    frames_in_state: int
    class_id:        int
    class_label:     str


class ConfirmationLayer:
    """Per-session multi-frame confirmation gate."""

    def __init__(
        self,
        confirm_frames: Optional[int] = None,
        clear_frames:   Optional[int] = None,
        settings:       Optional[ConfirmationSettings] = None,
    ) -> None:
        cfg = settings or get_settings().confirmation
        self._confirm = confirm_frames if confirm_frames is not None else cfg.confirm_frames
        self._clear   = clear_frames   if clear_frames   is not None else cfg.clear_frames
        self._state   = ConfirmationState.NORMAL
        self._count   = 0
        self._alert_class = 0
        # Temporal hysteresis for PENDING state: require multiple consecutive
        # Normal frames before resetting, to avoid a single jittery frame
        # destroying a genuine PENDING sequence.
        self._pending_patience = 3
        self._pending_normal_streak = 0

    @property
    def state(self) -> ConfirmationState:
        return self._state

    def update(self, class_id: int) -> Optional[ConfirmationEvent]:
        is_alert = class_id != 0

        if self._state == ConfirmationState.NORMAL:
            if is_alert:
                self._state = ConfirmationState.PENDING
                self._count = 1
                self._alert_class = class_id
                return ConfirmationEvent(
                    state=self._state,
                    frames_in_state=self._count,
                    class_id=class_id,
                    class_label=_CLASS_LABELS[class_id],
                )

        elif self._state == ConfirmationState.PENDING:
            if is_alert:
                self._pending_normal_streak = 0
                self._count += 1
                if self._count >= self._confirm:
                    confirmed_count = self._count
                    self._state = ConfirmationState.CONFIRMED
                    self._count = 0
                    return ConfirmationEvent(
                        state=self._state,
                        frames_in_state=confirmed_count,
                        class_id=self._alert_class,
                        class_label=_CLASS_LABELS[self._alert_class],
                    )
            else:
                # Don't instantly reset — require sustained Normal frames
                self._pending_normal_streak += 1
                if self._pending_normal_streak >= self._pending_patience:
                    self._state = ConfirmationState.NORMAL
                    self._count = 0
                    self._alert_class = 0
                    self._pending_normal_streak = 0
                    return ConfirmationEvent(
                        state=self._state,
                        frames_in_state=0,
                        class_id=0,
                        class_label=_CLASS_LABELS[0],
                    )

        elif self._state == ConfirmationState.CONFIRMED:
            if not is_alert:
                self._count += 1
                if self._count >= self._clear:
                    self._state = ConfirmationState.NORMAL
                    self._count = 0
                    self._alert_class = 0
                    return ConfirmationEvent(
                        state=self._state,
                        frames_in_state=0,
                        class_id=0,
                        class_label=_CLASS_LABELS[0],
                    )
            else:
                self._count = 0   # reset clear countdown on any new alert

        return None

    def reset(self) -> None:
        self._state = ConfirmationState.NORMAL
        self._count = 0
        self._alert_class = 0
        self._pending_normal_streak = 0