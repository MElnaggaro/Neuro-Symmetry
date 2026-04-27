"""
Confirmation Layer — multi-frame confirmation to suppress false positives.

State machine
-------------
  NORMAL    → any alert (class_id ≠ 0)            → PENDING
  PENDING   → CONFIRM_FRAMES consecutive alerts   → CONFIRMED
  PENDING   → one OK frame (class_id = 0)         → NORMAL  (resets)
  CONFIRMED → CLEAR_FRAMES consecutive OK frames  → NORMAL  (clears)
  CONFIRMED → any alert frame                     → stays CONFIRMED, resets clear countdown
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


CONFIRM_FRAMES = 5    # consecutive alert frames to confirm
CLEAR_FRAMES   = 10   # consecutive OK frames to clear

_CLASS_LABELS = ["Normal", "Mild", "Severe"]


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
    """
    Per-session multi-frame confirmation gate.

    Call ``update(class_id)`` once per frame.
    Returns a ``ConfirmationEvent`` on state transition, else None.
    """

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        clear_frames:   int = CLEAR_FRAMES,
    ) -> None:
        self._confirm = confirm_frames
        self._clear   = clear_frames
        self._state   = ConfirmationState.NORMAL
        self._count   = 0
        self._alert_class = 0

    @property
    def state(self) -> ConfirmationState:
        return self._state

    def update(self, class_id: int) -> Optional[ConfirmationEvent]:
        """Feed the latest classification result. Returns event on transition."""
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
                self._count += 1
                if self._count >= self._confirm:
                    self._state = ConfirmationState.CONFIRMED
                    self._count = 0
                    return ConfirmationEvent(
                        state=self._state,
                        frames_in_state=self._count,
                        class_id=self._alert_class,
                        class_label=_CLASS_LABELS[self._alert_class],
                    )
            else:
                prev_class = self._alert_class
                self._state = ConfirmationState.NORMAL
                self._count = 0
                self._alert_class = 0
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
