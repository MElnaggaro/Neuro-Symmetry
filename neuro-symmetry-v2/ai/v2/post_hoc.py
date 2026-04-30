"""
Post-hoc calibration, optimal thresholding, and temporal smoothing.

TemperatureScaling   — fit scalar T to minimise NLL; target ECE < 0.05
find_optimal_thresholds — per-class F1-maximising decision thresholds
threshold_predict    — apply per-class thresholds with severity priority
TemporalSmoother     — deque state machine; requires N consecutive frames
                        before confirming a pathological prediction
"""

from __future__ import annotations

from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Post-hoc confidence calibration via a single learnable temperature T.
    calibrated_probs = softmax(logits / T)
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(
        self,
        val_logits: Tensor,
        val_labels: Tensor,
        lr:         float = 0.01,
        max_iter:   int   = 100,
    ) -> float:
        """
        Fit T on validation set using LBFGS (NLL objective).
        Returns the optimised temperature value.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure() -> Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.item())

    @staticmethod
    def compute_ece(
        probs:  np.ndarray,   # (N, C) calibrated probabilities
        labels: np.ndarray,   # (N,) integer ground truth
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error — lower is better (target < 0.05)."""
        confs   = probs.max(axis=1)
        preds   = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        edges   = np.linspace(0.0, 1.0, n_bins + 1)
        ece     = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            ece += mask.mean() * abs(correct[mask].mean() - confs[mask].mean())
        return float(ece)


# ── Optimal Per-Class Thresholds ──────────────────────────────────────────────

def find_optimal_thresholds(
    probs:       np.ndarray,   # (N, C) softmax probabilities
    labels:      np.ndarray,   # (N,) integer ground truth
    num_classes: int   = 3,
    steps:       int   = 90,
) -> List[float]:
    """
    Grid-search per-class decision thresholds that maximise Macro-F1.
    Each class is treated as a binary OvR problem and searched independently.
    The resulting thresholds replace argmax at inference time.

    Returns
    -------
    List[float] of length num_classes — one threshold per class.
    """
    thresholds: List[float] = []
    for c in range(num_classes):
        binary_gt       = (labels == c).astype(int)
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, steps):
            f1 = f1_score(binary_gt, (probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds.append(best_t)
    return thresholds


def threshold_predict(
    probs:      np.ndarray,   # (C,) single-sample probabilities
    thresholds: List[float],
    priority:   Optional[List[int]] = None,
) -> int:
    """
    Apply per-class thresholds with priority ordering.
    Default: Severe (2) before Mild (1) before Normal (0).
    Falls back to argmax if no threshold is met.
    """
    if priority is None:
        priority = [2, 1, 0]
    for c in priority:
        if probs[c] >= thresholds[c]:
            return c
    return int(np.argmax(probs))


# ── Hard Negative Miner ───────────────────────────────────────────────────────

class HardNegativeMiner:
    """
    Identifies 'Hard Normals' — Normal-class samples the current model
    assigns high Mild/Severe probability — then builds a WeightedRandomSampler
    that oversamples them during the next epoch.

    Rebuilding the sampler every K epochs keeps the hard-set current as the
    model improves.
    """

    def __init__(
        self,
        features:     torch.Tensor,    # (N, D) — full training features on CPU
        labels:       np.ndarray,      # (N,)   — integer class labels
        model:        nn.Module,
        device:       torch.device,
        normal_class: int   = 0,
        hard_factor:  float = 8.0,
        fp_threshold: float = 0.30,
        batch_size:   int   = 4_096,
    ) -> None:
        self.weights = self._compute(
            features, labels, model, device,
            normal_class, hard_factor, fp_threshold, batch_size,
        )

    @torch.no_grad()
    def _compute(
        self,
        features:     torch.Tensor,
        labels:       np.ndarray,
        model:        nn.Module,
        device:       torch.device,
        normal_class: int,
        hard_factor:  float,
        threshold:    float,
        batch_size:   int,
    ) -> torch.Tensor:
        model.eval()
        logit_chunks = []
        for i in range(0, len(features), batch_size):
            chunk = features[i : i + batch_size].to(device)
            out   = model(chunk)
            # SymmetryNetV2 returns (logits, emb); handle both cases
            logit_chunks.append((out[0] if isinstance(out, tuple) else out).cpu())
        probs = F.softmax(torch.cat(logit_chunks, dim=0), dim=-1).numpy()

        weights = np.ones(len(features), dtype=np.float32)
        normal_mask      = labels == normal_class
        non_normal_prob  = 1.0 - probs[normal_mask, normal_class]
        hard_mask        = non_normal_prob > threshold

        # Scale weight linearly with how confidently the model misfires
        weights[normal_mask] = np.where(
            hard_mask,
            hard_factor * (non_normal_prob / (1.0 - threshold + 1e-8)),
            1.0,
        )
        return torch.FloatTensor(weights)

    def get_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            self.weights, num_samples=len(self.weights), replacement=True
        )


# ── Temporal State Machine ────────────────────────────────────────────────────

class TemporalSmoother:
    """
    Deque-based state machine for live video feeds.

    A pathological label (Mild=1 or Severe=2) is confirmed only after
    appearing in at least `required` of the last `window_size` frames.
    Normal is immediately restored when the window no longer supports
    a pathological decision.  Eliminates per-frame prediction flickering.
    """

    def __init__(self, window_size: int = 10, required: int = 5) -> None:
        if required > window_size:
            raise ValueError(f"required ({required}) must be <= window_size ({window_size})")
        self._window     = deque(maxlen=window_size)
        self._required   = required
        self.window_size = window_size

    def update(self, frame_pred: int) -> int:
        """
        Ingest one frame prediction.  Returns the smoothed (confirmed) label.
        Returns the raw frame prediction during the warm-up period so early
        pathological frames are not silently suppressed.
        """
        self._window.append(frame_pred)
        if len(self._window) < self.window_size:
            return frame_pred                          # insufficient history — pass through

        counts = Counter(self._window)
        for cls in [2, 1]:                             # Severe takes priority
            if counts.get(cls, 0) >= self._required:
                return cls
        return 0

    def reset(self) -> None:
        self._window.clear()

    @property
    def is_warmed_up(self) -> bool:
        return len(self._window) == self.window_size

    @property
    def state(self) -> dict:
        return {"window": list(self._window), "warmed_up": self.is_warmed_up}
