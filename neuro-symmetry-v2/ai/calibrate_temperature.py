"""
Temperature scaling calibration utilities.

Functions
---------
compute_ece(probs, labels, n_bins) → float
    Expected Calibration Error — lower is better (target < 0.05).

plot_reliability_diagram(probs_pre, probs_post, labels, path)
    Save a reliability diagram comparing pre/post calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def compute_ece(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Parameters
    ----------
    probs  : (N, C) softmax probabilities
    labels : (N,)   integer ground-truth class indices
    n_bins : number of confidence bins

    Returns
    -------
    ECE ∈ [0, 1]  (target < 0.05)
    """
    n = len(labels)
    confidences = probs.max(axis=1)           # (N,)
    predictions = probs.argmax(axis=1)        # (N,)
    correct     = (predictions == labels)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / n)


def plot_reliability_diagram(
    probs_pre:  np.ndarray,
    probs_post: np.ndarray,
    labels:     np.ndarray,
    path:       Path,
    n_bins:     int = 10,
) -> None:
    """
    Save reliability diagram (confidence vs. accuracy) comparing
    pre-calibration and post-calibration probabilities.
    Requires matplotlib; silently skips if not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for probs, label, color, ls in [
        (probs_pre,  "Before calibration", "steelblue", "--"),
        (probs_post, "After calibration",  "coral",     "-"),
    ]:
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        accs  = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confs > lo) & (confs <= hi)
            accs.append(float((preds[mask] == labels[mask]).mean()) if mask.sum() else float("nan"))
        ax.plot(bin_centers, accs, label=label, color=color, ls=ls, marker="o", markersize=4)

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
