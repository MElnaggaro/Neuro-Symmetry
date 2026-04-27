"""
Phase 4 training script.

Usage
-----
    python -m ai.train                       # defaults
    python -m ai.train --n 8000 --epochs 80  # larger run
    python -m ai.train --out ai/checkpoints/run2

Outputs (written to --out directory)
-------------------------------------
    model_calibrated.pt     — CalibratedModel weights + temperature
    model.onnx              — ONNX export for cross-platform inference
    reliability.png         — reliability diagram (pre vs. post calibration)
    metrics.json            — all evaluation metrics

Target metrics
--------------
    AUC-ROC     > 0.95
    Sensitivity > 0.92
    Specificity > 0.90
    F1 (macro)  > 0.91
    ECE         < 0.05
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 stdout/stderr so torch's ONNX exporter (which prints emoji) works on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from ai.calibrate_temperature import compute_ece, plot_reliability_diagram
from ai.model import CalibratedModel, NeuroSymmetryNet
from ai.synthetic_augment import build_dataset

# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[tuple, tuple, tuple]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def _tensors(X: np.ndarray, y: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.long,    device=device),
    )


def _eval_metrics(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels = y.cpu().numpy()
    preds  = probs.argmax(axis=1)

    acc = float((preds == labels).mean())
    f1  = float(f1_score(labels, preds, average="macro", zero_division=0))

    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    # Sensitivity (recall for class 1+2 — pathological) and specificity (class 0)
    n0 = (labels == 0).sum()
    tp_normal = ((preds == 0) & (labels == 0)).sum()
    spec = float(tp_normal / n0) if n0 else float("nan")
    path_mask = labels > 0
    sens = float((preds[path_mask] > 0).mean()) if path_mask.sum() else float("nan")

    ece = compute_ece(probs, labels)

    return dict(accuracy=acc, f1_macro=f1, auc_roc=auc,
                sensitivity=sens, specificity=spec, ece=ece,
                probs=probs, labels=labels)


# ── Main training function ────────────────────────────────────────────────────

def train(
    n_per_class: int = 6_000,
    epochs:      int = 60,
    batch_size:  int = 256,
    lr:          float = 3e-3,
    seed:        int = 42,
    out_dir:     Path = Path("ai/checkpoints"),
    verbose:     bool = True,
) -> dict:
    """
    Train NeuroSymmetryNet on synthetic data, calibrate temperature, export.

    Returns a metrics dict with all evaluation results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    torch.manual_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    if verbose:
        print(f"Generating synthetic dataset ({n_per_class} samples/class)…")
    X, y = build_dataset(n_per_class=n_per_class, seed=seed)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _split(X, y, seed=seed)

    Xt, yt   = _tensors(X_tr, y_tr, device)
    Xv, yv   = _tensors(X_val, y_val, device)
    Xte, yte = _tensors(X_te, y_te, device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"  Train {len(y_tr)}  Val {len(y_val)}  Test {len(y_te)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    base  = NeuroSymmetryNet()
    model = CalibratedModel(base).to(device)
    if verbose:
        print(f"  Parameters: {base.n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        base.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(base(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        # Validation
        base.eval()
        with torch.no_grad():
            val_loss = criterion(base(Xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in base.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            train_loss = total_loss / n_batches
            m = _eval_metrics(base, Xv, yv)
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc={m['accuracy']:.3f}  auc={m['auc_roc']:.3f}  "
                f"ece={m['ece']:.3f}"
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"Training done in {elapsed:.1f}s")

    # Restore best weights
    base.load_state_dict(best_state)

    # ── Temperature calibration ───────────────────────────────────────────────
    if verbose:
        print("Calibrating temperature…")

    pre_metrics  = _eval_metrics(base, Xte, yte)
    T = model.calibrate(Xv, yv)

    if verbose:
        print(f"  Temperature T = {T:.4f}")

    post_metrics = _eval_metrics(model, Xte, yte)

    # Reliability diagram
    plot_reliability_diagram(
        pre_metrics["probs"],
        post_metrics["probs"],
        post_metrics["labels"],
        path=out_dir / "reliability.png",
    )

    # ── Save model ────────────────────────────────────────────────────────────
    pt_path   = out_dir / "model_calibrated.pt"
    onnx_path = out_dir / "model.onnx"
    model.save(pt_path)
    model.to_onnx(onnx_path)

    if verbose:
        print(f"  Saved: {pt_path}  ({pt_path.stat().st_size / 1024:.1f} KB)")
        print(f"  Saved: {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # ── Report ────────────────────────────────────────────────────────────────
    results = {k: v for k, v in post_metrics.items() if k not in ("probs", "labels")}
    results["temperature"]     = T
    results["train_samples"]   = int(len(y_tr))
    results["elapsed_seconds"] = round(elapsed, 1)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    if verbose:
        print("\n── Test-set metrics (calibrated model) ──────────────────────")
        targets = {
            "auc_roc":     ("> 0.95", 0.95),
            "sensitivity": ("> 0.92", 0.92),
            "specificity": ("> 0.90", 0.90),
            "f1_macro":    ("> 0.91", 0.91),
            "ece":         ("< 0.05", None),
        }
        for key, (label, threshold) in targets.items():
            val = results[key]
            if threshold is None:
                met = val < 0.05
            else:
                met = val >= threshold
            tick = "✓" if met else "✗"
            print(f"  {tick}  {key:15s} {val:.4f}  (target {label})")
        print(f"\n  ECE before calibration: {pre_metrics['ece']:.4f}")
        print(f"  ECE after  calibration: {post_metrics['ece']:.4f}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroSymmetryNet")
    parser.add_argument("--n",      type=int,   default=6_000, help="Samples per class")
    parser.add_argument("--epochs", type=int,   default=60,    help="Training epochs")
    parser.add_argument("--lr",     type=float, default=3e-3,  help="Learning rate")
    parser.add_argument("--out",    type=str,   default="ai/checkpoints")
    args = parser.parse_args()

    train(
        n_per_class=args.n,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=Path(args.out),
    )
