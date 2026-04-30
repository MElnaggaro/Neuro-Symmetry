"""
V2 training script.

Loads the existing 50-D feature caches produced by train_real_data.ipynb,
trains SymmetryNetV2 with CombinedLoss + HardNegativeMiner, performs
post-hoc calibration (temperature + optimal thresholds), and exports.

Usage
-----
    cd neuro-symmetry-v2
    python -m ai.v2.train_v2                           # defaults
    python -m ai.v2.train_v2 --epochs 80 --gamma 2.5
    python -m ai.v2.train_v2 --input-dim 75            # if v2 geometric cache exists

Outputs  (ai/checkpoints/)
--------------------------
    model_v2_calibrated.pt    — CalibratedModelV2 (weights + T + thresholds)
    model_v2.onnx             — ONNX export
    metrics_v2.json           — evaluation metrics
    training_curves_v2.png    — loss / recall curves
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ai.calibrate_temperature import compute_ece
from ai.v2.losses import CombinedLoss, LabelSmoothingTransform
from ai.v2.model_v2 import CalibratedModelV2, SymmetryNetV2
from ai.v2.post_hoc import HardNegativeMiner, find_optimal_thresholds

# ── Module-level logger & constants ──────────────────────────────────────────
_log = logging.getLogger("neuro_symmetry.v2.train")

# Maximum samples per forward pass during evaluation.
# 2 048 × 50 features × 4 bytes = ~400 KB per batch — safe on any hardware.
# TokenAttention intermediate tensors scale as O(D²·embed_dim·batch), so
# a single pass over 99 K samples would attempt ~24 GB; 2 K caps it at ~500 MB.
EVAL_BATCH: int = 2_048

CACHE_DIR = Path("ai/cache")
OUT_DIR   = Path("ai/checkpoints")

CLASS_NAMES: list[str] = ["Normal", "Mild", "Severe"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_caches(input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and merge feature caches.  For 50-D mode uses the existing real-data
    caches (produced by train_real_data.ipynb).  For 75-D mode an additional
    v2_geometric_25d.npz cache is required (produced by a separate extraction run).
    """
    sources_50: list[tuple[str, Optional[int]]] = [
        ("celeba_aligned_normal",        0),
        ("celeba_aligned_png_normal",    0),
        ("celeba_wild_normal",           0),
        ("affectnet_all_classes_normal", 0),
        ("w300_normal",                  0),
        ("yfp_palsy_full",               None),   # mixed labels inside
    ]

    parts_f: list[np.ndarray] = []
    parts_l: list[np.ndarray] = []

    for fname, fixed_label in sources_50:
        p = CACHE_DIR / f"{fname}.npz"
        if not p.exists():
            _log.warning("Cache not found, skipping: %s", p.name)
            continue
        d = np.load(p)
        f = d["features"].astype(np.float32)
        l = (d["labels"].astype(np.int64) if fixed_label is None
             else np.full(len(f), fixed_label, dtype=np.int64))
        parts_f.append(f)
        parts_l.append(l)
        _log.info("Cache loaded: %-45s %8d samples", p.name, len(l))

    if not parts_f:
        raise FileNotFoundError(
            "No caches found in ai/cache/.  "
            "Run train_real_data.ipynb first to populate the cache directory."
        )

    X: np.ndarray = np.concatenate(parts_f, axis=0)
    y: np.ndarray = np.concatenate(parts_l, axis=0)

    if input_dim == 75:
        geo_path = CACHE_DIR / "v2_geometric_25d.npz"
        if not geo_path.exists():
            raise FileNotFoundError(
                f"{geo_path} not found.  Run the V2 geometric feature extraction "
                "step first (see ai/v2/feature_engineering.py)."
            )
        geo = np.load(geo_path)
        X_geo: np.ndarray = geo["features"].astype(np.float32)
        if len(X_geo) != len(X):
            raise ValueError(f"Cache size mismatch: 50D={len(X)}, 25D={len(X_geo)}")
        X = np.concatenate([X, X_geo], axis=1)
        _log.info("v2_geometric_25d.npz merged → %d-D features", X.shape[1])

    _log.info("Total corpus: %d samples  features=%d", len(y), X.shape[1])
    for c, name in enumerate(CLASS_NAMES):
        n = int((y == c).sum())
        _log.info("  %s: %d  (%.1f%%)", name, n, 100 * n / len(y))

    return X, y


# ── Memory-safe batched evaluation ───────────────────────────────────────────

@torch.no_grad()
def _eval(
    model: nn.Module,
    X: torch.Tensor,           # CPU tensor — mini-batches are moved to `device` inside
    y: torch.Tensor,           # CPU tensor
    device: torch.device,
    eval_batch_size: int = EVAL_BATCH,
    tag: str = "",
) -> dict:
    """
    Memory-safe batched evaluation.

    X and y must be on CPU.  Each mini-batch of `eval_batch_size` samples is
    sent to `device`, the forward pass runs, and the resulting probabilities
    are immediately moved back to CPU and accumulated as NumPy arrays.

    Peak device memory is bounded by:
        eval_batch_size × input_dim × sizeof(float32)   (feature tensor)
      + eval_batch_size × hidden_dim × embed_dim         (TokenAttention intermediates)
    At eval_batch_size=2048 this is ~500 MB — safe on any CPU or GPU.
    """
    model.eval()

    n: int = len(X)
    prob_chunks: list[np.ndarray] = []

    for start in range(0, n, eval_batch_size):
        xb: torch.Tensor = X[start : start + eval_batch_size].to(device)
        out = model(xb)
        logits_b: torch.Tensor = out[0] if isinstance(out, tuple) else out
        prob_chunks.append(torch.softmax(logits_b, dim=1).cpu().numpy())

    probs: np.ndarray  = np.concatenate(prob_chunks, axis=0)
    labels: np.ndarray = y.numpy()
    preds: np.ndarray  = probs.argmax(axis=1)

    # ── Verification ─────────────────────────────────────────────────────────
    assert len(probs) == n, (
        f"Batched eval produced {len(probs)} rows but expected {n}. "
        "Mini-batch slicing is broken — check EVAL_BATCH and tensor length."
    )
    assert probs.shape == (n, 3), (
        f"probs shape mismatch: got {probs.shape}, expected ({n}, 3)"
    )
    _log.debug(
        "Batched eval verified: %d samples in %d batches of ≤%d  [tag=%s]",
        n, len(prob_chunks), eval_batch_size, tag or "—",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc: float  = float((preds == labels).mean())
    f1: float   = float(f1_score(labels, preds, average="macro", zero_division=0))
    try:
        auc: float = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    n0: int    = int((labels == 0).sum())
    spec: float = (
        float(((preds == 0) & (labels == 0)).sum() / n0) if n0 else float("nan")
    )
    pm: np.ndarray = labels > 0
    sens: float = float((preds[pm] > 0).sum() / pm.sum()) if pm.sum() else float("nan")
    ece: float  = float(compute_ece(probs, labels))

    _log.info(
        "Eval [%s] n=%d | F1=%.4f  AUC=%.4f  sens=%.4f  spec=%.4f  ECE=%.4f",
        tag or "val", n, f1, auc, sens, spec, ece,
    )

    if tag:
        targets: list[tuple[str, float, str, Optional[float]]] = [
            ("F1 macro",    f1,   "> 0.92", 0.92),
            ("AUC-ROC",     auc,  "> 0.95", 0.95),
            ("Sensitivity", sens, "> 0.92", 0.92),
            ("Specificity", spec, "> 0.90", 0.90),
            ("ECE",         ece,  "< 0.05", None),
        ]
        for metric_name, val, label, thr in targets:
            ok: bool = (val < 0.05) if thr is None else (val >= thr)
            _log.info("  %s  %-14s %.4f  (target %s)", "✓" if ok else "✗", metric_name, val, label)
        _log.info("\n%s", confusion_matrix(labels, preds))
        _log.info("\n%s", classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    return dict(
        f1_macro=f1, auc_roc=auc, sensitivity=sens, specificity=spec,
        ece=ece, accuracy=acc, probs=probs, labels=labels,
    )


# ── Main training function ────────────────────────────────────────────────────

def train(
    input_dim:       int   = 50,
    hidden_dim:      int   = 128,
    bottleneck_dim:  int   = 64,
    epochs:          int   = 80,
    batch_size:      int   = 512,
    lr:              float = 3e-4,
    focal_gamma:     float = 2.0,
    arc_margin:      float = 0.45,
    focal_weight:    float = 0.60,
    label_smoothing: float = 0.10,
    hard_mine_every: int   = 15,
    synth_extra:     int   = 15_000,
    seed:            int   = 42,
    out_dir:         Path  = OUT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────────────
    _log.info("Loading feature caches …")
    X_raw, y_all = _load_caches(input_dim)

    if synth_extra > 0:
        from ai.synthetic_augment import build_dataset as _synth
        X_syn, y_syn = _synth(n_per_class=synth_extra, seed=seed + 1)
        if X_syn.shape[1] != input_dim:
            pad = np.zeros((len(X_syn), input_dim - X_syn.shape[1]), dtype=np.float32)
            X_syn = np.concatenate([X_syn, pad], axis=1)
        X_raw = np.concatenate([X_raw, X_syn])
        y_all = np.concatenate([y_all, y_syn])
        _log.info("+ %d synthetic samples", synth_extra * 3)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(y_all))
    X_all: np.ndarray = X_raw[perm].astype(np.float32)
    y_all              = y_all[perm]

    n: int      = len(y_all)
    n_te: int   = int(n * 0.15)
    n_val: int  = int(n * 0.15)
    X_te,  y_te  = X_all[:n_te],             y_all[:n_te]
    X_val, y_val = X_all[n_te:n_te + n_val], y_all[n_te:n_te + n_val]
    X_tr,  y_tr  = X_all[n_te + n_val:],     y_all[n_te + n_val:]
    _log.info("Split — Train %d  Val %d  Test %d", len(y_tr), len(y_val), len(y_te))

    scaler: StandardScaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)

    def _t(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    # All split tensors stay on CPU throughout training.
    # Only training mini-batches and eval mini-batches are moved to `device`.
    Xt, yt   = _t(X_tr,  y_tr)
    Xv, yv   = _t(X_val, y_val)
    Xte, yte = _t(X_te,  y_te)

    counts: np.ndarray = np.array([(y_tr == c).sum() for c in range(3)], dtype=np.float32)
    cw: torch.Tensor   = torch.tensor(len(y_tr) / (3.0 * counts), dtype=torch.float32)
    sampler = WeightedRandomSampler(cw[yt], num_samples=len(yt), replacement=True)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, sampler=sampler)
    _log.info("Class weights: %s", "  ".join(f"c{c}={w:.3f}" for c, w in enumerate(cw.tolist())))

    # ── Model & loss ─────────────────────────────────────────────────────────
    base: SymmetryNetV2       = SymmetryNetV2(input_dim, hidden_dim, bottleneck_dim).to(device)
    model: CalibratedModelV2  = CalibratedModelV2(base).to(device)
    _log.info("SymmetryNetV2  params=%d", base.n_params)

    focal_alpha: torch.Tensor = cw.to(device)
    criterion: CombinedLoss   = CombinedLoss(
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        arc_in_features=bottleneck_dim,
        arc_margin=arc_margin,
        focal_weight=focal_weight,
        arc_weight=1.0 - focal_weight,
    ).to(device)

    smoother: LabelSmoothingTransform = LabelSmoothingTransform(smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        list(base.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler_amp = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1:          float          = 0.0
    best_state:       Optional[dict] = None
    start_epoch:      int            = 1
    resume_ckpt_path: Path           = out_dir / "resume_checkpoint.pt"

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_ckpt_path.exists():
        _ckpt: dict = torch.load(resume_ckpt_path, map_location=device, weights_only=True)
        base.load_state_dict(_ckpt["model_state_dict"])
        # best_state_dict is saved alongside the required fields so that a
        # resumed run which finds no new best can still call base.load_state_dict.
        best_state  = _ckpt.get("best_state_dict") or {k: v.clone() for k, v in base.state_dict().items()}
        optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
        best_f1     = float(_ckpt["best_f1"])
        start_epoch = int(_ckpt["epoch"]) + 1
        _log.info(
            "Resuming training from epoch %d  (best_val_f1=%.4f  loaded from %s)",
            start_epoch, best_f1, resume_ckpt_path,
        )

    history: list[dict]   = []
    t0: float             = time.time()
    _interrupted_epoch: int = start_epoch - 1   # updated at the top of every iteration

    def _save_resume_ckpt(at_epoch: int) -> None:
        torch.save(
            {
                "epoch":                at_epoch,
                "model_state_dict":     {k: v.clone() for k, v in base.state_dict().items()},
                "best_state_dict":      best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1":              best_f1,
            },
            resume_ckpt_path,
        )

    try:
        for epoch in range(start_epoch, epochs + 1):
            _interrupted_epoch = epoch   # always valid in the except handler

            if epoch > 1 and (epoch - 1) % hard_mine_every == 0:
                _log.info("[ep%d] Rebuilding hard-negative sampler …", epoch)
                miner  = HardNegativeMiner(Xt, y_tr, base, device)
                loader = DataLoader(
                    TensorDataset(Xt, yt), batch_size=batch_size,
                    sampler=miner.get_sampler(),
                )

            base.train()
            criterion.train()
            tl: float = 0.0
            nb: int   = 0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                smooth: torch.Tensor = torch.stack([smoother(int(l)) for l in yb.cpu()]).to(device)

                optimizer.zero_grad(set_to_none=True)
                if scaler_amp:
                    with torch.amp.autocast("cuda"):
                        logits, emb = base(xb)
                        loss: torch.Tensor = criterion(logits, emb, yb, smooth)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(base.parameters(), 1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    logits, emb = base(xb)
                    loss = criterion(logits, emb, yb, smooth)
                    loss.backward()
                    nn.utils.clip_grad_norm_(base.parameters(), 1.0)
                    optimizer.step()

                tl += loss.item()
                nb += 1

            scheduler.step()

            # ── Validation: CPU tensors, device transfer happens inside _eval ─────
            m: dict = _eval(base, Xv, yv, device, tag="")
            history.append(dict(epoch=epoch, train_loss=tl / nb, f1=m["f1_macro"]))

            if m["f1_macro"] > best_f1:
                best_f1    = m["f1_macro"]
                best_state = {k: v.clone() for k, v in base.state_dict().items()}

                if best_f1 > 0.96:
                    ckpt_path: Path = out_dir / "checkpoint_f1_excellent.pt"
                    torch.save(best_state, ckpt_path)
                    _log.info(
                        "🔥 High-performing checkpoint saved: %s  (F1=%.4f)",
                        ckpt_path, best_f1,
                    )

            if epoch % 10 == 0 or epoch == 1:
                _log.info(
                    "ep%3d  loss=%.4f  val_f1=%.4f  sens=%.3f  spec=%.3f",
                    epoch, tl / nb, m["f1_macro"], m["sensitivity"], m["specificity"],
                )

            # ── End-of-epoch resume checkpoint (overwrites every epoch) ─────────
            _save_resume_ckpt(epoch)

    except KeyboardInterrupt:
        _log.info(
            "KeyboardInterrupt caught at epoch %d — saving resume checkpoint …",
            _interrupted_epoch,
        )
        _save_resume_ckpt(_interrupted_epoch)
        _log.info(
            "Resume checkpoint saved → %s  Proceeding to post-hoc calibration …",
            resume_ckpt_path,
        )

    elapsed: float = time.time() - t0
    _log.info("Training done in %.1fs  best_val_f1=%.4f", elapsed, best_f1)
    base.load_state_dict(best_state)

    # ── Post-hoc calibration ──────────────────────────────────────────────────
    # model.calibrate() is also batched (see model_v2.py) — pass CPU tensors.
    _log.info("Calibrating temperature …")
    T: float = model.calibrate(Xv, yv, device=device, eval_batch_size=EVAL_BATCH)
    _log.info("Temperature T = %.4f", T)

    # Collect calibrated probs in batches for threshold search
    cal_prob_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(Xv), EVAL_BATCH):
            xb = Xv[start : start + EVAL_BATCH].to(device)
            logits_b, _ = model(xb)
            cal_prob_chunks.append(torch.softmax(logits_b, dim=1).cpu().numpy())
    cal_probs: np.ndarray = np.concatenate(cal_prob_chunks, axis=0)

    assert len(cal_probs) == len(Xv), (
        f"Calibrated probs length {len(cal_probs)} != val set length {len(Xv)}"
    )
    _log.debug("Calibrated probs collected: %d rows, %.1f MB", len(cal_probs), cal_probs.nbytes / 1e6)

    thresholds: list[float] = find_optimal_thresholds(cal_probs, y_val)
    _log.info(
        "Optimal thresholds: Normal=%.3f  Mild=%.3f  Severe=%.3f",
        thresholds[0], thresholds[1], thresholds[2],
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    _eval(model, Xte, yte, device, tag="Post-calibration (test set)")

    # ── Save ──────────────────────────────────────────────────────────────────
    pt_path: Path   = out_dir / "model_v2_calibrated.pt"
    onnx_path: Path = out_dir / "model_v2.onnx"

    model.save(
        pt_path,
        thresholds=thresholds,
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.scale_.tolist(),
    )
    _log.info("Saved: %s  (%.1f KB)", pt_path, pt_path.stat().st_size / 1024)

    try:
        model.to_onnx(onnx_path)
        _log.info("Saved: %s  (%.1f KB)", onnx_path, onnx_path.stat().st_size / 1024)
    except Exception as exc:
        _log.warning("ONNX export failed (install onnx): %s", exc)

    try:
        import matplotlib.pyplot as plt
        eps  = [h["epoch"]      for h in history]
        loss = [h["train_loss"] for h in history]
        f1s  = [h["f1"]         for h in history]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(eps, loss); a1.set(title="Train Loss", xlabel="Epoch", ylabel="Combined Loss")
        a2.plot(eps, f1s, color="seagreen")
        a2.axhline(0.92, color="red", ls="--", lw=1, label="target 0.92")
        a2.set(title="Val F1 Macro", xlabel="Epoch", ylabel="F1"); a2.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves_v2.png", dpi=120)
        plt.close()
    except ImportError:
        pass

    # Metrics JSON — use batched eval (not a duplicate monolithic pass)
    final_m: dict = _eval(model, Xte, yte, device)
    results: dict = {k: v for k, v in final_m.items() if k not in ("probs", "labels")}
    results.update(dict(
        temperature=T,
        thresholds=thresholds,
        train_samples=int(len(y_tr)),
        elapsed_seconds=round(elapsed, 1),
        input_dim=input_dim,
        architecture="SymmetryNetV2",
    ))
    (out_dir / "metrics_v2.json").write_text(json.dumps(results, indent=2))
    _log.info("metrics_v2.json written.")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train SymmetryNetV2")
    parser.add_argument("--input-dim",    type=int,   default=50)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--bottleneck",   type=int,   default=64)
    parser.add_argument("--epochs",       type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--gamma",        type=float, default=2.0,  help="Focal Loss γ")
    parser.add_argument("--arc-margin",   type=float, default=0.45, help="ArcFace margin m")
    parser.add_argument("--synth-extra",  type=int,   default=15_000)
    parser.add_argument("--out",          type=str,   default="ai/checkpoints")
    args = parser.parse_args()

    train(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        focal_gamma=args.gamma,
        arc_margin=args.arc_margin,
        synth_extra=args.synth_extra,
        out_dir=Path(args.out),
    )
