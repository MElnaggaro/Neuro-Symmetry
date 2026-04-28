"""
SymmetryNetV2 — lightweight SE + token-attention MLP.

Architecture  (default dims, input_dim=50)
------------------------------------------
  Input(50) → BN
            → Linear(128) → LayerNorm → GELU
            → SqueezeExcitation(128, r=4)       ~8 K params
            → TokenAttention(128, d=8, h=4)     ~3 K params
            → Dropout(0.30) → Linear(64) → GELU → Dropout(0.15)
            → (logits(3), embedding(64))

Total ≈ 42 K parameters — comfortably edge-deployable.

ONNX export wraps the model to return only logits so the output shape
is identical to the V1 CalibratedModel export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Attention blocks ──────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """
    Channel-wise feature recalibration for flat 1-D vectors.
    output = x ⊙ σ(W₂ · ReLU(W₁ · x))
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid,      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid,      channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class TokenAttention(nn.Module):
    """
    Lightweight self-attention over feature dimensions.

    Each scalar feature is embedded to a d-dim vector ('token'),
    multi-head attention is applied, then projected back and added
    residually.  Lets e.g. eye-openness and mouth-tilt features
    interact before the classification head.

    Parameters at defaults (input_dim=128, d=8, heads=4):
        scalar_proj  : 128 × 8          =  1 024
        MHA Q/K/V    : 3 × 8² × 4 heads =    768   (approx)
        out_proj     : 128×8 → 128      =  1 024
        LayerNorm    : 128 × 2          =    256
        Total ≈ 3 K params
    """

    def __init__(
        self,
        input_dim:  int,
        embed_dim:  int   = 8,
        num_heads:  int   = 4,
        dropout:    float = 0.10,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim   = embed_dim
        self.scalar_proj = nn.Linear(1, embed_dim)
        self.attn        = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Per-token projection back to scalar — O(embed_dim) not O(D×embed_dim)
        self.out_proj    = nn.Linear(embed_dim, 1)
        self.norm        = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, D        = x.shape
        tokens      = self.scalar_proj(x.unsqueeze(-1))        # (B, D, embed_dim)
        attended, _ = self.attn(tokens, tokens, tokens)        # (B, D, embed_dim)
        out         = self.out_proj(attended).squeeze(-1)      # (B, D)
        return self.norm(x + out)


# ── Main model ────────────────────────────────────────────────────────────────

class SymmetryNetV2(nn.Module):
    """
    Upgraded 3-class facial-symmetry classifier.

    Returns (logits, embedding) — call forward_logits() for ONNX-compatible
    single-output inference.
    """

    def __init__(
        self,
        input_dim:      int   = 50,    # 50 = V1 features; 75 = V1 + V2 geometric
        hidden_dim:     int   = 128,
        bottleneck_dim: int   = 64,
        num_classes:    int   = 3,
        dropout:        float = 0.30,
        se_reduction:   int   = 4,
        attn_embed_dim: int   = 8,
        attn_heads:     int   = 4,
    ) -> None:
        super().__init__()

        self.input_bn    = nn.BatchNorm1d(input_dim)
        self.input_proj  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.se          = SqueezeExcitation(hidden_dim, se_reduction)
        self.attn        = TokenAttention(hidden_dim, attn_embed_dim, attn_heads)

        self.bottleneck  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.classifier  = nn.Linear(bottleneck_dim, num_classes)

        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_classes    = num_classes

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_embedding(self, x: Tensor) -> Tensor:
        x = self.input_bn(x)
        x = self.input_proj(x)
        x = self.se(x)
        x = self.attn(x)
        return self.bottleneck(x)                        # (B, bottleneck_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (logits, embedding). Both are needed during training."""
        emb    = self.forward_embedding(x)
        logits = self.classifier(emb)
        return logits, emb

    def forward_logits(self, x: Tensor) -> Tensor:
        """Logits only — used for ONNX export and V1-compatible inference."""
        return self.forward(x)[0]


# ── Temperature-calibrated wrapper ───────────────────────────────────────────

class CalibratedModelV2(nn.Module):
    """
    Wraps SymmetryNetV2 with a learnable temperature scalar T.
    calibrated_logits = raw_logits / T
    """

    def __init__(self, base: SymmetryNetV2) -> None:
        super().__init__()
        self.base        = base
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits, emb = self.base(x)
        return logits / self.temperature.clamp(min=0.05), emb

    def forward_logits(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        val_features:    Tensor,
        val_labels:      Tensor,
        max_iter:        int = 100,
        device:          Optional[torch.device] = None,
        eval_batch_size: int = 2_048,
    ) -> float:
        """Fit T on validation set using LBFGS + NLL. Returns T value.

        val_features and val_labels must be CPU tensors; batches are moved to
        *device* (defaults to wherever the model currently lives) one at a time
        so the full validation set never needs to reside on-device at once.
        """
        import logging
        _log = logging.getLogger("neuro_symmetry.v2.model")

        _device: torch.device = device or next(self.base.parameters()).device
        self.base.eval()

        logit_chunks: list[Tensor] = []
        n: int = len(val_features)
        with torch.no_grad():
            for start in range(0, n, eval_batch_size):
                xb: Tensor = val_features[start : start + eval_batch_size].to(_device)
                logits_b, _ = self.base(xb)
                logit_chunks.append(logits_b.cpu())

        logits_raw:     Tensor = torch.cat(logit_chunks, dim=0)   # (n, C) — CPU
        val_labels_cpu: Tensor = val_labels.cpu()

        _log.debug(
            "calibrate(): collected %d logit rows in %d batches of ≤%d",
            len(logits_raw), len(logit_chunks), eval_batch_size,
        )

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure() -> Tensor:
            optimizer.zero_grad()
            t_dev: torch.device = self.temperature.device
            loss = F.cross_entropy(
                logits_raw.to(t_dev) / self.temperature.clamp(min=0.05),
                val_labels_cpu.to(t_dev),
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        T: float = float(self.temperature.item())
        _log.info("Temperature calibrated to %.4f", T)
        return T

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Path,
        thresholds: Optional[list[float]] = None,
        scaler_mean: Optional[list[float]] = None,
        scaler_std:  Optional[list[float]] = None,
    ) -> None:
        torch.save(
            {
                "base_state":    self.base.state_dict(),
                "temperature":   float(self.temperature.item()),
                "input_dim":     self.base.input_dim,
                "hidden_dim":    self.base.hidden_dim,
                "bottleneck_dim": self.base.bottleneck_dim,
                "num_classes":   self.base.num_classes,
                "thresholds":    thresholds or [0.5, 0.5, 0.5],
                "scaler_mean":   scaler_mean,
                "scaler_std":    scaler_std,
                "v2":            True,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "CalibratedModelV2":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        base = SymmetryNetV2(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt.get("hidden_dim", 128),
            bottleneck_dim=ckpt.get("bottleneck_dim", 64),
            num_classes=ckpt.get("num_classes", 3),
        )
        base.load_state_dict(ckpt["base_state"])
        model = cls(base)
        model.temperature.data.fill_(ckpt["temperature"])
        model._ckpt_meta = ckpt   # stash for threshold retrieval
        return model

    @property
    def thresholds(self) -> list[float]:
        return getattr(self, "_ckpt_meta", {}).get("thresholds", [0.5, 0.5, 0.5])

    @property
    def scaler_mean(self) -> Optional[list[float]]:
        return getattr(self, "_ckpt_meta", {}).get("scaler_mean")

    @property
    def scaler_std(self) -> Optional[list[float]]:
        return getattr(self, "_ckpt_meta", {}).get("scaler_std")

    def to_onnx(self, path: Path) -> None:
        """Export logits-only model to ONNX (float32, opset 17)."""
        self.eval()
        dummy = torch.zeros(1, self.base.input_dim)

        # Wrap in a single-output module so ONNX sees only logits, not the
        # (logits, embedding) tuple that CalibratedModelV2.forward() returns.
        class _LogitsOnly(nn.Module):
            def __init__(self, m: "CalibratedModelV2") -> None:
                super().__init__()
                self.m = m
            def forward(self, x: Tensor) -> Tensor:
                return self.m.forward_logits(x)

        torch.onnx.export(
            _LogitsOnly(self),
            dummy,
            str(path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
