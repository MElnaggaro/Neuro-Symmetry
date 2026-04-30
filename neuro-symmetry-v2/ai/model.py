"""
V1 NeuroSymmetryNet — 50→128→64→32→3 MLP with BatchNorm and temperature scaling.

This is the V1 architecture kept for backward compatibility with existing tests
and the V1 ONNX checkpoint.  New training uses ai.v2.model_v2.SymmetryNetV2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


N_FEATURES: int = 50
N_CLASSES:  int = 3


class NeuroSymmetryNet(nn.Module):
    """
    V1 classifier: BatchNorm → 128 → 64 → 32 → 3.
    Returns raw logits (B, 3).
    """

    def __init__(
        self,
        input_dim:  int   = N_FEATURES,
        num_classes: int  = N_CLASSES,
        dropout:    float = 0.30,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CalibratedModel(nn.Module):
    """
    Wraps NeuroSymmetryNet with a learnable temperature scalar T.
    Output = raw_logits / T.
    """

    def __init__(self, base: NeuroSymmetryNet) -> None:
        super().__init__()
        self.base        = base
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) / self.temperature.clamp(min=0.05)

    def calibrate(
        self,
        val_features: Tensor,
        val_labels:   Tensor,
        max_iter:     int = 100,
    ) -> float:
        """Fit T on validation set using LBFGS + NLL. Returns T value."""
        self.base.eval()
        with torch.no_grad():
            logits_raw = self.base(val_features).detach()

        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure() -> Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(
                logits_raw / self.temperature.clamp(min=0.05), val_labels
            )
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.item())

    def save(
        self,
        path: Path,
        thresholds: Optional[list[float]] = None,
        scaler_mean: Optional[list[float]] = None,
        scaler_std:  Optional[list[float]] = None,
    ) -> None:
        torch.save(
            {
                "base_state":  self.base.state_dict(),
                "temperature": float(self.temperature.item()),
                "thresholds":  thresholds or [0.5, 0.5, 0.5],
                "scaler_mean": scaler_mean,
                "scaler_std":  scaler_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "CalibratedModel":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        base = NeuroSymmetryNet()
        base.load_state_dict(ckpt["base_state"])
        model = cls(base)
        model.temperature.data.fill_(ckpt["temperature"])
        return model

    def to_onnx(self, path: Path) -> None:
        self.eval()
        dummy = torch.zeros(1, N_FEATURES)

        class _LogitsOnly(nn.Module):
            def __init__(self, m: "CalibratedModel") -> None:
                super().__init__()
                self.m = m
            def forward(self, x: Tensor) -> Tensor:
                return self.m(x)

        torch.onnx.export(
            _LogitsOnly(self),
            dummy,
            str(path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )


__all__ = ["NeuroSymmetryNet", "CalibratedModel", "N_FEATURES", "N_CLASSES"]
