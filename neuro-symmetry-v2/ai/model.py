"""
NeuroSymmetry model — 3-class MLP + temperature-calibrated wrapper.

Architecture (NeuroSymmetryNet)
--------------------------------
  Input(50) → Linear(128) → BN → ReLU → Dropout(0.3)
            → Linear(64)  → BN → ReLU → Dropout(0.2)
            → Linear(32)  → BN → ReLU
            → Logits(3)

BatchNorm after every linear layer stabilises training on the 400 K+ dataset.

Label convention
----------------
  0 = Normal
  1 = Mild asymmetry
  2 = Severe asymmetry
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

N_FEATURES = 50
N_CLASSES  = 3


# ── Base model ────────────────────────────────────────────────────────────────

class NeuroSymmetryNet(nn.Module):
    """
    Lightweight MLP for 50-dim feature vectors.
    Architecture: 50 → 128 → 64 → 32 → 3 (logits).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_classes:  int = N_CLASSES,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.67),   # ≈ 0.2

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Calibrated wrapper ────────────────────────────────────────────────────────

class CalibratedModel(nn.Module):
    """
    Wraps NeuroSymmetryNet with a learnable scalar temperature T.

    Forward:  logits / T
    .calibrate() optimises T via LBFGS on a held-out validation set
    so that confidence ≈ accuracy (ECE minimisation).
    """

    def __init__(self, base: NeuroSymmetryNet) -> None:
        super().__init__()
        self.base = base
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) / self.temperature.clamp(min=1e-2)

    # ── temperature fitting ───────────────────────────────────────────────

    def calibrate(
        self,
        val_features: Tensor,
        val_labels:   Tensor,
        max_iter:     int = 50,
    ) -> float:
        """
        Fit temperature T on val set using LBFGS + NLL.
        Returns the optimised T value (float).
        """
        self.base.eval()
        nll = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self.base(val_features)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.01, max_iter=max_iter
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = nll(logits / self.temperature.clamp(min=1e-2), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        torch.save(
            {
                "base_state":  self.base.state_dict(),
                "temperature": float(self.temperature.item()),
                "n_features":  self.base.net[0].in_features,
                "n_classes":   self.base.net[-1].out_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "CalibratedModel":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        base = NeuroSymmetryNet(
            n_features=ckpt["n_features"],
            n_classes=ckpt["n_classes"],
        )
        base.load_state_dict(ckpt["base_state"])
        model = cls(base)
        model.temperature.data.fill_(ckpt["temperature"])
        return model

    def to_onnx(self, path: Path) -> None:
        """Export to ONNX (float32, opset 17)."""
        self.eval()
        dummy = torch.zeros(1, self.base.net[0].in_features)
        torch.onnx.export(
            self,
            dummy,
            str(path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
