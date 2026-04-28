"""
V2 loss functions.

FocalLoss    — focuses gradient on hard examples; down-weights easy Normals
ArcMarginHead — adds angular margin to force inter-class separation in embedding space
CombinedLoss  — weighted sum of Focal + ArcFace losses

Why this combination fixes the V1 bottleneck
--------------------------------------------
V1 used weighted cross-entropy.  The model learned to maximise recall (good)
but not precision for the Mild class — it over-fired on Normal samples because
the decision boundary was poorly defined in embedding space.

Focal Loss (γ=2): (1 − p_t)² multiplier exponentially suppresses easy
examples, concentrating gradient on the 1 958 hard Normal→Mild false positives.

ArcFace Margin:  normalises embeddings and class weight vectors, then adds
angular margin m to the target class's cosine score.  This forces all three
class centres to be separated by at least m radians (≈ 25° at m=0.45),
directly pushing the Normal and Mild clusters apart.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Supports hard (Long) and soft (Float, label-smoothed) targets.
    """

    def __init__(
        self,
        alpha:     Optional[Tensor] = None,  # (C,) per-class weight
        gamma:     float = 2.0,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() == 2:
            # Soft labels (label-smoothed): manual cross-entropy
            log_p = F.log_softmax(logits, dim=-1)
            ce    = -(targets * log_p).sum(dim=-1)
        else:
            ce = F.cross_entropy(
                logits, targets,
                weight=self.alpha,
                reduction="none",
            )

        focal = (1.0 - torch.exp(-ce)) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class ArcMarginHead(nn.Module):
    """
    ArcFace classification head.

    Adds angular margin m (radians) to the target-class cosine similarity,
    then scales by s before softmax.  Forces cluster separation of at least
    m radians between any two class centres in the embedding hypersphere.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss" CVPR 2019.
    """

    def __init__(
        self,
        in_features:  int,
        num_classes:  int   = 3,
        scale:        float = 30.0,
        margin:       float = 0.45,
    ) -> None:
        super().__init__()
        self.scale  = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trig constants to avoid redundant ops per forward
        cos_m, sin_m = math.cos(margin), math.sin(margin)
        self.register_buffer("_cos_m", torch.tensor(cos_m))
        self.register_buffer("_sin_m", torch.tensor(sin_m))
        # Stable boundary: below cos(π−m) the linear approximation kicks in
        self.register_buffer("_th",   torch.tensor(math.cos(math.pi - margin)))
        self.register_buffer("_mm",   torch.tensor(sin_m * margin))

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # (B, C)
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))

        # cos(θ + m) = cosθ·cosm − sinθ·sinm
        phi = cosine * self._cos_m - sine * self._sin_m
        # Use linear approx in the unstable region (θ > π − m)
        phi = torch.where(cosine > self._th, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        output  = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(output * self.scale, labels)


class CombinedLoss(nn.Module):
    """
    Focal Loss + ArcFace, weighted sum.

    Training call signature:
        loss = criterion(logits, embeddings, hard_labels, smooth_labels=None)

    hard_labels  : LongTensor (B,)  — required by ArcFace
    smooth_labels: FloatTensor (B, C) — if provided, Focal uses these;
                   otherwise Focal uses hard_labels
    """

    def __init__(
        self,
        focal_alpha:     Optional[Tensor] = None,
        focal_gamma:     float = 2.0,
        arc_in_features: int   = 64,
        arc_num_classes: int   = 3,
        arc_scale:       float = 30.0,
        arc_margin:      float = 0.45,
        focal_weight:    float = 0.60,
        arc_weight:      float = 0.40,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.arc   = ArcMarginHead(arc_in_features, arc_num_classes, arc_scale, arc_margin)
        self.fw    = focal_weight
        self.aw    = arc_weight

    def forward(
        self,
        logits:        Tensor,
        embeddings:    Tensor,
        hard_labels:   Tensor,
        smooth_labels: Optional[Tensor] = None,
    ) -> Tensor:
        arc_loss   = self.arc(embeddings, hard_labels)
        focal_in   = smooth_labels if smooth_labels is not None else hard_labels
        focal_loss = self.focal(logits, focal_in)
        return self.fw * focal_loss + self.aw * arc_loss


# ── Label smoothing transform ─────────────────────────────────────────────────

class LabelSmoothingTransform:
    """
    [1, 0, 0] → [confidence, fill, fill]
    where confidence = 1 − smoothing and fill = smoothing / (C − 1).
    """

    def __init__(self, num_classes: int = 3, smoothing: float = 0.10) -> None:
        self.num_classes = num_classes
        self.confidence  = 1.0 - smoothing
        self.fill        = smoothing / max(num_classes - 1, 1)

    def __call__(self, label: int) -> torch.Tensor:
        t       = torch.full((self.num_classes,), self.fill)
        t[label] = self.confidence
        return t
