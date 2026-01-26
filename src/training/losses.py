"""
Loss Functions for ICU Deterioration Prediction

Implements focal loss and weighted BCE for class imbalance handling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (batch,) or (batch, 1) - model outputs before sigmoid
            targets: (batch,) or (batch, 1) - binary labels (0 or 1)

        Returns:
            Scalar loss
        """
        # Flatten inputs
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # p_t = p if y=1 else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight: alpha if y=1 else 1-alpha
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.

    Applies higher weight to positive (minority) class.

    Args:
        pos_weight: Weight for positive class
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: (batch,) or (batch, 1) - model outputs before sigmoid
            targets: (batch,) or (batch, 1) - binary labels

        Returns:
            Scalar loss
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Create weight tensor
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight

        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            weight=weights,
            reduction=self.reduction,
        )

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components.

    Can combine classification loss with auxiliary losses.
    """

    def __init__(
        self,
        main_loss: nn.Module,
        auxiliary_losses: list[tuple[nn.Module, float]] | None = None,
    ):
        """
        Args:
            main_loss: Primary loss function
            auxiliary_losses: List of (loss_fn, weight) tuples
        """
        super().__init__()
        self.main_loss = main_loss
        self.auxiliary_losses = auxiliary_losses or []

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **auxiliary_inputs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            logits: Model outputs
            targets: Ground truth labels
            auxiliary_inputs: Inputs for auxiliary losses

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dict with individual loss components
        """
        # Main loss
        main = self.main_loss(logits, targets)

        loss_dict = {"main": main.item()}
        total = main

        # Auxiliary losses
        for loss_fn, weight in self.auxiliary_losses:
            aux_loss = loss_fn(**auxiliary_inputs)
            total = total + weight * aux_loss
            loss_dict[loss_fn.__class__.__name__] = aux_loss.item()

        loss_dict["total"] = total.item()

        return total, loss_dict


def create_loss(
    loss_type: str,
    config: dict,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: "focal" or "weighted_bce"
        config: Training configuration dict

    Returns:
        Loss module
    """
    loss_config = config.get("training", {})

    if loss_type == "focal":
        focal_config = loss_config.get("focal_loss", {})
        return FocalLoss(
            alpha=focal_config.get("alpha", 0.25),
            gamma=focal_config.get("gamma", 2.0),
        )

    elif loss_type == "weighted_bce":
        # Compute pos_weight from class imbalance
        # Typically pos_weight = n_neg / n_pos
        pos_weight = loss_config.get("pos_weight", 10.0)
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
