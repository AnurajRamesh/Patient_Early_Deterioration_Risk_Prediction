"""
Interpretability Module for ICU Deterioration Prediction

Visualizes attention weights, feature importance, and model explanations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Visualization imports (with fallback)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_temporal_attention(
    sample: dict,
    model: nn.Module,
    feature_names: list[str],
    output_path: Optional[Path] = None,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Visualize attention weights over temporal features for a single sample.

    Args:
        sample: Sample dict from dataset
        model: Trained model
        feature_names: Names of features (vitals + labs)
        output_path: Path to save figure (if None, shows plot)
        device: Device to run inference on

    Returns:
        Attention weights array (seq_len,) or None if visualization unavailable
    """
    model.eval()

    # Prepare inputs
    vitals = sample["vitals"].unsqueeze(0).to(device)
    labs = sample["labs"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)
    embedding = sample["embedding"].unsqueeze(0).to(device)
    has_notes = sample["has_notes"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_info = model(vitals, labs, mask, static, embedding, has_notes)

    # Get attention weights
    temporal_attn = attn_info.get("temporal_attn")
    if temporal_attn is None:
        return None

    attn_weights = temporal_attn.squeeze().cpu().numpy()

    # Get prediction
    prob = torch.sigmoid(logits).item()

    if not HAS_MATPLOTLIB:
        return attn_weights

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 3]})

    # Plot 1: Attention over time
    ax1 = axes[0]
    seq_len = len(attn_weights)
    ax1.bar(range(seq_len), attn_weights, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Attention")
    ax1.set_title(f"Temporal Attention Weights (Prediction: {prob:.3f})")

    # Plot 2: Feature values over time with attention overlay
    ax2 = axes[1]

    # Combine vitals and labs
    combined = torch.cat([sample["vitals"], sample["labs"]], dim=-1).numpy()

    # Heatmap of feature values
    im = ax2.imshow(
        combined.T,
        aspect="auto",
        cmap="RdYlBu_r",
        interpolation="nearest",
    )

    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Feature")
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Normalized Value")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return attn_weights


def visualize_note_attention(
    sample: dict,
    model: nn.Module,
    note_texts: list[str],
    output_path: Optional[Path] = None,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Visualize attention weights over clinical notes.

    Args:
        sample: Sample dict from dataset
        model: Trained model
        note_texts: List of note text snippets
        output_path: Path to save figure
        device: Device to run inference on

    Returns:
        Note attention weights or None
    """
    model.eval()

    # This would require the model to have note aggregation
    # For now, return None if not applicable
    return None


def get_feature_importance(
    model: nn.Module,
    dataloader,
    feature_names: list[str],
    n_samples: int = 1000,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Compute feature importance via gradient-based attribution.

    Uses integrated gradients approximation.

    Args:
        model: Trained model
        dataloader: Data loader
        feature_names: Names of features
        n_samples: Number of samples to use
        device: Device to run on

    Returns:
        Dict mapping feature name to importance score
    """
    model.eval()
    model.to(device)

    importances = np.zeros(len(feature_names))
    n_processed = 0

    for batch in dataloader:
        if n_processed >= n_samples:
            break

        # Move to device
        vitals = batch["vitals"].to(device).requires_grad_(True)
        labs = batch["labs"].to(device).requires_grad_(True)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device)
        embedding = batch["embedding"].to(device)
        has_notes = batch["has_notes"].to(device)

        # Forward pass
        logits, _ = model(vitals, labs, mask, static, embedding, has_notes)

        # Backward for gradients
        logits.sum().backward()

        # Compute importance (gradient * input)
        if vitals.grad is not None:
            vitals_importance = (vitals.grad.abs() * vitals.abs()).mean(dim=(0, 1))
            importances[:len(vitals_importance)] += vitals_importance.cpu().numpy()

        if labs.grad is not None:
            labs_importance = (labs.grad.abs() * labs.abs()).mean(dim=(0, 1))
            n_vitals = vitals.shape[-1]
            importances[n_vitals:n_vitals + len(labs_importance)] += labs_importance.cpu().numpy()

        n_processed += len(batch["vitals"])

        # Clear gradients
        model.zero_grad()

    # Normalize
    importances /= max(n_processed, 1)
    importances /= importances.sum() + 1e-8

    return {name: float(imp) for name, imp in zip(feature_names, importances)}


def plot_feature_importance(
    importance_dict: dict[str, float],
    output_path: Optional[Path] = None,
    top_k: int = 20,
):
    """
    Plot feature importance bar chart.

    Args:
        importance_dict: Dict mapping feature name to importance
        output_path: Path to save figure
        top_k: Number of top features to show
    """
    if not HAS_MATPLOTLIB:
        return

    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_k]

    names = [item[0] for item in top_items]
    values = [item[1] for item in top_items]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_k} Feature Importance")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    mean_predicted: np.ndarray,
    fraction_positive: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    output_path: Optional[Path] = None,
):
    """
    Plot reliability diagram (calibration curve).

    Args:
        mean_predicted: Mean predicted probability per bin
        fraction_positive: Actual positive fraction per bin
        bin_counts: Number of samples per bin
        ece: Expected Calibration Error
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    # Calibration curve
    nonzero = bin_counts > 0
    ax.plot(
        mean_predicted[nonzero],
        fraction_positive[nonzero],
        "o-",
        color="steelblue",
        label=f"Model (ECE={ece:.3f})",
    )

    # Histogram of predictions
    ax2 = ax.twinx()
    bins = np.linspace(0, 1, len(bin_counts) + 1)
    ax2.hist(
        mean_predicted[nonzero],
        bins=bins,
        weights=bin_counts[nonzero],
        alpha=0.3,
        color="gray",
    )
    ax2.set_ylabel("Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[Path] = None,
):
    """
    Plot ROC curves for multiple models.

    Args:
        results: Dict mapping model name to (fpr, tpr, auroc)
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Random classifier
    ax.plot([0, 1], [0, 1], "k--", label="Random")

    # Plot each model
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for (name, (fpr, tpr, auroc)), color in zip(results.items(), colors):
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auroc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
