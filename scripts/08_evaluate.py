#!/usr/bin/env python3
"""
Script 8: Evaluate Model

Evaluates trained model on test set with comprehensive metrics and visualizations.

Usage:
    python scripts/08_evaluate.py \
        --model-dir outputs/models/multimodal \
        --samples-dir outputs/samples \
        --output-dir outputs/results
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ICUDataset
from src.models.classifier import create_model
from src.evaluation.metrics import (
    compute_metrics,
    compute_calibration,
    evaluate_subgroups,
)
from src.evaluation.interpretability import (
    plot_calibration_curve,
    plot_feature_importance,
    get_feature_importance,
)


def load_model(model_dir: Path, config: dict, n_vitals: int, n_labs: int, device: str):
    """Load trained model from checkpoint."""
    # Get model type from summary or config
    summary_path = model_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        model_type = summary.get("model_type", "multimodal")
    else:
        model_type = "multimodal"

    # Create model
    model = create_model(
        model_type=model_type,
        n_vitals=n_vitals,
        n_labs=n_labs,
        config=config,
    )

    # Load weights
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: str,
) -> tuple[list[int], list[float]]:
    """Run inference on dataloader and collect predictions."""
    model.eval()

    all_probs = []
    all_labels = []

    for batch in dataloader:
        # Move to device
        vitals = batch["vitals"].to(device)
        labs = batch["labs"].to(device)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device)
        embedding = batch["embedding"].to(device)
        has_notes = batch["has_notes"].to(device)
        labels = batch["label"]

        # Forward pass
        logits, _ = model(vitals, labs, mask, static, embedding, has_notes)

        # Convert to probabilities
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()

        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained deterioration prediction model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory with trained model",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help="Directory with sample files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print("=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    print(f"Model dir: {args.model_dir}")
    print(f"Samples dir: {args.samples_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Load config
    config_path = args.model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create test dataset
    print("Loading test data...")
    test_dataset = ICUDataset(
        samples_dir=args.samples_dir,
        split="test",
        normalize=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Test samples: {len(test_dataset):,}")

    # Load model
    print("\nLoading model...")
    model = load_model(
        model_dir=args.model_dir,
        config=config,
        n_vitals=test_dataset.n_vitals,
        n_labs=test_dataset.n_labs,
        device=args.device,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    y_true, y_prob = evaluate_model(model, test_loader, args.device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_prob)

    print("\nTest Metrics:")
    print(f"  AUROC:       {metrics.auroc:.4f}")
    print(f"  AUPRC:       {metrics.auprc:.4f}")
    print(f"  F1:          {metrics.f1:.4f}")
    print(f"  Precision:   {metrics.precision:.4f}")
    print(f"  Recall:      {metrics.recall:.4f}")
    print(f"  Specificity: {metrics.specificity:.4f}")
    print(f"  Brier Score: {metrics.brier_score:.4f}")
    print(f"  Threshold:   {metrics.optimal_threshold:.3f}")

    # Calibration
    print("\nComputing calibration...")
    mean_pred, frac_pos, bin_counts, ece = compute_calibration(y_true, y_prob)
    print(f"  ECE: {ece:.4f}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    results = {
        "metrics": asdict(metrics),
        "calibration": {
            "ece": ece,
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
            "bin_counts": bin_counts.tolist(),
        },
    }

    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    predictions = {
        "y_true": y_true,
        "y_prob": y_prob,
    }
    np.savez(args.output_dir / "predictions.npz", **predictions)

    # Generate visualizations
    print("\nGenerating visualizations...")

    try:
        # Calibration curve
        plot_calibration_curve(
            mean_pred, frac_pos, bin_counts, ece,
            output_path=args.output_dir / "calibration_curve.png",
        )
        print("  - Saved calibration curve")

        # Feature importance
        feature_names = test_dataset.vitals_cols + test_dataset.labs_cols
        importance = get_feature_importance(
            model, test_loader, feature_names,
            n_samples=1000, device=args.device,
        )
        plot_feature_importance(
            importance,
            output_path=args.output_dir / "feature_importance.png",
        )
        print("  - Saved feature importance")

        # Save feature importance
        with open(args.output_dir / "feature_importance.json", "w") as f:
            json.dump(importance, f, indent=2)

    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey Results:")
    print(f"  AUROC: {metrics.auroc:.4f}")
    print(f"  AUPRC: {metrics.auprc:.4f}")
    print(f"  ECE:   {ece:.4f}")


if __name__ == "__main__":
    main()
