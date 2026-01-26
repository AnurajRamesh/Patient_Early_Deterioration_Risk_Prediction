#!/usr/bin/env python3
"""
Script 7: Train Model

Trains deterioration prediction models (structured-only, text-only, or multimodal).

Usage:
    # Train multimodal model
    python scripts/07_train_model.py \
        --samples-dir outputs/samples \
        --output-dir outputs/models/multimodal \
        --model multimodal

    # Train structured-only baseline
    python scripts/07_train_model.py \
        --samples-dir outputs/samples \
        --output-dir outputs/models/structured \
        --model structured

    # Train with transformer encoder
    python scripts/07_train_model.py \
        --samples-dir outputs/samples \
        --output-dir outputs/models/transformer \
        --model multimodal \
        --encoder transformer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ICUDataset, create_dataloaders
from src.models.classifier import create_model
from src.training.losses import create_loss
from src.training.trainer import train_model, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train deterioration prediction model"
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
        help="Output directory for model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["multimodal", "structured", "text"],
        default="multimodal",
        help="Model type (default: multimodal)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["lstm", "transformer"],
        default="lstm",
        help="Temporal encoder type (default: lstm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers (default: 4)",
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
    print("Training Deterioration Prediction Model")
    print("=" * 60)
    print(f"Samples dir: {args.samples_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model type: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()

    # Load config
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "model": {
                "temporal_encoder": args.encoder,
                "lstm": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
                "transformer": {"d_model": 128, "nhead": 4, "num_layers": 3},
                "text": {"projection_dim": 256, "dropout": 0.1},
                "fusion": {"hidden_dim": 128},
                "classifier": {"hidden_dim": 64, "dropout": 0.3},
            },
            "training": {
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": 1e-5,
                "max_epochs": args.epochs,
                "patience": 10,
                "gradient_clip": 1.0,
                "use_amp": args.device == "cuda",  # MPS doesn't support AMP
                "loss": "focal",
                "focal_loss": {"alpha": 0.25, "gamma": 2.0},
            },
        }

    # Override with command line args
    config["model"]["temporal_encoder"] = args.encoder
    config["training"]["batch_size"] = args.batch_size
    config["training"]["max_epochs"] = args.epochs
    config["training"]["learning_rate"] = args.lr

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        samples_dir=args.samples_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Get feature dimensions from dataset
    train_dataset = train_loader.dataset
    n_vitals = train_dataset.n_vitals
    n_labs = train_dataset.n_labs

    print(f"Features: {n_vitals} vitals, {n_labs} labs")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type=args.model,
        n_vitals=n_vitals,
        n_labs=n_labs,
        config=config,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

    # Create loss function
    criterion = create_loss(
        loss_type=config["training"].get("loss", "focal"),
        config=config,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer = train_model(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )

    # Save final summary
    summary = {
        "model_type": args.model,
        "encoder": args.encoder,
        "n_vitals": n_vitals,
        "n_labs": n_labs,
        "n_params": n_params,
        "best_val_auroc": trainer.best_val_auroc,
        "final_epoch": len(trainer.history),
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val AUROC: {trainer.best_val_auroc:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
