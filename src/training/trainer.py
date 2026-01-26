"""
Training Module for ICU Deterioration Prediction

Implements training loop with early stopping, mixed precision, and logging.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..evaluation.metrics import compute_auroc, compute_auprc


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    warmup_epochs: int = 3
    use_amp: bool = True
    log_every: int = 100
    checkpoint_every: int = 5
    save_best_only: bool = True


@dataclass
class TrainingMetrics:
    """Metrics from one epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_auroc: float
    val_auprc: float
    learning_rate: float
    epoch_time: float


class Trainer:
    """
    Trainer for ICU deterioration prediction models.

    Handles:
    - Training loop with gradient clipping
    - Mixed precision training (AMP)
    - Early stopping on validation AUROC
    - Learning rate scheduling
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        output_dir: Path,
        device: str = "cuda",
    ):
        """
        Args:
            model: Model to train
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Directory for checkpoints and logs
            device: Device to train on
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # Mixed precision (only for CUDA, not MPS)
        self.use_amp = config.use_amp and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Tracking
        self.best_val_auroc = 0.0
        self.epochs_without_improvement = 0
        self.history: list[TrainingMetrics] = []

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            vitals = batch["vitals"].to(self.device)
            labs = batch["labs"].to(self.device)
            mask = batch["mask"].to(self.device)
            static = batch["static"].to(self.device)
            embedding = batch["embedding"].to(self.device)
            has_notes = batch["has_notes"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP (CUDA only)
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(
                        vitals, labs, mask, static, embedding, has_notes
                    )
                    loss = self.criterion(logits.squeeze(), labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

                # Update
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard forward pass
                logits, _ = self.model(
                    vitals, labs, mask, static, embedding, has_notes
                )
                loss = self.criterion(logits.squeeze(), labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

                # Update
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Logging
            if batch_idx > 0 and batch_idx % self.config.log_every == 0:
                avg_loss = total_loss / n_batches
                print(f"    Batch {batch_idx}/{len(self.train_loader)}, Loss: {avg_loss:.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        """Validate model. Returns (loss, auroc, auprc)."""
        self.model.eval()

        total_loss = 0.0
        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            # Move to device
            vitals = batch["vitals"].to(self.device)
            labs = batch["labs"].to(self.device)
            mask = batch["mask"].to(self.device)
            static = batch["static"].to(self.device)
            embedding = batch["embedding"].to(self.device)
            has_notes = batch["has_notes"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            logits, _ = self.model(
                vitals, labs, mask, static, embedding, has_notes
            )
            loss = self.criterion(logits.squeeze(), labels)

            total_loss += loss.item()

            # Collect predictions
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        auroc = compute_auroc(all_labels, all_probs)
        auprc = compute_auprc(all_labels, all_probs)

        return avg_loss, auroc or 0.0, auprc or 0.0

    def train(self) -> list[TrainingMetrics]:
        """
        Full training loop.

        Returns:
            List of per-epoch metrics
        """
        print(f"Starting training for {self.config.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()

            # Train
            print(f"\nEpoch {epoch}/{self.config.max_epochs}")
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_auroc, val_auprc = self.validate()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auroc=val_auroc,
                val_auprc=val_auprc,
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
            self.history.append(metrics)

            # Print summary
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AUROC: {val_auroc:.4f}")
            print(f"  Val AUPRC: {val_auprc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")

            # Check for improvement
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.epochs_without_improvement = 0
                print(f"  New best AUROC! Saving model...")
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")

                # Save periodic checkpoint
                if epoch % self.config.checkpoint_every == 0:
                    if not self.config.save_best_only:
                        self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save training history
        self.save_history()

        print(f"\nTraining complete!")
        print(f"Best Val AUROC: {self.best_val_auroc:.4f}")

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auroc": self.best_val_auroc,
            "config": asdict(self.config),
        }

        if is_best:
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        history_data = [asdict(m) for m in self.history]
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        print(f"Saved training history to {history_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_auroc = checkpoint["best_val_auroc"]
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"]


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: dict,
    device: str = "cuda",
) -> Trainer:
    """
    Convenience function to train a model.

    Args:
        model: Model to train
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory for outputs
        config: Full configuration dict
        device: Device to train on

    Returns:
        Trained Trainer instance
    """
    # Extract training config
    train_config = config.get("training", {})

    training_config = TrainingConfig(
        batch_size=train_config.get("batch_size", 64),
        learning_rate=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 1e-5),
        max_epochs=train_config.get("max_epochs", 100),
        patience=train_config.get("patience", 10),
        gradient_clip=train_config.get("gradient_clip", 1.0),
        warmup_epochs=train_config.get("warmup_epochs", 3),
        use_amp=train_config.get("use_amp", True),
        log_every=train_config.get("log_every", 100),
        checkpoint_every=train_config.get("checkpoint_every", 5),
        save_best_only=train_config.get("save_best_only", True),
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        output_dir=output_dir,
        device=device,
    )

    trainer.train()

    return trainer
