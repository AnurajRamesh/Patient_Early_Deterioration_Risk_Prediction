#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly.
This simulates what the Colab notebook does.
"""
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 60)
print("TESTING TRAINING PIPELINE")
print("=" * 60)

# Check device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

# Configuration
SAMPLES_DIR = Path("outputs/samples")
EMBEDDINGS_DIR = Path("outputs/embeddings")
BATCH_SIZE = 64
NUM_TEST_BATCHES = 5  # Only test a few batches

print(f"\nSamples dir: {SAMPLES_DIR}")
print(f"Embeddings dir: {EMBEDDINGS_DIR}")

# Test 1: Load dataset
print("\n" + "=" * 60)
print("TEST 1: Loading Dataset")
print("=" * 60)

try:
    from src.data.dataset import ICUDataset, create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        samples_dir=SAMPLES_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 for debugging
        embeddings_dir=EMBEDDINGS_DIR,
    )

    train_dataset = train_loader.dataset
    n_vitals = train_dataset.n_vitals
    n_labs = train_dataset.n_labs

    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Val samples: {len(val_loader.dataset):,}")
    print(f"✓ Test samples: {len(test_loader.dataset):,}")
    print(f"✓ Features: {n_vitals} vitals, {n_labs} labs")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Get a batch
print("\n" + "=" * 60)
print("TEST 2: Getting Batches")
print("=" * 60)

try:
    batch = next(iter(train_loader))
    print(f"✓ Batch keys: {list(batch.keys())}")
    print(f"✓ Vitals shape: {batch['vitals'].shape}")
    print(f"✓ Labs shape: {batch['labs'].shape}")
    print(f"✓ Embedding shape: {batch['embedding'].shape}")
    print(f"✓ Labels shape: {batch['label'].shape}")
    print(f"✓ Labels sum (positive): {batch['label'].sum().item()}")

    # Check for NaN/Inf
    vitals_nan = torch.isnan(batch['vitals']).any().item()
    labs_nan = torch.isnan(batch['labs']).any().item()
    emb_nan = torch.isnan(batch['embedding']).any().item()

    print(f"✓ Vitals has NaN: {vitals_nan}")
    print(f"✓ Labs has NaN: {labs_nan}")
    print(f"✓ Embedding has NaN: {emb_nan}")

    if vitals_nan or labs_nan or emb_nan:
        print("✗ WARNING: NaN values detected in batch!")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create model
print("\n" + "=" * 60)
print("TEST 3: Creating Model")
print("=" * 60)

try:
    from src.models.classifier import create_model

    config = {
        "model": {
            "temporal_encoder": "lstm",
            "lstm": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
            "text": {"projection_dim": 256, "dropout": 0.1},
            "fusion": {"hidden_dim": 128},
            "classifier": {"hidden_dim": 64, "dropout": 0.3},
        },
        "training": {
            "loss": "focal",
            "focal_loss": {"alpha": 0.25, "gamma": 2.0},
        },
    }

    model = create_model(
        model_type="multimodal",
        n_vitals=n_vitals,
        n_labs=n_labs,
        config=config,
    )
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass
print("\n" + "=" * 60)
print("TEST 4: Forward Pass")
print("=" * 60)

try:
    model.eval()
    with torch.no_grad():
        vitals = batch["vitals"].to(DEVICE)
        labs = batch["labs"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        embedding = batch["embedding"].to(DEVICE)
        has_notes = batch["has_notes"].to(DEVICE)

        logits, attention = model(vitals, labs, mask, static, embedding, has_notes)

        print(f"✓ Logits shape: {logits.shape}")
        print(f"✓ Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"✓ Logits has NaN: {torch.isnan(logits).any().item()}")
        print(f"✓ Logits has Inf: {torch.isinf(logits).any().item()}")

        probs = torch.sigmoid(logits.squeeze())
        print(f"✓ Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Loss computation
print("\n" + "=" * 60)
print("TEST 5: Loss Computation")
print("=" * 60)

try:
    from src.training.losses import create_loss

    criterion = create_loss("focal", config)

    model.train()
    vitals = batch["vitals"].to(DEVICE)
    labs = batch["labs"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)
    static = batch["static"].to(DEVICE)
    embedding = batch["embedding"].to(DEVICE)
    has_notes = batch["has_notes"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    logits, _ = model(vitals, labs, mask, static, embedding, has_notes)
    loss = criterion(logits.squeeze(), labels)

    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Loss is NaN: {torch.isnan(loss).item()}")
    print(f"✓ Loss is Inf: {torch.isinf(loss).item()}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("✗ WARNING: Loss is NaN or Inf!")
        sys.exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Backward pass
print("\n" + "=" * 60)
print("TEST 6: Backward Pass")
print("=" * 60)

try:
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer.zero_grad()

    logits, _ = model(vitals, labs, mask, static, embedding, has_notes)
    loss = criterion(logits.squeeze(), labels)
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                print(f"✗ WARNING: {name} has NaN/Inf gradient!")

    print(f"✓ Gradients computed for {len(grad_norms)} parameters")
    print(f"✓ Grad norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"✓ Optimizer step completed")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Training loop (few batches)
print("\n" + "=" * 60)
print(f"TEST 7: Training Loop ({NUM_TEST_BATCHES} batches)")
print("=" * 60)

try:
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= NUM_TEST_BATCHES:
            break

        vitals = batch["vitals"].to(DEVICE)
        labs = batch["labs"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        embedding = batch["embedding"].to(DEVICE)
        has_notes = batch["has_notes"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(vitals, labs, mask, static, embedding, has_notes)
        loss = criterion(logits.squeeze(), labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"✗ Batch {batch_idx}: Loss is NaN/Inf!")
            sys.exit(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        print(f"  Batch {batch_idx + 1}/{NUM_TEST_BATCHES}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / NUM_TEST_BATCHES
    print(f"✓ Average loss: {avg_loss:.4f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Validation
print("\n" + "=" * 60)
print(f"TEST 8: Validation ({NUM_TEST_BATCHES} batches)")
print("=" * 60)

try:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= NUM_TEST_BATCHES:
                break

            vitals = batch["vitals"].to(DEVICE)
            labs = batch["labs"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            static = batch["static"].to(DEVICE)
            embedding = batch["embedding"].to(DEVICE)
            has_notes = batch["has_notes"].to(DEVICE)
            labels = batch["label"]

            logits, _ = model(vitals, labs, mask, static, embedding, has_notes)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Compute metrics
    if len(set(all_labels)) > 1:  # Need both classes for AUROC
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        print(f"✓ AUROC: {auroc:.4f}")
        print(f"✓ AUPRC: {auprc:.4f}")
    else:
        print(f"✓ Only one class in sample (need more batches for metrics)")

    print(f"✓ Predictions: {len(all_probs)}")
    print(f"✓ Positive rate: {sum(all_labels)/len(all_labels)*100:.1f}%")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe training pipeline is working correctly.")
print("You can now run the Colab notebook.")
