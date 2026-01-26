#!/usr/bin/env python3
"""
Create a smaller subset of data for Google Colab training.
"""
import json
import numpy as np
from pathlib import Path
import shutil

def create_subset(
    samples_dir: Path,
    output_dir: Path,
    train_size: int = 200000,
    val_size: int = 40000,
    test_size: int = 40000,
    seed: int = 42,
):
    """Create a subset of the data for Colab training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    print("Loading full dataset indices...")

    # Load indices
    train_indices = np.load(samples_dir / "train_indices.npy")
    val_indices = np.load(samples_dir / "val_indices.npy")
    test_indices = np.load(samples_dir / "test_indices.npy")

    print(f"Full dataset: train={len(train_indices):,}, val={len(val_indices):,}, test={len(test_indices):,}")

    # Sample subsets
    train_subset = np.random.choice(len(train_indices), min(train_size, len(train_indices)), replace=False)
    val_subset = np.random.choice(len(val_indices), min(val_size, len(val_indices)), replace=False)
    test_subset = np.random.choice(len(test_indices), min(test_size, len(test_indices)), replace=False)

    # Get global indices for subset
    train_global = train_indices[train_subset]
    val_global = val_indices[val_subset]
    test_global = test_indices[test_subset]

    # Combine all global indices
    all_global = np.concatenate([train_global, val_global, test_global])
    all_global_sorted = np.sort(all_global)

    # Create mapping from old global index to new global index
    old_to_new = {old: new for new, old in enumerate(all_global_sorted)}

    print(f"Subset: train={len(train_subset):,}, val={len(val_subset):,}, test={len(test_subset):,}")
    print(f"Total samples: {len(all_global_sorted):,}")

    # Load and subset sequences
    print("\nLoading sequences (this may take a while)...")
    sequences = np.load(samples_dir / "sequences.npy", mmap_mode='r')
    labels = np.load(samples_dir / "labels.npy", mmap_mode='r')

    print("Creating subset arrays...")
    subset_sequences = sequences[all_global_sorted].copy()
    subset_labels = labels[all_global_sorted].copy()

    # Save subset arrays
    print("Saving subset sequences...")
    np.save(output_dir / "sequences.npy", subset_sequences)
    np.save(output_dir / "labels.npy", subset_labels)

    # Create new indices (pointing to positions in subset arrays)
    new_train_indices = np.array([old_to_new[g] for g in train_global], dtype=np.int32)
    new_val_indices = np.array([old_to_new[g] for g in val_global], dtype=np.int32)
    new_test_indices = np.array([old_to_new[g] for g in test_global], dtype=np.int32)

    np.save(output_dir / "train_indices.npy", new_train_indices)
    np.save(output_dir / "val_indices.npy", new_val_indices)
    np.save(output_dir / "test_indices.npy", new_test_indices)

    # Load and subset metadata
    print("Subsetting metadata...")
    with open(samples_dir / "train_index.json", "r") as f:
        train_meta = json.load(f)
    with open(samples_dir / "val_index.json", "r") as f:
        val_meta = json.load(f)
    with open(samples_dir / "test_index.json", "r") as f:
        test_meta = json.load(f)

    subset_train_meta = [train_meta[i] for i in train_subset]
    subset_val_meta = [val_meta[i] for i in val_subset]
    subset_test_meta = [test_meta[i] for i in test_subset]

    with open(output_dir / "train_index.json", "w") as f:
        json.dump(subset_train_meta, f)
    with open(output_dir / "val_index.json", "w") as f:
        json.dump(subset_val_meta, f)
    with open(output_dir / "test_index.json", "w") as f:
        json.dump(subset_test_meta, f)

    # Copy feature info
    shutil.copy(samples_dir / "feature_info.json", output_dir / "feature_info.json")

    # Print statistics
    train_pos = sum(1 for m in subset_train_meta if m.get("label", 0) == 1)
    val_pos = sum(1 for m in subset_val_meta if m.get("label", 0) == 1)
    test_pos = sum(1 for m in subset_test_meta if m.get("label", 0) == 1)

    print("\n" + "=" * 50)
    print("SUBSET CREATED")
    print("=" * 50)
    print(f"Train: {len(subset_train_meta):,} samples ({train_pos/len(subset_train_meta)*100:.1f}% positive)")
    print(f"Val: {len(subset_val_meta):,} samples ({val_pos/len(subset_val_meta)*100:.1f}% positive)")
    print(f"Test: {len(subset_test_meta):,} samples ({test_pos/len(subset_test_meta)*100:.1f}% positive)")
    print(f"\nOutput saved to: {output_dir}")

    # Calculate sizes
    seq_size = subset_sequences.nbytes / (1024**3)
    print(f"Sequences size: {seq_size:.2f} GB")


if __name__ == "__main__":
    create_subset(
        samples_dir=Path("outputs/samples"),
        output_dir=Path("outputs/samples_colab"),
        train_size=200000,
        val_size=40000,
        test_size=40000,
    )
