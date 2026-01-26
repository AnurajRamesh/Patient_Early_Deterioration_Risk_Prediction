"""
PyTorch Dataset Class for ICU Deterioration Prediction

Memory-efficient data loading using memory-mapped arrays.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ICUDataset(Dataset):
    """
    PyTorch Dataset for ICU deterioration prediction.

    Uses memory-mapped arrays for efficient loading of large datasets.
    """

    def __init__(
        self,
        samples_dir: Path,
        split: str = "all",
        max_seq_len: int = 48,
        load_embeddings: bool = True,
        embeddings_dir: Path | None = None,
        normalize: bool = True,
        norm_stats: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Initialize dataset.

        Args:
            samples_dir: Directory containing sample files
            split: Split name (e.g., "train", "val", "test", "all")
            max_seq_len: Maximum sequence length
            load_embeddings: Whether to load BERT embeddings
            embeddings_dir: Directory with embeddings (if different from samples_dir)
            normalize: Whether to normalize features
            norm_stats: Pre-computed normalization stats (mean, std) per feature
        """
        self.samples_dir = Path(samples_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.norm_stats = norm_stats

        # Load sample index
        index_path = self.samples_dir / f"{split}_index.json"
        if not index_path.exists():
            index_path = self.samples_dir / "sample_index.json"

        with open(index_path, "r") as f:
            self.sample_index = json.load(f)

        # Load split indices if available
        indices_path = self.samples_dir / f"{split}_indices.npy"
        if indices_path.exists():
            self.indices = np.load(indices_path)
        else:
            self.indices = np.arange(len(self.sample_index))

        # Memory-map the sequences array
        self.sequences = np.load(
            self.samples_dir / "sequences.npy",
            mmap_mode="r",
        )

        # Load labels
        self.labels = np.load(
            self.samples_dir / "labels.npy",
            mmap_mode="r",
        )

        # Load feature info
        with open(self.samples_dir / "feature_info.json", "r") as f:
            self.feature_info = json.load(f)

        self.n_vitals = self.feature_info["n_vitals"]
        self.n_labs = self.feature_info["n_labs"]
        self.n_features = self.feature_info["n_features"]
        self.vitals_cols = self.feature_info["vitals_cols"]
        self.labs_cols = self.feature_info["labs_cols"]

        # Load embeddings if requested
        # IMPORTANT: Load into memory dict to avoid concurrent access issues with DataLoader workers
        self.embeddings = None
        if load_embeddings:
            emb_dir = embeddings_dir or samples_dir
            emb_path = Path(emb_dir) / "cls_embeddings.npz"
            if emb_path.exists():
                print(f"  Loading embeddings from {emb_path}...")
                npz_data = np.load(emb_path, allow_pickle=True)
                # Load all embeddings into memory as a dict to avoid lazy-load issues
                self.embeddings = {key: npz_data[key].copy() for key in npz_data.files}
                npz_data.close()
                print(f"  Loaded {len(self.embeddings)} embeddings into memory")

        # Compute normalization stats if needed
        if self.normalize and self.norm_stats is None:
            self.norm_stats = self._compute_norm_stats()

        print(f"Loaded {len(self)} samples for split '{split}'")
        print(f"  Sequence shape: ({self.max_seq_len}, {self.n_features})")
        print(f"  Vitals: {self.n_vitals}, Labs: {self.n_labs}")

    def _compute_norm_stats(self) -> dict[str, tuple[float, float]]:
        """Compute mean and std for each feature from training data."""
        # Sample a subset for efficiency
        n_samples = min(10000, len(self))
        sample_indices = np.random.choice(len(self), n_samples, replace=False)

        # Collect all values
        all_values = []
        for local_idx in sample_indices:
            global_idx = self.indices[local_idx]
            seq = self.sequences[global_idx]
            all_values.append(seq)

        all_values = np.concatenate(all_values, axis=0)

        # Compute stats per feature
        stats = {}
        all_cols = self.vitals_cols + self.labs_cols
        for i, col in enumerate(all_cols):
            values = all_values[:, i]
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                mean = float(np.mean(valid))
                std = float(np.std(valid))
                if std == 0:
                    std = 1.0
            else:
                mean = 0.0
                std = 1.0
            stats[col] = (mean, std)

        return stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample."""
        global_idx = self.indices[idx]
        # sample_index is locally indexed per split, so use idx (not global_idx)
        sample_meta = self.sample_index[idx]

        # Get sequence
        sequence = self.sequences[global_idx].copy()  # (seq_len, n_features)

        # Create mask (1 where valid, 0 where missing)
        mask = (~np.isnan(sequence)).astype(np.float32)

        # Normalize features
        if self.normalize and self.norm_stats:
            all_cols = self.vitals_cols + self.labs_cols
            for i, col in enumerate(all_cols):
                if col in self.norm_stats:
                    mean, std = self.norm_stats[col]
                    valid = ~np.isnan(sequence[:, i])
                    sequence[valid, i] = (sequence[valid, i] - mean) / std

        # Replace NaN/Inf with 0 and clamp to prevent extreme values
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        sequence = np.clip(sequence, -10.0, 10.0)

        # Split into vitals and labs
        vitals = sequence[:, :self.n_vitals]
        labs = sequence[:, self.n_vitals:]

        # Get label
        label = self.labels[global_idx]

        # Get static features
        age = sample_meta.get("age", 0.0) / 100.0  # Normalize age to [0, 1]
        gender = sample_meta.get("gender", 0)
        has_notes = 1.0 if sample_meta.get("has_notes", False) else 0.0

        static = np.array([age, gender, has_notes], dtype=np.float32)

        # Get embeddings if available
        stay_id = str(sample_meta["stay_id"])
        if self.embeddings is not None and stay_id in self.embeddings:
            embedding = self.embeddings[stay_id].astype(np.float32)
            # Handle NaN/Inf in embeddings
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            embedding = np.zeros(768, dtype=np.float32)

        return {
            "sample_id": sample_meta["sample_id"],
            "vitals": torch.from_numpy(vitals),
            "labs": torch.from_numpy(labs),
            "mask": torch.from_numpy(mask),
            "static": torch.from_numpy(static),
            "embedding": torch.from_numpy(embedding),
            "has_notes": torch.tensor(has_notes, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def create_dataloaders(
    samples_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    embeddings_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        samples_dir: Directory containing sample files
        batch_size: Batch size
        num_workers: Number of data loading workers
        embeddings_dir: Directory with embeddings

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ICUDataset(
        samples_dir,
        split="train",
        embeddings_dir=embeddings_dir,
        normalize=True,
    )

    # Use training stats for val/test
    norm_stats = train_dataset.norm_stats

    val_dataset = ICUDataset(
        samples_dir,
        split="val",
        embeddings_dir=embeddings_dir,
        normalize=True,
        norm_stats=norm_stats,
    )

    test_dataset = ICUDataset(
        samples_dir,
        split="test",
        embeddings_dir=embeddings_dir,
        normalize=True,
        norm_stats=norm_stats,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate function for batching samples."""
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "vitals": torch.stack([item["vitals"] for item in batch]),
        "labs": torch.stack([item["labs"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "static": torch.stack([item["static"] for item in batch]),
        "embedding": torch.stack([item["embedding"] for item in batch]),
        "has_notes": torch.stack([item["has_notes"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
