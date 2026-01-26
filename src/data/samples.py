"""
Hourly Sample Generation Module

Generate prediction samples at each hour during ICU stay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .labels import compute_deterioration_label


@dataclass
class HourlySample:
    """A single hourly prediction sample."""
    sample_id: str          # "{stay_id}_{hour}"
    stay_id: str
    subject_id: str
    hour: int               # Hours since ICU admission

    # Sequence data indices (for memory-mapped access)
    seq_start: int          # Start index in time-series array
    seq_end: int            # End index in time-series array

    # Static features
    age: float
    gender: int             # 0=F, 1=M
    admission_type: str

    # Note information
    has_notes: bool
    note_indices: list[int]  # Indices into notes array

    # Labels
    label: int              # 0 or 1
    time_to_event: float | None  # Hours until first critical event


def build_hourly_samples(
    cohort: pd.DataFrame,
    vitals_dir: Path,
    labs_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    max_seq_len: int = 48,
    prediction_horizon: int = 24,
    min_hours: int = 6,
    split_info: dict[str, set[str]] | None = None,
) -> dict[str, list[HourlySample]]:
    """
    Generate hourly prediction samples for all ICU stays.

    For each stay, create samples at hours [min_hours, los - prediction_horizon].

    Args:
        cohort: Cohort DataFrame with stay info and critical event times
        vitals_dir: Directory with vitals parquet files
        labs_dir: Directory with labs parquet files
        embeddings_dir: Directory with BERT embeddings
        output_dir: Directory to save samples
        max_seq_len: Maximum sequence length (hours of history)
        prediction_horizon: Hours ahead to predict
        min_hours: Start generating samples at this hour
        split_info: Optional dict mapping split name to set of subject_ids

    Returns:
        Dict mapping split name to list of samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all vitals into a dict
    print("Loading vitals...")
    vitals_by_stay = load_all_timeseries(vitals_dir, "vitals")
    print(f"Loaded vitals for {len(vitals_by_stay):,} stays")

    # Load all labs
    print("Loading labs...")
    labs_by_stay = load_all_timeseries(labs_dir, "labs")
    print(f"Loaded labs for {len(labs_by_stay):,} stays")

    # Load embeddings
    print("Loading embeddings...")
    embeddings_by_stay = load_embeddings(embeddings_dir)
    print(f"Loaded embeddings for {len(embeddings_by_stay):,} stays")

    # Get feature columns
    vitals_cols = None
    labs_cols = None
    for stay_id, df in vitals_by_stay.items():
        vitals_cols = [c for c in df.columns if c not in ["stay_id", "hour"]]
        break
    for stay_id, df in labs_by_stay.items():
        labs_cols = [c for c in df.columns if c not in ["stay_id", "hour"]]
        break

    if vitals_cols is None or labs_cols is None:
        raise ValueError("No vitals or labs data found")

    print(f"Vitals columns: {vitals_cols}")
    print(f"Labs columns: {labs_cols}")

    # Generate samples
    samples_by_split: dict[str, list[dict]] = {"all": []}
    n_features = len(vitals_cols) + len(labs_cols)

    # Arrays to store time-series data
    all_sequences = []
    all_labels = []
    sample_metadata = []

    n_processed = 0
    n_samples = 0

    for _, row in cohort.iterrows():
        stay_id = str(row["stay_id"])
        subject_id = str(row["subject_id"])

        # Get stay duration
        los_hours = int(row["los_hours"])

        # Get vitals and labs for this stay
        vitals = vitals_by_stay.get(stay_id)
        labs = labs_by_stay.get(stay_id)

        if vitals is None and labs is None:
            continue

        # Prepare feature matrix
        max_hours = los_hours + 1
        feature_matrix = np.full((max_hours, n_features), np.nan, dtype=np.float32)

        # Fill vitals
        if vitals is not None:
            vitals = vitals.set_index("hour") if "hour" in vitals.columns else vitals
            for i, col in enumerate(vitals_cols):
                if col in vitals.columns:
                    for hour in range(max_hours):
                        if hour in vitals.index:
                            val = vitals.loc[hour, col]
                            if pd.notna(val):
                                feature_matrix[hour, i] = val

        # Fill labs
        if labs is not None:
            labs = labs.set_index("hour") if "hour" in labs.columns else labs
            for i, col in enumerate(labs_cols):
                if col in labs.columns:
                    col_idx = len(vitals_cols) + i
                    for hour in range(max_hours):
                        if hour in labs.index:
                            val = labs.loc[hour, col]
                            if pd.notna(val):
                                feature_matrix[hour, col_idx] = val

        # Get critical event times
        death_time = row.get("icu_deathtime")
        vaso_time = row.get("first_vasopressor_time")
        intub_time = row.get("first_intubation_time")
        intime = row["intime"]

        # Convert to hours since admission
        def to_hours(t):
            if pd.isna(t):
                return None
            return (t - intime).total_seconds() / 3600

        death_hours = to_hours(death_time)
        vaso_hours = to_hours(vaso_time)
        intub_hours = to_hours(intub_time)

        # Get embeddings
        embeddings = embeddings_by_stay.get(stay_id)
        has_notes = embeddings is not None

        # Generate samples at each valid hour
        last_sample_hour = los_hours - prediction_horizon
        for hour in range(min_hours, max(min_hours, last_sample_hour + 1)):
            # Get sequence (past max_seq_len hours)
            seq_start = max(0, hour - max_seq_len + 1)
            seq_end = hour + 1
            sequence = feature_matrix[seq_start:seq_end]

            # Pad if needed
            if len(sequence) < max_seq_len:
                pad = np.full((max_seq_len - len(sequence), n_features), np.nan, dtype=np.float32)
                sequence = np.concatenate([pad, sequence], axis=0)

            # Compute label
            label = 0
            time_to_event = None

            # Check each event type
            for event_hours in [death_hours, vaso_hours, intub_hours]:
                if event_hours is not None:
                    if hour < event_hours <= hour + prediction_horizon:
                        label = 1
                        if time_to_event is None or (event_hours - hour) < time_to_event:
                            time_to_event = event_hours - hour

            # Create sample
            sample = {
                "sample_id": f"{stay_id}_{hour}",
                "stay_id": stay_id,
                "subject_id": subject_id,
                "hour": hour,
                "age": float(row.get("age", 0)),
                "gender": 1 if row.get("gender") == "M" else 0,
                "admission_type": str(row.get("admission_type", "")),
                "has_notes": has_notes,
                "label": label,
                "time_to_event": time_to_event,
            }

            all_sequences.append(sequence)
            all_labels.append(label)
            sample_metadata.append(sample)
            n_samples += 1

        n_processed += 1
        if n_processed % 5000 == 0:
            print(f"  Processed {n_processed:,} stays, {n_samples:,} samples...")

    print(f"\nTotal: {n_processed:,} stays, {n_samples:,} samples")

    # Convert to arrays
    sequences_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    # Compute label statistics
    n_pos = labels_array.sum()
    print(f"Positive samples: {n_pos:,} ({n_pos/len(labels_array)*100:.1f}%)")

    # Split samples if split_info provided
    if split_info is not None:
        for split_name, subject_ids in split_info.items():
            split_samples = [
                s for s in sample_metadata if s["subject_id"] in subject_ids
            ]
            samples_by_split[split_name] = split_samples
            print(f"  {split_name}: {len(split_samples):,} samples")
    else:
        samples_by_split["all"] = sample_metadata

    # Save samples
    save_samples(
        sequences_array,
        labels_array,
        sample_metadata,
        vitals_cols,
        labs_cols,
        output_dir,
        split_info,
    )

    return samples_by_split


def load_all_timeseries(
    data_dir: Path,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    """Load all time-series from batched parquet files."""
    result = {}

    for batch_path in sorted(data_dir.glob(f"{prefix}_batch_*.parquet")):
        batch_df = pd.read_parquet(batch_path)
        for stay_id, group in batch_df.groupby("stay_id"):
            result[str(stay_id)] = group.copy()

    return result


def load_embeddings(embeddings_dir: Path) -> dict[str, np.ndarray]:
    """Load BERT embeddings from NPZ files."""
    result = {}

    cls_path = embeddings_dir / "cls_embeddings.npz"
    if cls_path.exists():
        data = np.load(cls_path, allow_pickle=True)
        for key in data.files:
            result[key] = data[key]

    return result


def save_samples(
    sequences: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict],
    vitals_cols: list[str],
    labs_cols: list[str],
    output_dir: Path,
    split_info: dict[str, set[str]] | None = None,
) -> None:
    """Save samples to disk."""
    # Save full arrays
    np.save(output_dir / "sequences.npy", sequences)
    np.save(output_dir / "labels.npy", labels)

    # Save metadata
    with open(output_dir / "sample_index.json", "w") as f:
        json.dump(metadata, f)

    # Save feature info
    feature_info = {
        "vitals_cols": vitals_cols,
        "labs_cols": labs_cols,
        "n_vitals": len(vitals_cols),
        "n_labs": len(labs_cols),
        "n_features": len(vitals_cols) + len(labs_cols),
    }
    with open(output_dir / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    # If split_info provided, create split-specific indices
    if split_info is not None:
        sample_to_idx = {s["sample_id"]: i for i, s in enumerate(metadata)}

        for split_name, subject_ids in split_info.items():
            split_indices = []
            for i, s in enumerate(metadata):
                if s["subject_id"] in subject_ids:
                    split_indices.append(i)

            split_indices = np.array(split_indices, dtype=np.int32)
            np.save(output_dir / f"{split_name}_indices.npy", split_indices)

            split_metadata = [metadata[i] for i in split_indices]
            with open(output_dir / f"{split_name}_index.json", "w") as f:
                json.dump(split_metadata, f)

            print(f"  Saved {split_name}: {len(split_indices):,} samples")

    print(f"Saved samples to {output_dir}")
