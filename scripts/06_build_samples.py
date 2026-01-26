#!/usr/bin/env python3
"""
Script 6: Build Hourly Samples

Generates hourly prediction samples combining vitals, labs, and labels.

Usage:
    python scripts/06_build_samples.py \
        --cohort outputs/cohort/icu_cohort_with_events.parquet \
        --vitals-dir outputs/timeseries \
        --labs-dir outputs/timeseries \
        --embeddings-dir outputs/embeddings \
        --output-dir outputs/samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohort import load_cohort, split_cohort
from src.data.samples import build_hourly_samples


def main():
    parser = argparse.ArgumentParser(
        description="Build hourly prediction samples"
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        required=True,
        help="Path to cohort parquet file",
    )
    parser.add_argument(
        "--vitals-dir",
        type=Path,
        required=True,
        help="Directory with vitals batch files",
    )
    parser.add_argument(
        "--labs-dir",
        type=Path,
        default=None,
        help="Directory with labs batch files (defaults to vitals-dir)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory with BERT embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/samples"),
        help="Output directory for samples",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=48,
        help="Maximum sequence length in hours (default: 48)",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=24,
        help="Prediction horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=6,
        help="Start sampling at this hour (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    if args.labs_dir is None:
        args.labs_dir = args.vitals_dir

    print("=" * 60)
    print("Building Hourly Samples")
    print("=" * 60)
    print(f"Cohort: {args.cohort}")
    print(f"Vitals dir: {args.vitals_dir}")
    print(f"Labs dir: {args.labs_dir}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max seq length: {args.max_seq_len} hours")
    print(f"Prediction horizon: {args.prediction_horizon} hours")
    print(f"Min hours: {args.min_hours}")
    print()

    # Load cohort
    print("Loading cohort...")
    cohort = load_cohort(args.cohort)
    print(f"Loaded {len(cohort):,} ICU stays")

    # Split cohort
    print("\nSplitting cohort...")
    train_df, val_df, test_df = split_cohort(
        cohort,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        split_by="subject_id",
        seed=args.seed,
    )

    # Create split info
    split_info = {
        "train": set(train_df["subject_id"].astype(str)),
        "val": set(val_df["subject_id"].astype(str)),
        "test": set(test_df["subject_id"].astype(str)),
    }

    # Build samples
    print("\n" + "=" * 60)
    print("Building samples...")
    print("=" * 60)

    samples_by_split = build_hourly_samples(
        cohort=cohort,
        vitals_dir=args.vitals_dir,
        labs_dir=args.labs_dir,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        prediction_horizon=args.prediction_horizon,
        min_hours=args.min_hours,
        split_info=split_info,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, samples in samples_by_split.items():
        if samples:
            n_pos = sum(1 for s in samples if s.get("label", 0) == 1)
            print(f"  {split_name}: {len(samples):,} samples ({n_pos/len(samples)*100:.1f}% positive)")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
