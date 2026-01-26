#!/usr/bin/env python3
"""
Script 1: Build ICU Cohort

Constructs the study cohort from MIMIC-IV ICU stays with inclusion/exclusion criteria.

Usage:
    python scripts/01_build_cohort.py \
        --data-dir /path/to/mimic-iv/data \
        --output-dir outputs/cohort
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohort import build_icu_cohort, split_cohort
from src.data.labels import load_critical_events, compute_deterioration_labels


def main():
    parser = argparse.ArgumentParser(
        description="Build ICU cohort for deterioration prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to MIMIC-IV data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort"),
        help="Output directory for cohort files",
    )
    parser.add_argument(
        "--min-los-hours",
        type=float,
        default=24.0,
        help="Minimum ICU length of stay in hours (default: 24)",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=18,
        help="Minimum patient age (default: 18)",
    )
    parser.add_argument(
        "--exclusion-hours",
        type=float,
        default=6.0,
        help="Exclude deaths within first N hours (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building ICU Cohort")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min LOS: {args.min_los_hours} hours")
    print(f"Min age: {args.min_age}")
    print(f"Exclusion window: {args.exclusion_hours} hours")
    print()

    # Build base cohort
    cohort = build_icu_cohort(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_los_hours=args.min_los_hours,
        min_age=args.min_age,
        exclusion_hours=args.exclusion_hours,
    )

    # Load critical events (vasopressor, intubation)
    print("\n" + "=" * 60)
    print("Loading Critical Events")
    print("=" * 60)

    cohort = load_critical_events(args.data_dir, cohort)

    # Compute deterioration labels
    cohort = compute_deterioration_labels(cohort, prediction_horizon=24)

    # Save updated cohort with events
    output_path = args.output_dir / "icu_cohort_with_events.parquet"
    cohort.to_parquet(output_path, index=False)
    print(f"\nCohort with events saved to {output_path}")

    # Split into train/val/test
    print("\n" + "=" * 60)
    print("Splitting Cohort")
    print("=" * 60)

    train_df, val_df, test_df = split_cohort(
        cohort,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        split_by="subject_id",
        seed=args.seed,
    )

    # Save splits
    train_df.to_parquet(args.output_dir / "train_cohort.parquet", index=False)
    val_df.to_parquet(args.output_dir / "val_cohort.parquet", index=False)
    test_df.to_parquet(args.output_dir / "test_cohort.parquet", index=False)

    print(f"\nSaved split cohorts to {args.output_dir}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total ICU stays: {len(cohort):,}")
    print(f"  - Train: {len(train_df):,}")
    print(f"  - Val:   {len(val_df):,}")
    print(f"  - Test:  {len(test_df):,}")
    print(f"\nPositive rate (has critical event):")
    print(f"  - Overall: {cohort['has_critical_event'].mean()*100:.1f}%")
    print(f"  - Train:   {train_df['has_critical_event'].mean()*100:.1f}%")
    print(f"  - Val:     {val_df['has_critical_event'].mean()*100:.1f}%")
    print(f"  - Test:    {test_df['has_critical_event'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
