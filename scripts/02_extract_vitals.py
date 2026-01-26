#!/usr/bin/env python3
"""
Script 2: Extract Vital Signs

Extracts and processes vital signs from MIMIC-IV chartevents using streaming.

Usage:
    python scripts/02_extract_vitals.py \
        --data-dir /path/to/mimic-iv/data \
        --cohort outputs/cohort/icu_cohort_with_events.parquet \
        --output-dir outputs/timeseries
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohort import load_cohort
from src.data.vitals import extract_vitals, VITAL_ITEMIDS


def main():
    parser = argparse.ArgumentParser(
        description="Extract vital signs from MIMIC-IV chartevents"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to MIMIC-IV data directory",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        required=True,
        help="Path to cohort parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/timeseries"),
        help="Output directory for vitals",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for streaming (default: 500000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Extracting Vital Signs")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Cohort: {args.cohort}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunk size: {args.chunk_size:,}")
    print()

    # Load cohort
    print("Loading cohort...")
    cohort = load_cohort(args.cohort)
    print(f"Loaded {len(cohort):,} ICU stays")

    # Print vital signs being extracted
    print("\nVital signs to extract:")
    for itemid, name in VITAL_ITEMIDS.items():
        print(f"  {itemid}: {name}")

    # Extract vitals
    print("\n" + "=" * 60)
    print("Streaming chartevents...")
    print("=" * 60)

    chartevents_path = args.data_dir / "mimic-iv-3.1" / "icu" / "chartevents.csv.gz"

    vitals_by_stay = extract_vitals(
        chartevents_path=chartevents_path,
        cohort=cohort,
        output_dir=args.output_dir,
        item_ids=VITAL_ITEMIDS,
        chunk_size=args.chunk_size,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Extracted vitals for {len(vitals_by_stay):,} stays")
    print(f"Coverage: {len(vitals_by_stay)/len(cohort)*100:.1f}%")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
