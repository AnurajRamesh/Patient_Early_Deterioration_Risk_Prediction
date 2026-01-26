#!/usr/bin/env python3
"""
Script 3: Extract Laboratory Results

Extracts and processes lab results from MIMIC-IV labevents using streaming.

Usage:
    python scripts/03_extract_labs.py \
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
from src.data.labs import extract_labs, LAB_ITEMIDS


def main():
    parser = argparse.ArgumentParser(
        description="Extract lab results from MIMIC-IV labevents"
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
        help="Output directory for labs",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for streaming (default: 500000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Extracting Laboratory Results")
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

    # Print labs being extracted
    print("\nLaboratory tests to extract:")
    for itemid, name in LAB_ITEMIDS.items():
        print(f"  {itemid}: {name}")

    # Extract labs
    print("\n" + "=" * 60)
    print("Streaming labevents...")
    print("=" * 60)

    labevents_path = args.data_dir / "mimic-iv-3.1" / "hosp" / "labevents.csv.gz"

    labs_by_stay = extract_labs(
        labevents_path=labevents_path,
        cohort=cohort,
        output_dir=args.output_dir,
        item_ids=LAB_ITEMIDS,
        chunk_size=args.chunk_size,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Extracted labs for {len(labs_by_stay):,} stays")
    print(f"Coverage: {len(labs_by_stay)/len(cohort)*100:.1f}%")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
