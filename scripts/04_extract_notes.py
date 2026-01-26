#!/usr/bin/env python3
"""
Script 4: Extract Clinical Notes

Extracts and preprocesses clinical notes (radiology reports) from MIMIC-IV-Note.

Usage:
    python scripts/04_extract_notes.py \
        --data-dir /path/to/mimic-iv/data \
        --cohort outputs/cohort/icu_cohort_with_events.parquet \
        --output-dir outputs/notes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohort import load_cohort
from src.data.notes import extract_notes


def main():
    parser = argparse.ArgumentParser(
        description="Extract clinical notes from MIMIC-IV-Note"
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
        default=Path("outputs/notes"),
        help="Output directory for notes",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum note length to include (default: 50)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for streaming (default: 100000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Extracting Clinical Notes")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Cohort: {args.cohort}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum length: {args.min_length}")
    print()

    # Load cohort
    print("Loading cohort...")
    cohort = load_cohort(args.cohort)
    print(f"Loaded {len(cohort):,} ICU stays")

    # Extract notes
    print("\n" + "=" * 60)
    print("Streaming radiology notes...")
    print("=" * 60)

    notes_path = args.data_dir / "mimic-iv-note-2.2" / "note" / "radiology.csv.gz"

    notes_by_stay = extract_notes(
        notes_path=notes_path,
        cohort=cohort,
        output_dir=args.output_dir,
        min_length=args.min_length,
        chunk_size=args.chunk_size,
    )

    # Count stays with notes
    stays_with_notes = sum(1 for notes in notes_by_stay.values() if notes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total stays: {len(cohort):,}")
    print(f"Stays with notes: {stays_with_notes:,} ({stays_with_notes/len(cohort)*100:.1f}%)")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
