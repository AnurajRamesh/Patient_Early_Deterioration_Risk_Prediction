"""
Laboratory Results Extraction Module

Streaming extraction of lab results from MIMIC-IV labevents.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Callable

import pandas as pd
import numpy as np


# Key laboratory test item IDs (from MIMIC-IV d_labitems)
LAB_ITEMIDS = {
    # Blood Gas
    50813: "lactate",
    50820: "ph",
    50821: "po2",
    50818: "pco2",

    # Chemistry
    50912: "creatinine",
    50971: "potassium",
    50983: "sodium",
    50882: "bicarbonate",
    50893: "calcium",

    # Hematology
    51301: "wbc",
    51222: "hemoglobin",
    51265: "platelets",

    # Coagulation
    51237: "inr",
    51275: "ptt",

    # Liver
    50885: "bilirubin_total",

    # Metabolic
    50931: "glucose",
}

# Critical labs that need shorter forward-fill windows
CRITICAL_LABS = {"lactate", "ph", "po2", "pco2"}


def stream_labevents(
    labevents_path: Path,
    hadm_ids: set[str],
    item_ids: set[int] | None = None,
    chunk_size: int = 500_000,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream labevents in chunks, filtering to relevant admissions and items.

    Args:
        labevents_path: Path to labevents.csv.gz
        hadm_ids: Set of hadm_ids to include
        item_ids: Set of itemids to include (None = all labs)
        chunk_size: Rows per chunk
        progress_callback: Called with total rows processed

    Yields:
        DataFrames with filtered labevents
    """
    if item_ids is None:
        item_ids = set(LAB_ITEMIDS.keys())

    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        labevents_path,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        dtype={"hadm_id": str, "itemid": int},
        parse_dates=["charttime"],
    ):
        total_rows += len(chunk)

        # Filter to relevant admissions and items
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids) &
            chunk["itemid"].isin(item_ids) &
            chunk["valuenum"].notna()
        ]

        matched_rows += len(chunk)

        if progress_callback:
            progress_callback(total_rows)

        if len(chunk) > 0:
            yield chunk

    print(f"Streamed {total_rows:,} rows, matched {matched_rows:,}")


def extract_labs(
    labevents_path: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    item_ids: dict[int, str] | None = None,
    chunk_size: int = 500_000,
) -> dict[str, pd.DataFrame]:
    """
    Extract lab results for cohort stays, aligned to ICU stay times.

    Labs are associated with hadm_id but we need to align with ICU stay times.

    Args:
        labevents_path: Path to labevents.csv.gz
        cohort: Cohort DataFrame with stay_id, hadm_id, intime, outtime
        output_dir: Directory to save output
        item_ids: Dict mapping itemid to lab name
        chunk_size: Rows per chunk for streaming

    Returns:
        Dict mapping stay_id to hourly labs DataFrame
    """
    if item_ids is None:
        item_ids = LAB_ITEMIDS

    hadm_ids = set(cohort["hadm_id"].astype(str))
    print(f"Extracting labs for {len(hadm_ids):,} admissions...")

    # Collect all lab chunks
    lab_chunks = []
    rows_processed = 0

    def progress(n):
        nonlocal rows_processed
        if n - rows_processed >= 1_000_000:
            print(f"  Processed {n:,} rows...")
            rows_processed = n

    for chunk in stream_labevents(
        labevents_path,
        hadm_ids,
        set(item_ids.keys()),
        chunk_size,
        progress,
    ):
        lab_chunks.append(chunk)

    if not lab_chunks:
        print("No labs found!")
        return {}

    # Combine all chunks
    labs_df = pd.concat(lab_chunks, ignore_index=True)
    print(f"Total lab observations: {len(labs_df):,}")

    # Map itemid to lab name
    labs_df["lab_name"] = labs_df["itemid"].map(item_ids)

    # Create mapping from hadm_id to stay info
    # (one hadm can have multiple ICU stays, so we need stay-level alignment)
    stay_info = {}
    for _, row in cohort.iterrows():
        stay_id = str(row["stay_id"])
        hadm_id = str(row["hadm_id"])
        stay_info[stay_id] = {
            "hadm_id": hadm_id,
            "intime": row["intime"],
            "outtime": row["outtime"],
        }

    # Create reverse mapping: hadm_id -> [stay_ids]
    hadm_to_stays = {}
    for stay_id, info in stay_info.items():
        hadm_id = info["hadm_id"]
        if hadm_id not in hadm_to_stays:
            hadm_to_stays[hadm_id] = []
        hadm_to_stays[hadm_id].append(stay_id)

    # Process each stay
    output_dir.mkdir(parents=True, exist_ok=True)
    labs_by_stay = {}
    n_processed = 0

    for stay_id, info in stay_info.items():
        hadm_id = info["hadm_id"]
        intime = info["intime"]
        outtime = info["outtime"]

        # Get labs for this hadm
        hadm_labs = labs_df[labs_df["hadm_id"] == hadm_id].copy()

        if len(hadm_labs) == 0:
            continue

        # Filter to within ICU stay window
        stay_labs = hadm_labs[
            (hadm_labs["charttime"] >= intime) &
            (hadm_labs["charttime"] <= outtime)
        ].copy()

        if len(stay_labs) == 0:
            continue

        # Calculate hours since admission
        stay_labs["hours_since_admit"] = (
            (stay_labs["charttime"] - intime).dt.total_seconds() / 3600
        )

        # Create hourly bins
        stay_labs["hour_bin"] = stay_labs["hours_since_admit"].astype(int)

        # Pivot to wide format (use last value per hour if multiple)
        hourly_labs = (
            stay_labs.groupby(["hour_bin", "lab_name"])["valuenum"]
            .last()  # Use last value in hour
            .unstack(fill_value=np.nan)
        )

        # Reindex to have all hours from 0 to max
        max_hour = int((outtime - intime).total_seconds() / 3600)
        full_index = range(max_hour + 1)
        hourly_labs = hourly_labs.reindex(full_index)

        # Ensure all lab columns exist
        for lab_name in item_ids.values():
            if lab_name not in hourly_labs.columns:
                hourly_labs[lab_name] = np.nan

        labs_by_stay[stay_id] = hourly_labs

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"  Processed {n_processed:,} stays...")

    print(f"Extracted labs for {len(labs_by_stay):,} stays")

    # Save to parquet files
    save_labs_batch(labs_by_stay, output_dir)

    return labs_by_stay


def save_labs_batch(
    labs_by_stay: dict[str, pd.DataFrame],
    output_dir: Path,
    batch_size: int = 1000,
) -> None:
    """Save labs in batched parquet files."""
    stay_ids = list(labs_by_stay.keys())
    n_batches = (len(stay_ids) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(stay_ids))
        batch_ids = stay_ids[start:end]

        # Combine into long format with stay_id
        records = []
        for stay_id in batch_ids:
            df = labs_by_stay[stay_id].copy()
            df["stay_id"] = stay_id
            df["hour"] = df.index
            df = df.reset_index(drop=True)
            records.append(df)

        batch_df = pd.concat(records, ignore_index=True)
        batch_path = output_dir / f"labs_batch_{batch_idx:04d}.parquet"
        batch_df.to_parquet(batch_path, index=False)

    print(f"Saved {n_batches} labs batch files to {output_dir}")


def load_labs(
    labs_dir: Path,
    stay_id: str,
) -> pd.DataFrame | None:
    """Load labs for a specific stay from batched files."""
    for batch_path in labs_dir.glob("labs_batch_*.parquet"):
        batch_df = pd.read_parquet(batch_path)
        stay_labs = batch_df[batch_df["stay_id"] == stay_id]
        if len(stay_labs) > 0:
            return stay_labs.set_index("hour").drop(columns=["stay_id"])
    return None


def forward_fill_labs(
    labs: pd.DataFrame,
    routine_gap_hours: int = 24,
    critical_gap_hours: int = 6,
    critical_labs: set[str] | None = None,
) -> pd.DataFrame:
    """
    Forward-fill missing lab values with lab-specific constraints.

    Routine labs (CBC, BMP) can be forward-filled longer than critical labs (ABG, lactate).

    Args:
        labs: Hourly labs DataFrame (index = hour)
        routine_gap_hours: Max hours for routine labs
        critical_gap_hours: Max hours for critical labs
        critical_labs: Set of critical lab names

    Returns:
        Forward-filled DataFrame
    """
    if critical_labs is None:
        critical_labs = CRITICAL_LABS

    filled = labs.copy()

    for col in filled.columns:
        if col in critical_labs:
            max_gap = critical_gap_hours
        else:
            max_gap = routine_gap_hours
        filled[col] = filled[col].ffill(limit=max_gap)

    return filled


def normalize_labs(
    labs: pd.DataFrame,
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Z-score normalize labs.

    Args:
        labs: Labs DataFrame
        stats: Pre-computed (mean, std) per column. If None, compute from data.

    Returns:
        (normalized_df, stats_dict)
    """
    if stats is None:
        stats = {}
        for col in labs.columns:
            mean = labs[col].mean()
            std = labs[col].std()
            if pd.isna(std) or std == 0:
                std = 1.0
            stats[col] = (mean, std)

    normalized = labs.copy()
    for col in normalized.columns:
        if col in stats:
            mean, std = stats[col]
            normalized[col] = (normalized[col] - mean) / std

    return normalized, stats
