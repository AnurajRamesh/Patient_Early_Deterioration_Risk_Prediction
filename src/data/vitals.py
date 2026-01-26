"""
Vital Signs Extraction Module

Streaming extraction of vital signs from MIMIC-IV chartevents.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Callable

import pandas as pd
import numpy as np


# Key vital sign item IDs (from MIMIC-IV d_items)
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220050: "abp_systolic",
    220051: "abp_diastolic",
    220052: "abp_mean",
    220179: "nibp_systolic",
    220180: "nibp_diastolic",
    220210: "respiratory_rate",
    223761: "temperature_f",
    223762: "temperature_c",
    220277: "spo2",
}


def stream_chartevents(
    chartevents_path: Path,
    stay_ids: set[str],
    item_ids: set[int] | None = None,
    chunk_size: int = 500_000,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream chartevents in chunks, filtering to relevant stays and items.

    This is memory-efficient for the large chartevents file (3.3GB).

    Args:
        chartevents_path: Path to chartevents.csv.gz
        stay_ids: Set of stay_ids to include
        item_ids: Set of itemids to include (None = all vitals)
        chunk_size: Rows per chunk
        progress_callback: Called with total rows processed

    Yields:
        DataFrames with filtered chartevents
    """
    if item_ids is None:
        item_ids = set(VITAL_ITEMIDS.keys())

    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        chartevents_path,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": str, "itemid": int},
        parse_dates=["charttime"],
    ):
        total_rows += len(chunk)

        # Filter to relevant stays and items
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids) &
            chunk["itemid"].isin(item_ids) &
            chunk["valuenum"].notna()
        ]

        matched_rows += len(chunk)

        if progress_callback:
            progress_callback(total_rows)

        if len(chunk) > 0:
            yield chunk

    print(f"Streamed {total_rows:,} rows, matched {matched_rows:,}")


def extract_vitals(
    chartevents_path: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    item_ids: dict[int, str] | None = None,
    chunk_size: int = 500_000,
) -> dict[str, pd.DataFrame]:
    """
    Extract vital signs for cohort stays, aggregated hourly.

    Args:
        chartevents_path: Path to chartevents.csv.gz
        cohort: Cohort DataFrame with stay_id, intime, outtime
        output_dir: Directory to save output
        item_ids: Dict mapping itemid to vital name
        chunk_size: Rows per chunk for streaming

    Returns:
        Dict mapping stay_id to hourly vitals DataFrame
    """
    if item_ids is None:
        item_ids = VITAL_ITEMIDS

    stay_ids = set(cohort["stay_id"].astype(str))
    print(f"Extracting vitals for {len(stay_ids):,} stays...")

    # Collect all vitals chunks
    vitals_chunks = []
    rows_processed = 0

    def progress(n):
        nonlocal rows_processed
        if n - rows_processed >= 1_000_000:
            print(f"  Processed {n:,} rows...")
            rows_processed = n

    for chunk in stream_chartevents(
        chartevents_path,
        stay_ids,
        set(item_ids.keys()),
        chunk_size,
        progress,
    ):
        vitals_chunks.append(chunk)

    if not vitals_chunks:
        print("No vitals found!")
        return {}

    # Combine all chunks
    vitals_df = pd.concat(vitals_chunks, ignore_index=True)
    print(f"Total vital observations: {len(vitals_df):,}")

    # Map itemid to vital name
    vitals_df["vital_name"] = vitals_df["itemid"].map(item_ids)

    # Get stay info for time alignment
    stay_info = cohort.set_index("stay_id")[["intime", "outtime"]].to_dict("index")

    # Process each stay
    output_dir.mkdir(parents=True, exist_ok=True)
    vitals_by_stay = {}
    n_processed = 0

    for stay_id, stay_vitals in vitals_df.groupby("stay_id"):
        if stay_id not in stay_info:
            continue

        info = stay_info[stay_id]
        intime = info["intime"]
        outtime = info["outtime"]

        # Filter to within ICU stay
        stay_vitals = stay_vitals[
            (stay_vitals["charttime"] >= intime) &
            (stay_vitals["charttime"] <= outtime)
        ].copy()

        if len(stay_vitals) == 0:
            continue

        # Calculate hours since admission
        stay_vitals["hours_since_admit"] = (
            (stay_vitals["charttime"] - intime).dt.total_seconds() / 3600
        )

        # Create hourly bins
        stay_vitals["hour_bin"] = stay_vitals["hours_since_admit"].astype(int)

        # Pivot to wide format (one column per vital)
        hourly_vitals = (
            stay_vitals.groupby(["hour_bin", "vital_name"])["valuenum"]
            .mean()
            .unstack(fill_value=np.nan)
        )

        # Reindex to have all hours from 0 to max
        max_hour = int((outtime - intime).total_seconds() / 3600)
        full_index = range(max_hour + 1)
        hourly_vitals = hourly_vitals.reindex(full_index)

        # Ensure all vital columns exist
        for vital_name in item_ids.values():
            if vital_name not in hourly_vitals.columns:
                hourly_vitals[vital_name] = np.nan

        vitals_by_stay[stay_id] = hourly_vitals

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"  Processed {n_processed:,} stays...")

    print(f"Extracted vitals for {len(vitals_by_stay):,} stays")

    # Save to parquet files (one per stay would be too many, so batch)
    save_vitals_batch(vitals_by_stay, output_dir)

    return vitals_by_stay


def save_vitals_batch(
    vitals_by_stay: dict[str, pd.DataFrame],
    output_dir: Path,
    batch_size: int = 1000,
) -> None:
    """Save vitals in batched parquet files."""
    stay_ids = list(vitals_by_stay.keys())
    n_batches = (len(stay_ids) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(stay_ids))
        batch_ids = stay_ids[start:end]

        # Combine into long format with stay_id
        records = []
        for stay_id in batch_ids:
            df = vitals_by_stay[stay_id].copy()
            df["stay_id"] = stay_id
            df["hour"] = df.index
            df = df.reset_index(drop=True)
            records.append(df)

        batch_df = pd.concat(records, ignore_index=True)
        batch_path = output_dir / f"vitals_batch_{batch_idx:04d}.parquet"
        batch_df.to_parquet(batch_path, index=False)

    print(f"Saved {n_batches} vitals batch files to {output_dir}")


def load_vitals(
    vitals_dir: Path,
    stay_id: str,
) -> pd.DataFrame | None:
    """Load vitals for a specific stay from batched files."""
    for batch_path in vitals_dir.glob("vitals_batch_*.parquet"):
        batch_df = pd.read_parquet(batch_path)
        stay_vitals = batch_df[batch_df["stay_id"] == stay_id]
        if len(stay_vitals) > 0:
            return stay_vitals.set_index("hour").drop(columns=["stay_id"])
    return None


def forward_fill_vitals(
    vitals: pd.DataFrame,
    max_gap_hours: int = 4,
) -> pd.DataFrame:
    """
    Forward-fill missing vital values with max gap constraint.

    Args:
        vitals: Hourly vitals DataFrame (index = hour)
        max_gap_hours: Maximum hours to forward-fill

    Returns:
        Forward-filled DataFrame
    """
    filled = vitals.copy()

    for col in filled.columns:
        # Forward fill with limit
        filled[col] = filled[col].ffill(limit=max_gap_hours)

    return filled


def normalize_vitals(
    vitals: pd.DataFrame,
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Z-score normalize vitals.

    Args:
        vitals: Vitals DataFrame
        stats: Pre-computed (mean, std) per column. If None, compute from data.

    Returns:
        (normalized_df, stats_dict)
    """
    if stats is None:
        stats = {}
        for col in vitals.columns:
            mean = vitals[col].mean()
            std = vitals[col].std()
            if pd.isna(std) or std == 0:
                std = 1.0
            stats[col] = (mean, std)

    normalized = vitals.copy()
    for col in normalized.columns:
        if col in stats:
            mean, std = stats[col]
            normalized[col] = (normalized[col] - mean) / std

    return normalized, stats
