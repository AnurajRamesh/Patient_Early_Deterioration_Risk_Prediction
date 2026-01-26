"""
Deterioration Label Generation Module

Computes binary labels for deterioration events (mortality, vasopressor, intubation).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np


# Critical event item IDs (from MIMIC-IV d_items)
VASOPRESSOR_ITEMIDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221749: "phenylephrine",
    222315: "vasopressin",
    221653: "dobutamine",
}

INTUBATION_ITEMIDS = {
    224385: "intubation",
    225792: "invasive_ventilation",
}


def load_vasopressor_events(
    data_dir: Path,
    stay_ids: set[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Load vasopressor administration events from inputevents.

    Args:
        data_dir: Path to MIMIC-IV data directory
        stay_ids: Set of stay_ids to filter
        chunk_size: Rows per chunk for streaming

    Returns:
        DataFrame with first vasopressor time per stay
    """
    path = data_dir / "mimic-iv-3.1" / "icu" / "inputevents.csv.gz"
    print(f"Loading vasopressor events from {path}...")

    vasopressor_itemids = set(VASOPRESSOR_ITEMIDS.keys())
    events = []

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["stay_id", "starttime", "itemid"],
        dtype={"stay_id": str, "itemid": int},
        parse_dates=["starttime"],
    ):
        # Filter to relevant stays and vasopressors
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids) &
            chunk["itemid"].isin(vasopressor_itemids)
        ]
        events.append(chunk)

    if not events:
        return pd.DataFrame(columns=["stay_id", "first_vasopressor_time"])

    events_df = pd.concat(events, ignore_index=True)

    # Get first vasopressor time per stay
    first_vaso = (
        events_df.groupby("stay_id")["starttime"]
        .min()
        .reset_index()
        .rename(columns={"starttime": "first_vasopressor_time"})
    )

    print(f"Found vasopressor events for {len(first_vaso):,} stays")
    return first_vaso


def load_intubation_events(
    data_dir: Path,
    stay_ids: set[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Load intubation/ventilation events from procedureevents.

    Args:
        data_dir: Path to MIMIC-IV data directory
        stay_ids: Set of stay_ids to filter
        chunk_size: Rows per chunk for streaming

    Returns:
        DataFrame with first intubation time per stay
    """
    path = data_dir / "mimic-iv-3.1" / "icu" / "procedureevents.csv.gz"
    print(f"Loading intubation events from {path}...")

    intubation_itemids = set(INTUBATION_ITEMIDS.keys())
    events = []

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["stay_id", "starttime", "itemid"],
        dtype={"stay_id": str, "itemid": int},
        parse_dates=["starttime"],
    ):
        # Filter to relevant stays and intubation events
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids) &
            chunk["itemid"].isin(intubation_itemids)
        ]
        events.append(chunk)

    if not events:
        return pd.DataFrame(columns=["stay_id", "first_intubation_time"])

    events_df = pd.concat(events, ignore_index=True)

    # Get first intubation time per stay
    first_intub = (
        events_df.groupby("stay_id")["starttime"]
        .min()
        .reset_index()
        .rename(columns={"starttime": "first_intubation_time"})
    )

    print(f"Found intubation events for {len(first_intub):,} stays")
    return first_intub


def load_critical_events(
    data_dir: Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load all critical events and merge with cohort.

    Adds columns:
    - first_vasopressor_time
    - first_intubation_time
    - first_critical_event_time

    Args:
        data_dir: Path to MIMIC-IV data directory
        cohort: Cohort DataFrame with stay_id, icu_deathtime

    Returns:
        Cohort with critical event times
    """
    stay_ids = set(cohort["stay_id"].astype(str))

    # Load events
    vaso_events = load_vasopressor_events(data_dir, stay_ids)
    intub_events = load_intubation_events(data_dir, stay_ids)

    # Merge with cohort
    cohort = cohort.copy()
    cohort["stay_id"] = cohort["stay_id"].astype(str)

    cohort = cohort.merge(vaso_events, on="stay_id", how="left")
    cohort = cohort.merge(intub_events, on="stay_id", how="left")

    # Compute first critical event time (earliest of death, vasopressor, intubation)
    cohort["first_critical_event_time"] = cohort[
        ["icu_deathtime", "first_vasopressor_time", "first_intubation_time"]
    ].min(axis=1)

    # Print statistics
    n_vaso = cohort["first_vasopressor_time"].notna().sum()
    n_intub = cohort["first_intubation_time"].notna().sum()
    n_death = cohort["icu_deathtime"].notna().sum()
    n_any = cohort["first_critical_event_time"].notna().sum()

    print("\nCritical event statistics:")
    print(f"  Vasopressor initiation: {n_vaso:,} ({n_vaso/len(cohort)*100:.1f}%)")
    print(f"  Intubation: {n_intub:,} ({n_intub/len(cohort)*100:.1f}%)")
    print(f"  ICU death: {n_death:,} ({n_death/len(cohort)*100:.1f}%)")
    print(f"  Any critical event: {n_any:,} ({n_any/len(cohort)*100:.1f}%)")

    return cohort


def compute_deterioration_label(
    prediction_time: datetime,
    prediction_horizon: int,  # hours
    death_time: datetime | None,
    vasopressor_time: datetime | None,
    intubation_time: datetime | None,
) -> int:
    """
    Compute binary deterioration label for a single prediction time.

    Label = 1 if ANY critical event occurs within [prediction_time, prediction_time + horizon]

    Args:
        prediction_time: Time of prediction
        prediction_horizon: Prediction window in hours
        death_time: ICU death time (or None)
        vasopressor_time: First vasopressor time (or None)
        intubation_time: First intubation time (or None)

    Returns:
        1 if deterioration occurs, 0 otherwise
    """
    window_end = prediction_time + timedelta(hours=prediction_horizon)

    events = [death_time, vasopressor_time, intubation_time]
    for event_time in events:
        if event_time is not None:
            if prediction_time < event_time <= window_end:
                return 1

    return 0


def compute_deterioration_labels(
    cohort: pd.DataFrame,
    prediction_horizon: int = 24,
) -> pd.DataFrame:
    """
    Add deterioration flags based on critical events.

    Adds columns:
    - has_critical_event: Whether any critical event occurred during stay
    - hours_to_event: Hours from ICU admission to first critical event

    Args:
        cohort: Cohort DataFrame with critical event times
        prediction_horizon: Prediction window in hours

    Returns:
        Cohort with deterioration flags
    """
    cohort = cohort.copy()

    # Flag stays with any critical event
    cohort["has_critical_event"] = cohort["first_critical_event_time"].notna()

    # Hours from admission to first event
    cohort["hours_to_event"] = np.nan
    mask = cohort["first_critical_event_time"].notna()
    cohort.loc[mask, "hours_to_event"] = (
        (cohort.loc[mask, "first_critical_event_time"] - cohort.loc[mask, "intime"])
        .dt.total_seconds() / 3600
    )

    # Print label statistics
    print("\nDeterioriation label statistics:")
    print(f"  Total stays: {len(cohort):,}")
    print(f"  Stays with critical event: {cohort['has_critical_event'].sum():,} ({cohort['has_critical_event'].mean()*100:.1f}%)")

    if cohort["hours_to_event"].notna().sum() > 0:
        print(f"  Hours to event (median): {cohort['hours_to_event'].median():.1f}")
        print(f"  Hours to event (mean): {cohort['hours_to_event'].mean():.1f}")

    return cohort
