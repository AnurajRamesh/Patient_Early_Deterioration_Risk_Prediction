"""
ICU Cohort Construction Module

Builds the study cohort from MIMIC-IV ICU stays with inclusion/exclusion criteria.
"""
from __future__ import annotations

import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np


def parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime string in common formats."""
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def load_icustays(data_dir: Path) -> pd.DataFrame:
    """
    Load ICU stays from MIMIC-IV.

    Returns DataFrame with columns:
    - stay_id, hadm_id, subject_id
    - first_careunit, last_careunit
    - intime, outtime, los
    """
    path = data_dir / "mimic-iv-3.1" / "icu" / "icustays.csv.gz"
    print(f"Loading ICU stays from {path}...")

    df = pd.read_csv(
        path,
        compression="gzip",
        dtype={
            "stay_id": str,
            "hadm_id": str,
            "subject_id": str,
            "first_careunit": str,
            "last_careunit": str,
        },
        parse_dates=["intime", "outtime"],
    )

    print(f"Loaded {len(df):,} ICU stays")
    return df


def load_admissions(data_dir: Path) -> pd.DataFrame:
    """
    Load hospital admissions with mortality information.

    Returns DataFrame with columns:
    - hadm_id, subject_id
    - admittime, dischtime, deathtime
    - hospital_expire_flag
    """
    path = data_dir / "mimic-iv-3.1" / "hosp" / "admissions.csv.gz"
    print(f"Loading admissions from {path}...")

    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "hadm_id", "subject_id", "admittime", "dischtime",
            "deathtime", "hospital_expire_flag", "admission_type",
        ],
        dtype={
            "hadm_id": str,
            "subject_id": str,
            "hospital_expire_flag": int,
            "admission_type": str,
        },
        parse_dates=["admittime", "dischtime", "deathtime"],
    )

    print(f"Loaded {len(df):,} admissions")
    return df


def load_patients(data_dir: Path) -> pd.DataFrame:
    """
    Load patient demographics.

    Returns DataFrame with columns:
    - subject_id, gender, anchor_age, dod
    """
    path = data_dir / "mimic-iv-3.1" / "hosp" / "patients.csv.gz"
    print(f"Loading patients from {path}...")

    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        dtype={
            "subject_id": str,
            "gender": str,
            "anchor_age": int,
            "anchor_year": int,
        },
    )

    # Parse dod (date of death)
    df["dod"] = pd.to_datetime(df["dod"], errors="coerce")

    print(f"Loaded {len(df):,} patients")
    return df


def build_icu_cohort(
    data_dir: Path,
    output_dir: Path,
    min_los_hours: float = 24.0,
    min_age: int = 18,
    exclusion_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Build ICU cohort with inclusion/exclusion criteria.

    Inclusion criteria:
    - ICU stay >= min_los_hours (default 24h)
    - Age >= min_age (default 18)

    Exclusion criteria:
    - Death within exclusion_hours of ICU admission (default 6h)
    - Missing intime or outtime

    Args:
        data_dir: Path to MIMIC-IV data directory
        output_dir: Path to save cohort files
        min_los_hours: Minimum ICU length of stay in hours
        min_age: Minimum patient age
        exclusion_hours: Exclude deaths within first N hours

    Returns:
        DataFrame with cohort information
    """
    # Load source data
    icu_stays = load_icustays(data_dir)
    admissions = load_admissions(data_dir)
    patients = load_patients(data_dir)

    # Merge ICU stays with admissions and patients
    print("\nMerging data sources...")
    cohort = icu_stays.merge(
        admissions[["hadm_id", "hospital_expire_flag", "deathtime", "admission_type"]],
        on="hadm_id",
        how="left",
    )
    cohort = cohort.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    )

    print(f"After merge: {len(cohort):,} ICU stays")

    # Calculate length of stay in hours
    cohort["los_hours"] = (
        (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
    )

    # Determine if death occurred during ICU stay
    cohort["died_in_icu"] = False
    mask_died = (
        (cohort["hospital_expire_flag"] == 1) &
        cohort["deathtime"].notna() &
        (cohort["deathtime"] >= cohort["intime"]) &
        (cohort["deathtime"] <= cohort["outtime"])
    )
    cohort.loc[mask_died, "died_in_icu"] = True

    # Calculate hours from ICU admission to death
    cohort["hours_to_death"] = np.nan
    cohort.loc[mask_died, "hours_to_death"] = (
        (cohort.loc[mask_died, "deathtime"] - cohort.loc[mask_died, "intime"])
        .dt.total_seconds() / 3600
    )

    # Apply inclusion criteria
    print("\nApplying inclusion criteria...")
    n_before = len(cohort)

    # Missing times
    cohort = cohort.dropna(subset=["intime", "outtime"])
    print(f"  - After removing missing times: {len(cohort):,} ({n_before - len(cohort):,} removed)")
    n_before = len(cohort)

    # Minimum LOS
    cohort = cohort[cohort["los_hours"] >= min_los_hours]
    print(f"  - After LOS >= {min_los_hours}h: {len(cohort):,} ({n_before - len(cohort):,} removed)")
    n_before = len(cohort)

    # Minimum age
    cohort = cohort[cohort["anchor_age"] >= min_age]
    print(f"  - After age >= {min_age}: {len(cohort):,} ({n_before - len(cohort):,} removed)")
    n_before = len(cohort)

    # Apply exclusion criteria
    print("\nApplying exclusion criteria...")

    # Exclude early deaths (within exclusion_hours)
    early_death_mask = (
        cohort["died_in_icu"] &
        (cohort["hours_to_death"] < exclusion_hours)
    )
    cohort = cohort[~early_death_mask]
    print(f"  - After excluding deaths < {exclusion_hours}h: {len(cohort):,} ({n_before - len(cohort):,} removed)")

    # Rename columns for clarity
    cohort = cohort.rename(columns={
        "anchor_age": "age",
        "deathtime": "icu_deathtime",
    })

    # Select and order columns
    cohort = cohort[[
        "stay_id", "hadm_id", "subject_id",
        "intime", "outtime", "los_hours",
        "first_careunit", "last_careunit",
        "age", "gender", "admission_type",
        "hospital_expire_flag", "died_in_icu", "icu_deathtime",
        "hours_to_death",
    ]]

    # Print summary statistics
    print("\n" + "=" * 60)
    print("COHORT SUMMARY")
    print("=" * 60)
    print(f"Total ICU stays: {len(cohort):,}")
    print(f"Unique patients: {cohort['subject_id'].nunique():,}")
    print(f"ICU mortality: {cohort['died_in_icu'].sum():,} ({cohort['died_in_icu'].mean()*100:.1f}%)")
    print(f"\nAge distribution:")
    print(f"  Mean: {cohort['age'].mean():.1f}")
    print(f"  Median: {cohort['age'].median():.1f}")
    print(f"  Min: {cohort['age'].min()}")
    print(f"  Max: {cohort['age'].max()}")
    print(f"\nGender distribution:")
    print(cohort["gender"].value_counts())
    print(f"\nLOS (hours) distribution:")
    print(f"  Mean: {cohort['los_hours'].mean():.1f}")
    print(f"  Median: {cohort['los_hours'].median():.1f}")
    print(f"  Min: {cohort['los_hours'].min():.1f}")
    print(f"  Max: {cohort['los_hours'].max():.1f}")

    # Save cohort
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "icu_cohort.parquet"
    cohort.to_parquet(output_path, index=False)
    print(f"\nCohort saved to {output_path}")

    # Also save as CSV for inspection
    csv_path = output_dir / "icu_cohort.csv"
    cohort.to_csv(csv_path, index=False)
    print(f"CSV copy saved to {csv_path}")

    return cohort


def load_cohort(cohort_path: Path) -> pd.DataFrame:
    """Load cohort from parquet file."""
    df = pd.read_parquet(cohort_path)

    # Ensure datetime columns are parsed
    for col in ["intime", "outtime", "icu_deathtime"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col])

    return df


def split_cohort(
    cohort: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_by: str = "subject_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cohort into train/val/test sets by patient.

    Args:
        cohort: Full cohort DataFrame
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        split_by: Column to split by (default: subject_id for patient-level)
        seed: Random seed

    Returns:
        (train_df, val_df, test_df)
    """
    # Get unique split units (patients)
    units = cohort[split_by].unique()
    n_units = len(units)

    # Shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(units)

    # Calculate split sizes
    n_test = int(n_units * test_frac)
    n_val = int(n_units * val_frac)

    test_units = set(units[:n_test])
    val_units = set(units[n_test:n_test + n_val])
    train_units = set(units[n_test + n_val:])

    # Split data
    train_df = cohort[cohort[split_by].isin(train_units)].copy()
    val_df = cohort[cohort[split_by].isin(val_units)].copy()
    test_df = cohort[cohort[split_by].isin(test_units)].copy()

    print(f"Split by {split_by}:")
    print(f"  Train: {len(train_df):,} stays ({len(train_units):,} patients)")
    print(f"  Val:   {len(val_df):,} stays ({len(val_units):,} patients)")
    print(f"  Test:  {len(test_df):,} stays ({len(test_units):,} patients)")

    return train_df, val_df, test_df
