# ICU Deterioration Risk Prediction System

A multimodal deep learning system that predicts patient deterioration in the ICU by combining structured EHR time-series data (vitals/labs) with clinical notes from MIMIC-IV.

## Overview

This repo contains an end-to-end pipeline:

1. Build an ICU cohort and compute deterioration labels (critical events such as vasopressor use / intubation).
2. Extract and preprocess hourly vitals and lab time series.
3. Extract radiology notes (MIMIC-IV-Note) and generate ClinicalBERT embeddings.
4. Build hourly prediction samples for train/val/test.
5. Train structured-only, text-only, or multimodal models (LSTM or Transformer temporal encoder).
6. Evaluate on a held-out test set with metrics and plots (calibration + feature importance).

## Setup

- Python: 3.9+
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

You need access to:

- MIMIC-IV (expected folder name: `mimic-iv-3.1/`)
- MIMIC-IV-Note (expected folder name: `mimic-iv-note-2.2/`)

Scripts read compressed CSVs (`.csv.gz`) directly. The expected relative paths are listed in `configs/features.yaml`.

## Run the pipeline

Set `DATA_DIR` to the directory that contains `mimic-iv-3.1/` (and optionally `mimic-iv-note-2.2/`):

```bash
export DATA_DIR=/path/to/mimic-data
```

1) Build cohort + labels:

```bash
python scripts/01_build_cohort.py --data-dir "$DATA_DIR" --output-dir outputs/cohort
```

2) Extract vitals:

```bash
python scripts/02_extract_vitals.py \
  --data-dir "$DATA_DIR" \
  --cohort outputs/cohort/icu_cohort_with_events.parquet \
  --output-dir outputs/timeseries
```

3) Extract labs:

```bash
python scripts/03_extract_labs.py \
  --data-dir "$DATA_DIR" \
  --cohort outputs/cohort/icu_cohort_with_events.parquet \
  --output-dir outputs/timeseries
```

4) Extract notes (radiology):

```bash
python scripts/04_extract_notes.py \
  --data-dir "$DATA_DIR" \
  --cohort outputs/cohort/icu_cohort_with_events.parquet \
  --output-dir outputs/notes
```

5) Generate ClinicalBERT embeddings (requires downloading the model from Hugging Face the first time):

```bash
python scripts/05_generate_embeddings.py \
  --notes-path outputs/notes/notes_by_stay.json \
  --output-dir outputs/embeddings
```

6) Build hourly samples:

```bash
python scripts/06_build_samples.py \
  --cohort outputs/cohort/icu_cohort_with_events.parquet \
  --vitals-dir outputs/timeseries \
  --labs-dir outputs/timeseries \
  --embeddings-dir outputs/embeddings \
  --output-dir outputs/samples
```

7) Train a model:

```bash
# Multimodal (default)
python scripts/07_train_model.py --samples-dir outputs/samples --output-dir outputs/models/multimodal --model multimodal

# Structured-only baseline
python scripts/07_train_model.py --samples-dir outputs/samples --output-dir outputs/models/structured --model structured

# Use the full training config
python scripts/07_train_model.py --config configs/training.yaml --samples-dir outputs/samples --output-dir outputs/models/multimodal --model multimodal
```

8) Evaluate:

```bash
python scripts/08_evaluate.py --model-dir outputs/models/multimodal --samples-dir outputs/samples --output-dir outputs/results
```

## Repository layout

- `scripts/`: pipeline entry points (`01_...py` → `08_...py`)
- `src/data/`: cohort creation, extraction, sample building, dataset/dataloaders
- `src/models/`: temporal encoders, text encoder, fusion, classifier
- `src/training/`: losses + trainer
- `src/evaluation/`: metrics + interpretability plots
- `configs/`: feature item IDs and training configuration

## Notes

- `outputs/` (and other large artifacts like `.parquet`/`.npz`) are ignored by git via `.gitignore`.
- Device selection is auto-detected (CUDA → MPS → CPU). You can override with `--device cuda|mps|cpu` in training/evaluation scripts.

