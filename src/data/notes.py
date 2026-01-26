"""
Clinical Notes Extraction Module

Extract and preprocess clinical notes (radiology reports) from MIMIC-IV-Note.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class NoteRecord:
    """A clinical note record."""
    note_id: str
    stay_id: str
    hadm_id: str
    subject_id: str
    charttime: datetime
    text: str


# De-identification patterns in MIMIC notes
DEID_PATTERNS = [
    r"\[\*\*[^\]]*\*\*\]",  # [**Name**], [**Date**], etc.
    r"___+",                # Triple underscores (redacted info)
]
DEID_RE = re.compile("|".join(DEID_PATTERNS))

# Clinical abbreviations that shouldn't end a sentence
CLINICAL_ABBREVS = {
    "dr", "mr", "mrs", "ms", "vs", "pt", "mg", "ml", "cm", "mm", "kg",
    "iv", "po", "prn", "bid", "tid", "qid", "qd", "qhs", "qam", "qpm",
    "no", "approx", "inc", "dept", "hosp", "adm", "disch",
    "dx", "hx", "rx", "tx", "fx", "sx", "px", "ax", "bx", "cx",
    "l", "r", "lt", "rt", "bilat",
}


def preprocess_clinical_text(text: str) -> str:
    """
    Clean clinical text by removing de-identification placeholders.

    - Removes [**...**] patterns
    - Removes ___ placeholders
    - Normalizes whitespace
    - Preserves medical abbreviations
    """
    # Remove de-identification patterns
    text = DEID_RE.sub(" ", text)

    # Normalize whitespace (but preserve paragraph breaks)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Join with single newlines, collapse multiple blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split clinical text into sentences using clinical-aware rules.

    Handles:
    - Medical abbreviations (e.g., "Dr.", "mg.", "q.d.")
    - Numbered lists
    - Section headers
    """
    sentences: list[str] = []

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        lines = para.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a header (all caps, short, or ends with colon)
            if len(line) < 60 and (line.isupper() or line.endswith(":")):
                if line:
                    sentences.append(line)
                continue

            # Split by sentence terminators
            parts = re.split(r"(?<=[.!?])\s+", line)

            current_sentence = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Check if previous part ended with an abbreviation
                if current_sentence:
                    words = current_sentence.split()
                    if words:
                        last_word = words[-1].rstrip(".").lower()
                        if last_word in CLINICAL_ABBREVS:
                            current_sentence += " " + part
                            continue

                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = part

            if current_sentence:
                sentences.append(current_sentence)

    # Filter out very short sentences (likely noise)
    sentences = [s for s in sentences if len(s) >= 10]

    return sentences


def extract_notes(
    notes_path: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    min_length: int = 50,
    chunk_size: int = 100_000,
) -> dict[str, list[NoteRecord]]:
    """
    Extract radiology notes within ICU stay window for cohort patients.

    Args:
        notes_path: Path to radiology.csv.gz
        cohort: Cohort DataFrame with stay_id, hadm_id, intime, outtime
        output_dir: Directory to save output
        min_length: Minimum note length to include
        chunk_size: Rows per chunk for streaming

    Returns:
        Dict mapping stay_id to list of NoteRecords
    """
    # Build mapping from hadm_id to stay info
    stay_info = {}
    hadm_to_stays = {}

    for _, row in cohort.iterrows():
        stay_id = str(row["stay_id"])
        hadm_id = str(row["hadm_id"])
        stay_info[stay_id] = {
            "hadm_id": hadm_id,
            "subject_id": str(row["subject_id"]),
            "intime": row["intime"],
            "outtime": row["outtime"],
        }
        if hadm_id not in hadm_to_stays:
            hadm_to_stays[hadm_id] = []
        hadm_to_stays[hadm_id].append(stay_id)

    hadm_ids = set(hadm_to_stays.keys())
    print(f"Extracting notes for {len(hadm_ids):,} admissions...")

    # Initialize notes storage
    notes_by_stay: dict[str, list[NoteRecord]] = {
        stay_id: [] for stay_id in stay_info
    }

    total_notes = 0
    matched_notes = 0
    in_window_notes = 0

    # Stream through notes file
    for chunk in pd.read_csv(
        notes_path,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        dtype={"note_id": str, "subject_id": str, "hadm_id": str},
        parse_dates=["charttime"],
    ):
        total_notes += len(chunk)

        # Filter to relevant admissions
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if len(chunk) == 0:
            continue

        matched_notes += len(chunk)

        # Process each note
        for _, row in chunk.iterrows():
            hadm_id = str(row["hadm_id"])
            charttime = row["charttime"]

            # Skip notes without charttime
            if pd.isna(charttime):
                continue

            # Get text
            text = str(row["text"]).strip()
            if len(text) < min_length:
                continue

            # Find matching stays for this hadm
            stay_ids_for_hadm = hadm_to_stays.get(hadm_id, [])

            for stay_id in stay_ids_for_hadm:
                info = stay_info[stay_id]
                intime = info["intime"]
                outtime = info["outtime"]

                # Check if note is within ICU stay window
                if charttime < intime or charttime > outtime:
                    continue

                in_window_notes += 1

                note = NoteRecord(
                    note_id=str(row["note_id"]),
                    stay_id=stay_id,
                    hadm_id=hadm_id,
                    subject_id=str(row["subject_id"]),
                    charttime=charttime,
                    text=text,
                )
                notes_by_stay[stay_id].append(note)

        if total_notes % 500_000 == 0:
            print(f"  Processed {total_notes:,} notes, {in_window_notes:,} in window...")

    print(f"\nTotal notes: {total_notes:,}")
    print(f"Matched to cohort: {matched_notes:,}")
    print(f"Within ICU window: {in_window_notes:,}")

    # Count stays with notes
    stays_with_notes = sum(1 for notes in notes_by_stay.values() if notes)
    print(f"Stays with notes: {stays_with_notes:,} ({stays_with_notes/len(stay_info)*100:.1f}%)")

    # Save preprocessed notes
    output_dir.mkdir(parents=True, exist_ok=True)
    save_notes(notes_by_stay, output_dir)

    return notes_by_stay


def save_notes(
    notes_by_stay: dict[str, list[NoteRecord]],
    output_dir: Path,
) -> None:
    """Save preprocessed notes to JSON file."""
    result = {}

    for stay_id, notes in notes_by_stay.items():
        if not notes:
            continue

        # Sort by charttime
        notes.sort(key=lambda n: n.charttime)

        # Clean and combine texts
        combined_texts = []
        note_ids = []

        for note in notes:
            cleaned = preprocess_clinical_text(note.text)
            if cleaned:
                combined_texts.append(cleaned)
                note_ids.append(note.note_id)

        if not combined_texts:
            continue

        full_text = "\n\n".join(combined_texts)
        sentences = split_into_sentences(full_text)

        result[stay_id] = {
            "full_text": full_text,
            "sentences": sentences,
            "note_count": len(notes),
            "note_ids": note_ids,
            "charttimes": [n.charttime.isoformat() for n in notes],
        }

    # Save to JSON
    output_path = output_dir / "notes_by_stay.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved notes for {len(result)} stays to {output_path}")


def load_notes(notes_dir: Path) -> dict[str, dict]:
    """Load preprocessed notes from JSON file."""
    notes_path = notes_dir / "notes_by_stay.json"
    with open(notes_path, "r", encoding="utf-8") as f:
        return json.load(f)
