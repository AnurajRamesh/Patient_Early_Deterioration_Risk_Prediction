#!/usr/bin/env python3
"""
Script 5: Generate BERT Embeddings

Generates ClinicalBERT embeddings for clinical notes.

Usage:
    python scripts/05_generate_embeddings.py \
        --notes-path outputs/notes/notes_by_stay.json \
        --output-dir outputs/embeddings \
        --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_bioclinical_bert(model_name: str, device: str):
    """Load Bio_ClinicalBERT model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print(f"Model loaded. Hidden size: {model.config.hidden_size}")
    return tokenizer, model


def generate_cls_embedding(
    text: str,
    tokenizer,
    model,
    device: str,
    max_length: int = 512,
) -> np.ndarray:
    """Generate [CLS] embedding for a single text."""
    # Tokenize
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )

    input_ids = tokens["input_ids"][0]
    total_tokens = len(input_ids)

    if total_tokens <= max_length:
        # Single pass
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embedding[0]

    else:
        # Sliding window for long documents
        stride = max_length // 2
        cls_embeddings = []

        for start in range(0, total_tokens, stride):
            end = min(start + max_length, total_tokens)
            chunk_ids = input_ids[start:end].unsqueeze(0)
            attention_mask = torch.ones_like(chunk_ids)

            inputs = {
                "input_ids": chunk_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }

            with torch.no_grad():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                cls_embeddings.append(cls_emb[0])

            if end >= total_tokens:
                break

        return np.mean(cls_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ClinicalBERT embeddings for clinical notes"
    )
    parser.add_argument(
        "--notes-path",
        type=Path,
        required=True,
        help="Path to notes_by_stay.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N admissions",
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print("=" * 60)
    print("Generating BERT Embeddings")
    print("=" * 60)
    print(f"Notes path: {args.notes_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # Load notes
    print("Loading notes...")
    with open(args.notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)
    print(f"Loaded notes for {len(notes_data):,} stays")

    # Load model
    tokenizer, model = load_bioclinical_bert(args.model, args.device)
    hidden_size = model.config.hidden_size

    # Generate embeddings
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cls_embeddings = {}
    stay_ids = list(notes_data.keys())
    total = len(stay_ids)

    for idx, stay_id in enumerate(stay_ids):
        note_info = notes_data[stay_id]
        full_text = note_info.get("full_text", "")

        if full_text:
            cls_emb = generate_cls_embedding(
                full_text, tokenizer, model, args.device
            )
        else:
            cls_emb = np.zeros(hidden_size)

        cls_embeddings[stay_id] = cls_emb

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} stays...")

        # Checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            np.savez_compressed(
                args.output_dir / "cls_embeddings.npz",
                **cls_embeddings,
            )

    # Final save
    np.savez_compressed(
        args.output_dir / "cls_embeddings.npz",
        **cls_embeddings,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated embeddings for {len(cls_embeddings):,} stays")
    print(f"Embedding dimension: {hidden_size}")
    print(f"Output saved to: {args.output_dir / 'cls_embeddings.npz'}")


if __name__ == "__main__":
    main()
