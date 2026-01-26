"""
Text Encoder Models for Clinical Notes

Uses ClinicalBERT to encode clinical notes and aggregate multiple notes.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    ClinicalBERT-based text encoder for clinical notes.

    Wraps a pre-trained BERT model and projects to desired dimension.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        projection_dim: int = 256,
        freeze_bert: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            projection_dim: Output dimension after projection
            freeze_bert: Whether to freeze BERT parameters
            dropout: Dropout probability
        """
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze_bert = freeze_bert

        # Load BERT model (lazy loading to avoid import errors if not needed)
        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
            self.bert_dim = self.bert.config.hidden_size  # 768 for BERT
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = projection_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) - tokenized input
            attention_mask: (batch, seq_len) - attention mask

        Returns:
            (batch, projection_dim) - text embedding
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, bert_dim)

        # Project to desired dimension
        return self.projection(cls_embedding)


class PrecomputedTextEncoder(nn.Module):
    """
    Text encoder that uses pre-computed BERT embeddings.

    More efficient for training when embeddings are pre-computed.
    """

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of pre-computed embeddings (768 for BERT)
            projection_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = projection_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embedding: (batch, input_dim) - pre-computed BERT embedding

        Returns:
            (batch, projection_dim) - projected embedding
        """
        return self.projection(embedding)


class NoteAggregator(nn.Module):
    """
    Aggregates embeddings from multiple clinical notes using attention.

    Used when a patient has multiple notes at prediction time.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.output_dim = embed_dim

    def forward(
        self,
        note_embeddings: torch.Tensor,
        note_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            note_embeddings: (batch, max_notes, embed_dim) - note embeddings
            note_mask: (batch, max_notes) - 1 where valid note, 0 for padding

        Returns:
            output: (batch, embed_dim) - aggregated representation
            attn_weights: (batch, max_notes) - attention weights
        """
        batch_size, max_notes, _ = note_embeddings.shape

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)

        # Create attention mask (True = ignore)
        if note_mask is not None:
            key_padding_mask = (note_mask == 0)
        else:
            key_padding_mask = None

        # Attention
        attn_out, attn_weights = self.attention(
            query, note_embeddings, note_embeddings,
            key_padding_mask=key_padding_mask,
        )

        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)  # (batch, embed_dim)

        # Layer norm and projection
        output = self.layer_norm(attn_out)
        output = self.output_proj(output)

        # Squeeze attention weights
        attn_weights = attn_weights.squeeze(1)  # (batch, max_notes)

        return output, attn_weights


class TextEncoderWithAggregation(nn.Module):
    """
    Combined text encoder with multi-note aggregation.

    Handles both single notes and multiple notes per patient.
    """

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_aggregation: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            projection_dim: Output dimension
            num_heads: Number of attention heads for aggregation
            dropout: Dropout probability
            use_aggregation: Whether to use attention aggregation for multiple notes
        """
        super().__init__()

        self.text_encoder = PrecomputedTextEncoder(
            input_dim=input_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        self.use_aggregation = use_aggregation
        if use_aggregation:
            self.note_aggregator = NoteAggregator(
                embed_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.output_dim = projection_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        note_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            embeddings: (batch, input_dim) for single notes,
                       or (batch, max_notes, input_dim) for multiple notes
            note_mask: (batch, max_notes) - mask for multiple notes

        Returns:
            output: (batch, projection_dim)
            attn_weights: (batch, max_notes) or None for single notes
        """
        if embeddings.dim() == 2:
            # Single note per patient
            output = self.text_encoder(embeddings)
            return output, None

        elif embeddings.dim() == 3:
            # Multiple notes per patient
            batch_size, max_notes, input_dim = embeddings.shape

            # Project each note
            # Reshape: (batch * max_notes, input_dim)
            flat_embeddings = embeddings.view(-1, input_dim)
            projected = self.text_encoder(flat_embeddings)
            # Reshape back: (batch, max_notes, projection_dim)
            projected = projected.view(batch_size, max_notes, -1)

            if self.use_aggregation:
                # Aggregate with attention
                output, attn_weights = self.note_aggregator(projected, note_mask)
                return output, attn_weights
            else:
                # Simple average pooling
                if note_mask is not None:
                    mask = note_mask.unsqueeze(-1).float()
                    output = (projected * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                else:
                    output = projected.mean(dim=1)
                return output, None

        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got {embeddings.dim()}D")
