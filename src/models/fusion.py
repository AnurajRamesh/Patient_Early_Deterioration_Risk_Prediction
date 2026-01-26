"""
Cross-Modal Fusion Module

Combines structured EHR time-series representation with clinical note representation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Cross-modal attention fusion for structured and text representations.

    Uses gated attention to adaptively combine information from both modalities.
    Handles cases where text is missing gracefully.
    """

    def __init__(
        self,
        structured_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            structured_dim: Dimension of structured data representation
            text_dim: Dimension of text representation
            hidden_dim: Hidden/output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.structured_dim = structured_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Project both modalities to common dimension
        self.struct_proj = nn.Linear(structured_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-attention: text attends to structured
        self.text_to_struct_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: structured attends to text
        self.struct_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Layer norms
        self.ln_struct = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_fused = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        struct_repr: torch.Tensor,
        text_repr: torch.Tensor,
        has_notes_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            struct_repr: (batch, structured_dim) - structured data representation
            text_repr: (batch, text_dim) - text representation (can be zeros if no notes)
            has_notes_mask: (batch,) - 1 if patient has notes, 0 otherwise

        Returns:
            fused: (batch, hidden_dim) - fused representation
            attn_info: dict with attention weights and gate values
        """
        batch_size = struct_repr.size(0)

        # Project to common dimension
        struct_h = self.struct_proj(struct_repr)  # (batch, hidden_dim)
        text_h = self.text_proj(text_repr)        # (batch, hidden_dim)

        # Add sequence dimension for attention
        struct_h = struct_h.unsqueeze(1)  # (batch, 1, hidden_dim)
        text_h = text_h.unsqueeze(1)      # (batch, 1, hidden_dim)

        # Cross-attention: text representation attends to structured
        text_attended, text_attn = self.text_to_struct_attn(
            text_h, struct_h, struct_h
        )  # (batch, 1, hidden_dim)

        # Cross-attention: structured representation attends to text
        struct_attended, struct_attn = self.struct_to_text_attn(
            struct_h, text_h, text_h
        )  # (batch, 1, hidden_dim)

        # Remove sequence dimension
        text_attended = text_attended.squeeze(1)    # (batch, hidden_dim)
        struct_attended = struct_attended.squeeze(1)  # (batch, hidden_dim)

        # Layer norm with residual
        text_attended = self.ln_text(text_h.squeeze(1) + text_attended)
        struct_attended = self.ln_struct(struct_h.squeeze(1) + struct_attended)

        # Compute gate
        combined = torch.cat([struct_attended, text_attended], dim=-1)
        gate = self.gate(combined)  # (batch, hidden_dim)

        # Apply mask for patients without notes
        if has_notes_mask is not None:
            # Expand mask to hidden_dim
            mask = has_notes_mask.unsqueeze(-1).float()  # (batch, 1)
            # For patients without notes, gate should be 0 (use struct only)
            gate = gate * mask

        # Gated fusion
        fused = gate * text_attended + (1 - gate) * struct_attended

        # Layer norm and output projection
        fused = self.ln_fused(fused)
        fused = self.output_proj(fused)

        # Collect attention info
        attn_info = {
            "text_to_struct_attn": text_attn,
            "struct_to_text_attn": struct_attn,
            "gate": gate,
        }

        return fused, attn_info


class SimpleFusion(nn.Module):
    """
    Simple concatenation-based fusion.

    Concatenates structured and text representations, then projects.
    """

    def __init__(
        self,
        structured_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            structured_dim: Dimension of structured data representation
            text_dim: Dimension of text representation
            hidden_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.structured_dim = structured_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(structured_dim + text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        struct_repr: torch.Tensor,
        text_repr: torch.Tensor,
        has_notes_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            struct_repr: (batch, structured_dim)
            text_repr: (batch, text_dim)
            has_notes_mask: (batch,) - 1 if patient has notes, 0 otherwise

        Returns:
            fused: (batch, hidden_dim)
            attn_info: empty dict
        """
        # Zero out text for patients without notes
        if has_notes_mask is not None:
            mask = has_notes_mask.unsqueeze(-1).float()
            text_repr = text_repr * mask

        # Concatenate and fuse
        combined = torch.cat([struct_repr, text_repr], dim=-1)
        fused = self.fusion(combined)

        return fused, {}


class LateFusion(nn.Module):
    """
    Late fusion: separate predictions combined at output level.

    Learns optimal blending weights for structured and text predictions.
    """

    def __init__(
        self,
        structured_dim: int,
        text_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            structured_dim: Dimension of structured data representation
            text_dim: Dimension of text representation
            dropout: Dropout probability
        """
        super().__init__()

        # Separate classifiers
        self.struct_classifier = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Learnable blending weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))

        self.output_dim = 1

    def forward(
        self,
        struct_repr: torch.Tensor,
        text_repr: torch.Tensor,
        has_notes_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            struct_repr: (batch, structured_dim)
            text_repr: (batch, text_dim)
            has_notes_mask: (batch,) - 1 if patient has notes

        Returns:
            logits: (batch, 1)
            attn_info: dict with individual predictions
        """
        # Get individual predictions
        struct_logits = self.struct_classifier(struct_repr)  # (batch, 1)
        text_logits = self.text_classifier(text_repr)        # (batch, 1)

        # Blend weight (clamped to [0, 1])
        w = torch.sigmoid(self.blend_weight)

        # Blend predictions
        if has_notes_mask is not None:
            # Only use text when notes available
            mask = has_notes_mask.unsqueeze(-1).float()
            # For patients with notes: weighted blend
            # For patients without notes: struct only
            blended = mask * (w * text_logits + (1 - w) * struct_logits) + \
                     (1 - mask) * struct_logits
        else:
            blended = w * text_logits + (1 - w) * struct_logits

        attn_info = {
            "struct_logits": struct_logits,
            "text_logits": text_logits,
            "blend_weight": w,
        }

        return blended, attn_info
