"""
Temporal Encoder Models for Time-Series EHR Data

Implements both LSTM and Transformer encoders for vital signs and lab sequences.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with self-attention for temporal EHR data.

    Processes sequences of vitals/labs and outputs a fixed-size representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features (vitals + labs)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) - input sequence
            mask: (batch, seq_len) - 1 where valid, 0 where padding

        Returns:
            output: (batch, output_dim) - sequence representation
            attn_weights: (batch, seq_len) - attention weights over time
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)
        x = F.relu(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, output_dim)

        # Self-attention
        # Create attention mask (True = ignore position)
        if mask is not None:
            # Sum across features to get per-timestep mask
            if mask.dim() == 3:
                timestep_mask = (mask.sum(dim=-1) == 0)  # (batch, seq_len)
            else:
                timestep_mask = (mask == 0)
        else:
            timestep_mask = None

        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=timestep_mask,
        )

        # Residual connection and layer norm
        attn_out = self.layer_norm(lstm_out + attn_out)
        attn_out = self.dropout(attn_out)

        # Global average pooling (with mask)
        if mask is not None:
            # Expand mask to match output dim
            if mask.dim() == 3:
                valid_mask = (mask.sum(dim=-1, keepdim=True) > 0).float()
            else:
                valid_mask = mask.unsqueeze(-1).float()

            # Masked mean
            attn_out = attn_out * valid_mask
            output = attn_out.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        else:
            output = attn_out.mean(dim=1)

        # Get attention weights (average over heads if multi-head)
        attn_weights = attn_weights.mean(dim=1) if attn_weights.dim() == 3 else attn_weights

        return output, attn_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal EHR data.

    Uses positional encoding and multi-head self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 168,
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output dimension
        self.output_dim = d_model

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) - input sequence
            mask: (batch, seq_len) - 1 where valid, 0 where padding

        Returns:
            output: (batch, d_model) - sequence representation
            attn_weights: (batch, seq_len) - not actual weights, just placeholder
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask
        if mask is not None:
            if mask.dim() == 3:
                src_mask = (mask.sum(dim=-1) == 0)  # (batch, seq_len)
            else:
                src_mask = (mask == 0)
            # Add mask for CLS token (always attend)
            cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=torch.bool)
            src_mask = torch.cat([cls_mask, src_mask], dim=1)
        else:
            src_mask = None

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_mask)

        # Apply layer norm
        x = self.layer_norm(x)

        # Use CLS token output as sequence representation
        output = x[:, 0, :]  # (batch, d_model)

        # Return placeholder attention weights
        attn_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len

        return output, attn_weights


class TemporalEncoder(nn.Module):
    """
    Wrapper class that can use either LSTM or Transformer encoder.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_type: str = "lstm",
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        nhead: int = 4,
        max_seq_len: int = 168,
    ):
        """
        Args:
            input_dim: Number of input features
            encoder_type: "lstm" or "transformer"
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout probability
            nhead: Number of attention heads (transformer only)
            max_seq_len: Maximum sequence length (transformer only)
        """
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            self.output_dim = self.encoder.output_dim
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            self.output_dim = self.encoder.output_dim
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) or (batch, seq_len, input_dim)

        Returns:
            output: (batch, output_dim)
            attn_weights: (batch, seq_len)
        """
        return self.encoder(x, mask)
