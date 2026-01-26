"""
Deterioration Classifier Models

Full multimodal classification models combining temporal encoder, text encoder, and fusion.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import TemporalEncoder
from .text import PrecomputedTextEncoder, TextEncoderWithAggregation
from .fusion import CrossModalFusion, SimpleFusion


class DeteriorationClassifier(nn.Module):
    """
    Full multimodal classifier for ICU deterioration prediction.

    Combines:
    - Temporal encoder for vitals/labs time-series
    - Text encoder for clinical notes (optional)
    - Cross-modal fusion (optional)
    - Classification head
    """

    def __init__(
        self,
        n_vitals: int,
        n_labs: int,
        n_static: int = 3,
        temporal_encoder_type: str = "lstm",
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        temporal_dropout: float = 0.3,
        use_text: bool = True,
        text_input_dim: int = 768,
        text_projection_dim: int = 256,
        text_dropout: float = 0.1,
        fusion_type: str = "cross_modal",
        fusion_hidden: int = 128,
        fusion_heads: int = 4,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.3,
    ):
        """
        Args:
            n_vitals: Number of vital sign features
            n_labs: Number of lab features
            n_static: Number of static features (age, gender, has_notes)
            temporal_encoder_type: "lstm" or "transformer"
            temporal_hidden: Hidden dim for temporal encoder
            temporal_layers: Number of temporal encoder layers
            temporal_dropout: Dropout for temporal encoder
            use_text: Whether to use text modality
            text_input_dim: Dimension of text embeddings (768 for BERT)
            text_projection_dim: Text projection dimension
            text_dropout: Dropout for text encoder
            fusion_type: "cross_modal" or "simple"
            fusion_hidden: Hidden dim for fusion
            fusion_heads: Number of attention heads for cross-modal fusion
            classifier_hidden: Hidden dim for classifier
            classifier_dropout: Dropout for classifier
        """
        super().__init__()

        self.n_vitals = n_vitals
        self.n_labs = n_labs
        self.n_static = n_static
        self.use_text = use_text
        self.fusion_type = fusion_type

        # Temporal encoder for vitals + labs
        input_dim = n_vitals + n_labs
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            encoder_type=temporal_encoder_type,
            hidden_dim=temporal_hidden,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        temporal_output_dim = self.temporal_encoder.output_dim

        # Add static features dimension
        structured_dim = temporal_output_dim + n_static

        # Text encoder (optional)
        if use_text:
            self.text_encoder = PrecomputedTextEncoder(
                input_dim=text_input_dim,
                projection_dim=text_projection_dim,
                dropout=text_dropout,
            )
            text_dim = text_projection_dim

            # Fusion layer
            if fusion_type == "cross_modal":
                self.fusion = CrossModalFusion(
                    structured_dim=structured_dim,
                    text_dim=text_dim,
                    hidden_dim=fusion_hidden,
                    num_heads=fusion_heads,
                    dropout=text_dropout,
                )
            else:
                self.fusion = SimpleFusion(
                    structured_dim=structured_dim,
                    text_dim=text_dim,
                    hidden_dim=fusion_hidden,
                    dropout=text_dropout,
                )

            classifier_input_dim = self.fusion.output_dim
        else:
            self.text_encoder = None
            self.fusion = None
            classifier_input_dim = structured_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        vitals: torch.Tensor,
        labs: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        has_notes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.

        Args:
            vitals: (batch, seq_len, n_vitals) - vital signs sequence
            labs: (batch, seq_len, n_labs) - lab values sequence
            mask: (batch, seq_len, n_vitals + n_labs) - validity mask
            static: (batch, n_static) - static features
            embedding: (batch, 768) - pre-computed BERT embedding (optional)
            has_notes: (batch,) - mask indicating which samples have notes

        Returns:
            logits: (batch, 1) - prediction logits
            attn_info: dict with attention weights and intermediate outputs
        """
        # Combine vitals and labs
        temporal_input = torch.cat([vitals, labs], dim=-1)  # (batch, seq_len, n_vitals + n_labs)

        # Temporal encoding
        temporal_repr, temporal_attn = self.temporal_encoder(temporal_input, mask)

        # Combine with static features
        structured_repr = torch.cat([temporal_repr, static], dim=-1)

        # Multimodal fusion
        if self.use_text and embedding is not None:
            # Encode text
            text_repr = self.text_encoder(embedding)

            # Fuse modalities
            fused_repr, fusion_attn_info = self.fusion(
                structured_repr, text_repr, has_notes
            )
        else:
            fused_repr = structured_repr
            fusion_attn_info = {}

        # Classification
        logits = self.classifier(fused_repr)

        # Collect attention info
        attn_info = {
            "temporal_attn": temporal_attn,
            **fusion_attn_info,
        }

        return logits, attn_info


class StructuredOnlyClassifier(nn.Module):
    """
    Structured data only classifier (baseline).

    Uses only vitals, labs, and static features.
    """

    def __init__(
        self,
        n_vitals: int,
        n_labs: int,
        n_static: int = 3,
        temporal_encoder_type: str = "lstm",
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        temporal_dropout: float = 0.3,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = DeteriorationClassifier(
            n_vitals=n_vitals,
            n_labs=n_labs,
            n_static=n_static,
            temporal_encoder_type=temporal_encoder_type,
            temporal_hidden=temporal_hidden,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            use_text=False,
            classifier_hidden=classifier_hidden,
            classifier_dropout=classifier_dropout,
        )

    def forward(
        self,
        vitals: torch.Tensor,
        labs: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.classifier(vitals, labs, mask, static)


class TextOnlyClassifier(nn.Module):
    """
    Text only classifier (baseline).

    Uses only clinical note embeddings.
    """

    def __init__(
        self,
        text_input_dim: int = 768,
        text_projection_dim: int = 256,
        text_dropout: float = 0.1,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.text_encoder = PrecomputedTextEncoder(
            input_dim=text_input_dim,
            projection_dim=text_projection_dim,
            dropout=text_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(text_projection_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        has_notes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.

        Args:
            embedding: (batch, 768) - BERT embedding
            has_notes: (batch,) - mask for patients with notes

        Returns:
            logits: (batch, 1)
            attn_info: empty dict
        """
        text_repr = self.text_encoder(embedding)
        logits = self.classifier(text_repr)

        # Set logits to 0 for patients without notes
        if has_notes is not None:
            mask = has_notes.unsqueeze(-1)
            logits = logits * mask

        return logits, {}


def create_model(
    model_type: str,
    n_vitals: int,
    n_labs: int,
    config: dict,
) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: "multimodal", "structured", or "text"
        n_vitals: Number of vital features
        n_labs: Number of lab features
        config: Model configuration dict

    Returns:
        Model instance
    """
    model_config = config.get("model", {})

    if model_type == "multimodal":
        return DeteriorationClassifier(
            n_vitals=n_vitals,
            n_labs=n_labs,
            temporal_encoder_type=model_config.get("temporal_encoder", "lstm"),
            temporal_hidden=model_config.get("lstm", {}).get("hidden_dim", 128),
            temporal_layers=model_config.get("lstm", {}).get("num_layers", 2),
            temporal_dropout=model_config.get("lstm", {}).get("dropout", 0.3),
            use_text=True,
            text_input_dim=768,
            text_projection_dim=model_config.get("text", {}).get("projection_dim", 256),
            text_dropout=model_config.get("text", {}).get("dropout", 0.1),
            fusion_hidden=model_config.get("fusion", {}).get("hidden_dim", 128),
            classifier_hidden=model_config.get("classifier", {}).get("hidden_dim", 64),
            classifier_dropout=model_config.get("classifier", {}).get("dropout", 0.3),
        )

    elif model_type == "structured":
        return StructuredOnlyClassifier(
            n_vitals=n_vitals,
            n_labs=n_labs,
            temporal_encoder_type=model_config.get("temporal_encoder", "lstm"),
            temporal_hidden=model_config.get("lstm", {}).get("hidden_dim", 128),
            temporal_layers=model_config.get("lstm", {}).get("num_layers", 2),
            temporal_dropout=model_config.get("lstm", {}).get("dropout", 0.3),
            classifier_hidden=model_config.get("classifier", {}).get("hidden_dim", 64),
            classifier_dropout=model_config.get("classifier", {}).get("dropout", 0.3),
        )

    elif model_type == "text":
        return TextOnlyClassifier(
            text_input_dim=768,
            text_projection_dim=model_config.get("text", {}).get("projection_dim", 256),
            text_dropout=model_config.get("text", {}).get("dropout", 0.1),
            classifier_hidden=model_config.get("classifier", {}).get("hidden_dim", 64),
            classifier_dropout=model_config.get("classifier", {}).get("dropout", 0.3),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
