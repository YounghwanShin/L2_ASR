"""Unified model architecture for pronunciation assessment.

This module implements the main model combining Wav2Vec2 encoder,
feature processing, and task-specific output heads for canonical phoneme,
perceived phoneme, and error detection.
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import CanonicalHead, PerceivedHead, ErrorDetectionHead


class UnifiedModel(nn.Module):
    """Unified model for multi-task pronunciation assessment.
    
    The model consists of three main components:
    1. Wav2Vec2 encoder for audio feature extraction
    2. Feature encoder (Simple or Transformer) for enhancement
    3. Task-specific output heads for three prediction tasks
    
    Attributes:
        encoder: Wav2Vec2 encoder for audio processing.
        feature_encoder: Additional encoding layer.
        canonical_head: Output head for canonical phoneme prediction.
        perceived_head: Output head for perceived phoneme prediction.
        error_head: Output head for error detection.
    """
    
    def __init__(self,
                 pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
                 hidden_dim: int = 512,
                 num_phonemes: int = 42,
                 num_error_types: int = 5,
                 dropout: float = 0.1,
                 use_transformer: bool = False,
                 num_layers: int = 2,
                 num_heads: int = 8):
        """Initializes the unified model.
        
        Args:
            pretrained_model_name: Pretrained Wav2Vec2 model identifier.
            hidden_dim: Hidden dimension for feature encoder.
            num_phonemes: Number of phoneme classes.
            num_error_types: Number of error types.
            dropout: Dropout rate.
            use_transformer: Whether to use Transformer encoder.
            num_layers: Number of Transformer layers.
            num_heads: Number of attention heads.
        """
        super().__init__()

        # Wav2Vec2 encoder
        self.encoder = Wav2VecEncoder(pretrained_model_name)

        # Get Wav2Vec2 output dimension
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size

        # Feature encoder selection
        if use_transformer:
            self.feature_encoder = TransformerEncoder(
                wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
            )
        else:
            self.feature_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)

        # Output heads for three tasks
        self.canonical_head = CanonicalHead(hidden_dim, num_phonemes, dropout)
        self.perceived_head = PerceivedHead(hidden_dim, num_phonemes, dropout)
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)

    def forward(self, waveform, attention_mask=None, training_mode='phoneme_only'):
        """Forward pass through the model.
        
        Args:
            waveform: Input audio waveform [batch_size, sequence_length].
            attention_mask: Attention mask for padding [batch_size, sequence_length].
            training_mode: Training mode selection.
            
        Returns:
            Dictionary containing task-specific logits based on training mode.
        """
        # Extract Wav2Vec2 features
        features = self.encoder(waveform, attention_mask)

        # Enhance features
        if hasattr(self.feature_encoder, 'transformer'):
            enhanced_features = self.feature_encoder(features, attention_mask)
        else:
            enhanced_features = self.feature_encoder(features)

        # Compute outputs based on training mode
        outputs = {}
        
        if training_mode == 'phoneme_only':
            # Only perceived phonemes (backward compatibility)
            outputs['phoneme_logits'] = self.perceived_head(enhanced_features)
            
        elif training_mode == 'phoneme_error':
            # Perceived phonemes and error detection
            outputs['phoneme_logits'] = self.perceived_head(enhanced_features)
            outputs['error_logits'] = self.error_head(enhanced_features)
            
        elif training_mode == 'multitask':
            # All three tasks
            outputs['canonical_logits'] = self.canonical_head(enhanced_features)
            outputs['perceived_logits'] = self.perceived_head(enhanced_features)
            outputs['error_logits'] = self.error_head(enhanced_features)

        return outputs
