"""Unified model architecture for pronunciation assessment.

This module implements the main model combining Wav2Vec2 encoder,
feature processing, and task-specific output heads for phoneme recognition
and error detection.
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import ErrorDetectionHead, PhonemeRecognitionHead


class UnifiedModel(nn.Module):
    """Unified model for phoneme recognition and error detection.
    
    The model consists of three main components:
    1. Wav2Vec2 encoder for audio feature extraction
    2. Feature encoder (Simple or Transformer) for feature enhancement
    3. Task-specific output heads for phoneme and error prediction
    
    Attributes:
        encoder: Wav2Vec2 encoder for audio processing.
        feature_encoder: Additional encoding layer (Simple or Transformer).
        error_head: Output head for error detection.
        phoneme_head: Output head for phoneme recognition.
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
            pretrained_model_name: Name of pretrained Wav2Vec2 model.
            hidden_dim: Hidden dimension for feature encoder.
            num_phonemes: Number of phoneme classes.
            num_error_types: Number of error types (blank, D, I, S, C).
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

        # Output heads
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)

    def forward(self, x, attention_mask=None, training_mode='phoneme_only'):
        """Forward pass through the model.
        
        Args:
            x: Input audio waveform [batch_size, seq_len].
            attention_mask: Attention mask for padding [batch_size, seq_len].
            training_mode: Training mode ('phoneme_only' or 'phoneme_error').
            
        Returns:
            Dictionary containing 'phoneme_logits' and optionally 'error_logits'.
        """
        # Extract Wav2Vec2 features
        features = self.encoder(x, attention_mask)

        # Enhance features
        if hasattr(self.feature_encoder, 'transformer'):
            enhanced_features = self.feature_encoder(features, attention_mask)
        else:
            enhanced_features = self.feature_encoder(features)

        # Compute outputs
        outputs = {}
        outputs['phoneme_logits'] = self.phoneme_head(enhanced_features)

        if training_mode == 'phoneme_error':
            outputs['error_logits'] = self.error_head(enhanced_features)

        return outputs
