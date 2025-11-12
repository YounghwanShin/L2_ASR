"""Unified model architecture for pronunciation assessment.

This module implements the main model combining Wav2Vec2 encoder,
feature processing, and task-specific output heads with automatic
architecture adaptation.
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import PhonemeHead, ErrorDetectionHead


class UnifiedModel(nn.Module):
  """Unified model for multitask pronunciation assessment.
  
  Architecture:
    1. Wav2Vec2 encoder: Extracts audio features
    2. Feature encoder: Enhances features (Simple or Transformer)
    3. Task-specific heads: Canonical, perceived, error detection
  
  The model automatically adapts its dimensions based on the pretrained
  Wav2Vec2 model configuration.
  """
  
  def __init__(
      self,
      pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
      hidden_dim: int = 1024,
      num_phonemes: int = 42,
      num_error_types: int = 5,
      dropout: float = 0.1,
      use_transformer: bool = False,
      num_layers: int = 2,
      num_heads: int = 8
  ):
    """Initializes unified model.
    
    Args:
      pretrained_model_name: Pretrained Wav2Vec2 model name.
      hidden_dim: Hidden dimension for feature encoder.
      num_phonemes: Number of phoneme classes.
      num_error_types: Number of error types (blank, D, I, S, C).
      dropout: Dropout probability.
      use_transformer: Whether to use Transformer encoder.
      num_layers: Number of Transformer layers (if applicable).
      num_heads: Number of attention heads (if applicable).
    """
    super().__init__()

    # Wav2Vec2 audio encoder
    self.encoder = Wav2VecEncoder(pretrained_model_name)

    # Get Wav2Vec2 output dimension from configuration
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    wav2vec_dim = config.hidden_size

    # Feature encoder (Simple or Transformer)
    if use_transformer:
      self.feature_encoder = TransformerEncoder(
          wav2vec_dim, 
          hidden_dim, 
          num_layers, 
          num_heads, 
          dropout
      )
    else:
      self.feature_encoder = SimpleEncoder(
          wav2vec_dim, 
          hidden_dim, 
          dropout
      )

    # Task-specific output heads
    self.canonical_head = PhonemeHead(
        hidden_dim, 
        num_phonemes, 
        dropout
    )
    self.perceived_head = PhonemeHead(
        hidden_dim, 
        num_phonemes, 
        dropout
    )
    self.error_head = ErrorDetectionHead(
        hidden_dim, 
        num_error_types, 
        dropout
    )

  def forward(
      self, 
      waveform, 
      attention_mask=None, 
      training_mode='multitask'
  ):
    """Forward pass through the model.
    
    Args:
      waveform: Input audio of shape [batch_size, seq_len].
      attention_mask: Attention mask of shape [batch_size, seq_len].
      training_mode: Training mode determining active heads.
      
    Returns:
      Dictionary containing logits from active heads.
    """
    # Extract audio features
    features = self.encoder(waveform, attention_mask)

    # Enhance features
    if hasattr(self.feature_encoder, 'transformer'):
      enhanced_features = self.feature_encoder(features, attention_mask)
    else:
      enhanced_features = self.feature_encoder(features)

    # Compute outputs based on training mode
    outputs = {}
    
    if training_mode in ['phoneme_only', 'phoneme_error', 'multitask']:
      outputs['perceived_logits'] = self.perceived_head(enhanced_features)
    
    if training_mode == 'multitask':
      outputs['canonical_logits'] = self.canonical_head(enhanced_features)
    
    if training_mode in ['phoneme_error', 'multitask']:
      outputs['error_logits'] = self.error_head(enhanced_features)

    return outputs