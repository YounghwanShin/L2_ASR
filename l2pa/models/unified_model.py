"""Unified model architecture for pronunciation assessment.

This module implements the main model combining Wav2Vec2 encoder,
feature processing, and task-specific output heads.
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import PhonemeHead, ErrorDetectionHead


class UnifiedModel(nn.Module):
  """Unified model for multitask pronunciation assessment.
  
  The model consists of:
    1. Wav2Vec2 encoder for audio feature extraction
    2. Feature encoder (Simple or Transformer) for enhancement
    3. Task-specific heads: canonical, perceived, and error
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
    """Initialize unified model.
    
    Args:
      pretrained_model_name: Pretrained Wav2Vec2 model name.
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

    # Feature encoder
    if use_transformer:
      self.feature_encoder = TransformerEncoder(
          wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
      )
    else:
      self.feature_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)

    # Output heads
    self.canonical_head = PhonemeHead(hidden_dim, num_phonemes, dropout)
    self.perceived_head = PhonemeHead(hidden_dim, num_phonemes, dropout)
    self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)

  def forward(self, waveform, attention_mask=None, training_mode='multitask'):
    """Forward pass through the model.
    
    Args:
      waveform: Input audio [batch_size, seq_len].
      attention_mask: Attention mask [batch_size, seq_len].
      training_mode: Training mode determining which heads to use.
      
    Returns:
      Dictionary containing logits from active heads.
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
    
    if training_mode in ['phoneme_only', 'phoneme_error', 'multitask']:
      outputs['perceived_logits'] = self.perceived_head(enhanced_features)
    
    if training_mode in ['multitask']:
      outputs['canonical_logits'] = self.canonical_head(enhanced_features)
    
    if training_mode in ['phoneme_error', 'multitask']:
      outputs['error_logits'] = self.error_head(enhanced_features)

    return outputs
