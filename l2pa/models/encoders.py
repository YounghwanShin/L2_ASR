"""Encoder architectures for pronunciation assessment.

This module implements Wav2Vec2-based audio encoding and feature enhancement
layers including simple feedforward and Transformer encoders.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
  """Wav2Vec2-based audio encoder for robust feature extraction."""
  
  def __init__(self, pretrained_model_name: str = 'facebook/wav2vec2-large-xlsr-53'):
    """Initializes the Wav2Vec2 encoder.
    
    Args:
      pretrained_model_name: Name of pretrained Wav2Vec2 model.
    """
    super().__init__()
    
    # Load config and disable masking for finetuning
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    config.mask_time_prob = 0.0
    config.mask_feature_prob = 0.0
    
    self.wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_model_name,
        config=config
    )
  
  def forward(self, waveform, attention_mask=None):
    """Encodes audio waveform.
    
    Args:
      waveform: Input audio of shape [batch_size, seq_len].
      attention_mask: Attention mask of shape [batch_size, seq_len].
    
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
    return outputs.last_hidden_state


class SimpleEncoder(nn.Module):
  """Simple feedforward encoder for feature transformation."""
  
  def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
    """Initializes the simple encoder.
    
    Args:
      input_dim: Input feature dimension.
      hidden_dim: Hidden layer dimension.
      dropout: Dropout rate.
    """
    super().__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, features):
    """Transforms features with feedforward layers.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
    
    Returns:
      Transformed features of shape [batch_size, seq_len, hidden_dim].
    """
    hidden = F.relu(self.layer1(features))
    hidden = self.dropout(hidden)
    output = self.layer2(hidden)
    return self.layer_norm(output)


class TransformerEncoder(nn.Module):
  """Transformer-based encoder for contextual feature enhancement."""
  
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      num_layers: int = 2,
      num_heads: int = 8,
      dropout: float = 0.1
  ):
    """Initializes the Transformer encoder.
    
    Args:
      input_dim: Input feature dimension.
      hidden_dim: Hidden dimension for Transformer.
      num_layers: Number of Transformer layers.
      num_heads: Number of attention heads.
      dropout: Dropout rate.
    """
    super().__init__()
    
    self.input_projection = nn.Linear(input_dim, hidden_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
        activation='relu',
        batch_first=True
    )
    
    self.transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers
    )
    
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, features, attention_mask=None):
    """Encodes features with Transformer.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
      attention_mask: Optional attention mask.
    
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    # Project to hidden dimension
    projected = self.input_projection(features)
    projected = self.dropout(projected)
    
    # Apply Transformer
    encoded = self.transformer(projected)
    
    return self.layer_norm(encoded)
