"""Encoder architectures for pronunciation assessment.

This module implements audio encoders including Wav2Vec2-based encoder,
simple feed-forward encoder, and Transformer encoder for feature enhancement.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
  """Wav2Vec2-based audio encoder for robust feature extraction.
  
  Uses pretrained Wav2Vec2 model to extract contextualized audio features.
  """
  
  def __init__(self, pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"):
    """Initializes Wav2Vec2 encoder.
    
    Args:
      pretrained_model_name: Name or path of pretrained Wav2Vec2 model.
    """
    super().__init__()
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    # Disable masking for inference stability
    config.mask_time_prob = 0.0
    config.mask_feature_prob = 0.0
    self.wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_model_name,
        config=config
    )

  def forward(self, waveform, attention_mask=None):
    """Encodes audio waveform to features.
    
    Args:
      waveform: Input audio tensor of shape [batch_size, seq_len].
      attention_mask: Optional attention mask of shape [batch_size, seq_len].
      
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
    return outputs.last_hidden_state


class SimpleEncoder(nn.Module):
  """Simple feed-forward encoder for feature transformation.
  
  Applies two-layer MLP with residual connection for feature enhancement.
  """
  
  def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
    """Initializes simple encoder.
    
    Args:
      input_dim: Input feature dimension.
      hidden_dim: Hidden layer dimension.
      dropout: Dropout probability.
    """
    super().__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, features):
    """Encodes features through feed-forward layers.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
      
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    hidden = F.relu(self.layer1(features))
    hidden = self.dropout(hidden)
    output = self.layer2(hidden)
    return self.layer_norm(output)


class TransformerEncoder(nn.Module):
  """Transformer-based encoder for contextual feature enhancement.
  
  Uses multi-head self-attention to capture long-range dependencies.
  """
  
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      num_layers: int = 2,
      num_heads: int = 8,
      dropout: float = 0.1
  ):
    """Initializes Transformer encoder.
    
    Args:
      input_dim: Input feature dimension.
      hidden_dim: Hidden dimension for Transformer.
      num_layers: Number of Transformer layers.
      num_heads: Number of attention heads.
      dropout: Dropout probability.
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
    """Encodes features through Transformer layers.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
      attention_mask: Optional attention mask of shape [batch_size, seq_len].
      
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    projected = self.input_projection(features)
    projected = self.dropout(projected)
    encoded = self.transformer(projected)
    return self.layer_norm(encoded)
