"""Encoder architectures for pronunciation assessment.

Provides audio encoding and feature enhancement layers:
  - Wav2Vec2-based audio encoder for robust feature extraction
  - Simple feedforward encoder for efficient feature transformation
  - Transformer encoder for contextual feature modeling
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
  """Wav2Vec2-based audio encoder for robust feature extraction.
  
  Wraps a pretrained Wav2Vec2 model and disables masking for fine-tuning.
  Extracts contextualized audio representations from raw waveforms.
  
  Attributes:
    wav2vec2: Pretrained Wav2Vec2 model instance.
  """
  
  def __init__(self, pretrained_model_name: str = 'facebook/wav2vec2-large-xlsr-53'):
    """Initializes the Wav2Vec2 encoder.
    
    Args:
      pretrained_model_name: HuggingFace model identifier for Wav2Vec2.
    """
    super().__init__()
    
    # Load configuration and disable masking for supervised fine-tuning
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    config.mask_time_prob = 0.0
    config.mask_feature_prob = 0.0
    
    self.wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_model_name,
        config=config
    )
  
  def forward(self, waveforms, attention_mask=None):
    """Encodes audio waveforms into feature representations.
    
    Args:
      waveforms: Input audio tensor of shape [batch_size, audio_length].
      attention_mask: Optional attention mask of shape [batch_size, audio_length].
    
    Returns:
      Encoded audio features of shape [batch_size, sequence_length, hidden_dim].
    """
    outputs = self.wav2vec2(waveforms, attention_mask=attention_mask)
    return outputs.last_hidden_state


class SimpleEncoder(nn.Module):
  """Simple feedforward encoder for feature transformation.
  
  Applies two-layer feedforward network with ReLU activation and
  layer normalization for efficient feature enhancement.
  
  Attributes:
    projection_layer: First linear transformation layer.
    output_layer: Second linear transformation layer.
    layer_norm: Layer normalization for stable training.
    dropout: Dropout layer for regularization.
  """
  
  def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
    """Initializes the simple encoder.
    
    Args:
      input_dim: Dimension of input features.
      hidden_dim: Dimension of hidden and output features.
      dropout: Dropout probability.
    """
    super().__init__()
    
    self.projection_layer = nn.Linear(input_dim, hidden_dim)
    self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, features, attention_mask=None):
    """Transforms features through feedforward network.
    
    Args:
      features: Input features of shape [batch_size, sequence_length, input_dim].
      attention_mask: Optional attention mask (not used in simple encoder).
    
    Returns:
      Transformed features of shape [batch_size, sequence_length, hidden_dim].
    """
    # First projection with ReLU activation
    hidden = F.relu(self.projection_layer(features))
    hidden = self.dropout(hidden)
    
    # Second projection
    output = self.output_layer(hidden)
    
    # Layer normalization for stable training
    return self.layer_norm(output)


class TransformerEncoder(nn.Module):
  """Transformer-based encoder for contextual feature modeling.
  
  Uses multi-head self-attention to model long-range dependencies in
  audio features, enhancing the representations for downstream tasks.
  
  Attributes:
    input_projection: Projects input features to hidden dimension.
    transformer_encoder: PyTorch Transformer encoder layers.
    layer_norm: Final layer normalization.
    dropout: Dropout for regularization.
  """
  
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
      input_dim: Dimension of input features.
      hidden_dim: Hidden dimension for Transformer layers.
      num_layers: Number of Transformer encoder layers.
      num_heads: Number of attention heads per layer.
      dropout: Dropout probability.
    """
    super().__init__()
    
    # Project input to hidden dimension
    self.input_projection = nn.Linear(input_dim, hidden_dim)
    
    # Transformer encoder layers
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
        activation='relu',
        batch_first=True  # Use batch-first format
    )
    
    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers
    )
    
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, features, attention_mask=None):
    """Encodes features with Transformer self-attention.
    
    Args:
      features: Input features of shape [batch_size, sequence_length, input_dim].
      attention_mask: Optional attention mask (not used in current implementation).
    
    Returns:
      Encoded features of shape [batch_size, sequence_length, hidden_dim].
    """
    # Project to hidden dimension
    projected = self.input_projection(features)
    projected = self.dropout(projected)
    
    # Apply Transformer layers
    encoded = self.transformer_encoder(projected)
    
    # Final normalization
    return self.layer_norm(encoded)
