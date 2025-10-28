"""Encoder architectures for pronunciation assessment.

This module implements encoder architectures including Wav2Vec2-based audio
encoder, simple feed-forward encoder, and Transformer-based encoder.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
    """Wav2Vec2-based audio encoder.
    
    Uses pretrained Wav2Vec2 model for robust audio feature extraction.
    Masking is disabled to provide consistent features during training and inference.
    
    Attributes:
        wav2vec2: Pretrained Wav2Vec2 model.
    """
    
    def __init__(self, pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"):
        """Initializes the Wav2Vec2 encoder.
        
        Args:
            pretrained_model_name: Name of pretrained Wav2Vec2 model to use.
        """
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)

    def forward(self, waveform, attention_mask=None):
        """Performs Wav2Vec2 encoding.
        
        Args:
            waveform: Input audio waveform [batch_size, sequence_length].
            attention_mask: Attention mask for padding [batch_size, sequence_length].
            
        Returns:
            Encoded features [batch_size, sequence_length, hidden_dim].
        """
        outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
        return outputs.last_hidden_state


class SimpleEncoder(nn.Module):
    """Simple feed-forward encoder.
    
    Uses two linear layers with ReLU activation and layer normalization
    for feature transformation.
    
    Attributes:
        layer1: First linear transformation.
        layer2: Second linear transformation.
        layer_norm: Layer normalization.
        dropout: Dropout layer.
    """
    
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
        """Performs simple encoding.
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim].
            
        Returns:
            Encoded features [batch_size, sequence_length, hidden_dim].
        """
        hidden = F.relu(self.layer1(features))
        hidden = self.dropout(hidden)
        output = self.layer2(hidden)
        return self.layer_norm(output)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder.
    
    Uses multi-head self-attention and feed-forward layers for
    contextual feature enhancement.
    
    Attributes:
        input_projection: Projects input to hidden dimension.
        transformer: Transformer encoder layers.
        layer_norm: Layer normalization.
        dropout: Dropout layer.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 2, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, attention_mask=None):
        """Performs Transformer encoding.
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim].
            attention_mask: Attention mask for padding [batch_size, sequence_length].
            
        Returns:
            Encoded features [batch_size, sequence_length, hidden_dim].
        """
        projected = self.input_projection(features)
        projected = self.dropout(projected)
        encoded = self.transformer(projected)
        return self.layer_norm(encoded)