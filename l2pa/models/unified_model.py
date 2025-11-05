"""Unified model architecture for multitask pronunciation assessment.

Combines Wav2Vec2 audio encoding with task-specific output heads for:
  - Canonical phoneme recognition
  - Perceived phoneme recognition
  - Pronunciation error classification
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import PhonemeRecognitionHead, ErrorClassificationHead


class MultitaskPronunciationModel(nn.Module):
  """Multitask model for L2 pronunciation assessment.
  
  Architecture pipeline:
    1. Wav2Vec2 encoder extracts robust audio features
    2. Feature encoder enhances representations (Simple or Transformer)
    3. Task-specific heads produce predictions for each task
  
  The model supports three training modes:
    - phoneme_only: Only perceived phoneme recognition
    - phoneme_error: Perceived phonemes + error classification
    - multitask: All three tasks (canonical, perceived, errors)
  
  Attributes:
    wav2vec_encoder: Wav2Vec2-based audio feature extractor.
    feature_encoder: Feature enhancement layer (Simple or Transformer).
    canonical_head: Output head for canonical phoneme recognition.
    perceived_head: Output head for perceived phoneme recognition.
    error_head: Output head for error type classification.
  """
  
  def __init__(
      self,
      pretrained_model_name: str = 'facebook/wav2vec2-large-xlsr-53',
      hidden_dim: int = 1024,
      num_phonemes: int = 42,
      num_error_types: int = 5,
      dropout: float = 0.1,
      use_transformer: bool = True,
      num_transformer_layers: int = 2,
      num_attention_heads: int = 8
  ):
    """Initializes the multitask pronunciation assessment model.
    
    Args:
      pretrained_model_name: Identifier for pretrained Wav2Vec2 model.
      hidden_dim: Hidden dimension for feature encoder and task heads.
      num_phonemes: Number of phoneme classes (including blank).
      num_error_types: Number of error types (blank + D/I/S/C).
      dropout: Dropout probability for regularization.
      use_transformer: Whether to use Transformer encoder for features.
      num_transformer_layers: Number of Transformer encoder layers.
      num_attention_heads: Number of attention heads in Transformer.
    """
    super().__init__()
    
    # Wav2Vec2 audio encoder for robust feature extraction
    self.wav2vec_encoder = Wav2VecEncoder(pretrained_model_name)
    
    # Get Wav2Vec2 output dimension
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    wav2vec_output_dim = config.hidden_size
    
    # Feature encoder for enhancement
    if use_transformer:
      self.feature_encoder = TransformerEncoder(
          input_dim=wav2vec_output_dim,
          hidden_dim=hidden_dim,
          num_layers=num_transformer_layers,
          num_heads=num_attention_heads,
          dropout=dropout
      )
    else:
      self.feature_encoder = SimpleEncoder(
          input_dim=wav2vec_output_dim,
          hidden_dim=hidden_dim,
          dropout=dropout
      )
    
    # Task-specific output heads
    self.canonical_head = PhonemeRecognitionHead(
        input_dim=hidden_dim,
        num_phonemes=num_phonemes,
        dropout=dropout
    )
    self.perceived_head = PhonemeRecognitionHead(
        input_dim=hidden_dim,
        num_phonemes=num_phonemes,
        dropout=dropout
    )
    self.error_head = ErrorClassificationHead(
        input_dim=hidden_dim,
        num_error_types=num_error_types,
        dropout=dropout
    )
  
  def forward(self, waveforms, attention_mask=None, training_mode='multitask'):
    """Performs forward pass through the model.
    
    Args:
      waveforms: Input audio tensor of shape [batch_size, audio_length].
      attention_mask: Attention mask of shape [batch_size, audio_length].
      training_mode: Mode determining which task heads are active.
    
    Returns:
      Dictionary containing logits for active tasks. Keys are:
        - 'canonical_logits': Canonical phoneme predictions (multitask only)
        - 'perceived_logits': Perceived phoneme predictions (all modes)
        - 'error_logits': Error type predictions (phoneme_error, multitask)
    """
    # Extract audio features with Wav2Vec2
    audio_features = self.wav2vec_encoder(waveforms, attention_mask)
    
    # Enhance features with additional encoder
    enhanced_features = self.feature_encoder(audio_features, attention_mask)
    
    # Generate predictions from active task heads
    outputs = {}
    
    # Perceived phoneme recognition (active in all modes)
    if training_mode in ['phoneme_only', 'phoneme_error', 'multitask']:
      outputs['perceived_logits'] = self.perceived_head(enhanced_features)
    
    # Canonical phoneme recognition (only in multitask mode)
    if training_mode == 'multitask':
      outputs['canonical_logits'] = self.canonical_head(enhanced_features)
    
    # Error classification (in phoneme_error and multitask modes)
    if training_mode in ['phoneme_error', 'multitask']:
      outputs['error_logits'] = self.error_head(enhanced_features)
    
    return outputs
