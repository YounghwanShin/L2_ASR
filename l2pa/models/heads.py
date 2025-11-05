"""Output heads for pronunciation assessment tasks.

This module implements task-specific classification heads for phoneme
recognition and error detection.
"""

import torch.nn as nn


class PhonemeHead(nn.Module):
  """Output head for phoneme recognition with CTC decoding."""
  
  def __init__(self, input_dim: int, num_phonemes: int, dropout: float = 0.1):
    """Initializes the phoneme recognition head.
    
    Args:
      input_dim: Input feature dimension.
      num_phonemes: Number of phoneme classes (including blank).
      dropout: Dropout rate.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_phonemes)
  
  def forward(self, features):
    """Computes phoneme logits.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
    
    Returns:
      Phoneme logits of shape [batch_size, seq_len, num_phonemes].
    """
    return self.classifier(self.dropout(features))


class ErrorDetectionHead(nn.Module):
  """Output head for error type classification."""
  
  def __init__(
      self,
      input_dim: int,
      num_error_types: int = 5,
      dropout: float = 0.1
  ):
    """Initializes the error detection head.
    
    Args:
      input_dim: Input feature dimension.
      num_error_types: Number of error types (blank, D, I, S, C).
      dropout: Dropout rate.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_error_types)
  
  def forward(self, features):
    """Computes error type logits.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
    
    Returns:
      Error logits of shape [batch_size, seq_len, num_error_types].
    """
    return self.classifier(self.dropout(features))
