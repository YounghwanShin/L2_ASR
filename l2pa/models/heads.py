"""Output heads for pronunciation assessment tasks.

This module implements task-specific output heads for canonical phoneme
recognition, perceived phoneme recognition, and error detection.
"""

import torch.nn as nn


class PhonemeHead(nn.Module):
  """Output head for phoneme recognition.
  
  Classifies each time step into phoneme classes for CTC decoding.
  """
  
  def __init__(self, input_dim: int, num_phonemes: int, dropout: float = 0.1):
    """Initialize phoneme recognition head.
    
    Args:
      input_dim: Input feature dimension.
      num_phonemes: Number of phoneme classes.
      dropout: Dropout rate.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_phonemes)

  def forward(self, features):
    """Compute phoneme logits.
    
    Args:
      features: Input features [batch_size, seq_len, input_dim].
      
    Returns:
      Phoneme logits [batch_size, seq_len, num_phonemes].
    """
    return self.classifier(self.dropout(features))


class ErrorDetectionHead(nn.Module):
  """Output head for error detection.
  
  Classifies each time step into error types: blank, deletion (D),
  insertion (I), substitution (S), or correct (C).
  """
  
  def __init__(self, input_dim: int, num_error_types: int = 5, dropout: float = 0.1):
    """Initialize error detection head.
    
    Args:
      input_dim: Input feature dimension.
      num_error_types: Number of error types (blank, D, I, S, C).
      dropout: Dropout rate.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_error_types)

  def forward(self, features):
    """Compute error detection logits.
    
    Args:
      features: Input features [batch_size, seq_len, input_dim].
      
    Returns:
      Error logits [batch_size, seq_len, num_error_types].
    """
    return self.classifier(self.dropout(features))
