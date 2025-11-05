"""Output heads for pronunciation assessment tasks.

This module implements task-specific output heads for phoneme recognition
and error detection.
"""

import torch.nn as nn


class PhonemeHead(nn.Module):
  """Output head for phoneme recognition.
  
  Classifies each time step into phoneme classes using CTC decoding.
  """
  
  def __init__(self, input_dim: int, num_phonemes: int, dropout: float = 0.1):
    """Initializes phoneme recognition head.
    
    Args:
      input_dim: Input feature dimension.
      num_phonemes: Number of phoneme classes.
      dropout: Dropout probability.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_phonemes)

  def forward(self, features):
    """Computes phoneme logits for each time step.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
      
    Returns:
      Phoneme logits of shape [batch_size, seq_len, num_phonemes].
    """
    return self.classifier(self.dropout(features))


class ErrorDetectionHead(nn.Module):
  """Output head for error detection.
  
  Classifies each time step into error types: blank, deletion (D),
  insertion (I), substitution (S), or correct (C).
  """
  
  def __init__(
      self, 
      input_dim: int, 
      num_error_types: int = 5, 
      dropout: float = 0.1
  ):
    """Initializes error detection head.
    
    Args:
      input_dim: Input feature dimension.
      num_error_types: Number of error types (blank, D, I, S, C).
      dropout: Dropout probability.
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_error_types)

  def forward(self, features):
    """Computes error detection logits for each time step.
    
    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].
      
    Returns:
      Error logits of shape [batch_size, seq_len, num_error_types].
    """
    return self.classifier(self.dropout(features))
