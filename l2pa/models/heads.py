"""Output heads for pronunciation assessment tasks.

Provides task-specific classification heads for:
  - Phoneme recognition with CTC decoding
  - Error type classification (D/I/S/C)
"""

import torch.nn as nn


class PhonemeRecognitionHead(nn.Module):
  """Output head for phoneme recognition with CTC decoding.
  
  Applies linear projection to predict phoneme probabilities at each
  time step. Works with both canonical and perceived phoneme targets.
  
  Attributes:
    dropout: Dropout layer for regularization.
    classifier: Linear layer for phoneme classification.
  """
  
  def __init__(self, input_dim: int, num_phonemes: int, dropout: float = 0.1):
    """Initializes the phoneme recognition head.
    
    Args:
      input_dim: Dimension of input features.
      num_phonemes: Number of phoneme classes (including blank token).
      dropout: Dropout probability.
    """
    super().__init__()
    
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_phonemes)
  
  def forward(self, features):
    """Computes phoneme logits for each time step.
    
    Args:
      features: Input features of shape [batch_size, sequence_length, input_dim].
    
    Returns:
      Phoneme logits of shape [batch_size, sequence_length, num_phonemes].
    """
    features = self.dropout(features)
    return self.classifier(features)


class ErrorClassificationHead(nn.Module):
  """Output head for pronunciation error type classification.
  
  Classifies each time step into one of: Deletion (D), Insertion (I),
  Substitution (S), Correct (C), or blank.
  
  Attributes:
    dropout: Dropout layer for regularization.
    classifier: Linear layer for error type classification.
  """
  
  def __init__(
      self,
      input_dim: int,
      num_error_types: int = 5,
      dropout: float = 0.1
  ):
    """Initializes the error classification head.
    
    Args:
      input_dim: Dimension of input features.
      num_error_types: Number of error types (blank + D/I/S/C = 5).
      dropout: Dropout probability.
    """
    super().__init__()
    
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(input_dim, num_error_types)
  
  def forward(self, features):
    """Computes error type logits for each time step.
    
    Args:
      features: Input features of shape [batch_size, sequence_length, input_dim].
    
    Returns:
      Error logits of shape [batch_size, sequence_length, num_error_types].
    """
    features = self.dropout(features)
    return self.classifier(features)
