"""Output heads for pronunciation assessment model.

This module implements task-specific output heads for error detection
and phoneme recognition.
"""

import torch.nn as nn


class ErrorDetectionHead(nn.Module):
    """Output head for error detection.
    
    Classifies each time step into error types: blank, deletion (D),
    insertion (I), substitution (S), or correct (C).
    
    Attributes:
        linear: Linear layer for classification.
        dropout: Dropout layer.
    """
    
    def __init__(self, input_dim: int, num_error_types: int = 5, dropout: float = 0.1):
        """Initializes the error detection head.
        
        Args:
            input_dim: Input feature dimension.
            num_error_types: Number of error types (blank, D, I, S, C).
            dropout: Dropout rate.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_error_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Computes error detection logits.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim].
            
        Returns:
            Error logits [batch_size, seq_len, num_error_types].
        """
        x = self.dropout(x)
        return self.linear(x)


class PhonemeRecognitionHead(nn.Module):
    """Output head for phoneme recognition.
    
    Classifies each time step into phoneme classes for CTC decoding.
    
    Attributes:
        linear: Linear layer for classification.
        dropout: Dropout layer.
    """
    
    def __init__(self, input_dim: int, num_phonemes: int = 42, dropout: float = 0.1):
        """Initializes the phoneme recognition head.
        
        Args:
            input_dim: Input feature dimension.
            num_phonemes: Number of phoneme classes.
            dropout: Dropout rate.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Computes phoneme recognition logits.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim].
            
        Returns:
            Phoneme logits [batch_size, seq_len, num_phonemes].
        """
        x = self.dropout(x)
        return self.linear(x)
