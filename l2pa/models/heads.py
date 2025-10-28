"""Output heads for pronunciation assessment tasks.

This module implements task-specific output heads for canonical phoneme
prediction, perceived phoneme prediction, and error detection.
"""

import torch.nn as nn


class CanonicalHead(nn.Module):
    """Output head for canonical phoneme prediction.
    
    Predicts the correct canonical phoneme sequence that should be produced
    for the given text, classified at each time step for CTC decoding.
    
    Attributes:
        classifier: Linear layer for classification.
        dropout: Dropout layer for regularization.
    """
    
    def __init__(self, input_dim: int, num_phonemes: int = 42, dropout: float = 0.1):
        """Initializes the canonical phoneme head.
        
        Args:
            input_dim: Input feature dimension.
            num_phonemes: Number of phoneme classes.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """Computes canonical phoneme logits.
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim].
            
        Returns:
            Canonical phoneme logits [batch_size, sequence_length, num_phonemes].
        """
        features = self.dropout(features)
        return self.classifier(features)


class PerceivedHead(nn.Module):
    """Output head for perceived phoneme prediction.
    
    Predicts the actual phoneme sequence produced by the learner,
    classified at each time step for CTC decoding.
    
    Attributes:
        classifier: Linear layer for classification.
        dropout: Dropout layer for regularization.
    """
    
    def __init__(self, input_dim: int, num_phonemes: int = 42, dropout: float = 0.1):
        """Initializes the perceived phoneme head.
        
        Args:
            input_dim: Input feature dimension.
            num_phonemes: Number of phoneme classes.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """Computes perceived phoneme logits.
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim].
            
        Returns:
            Perceived phoneme logits [batch_size, sequence_length, num_phonemes].
        """
        features = self.dropout(features)
        return self.classifier(features)


class ErrorDetectionHead(nn.Module):
    """Output head for error detection.
    
    Classifies each time step into error types: blank, deletion (D),
    insertion (I), substitution (S), or correct (C).
    
    Attributes:
        classifier: Linear layer for classification.
        dropout: Dropout layer for regularization.
    """
    
    def __init__(self, input_dim: int, num_error_types: int = 5, dropout: float = 0.1):
        """Initializes the error detection head.
        
        Args:
            input_dim: Input feature dimension.
            num_error_types: Number of error types (blank, D, I, S, C).
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_error_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """Computes error detection logits.
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim].
            
        Returns:
            Error logits [batch_size, sequence_length, num_error_types].
        """
        features = self.dropout(features)
        return self.classifier(features)


# Backward compatibility alias
PhonemeRecognitionHead = PerceivedHead
