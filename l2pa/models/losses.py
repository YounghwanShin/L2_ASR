"""Loss functions for pronunciation assessment model.

This module implements Focal CTC Loss for handling class imbalance and
a unified loss class for multitask learning.
"""

import torch
import torch.nn as nn


class FocalCTCLoss(nn.Module):
    """Focal Loss applied to CTC Loss for handling class imbalance.
    
    Focal Loss helps the model focus on hard-to-classify samples by down-weighting
    easy examples. This is particularly useful for pronunciation assessment where
    some phoneme or error classes are much more frequent than others.
    
    Attributes:
        alpha: Weight factor for class balancing.
        gamma: Focusing parameter for hard examples.
        ctc_loss: Underlying CTC loss function.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, blank: int = 0, zero_infinity: bool = True):
        """Initializes the Focal CTC Loss.
        
        Args:
            alpha: Weight factor for class balancing.
            gamma: Focusing parameter (higher values focus more on hard examples).
            blank: Index of the CTC blank token.
            zero_infinity: Whether to set infinite losses to zero.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Computes the Focal CTC Loss.
        
        Args:
            log_probs: Log probabilities from the model [T, N, C] where T is
                sequence length, N is batch size, C is number of classes.
            targets: Target sequences [N, S] where S is target sequence length.
            input_lengths: Length of each input sequence [N].
            target_lengths: Length of each target sequence [N].
            
        Returns:
            Scalar tensor containing the mean focal CTC loss.
        """
        # Compute standard CTC loss per sample
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        
        # Compute focal weight: (1 - p_t)^gamma where p_t = exp(-loss)
        probability = torch.exp(-ctc_losses)
        probability = torch.clamp(probability, min=1e-6, max=1.0)
        focal_losses = ctc_losses * (self.alpha * (1 - probability) ** self.gamma)

        return focal_losses.mean()


class UnifiedLoss(nn.Module):
    """Unified loss function for multitask pronunciation assessment.
    
    Combines phoneme recognition and error detection losses with configurable
    weights. Uses Focal CTC Loss for both tasks to handle class imbalance.
    
    Attributes:
        training_mode: Current training mode ('phoneme_only' or 'phoneme_error').
        error_weight: Weight for error detection loss.
        phoneme_weight: Weight for phoneme recognition loss.
        error_criterion: Focal CTC loss for error detection.
        phoneme_criterion: Focal CTC loss for phoneme recognition.
    """
    
    def __init__(self, 
                 training_mode: str = 'phoneme_only', 
                 error_weight: float = 1.0, 
                 phoneme_weight: float = 1.0, 
                 focal_alpha: float = 1.0, 
                 focal_gamma: float = 2.0):
        """Initializes the Unified Loss.
        
        Args:
            training_mode: Training mode ('phoneme_only' or 'phoneme_error').
            error_weight: Weight for error detection loss.
            phoneme_weight: Weight for phoneme recognition loss.
            focal_alpha: Alpha parameter for Focal Loss.
            focal_gamma: Gamma parameter for Focal Loss.
        """
        super().__init__()
        self.training_mode = training_mode
        self.error_weight = error_weight
        self.phoneme_weight = phoneme_weight
        
        # Initialize Focal CTC Loss for both tasks
        self.error_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )
        self.phoneme_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )

    def forward(self, outputs, phoneme_targets, phoneme_input_lengths, phoneme_target_lengths,
                error_targets=None, error_input_lengths=None, error_target_lengths=None):
        """Computes the unified loss.
        
        Args:
            outputs: Dictionary containing model outputs with keys 'phoneme_logits'
                and optionally 'error_logits'.
            phoneme_targets: Target phoneme sequences.
            phoneme_input_lengths: Length of each input sequence for phoneme task.
            phoneme_target_lengths: Length of each target phoneme sequence.
            error_targets: Target error sequences (optional, for phoneme_error mode).
            error_input_lengths: Length of each input sequence for error task (optional).
            error_target_lengths: Length of each target error sequence (optional).
            
        Returns:
            Tuple of (total_loss, loss_dict) where total_loss is the weighted sum of
            individual losses and loss_dict contains breakdown of loss components.
        """
        total_loss = 0.0
        loss_dict = {}

        # Compute phoneme recognition loss
        phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
        phoneme_loss = self.phoneme_criterion(
            phoneme_log_probs.transpose(0, 1),
            phoneme_targets,
            phoneme_input_lengths,
            phoneme_target_lengths
        )
        weighted_phoneme_loss = self.phoneme_weight * phoneme_loss
        total_loss += weighted_phoneme_loss
        loss_dict['phoneme_loss'] = phoneme_loss.item()

        # Compute error detection loss if in phoneme_error mode
        if self.training_mode == 'phoneme_error' and 'error_logits' in outputs:
            if error_targets is not None and error_input_lengths is not None and error_target_lengths is not None:
                error_log_probs = torch.log_softmax(outputs['error_logits'], dim=-1)
                error_loss = self.error_criterion(
                    error_log_probs.transpose(0, 1),
                    error_targets,
                    error_input_lengths,
                    error_target_lengths
                )
                weighted_error_loss = self.error_weight * error_loss
                total_loss += weighted_error_loss
                loss_dict['error_loss'] = error_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict