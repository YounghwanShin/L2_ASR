"""Loss functions for pronunciation assessment model.

This module implements Focal CTC Loss for handling class imbalance and
a unified loss class for multi-task learning with up to three objectives.
"""

import torch
import torch.nn as nn


class FocalCTCLoss(nn.Module):
    """Focal Loss applied to CTC Loss for handling class imbalance.
    
    Focal Loss helps focus on hard-to-classify samples by down-weighting
    easy examples, particularly useful for pronunciation assessment where
    some classes are more frequent than others.
    
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
            log_probs: Log probabilities from model [T, N, C].
            targets: Target sequences [N, S].
            input_lengths: Length of each input sequence [N].
            target_lengths: Length of each target sequence [N].
            
        Returns:
            Scalar tensor containing the mean focal CTC loss.
        """
        # Compute standard CTC loss per sample
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        
        # Compute focal weight
        probability = torch.exp(-ctc_losses)
        probability = torch.clamp(probability, min=1e-6, max=1.0)
        focal_losses = ctc_losses * (self.alpha * (1 - probability) ** self.gamma)

        return focal_losses.mean()


class UnifiedLoss(nn.Module):
    """Unified loss function for multi-task pronunciation assessment.
    
    Combines canonical phoneme, perceived phoneme, and error detection
    losses with configurable weights. Uses Focal CTC Loss for all tasks.
    
    Attributes:
        training_mode: Current training mode.
        canonical_weight: Weight for canonical phoneme loss.
        perceived_weight: Weight for perceived phoneme loss.
        error_weight: Weight for error detection loss.
        canonical_criterion: Focal CTC loss for canonical prediction.
        perceived_criterion: Focal CTC loss for perceived prediction.
        error_criterion: Focal CTC loss for error detection.
    """
    
    def __init__(self, 
                 training_mode: str = 'phoneme_only', 
                 canonical_weight: float = 0.3,
                 perceived_weight: float = 0.3,
                 error_weight: float = 0.4, 
                 focal_alpha: float = 1.0, 
                 focal_gamma: float = 2.0):
        """Initializes the Unified Loss.
        
        Args:
            training_mode: Training mode selection.
            canonical_weight: Weight for canonical phoneme loss.
            perceived_weight: Weight for perceived phoneme loss.
            error_weight: Weight for error detection loss.
            focal_alpha: Alpha parameter for Focal Loss.
            focal_gamma: Gamma parameter for Focal Loss.
        """
        super().__init__()
        self.training_mode = training_mode
        self.canonical_weight = canonical_weight
        self.perceived_weight = perceived_weight
        self.error_weight = error_weight
        
        # Initialize Focal CTC Loss for all tasks
        self.canonical_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )
        self.perceived_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )
        self.error_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )

    def forward(self, outputs, 
                canonical_targets=None, canonical_input_lengths=None, canonical_target_lengths=None,
                perceived_targets=None, perceived_input_lengths=None, perceived_target_lengths=None,
                error_targets=None, error_input_lengths=None, error_target_lengths=None):
        """Computes the unified loss.
        
        Args:
            outputs: Dictionary containing model outputs.
            canonical_targets: Target canonical phoneme sequences.
            canonical_input_lengths: Input sequence lengths for canonical task.
            canonical_target_lengths: Target sequence lengths for canonical task.
            perceived_targets: Target perceived phoneme sequences.
            perceived_input_lengths: Input sequence lengths for perceived task.
            perceived_target_lengths: Target sequence lengths for perceived task.
            error_targets: Target error sequences.
            error_input_lengths: Input sequence lengths for error task.
            error_target_lengths: Target sequence lengths for error task.
            
        Returns:
            Tuple of (total_loss, loss_dict) where total_loss is the weighted sum
            and loss_dict contains breakdown of individual losses.
        """
        total_loss = 0.0
        loss_dict = {}

        # Canonical phoneme loss
        if self.training_mode == 'multitask' and 'canonical_logits' in outputs:
            if canonical_targets is not None and canonical_input_lengths is not None:
                canonical_log_probs = torch.log_softmax(outputs['canonical_logits'], dim=-1)
                canonical_loss = self.canonical_criterion(
                    canonical_log_probs.transpose(0, 1),
                    canonical_targets,
                    canonical_input_lengths,
                    canonical_target_lengths
                )
                weighted_canonical_loss = self.canonical_weight * canonical_loss
                total_loss += weighted_canonical_loss
                loss_dict['canonical_loss'] = canonical_loss.item()

        # Perceived phoneme loss
        if 'perceived_logits' in outputs:
            logits_key = 'perceived_logits'
        elif 'phoneme_logits' in outputs:
            # Backward compatibility
            logits_key = 'phoneme_logits'
        else:
            logits_key = None

        if logits_key and perceived_targets is not None and perceived_input_lengths is not None:
            perceived_log_probs = torch.log_softmax(outputs[logits_key], dim=-1)
            perceived_loss = self.perceived_criterion(
                perceived_log_probs.transpose(0, 1),
                perceived_targets,
                perceived_input_lengths,
                perceived_target_lengths
            )
            weighted_perceived_loss = self.perceived_weight * perceived_loss
            total_loss += weighted_perceived_loss
            loss_dict['perceived_loss'] = perceived_loss.item()
            # Backward compatibility
            loss_dict['phoneme_loss'] = perceived_loss.item()

        # Error detection loss
        if self.training_mode in ['phoneme_error', 'multitask'] and 'error_logits' in outputs:
            if error_targets is not None and error_input_lengths is not None:
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
