"""Loss functions for pronunciation assessment model.

This module implements Focal CTC Loss and a unified loss class for
multitask learning with canonical, perceived, and error components.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class FocalCTCLoss(nn.Module):
  """Focal Loss applied to CTC Loss for handling class imbalance."""
  
  def __init__(self,
               alpha: float = 1.0,
               gamma: float = 2.0,
               blank: int = 0,
               zero_infinity: bool = True):
    """Initialize Focal CTC Loss.
    
    Args:
      alpha: Weight factor for class balancing.
      gamma: Focusing parameter for hard examples.
      blank: Index of the CTC blank token.
      zero_infinity: Whether to set infinite losses to zero.
    """
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none',
                                zero_infinity=zero_infinity)

  def forward(self,
              log_probs: torch.Tensor,
              targets: torch.Tensor,
              input_lengths: torch.Tensor,
              target_lengths: torch.Tensor) -> torch.Tensor:
    """Compute Focal CTC Loss.
    
    Args:
      log_probs: Log probabilities [T, N, C].
      targets: Target sequences [N, S].
      input_lengths: Input sequence lengths [N].
      target_lengths: Target sequence lengths [N].
      
    Returns:
      Scalar loss tensor.
    """
    # Compute CTC loss per sample
    ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    ctc_losses = torch.clamp(ctc_losses, min=1e-6)
    
    # Apply focal weight
    probability = torch.exp(-ctc_losses)
    probability = torch.clamp(probability, min=1e-6, max=1.0)
    focal_losses = ctc_losses * (self.alpha * (1 - probability) ** self.gamma)

    return focal_losses.mean()


class UnifiedLoss(nn.Module):
  """Unified loss function for multitask pronunciation assessment.
  
  Combines canonical phoneme, perceived phoneme, and error detection
  losses with configurable weights.
  """
  
  def __init__(self,
               training_mode: str = 'multitask',
               canonical_weight: float = 0.3,
               perceived_weight: float = 0.3,
               error_weight: float = 0.4,
               focal_alpha: float = 1.0,
               focal_gamma: float = 2.0):
    """Initialize unified loss.
    
    Args:
      training_mode: Training mode determining active losses.
      canonical_weight: Weight for canonical phoneme loss.
      perceived_weight: Weight for perceived phoneme loss.
      error_weight: Weight for error detection loss.
      focal_alpha: Focal loss alpha parameter.
      focal_gamma: Focal loss gamma parameter.
    """
    super().__init__()
    self.training_mode = training_mode
    self.canonical_weight = canonical_weight
    self.perceived_weight = perceived_weight
    self.error_weight = error_weight
    
    # Initialize loss functions
    self.canonical_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)
    self.perceived_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)
    self.error_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)

  def forward(self,
              outputs: Dict[str, torch.Tensor],
              canonical_targets: Optional[torch.Tensor] = None,
              canonical_input_lengths: Optional[torch.Tensor] = None,
              canonical_target_lengths: Optional[torch.Tensor] = None,
              perceived_targets: Optional[torch.Tensor] = None,
              perceived_input_lengths: Optional[torch.Tensor] = None,
              perceived_target_lengths: Optional[torch.Tensor] = None,
              error_targets: Optional[torch.Tensor] = None,
              error_input_lengths: Optional[torch.Tensor] = None,
              error_target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute unified loss.
    
    Args:
      outputs: Model outputs with logits.
      canonical_targets: Canonical phoneme targets.
      canonical_input_lengths: Canonical input lengths.
      canonical_target_lengths: Canonical target lengths.
      perceived_targets: Perceived phoneme targets.
      perceived_input_lengths: Perceived input lengths.
      perceived_target_lengths: Perceived target lengths.
      error_targets: Error detection targets.
      error_input_lengths: Error input lengths.
      error_target_lengths: Error target lengths.
      
    Returns:
      Tuple of (total_loss, loss_dict).
    """
    total_loss = 0.0
    loss_dict = {}

    # Canonical phoneme loss
    if self.training_mode in ['multitask'] and 'canonical_logits' in outputs:
      if canonical_targets is not None:
        log_probs = torch.log_softmax(outputs['canonical_logits'], dim=-1)
        canonical_loss = self.canonical_criterion(
            log_probs.transpose(0, 1),
            canonical_targets,
            canonical_input_lengths,
            canonical_target_lengths
        )
        total_loss += self.canonical_weight * canonical_loss
        loss_dict['canonical_loss'] = canonical_loss.item()

    # Perceived phoneme loss
    if 'perceived_logits' in outputs:
      if perceived_targets is not None:
        log_probs = torch.log_softmax(outputs['perceived_logits'], dim=-1)
        perceived_loss = self.perceived_criterion(
            log_probs.transpose(0, 1),
            perceived_targets,
            perceived_input_lengths,
            perceived_target_lengths
        )
        total_loss += self.perceived_weight * perceived_loss
        loss_dict['perceived_loss'] = perceived_loss.item()

    # Error detection loss
    if self.training_mode in ['phoneme_error', 'multitask'] and 'error_logits' in outputs:
      if error_targets is not None:
        log_probs = torch.log_softmax(outputs['error_logits'], dim=-1)
        error_loss = self.error_criterion(
            log_probs.transpose(0, 1),
            error_targets,
            error_input_lengths,
            error_target_lengths
        )
        total_loss += self.error_weight * error_loss
        loss_dict['error_loss'] = error_loss.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict
