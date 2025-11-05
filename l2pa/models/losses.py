"""Loss functions for multitask pronunciation assessment.

Implements specialized loss functions for training:
  - Focal CTC Loss for handling class imbalance
  - Unified multitask loss combining all learning objectives
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class FocalCTCLoss(nn.Module):
  """Focal Loss variant for CTC to handle class imbalance.
  
  Applies focal loss weighting to CTC loss, down-weighting easy examples
  and focusing on hard examples. Particularly useful for pronunciation
  assessment where blank and correct tokens are much more frequent.
  
  Attributes:
    alpha: Balancing factor for positive/negative samples.
    gamma: Focusing parameter (higher = more focus on hard examples).
    ctc_loss: Base CTC loss function.
  """
  
  def __init__(
      self,
      alpha: float = 1.0,
      gamma: float = 2.0,
      blank_id: int = 0,
      zero_infinity: bool = True
  ):
    """Initializes Focal CTC Loss.
    
    Args:
      alpha: Weighting factor for class balancing.
      gamma: Focusing parameter (typically 2.0).
      blank_id: Index of the CTC blank token.
      zero_infinity: Whether to zero out infinite losses.
    """
    super().__init__()
    
    self.alpha = alpha
    self.gamma = gamma
    self.ctc_loss = nn.CTCLoss(
        blank=blank_id,
        reduction='none',  # Per-sample loss for focal weighting
        zero_infinity=zero_infinity
    )
  
  def forward(
      self,
      log_probs: torch.Tensor,
      targets: torch.Tensor,
      input_lengths: torch.Tensor,
      target_lengths: torch.Tensor
  ) -> torch.Tensor:
    """Computes Focal CTC Loss.
    
    Args:
      log_probs: Log probabilities of shape [sequence_length, batch_size, num_classes].
      targets: Target sequences of shape [batch_size, target_length].
      input_lengths: Valid input lengths of shape [batch_size].
      target_lengths: Valid target lengths of shape [batch_size].
    
    Returns:
      Scalar loss tensor.
    """
    # Compute per-sample CTC losses
    ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    ctc_losses = torch.clamp(ctc_losses, min=1e-6)
    
    # Compute focal weights: higher weight for harder examples
    probability = torch.exp(-ctc_losses)
    probability = torch.clamp(probability, min=1e-6, max=1.0)
    focal_weight = self.alpha * (1 - probability) ** self.gamma
    
    # Apply focal weighting and average
    focal_losses = ctc_losses * focal_weight
    return focal_losses.mean()


class UnifiedMultitaskLoss(nn.Module):
  """Unified loss function for multitask pronunciation assessment.
  
  Combines losses from multiple tasks with configurable weights:
    - Canonical phoneme recognition (ground truth)
    - Perceived phoneme recognition (actual pronunciation)
    - Error type classification (D/I/S/C)
  
  Attributes:
    training_mode: Mode determining which tasks are active.
    canonical_weight: Weight for canonical phoneme loss.
    perceived_weight: Weight for perceived phoneme loss.
    error_weight: Weight for error classification loss.
    canonical_criterion: Loss function for canonical phonemes.
    perceived_criterion: Loss function for perceived phonemes.
    error_criterion: Loss function for error classification.
  """
  
  def __init__(
      self,
      training_mode: str = 'multitask',
      canonical_weight: float = 0.3,
      perceived_weight: float = 0.3,
      error_weight: float = 0.4,
      focal_alpha: float = 1.0,
      focal_gamma: float = 2.0
  ):
    """Initializes unified multitask loss.
    
    Args:
      training_mode: Training mode determining active losses.
      canonical_weight: Weight for canonical phoneme loss.
      perceived_weight: Weight for perceived phoneme loss.
      error_weight: Weight for error classification loss.
      focal_alpha: Alpha parameter for focal loss.
      focal_gamma: Gamma parameter for focal loss.
    """
    super().__init__()
    
    self.training_mode = training_mode
    self.canonical_weight = canonical_weight
    self.perceived_weight = perceived_weight
    self.error_weight = error_weight
    
    # Initialize task-specific loss functions
    self.canonical_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)
    self.perceived_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)
    self.error_criterion = FocalCTCLoss(focal_alpha, focal_gamma, 0, True)
  
  def forward(
      self,
      model_outputs: Dict[str, torch.Tensor],
      canonical_targets: Optional[torch.Tensor] = None,
      canonical_input_lengths: Optional[torch.Tensor] = None,
      canonical_target_lengths: Optional[torch.Tensor] = None,
      perceived_targets: Optional[torch.Tensor] = None,
      perceived_input_lengths: Optional[torch.Tensor] = None,
      perceived_target_lengths: Optional[torch.Tensor] = None,
      error_targets: Optional[torch.Tensor] = None,
      error_input_lengths: Optional[torch.Tensor] = None,
      error_target_lengths: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes weighted multitask loss.
    
    Args:
      model_outputs: Dictionary with model predictions.
      canonical_targets: Canonical phoneme targets.
      canonical_input_lengths: Canonical input sequence lengths.
      canonical_target_lengths: Canonical target sequence lengths.
      perceived_targets: Perceived phoneme targets.
      perceived_input_lengths: Perceived input sequence lengths.
      perceived_target_lengths: Perceived target sequence lengths.
      error_targets: Error type targets.
      error_input_lengths: Error input sequence lengths.
      error_target_lengths: Error target sequence lengths.
    
    Returns:
      Tuple of (total_loss, loss_components_dict).
    """
    device = model_outputs[list(model_outputs.keys())[0]].device
    total_loss = torch.tensor(0.0, device=device)
    loss_components = {}
    
    # Canonical phoneme recognition loss (multitask mode only)
    if (self.training_mode == 'multitask' and
        'canonical_logits' in model_outputs and
        canonical_targets is not None):
      log_probs = torch.log_softmax(
          model_outputs['canonical_logits'], dim=-1
      ).transpose(0, 1)  # [T, N, C]
      
      canonical_loss = self.canonical_criterion(
          log_probs,
          canonical_targets,
          canonical_input_lengths,
          canonical_target_lengths
      )
      total_loss += self.canonical_weight * canonical_loss
      loss_components['canonical_loss'] = canonical_loss.item()
    
    # Perceived phoneme recognition loss (all modes)
    if ('perceived_logits' in model_outputs and
        perceived_targets is not None):
      log_probs = torch.log_softmax(
          model_outputs['perceived_logits'], dim=-1
      ).transpose(0, 1)  # [T, N, C]
      
      perceived_loss = self.perceived_criterion(
          log_probs,
          perceived_targets,
          perceived_input_lengths,
          perceived_target_lengths
      )
      total_loss += self.perceived_weight * perceived_loss
      loss_components['perceived_loss'] = perceived_loss.item()
    
    # Error classification loss (phoneme_error and multitask modes)
    if (self.training_mode in ['phoneme_error', 'multitask'] and
        'error_logits' in model_outputs and
        error_targets is not None):
      log_probs = torch.log_softmax(
          model_outputs['error_logits'], dim=-1
      ).transpose(0, 1)  # [T, N, C]
      
      error_loss = self.error_criterion(
          log_probs,
          error_targets,
          error_input_lengths,
          error_target_lengths
      )
      total_loss += self.error_weight * error_loss
      loss_components['error_loss'] = error_loss.item()
    
    loss_components['total_loss'] = total_loss.item()
    return total_loss, loss_components
