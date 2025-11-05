"""Trainer class for multitask pronunciation assessment.

Handles training loop with:
  - Gradient accumulation for effective large batch training
  - Mixed precision training for efficiency
  - Separate learning rates for Wav2Vec2 and task-specific layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from ..utils.audio import (
    create_attention_mask,
    enable_specaugment,
    compute_wav2vec_output_lengths
)


class PronunciationTrainer:
  """Trainer for multitask pronunciation assessment models.
  
  Manages the training process including:
    - Optimizer setup with differential learning rates
    - Training and validation loops
    - Gradient accumulation
    - Mixed precision training
  
  Attributes:
    model: Model to train.
    config: Configuration object.
    device: Training device (cuda/cpu).
    logger: Logger instance for training progress.
    wav2vec_optimizer: Optimizer for Wav2Vec2 parameters.
    main_optimizer: Optimizer for task-specific parameters.
    scaler: Gradient scaler for mixed precision training.
  """
  
  def __init__(
      self,
      model: nn.Module,
      config,
      device: str = 'cuda',
      logger=None
  ):
    """Initializes the trainer.
    
    Args:
      model: Model instance to train.
      config: Configuration object with training parameters.
      device: Device for training ('cuda' or 'cpu').
      logger: Optional logger for training progress.
    """
    self.model = model
    self.config = config
    self.device = device
    self.logger = logger
    
    # Setup optimizers with different learning rates
    self._setup_optimizers()
    
    # Initialize gradient scaler for mixed precision training
    self.scaler = torch.amp.GradScaler('cuda')
  
  def _setup_optimizers(self):
    """Sets up optimizers with differential learning rates.
    
    Uses lower learning rate for pretrained Wav2Vec2 parameters and
    higher learning rate for task-specific parameters.
    """
    wav2vec_params = []
    main_params = []
    
    # Separate parameters into Wav2Vec2 and task-specific
    for name, param in self.model.named_parameters():
      if 'wav2vec_encoder.wav2vec2' in name:
        wav2vec_params.append(param)
      else:
        main_params.append(param)
    
    # Create optimizers with different learning rates
    self.wav2vec_optimizer = optim.AdamW(
        wav2vec_params,
        lr=self.config.wav2vec_learning_rate
    )
    self.main_optimizer = optim.AdamW(
        main_params,
        lr=self.config.main_learning_rate
    )
  
  def _prepare_loss_inputs(self, batch, output_lengths):
    """Prepares inputs for loss computation.
    
    Args:
      batch: Batch dictionary with labels and metadata.
      output_lengths: Sequence lengths after Wav2Vec2 encoding.
    
    Returns:
      Dictionary with arguments for loss function.
    """
    loss_inputs = {}
    
    # Canonical phoneme targets
    if 'canonical_labels' in batch:
      loss_inputs.update({
          'canonical_targets': batch['canonical_labels'].to(self.device),
          'canonical_input_lengths': output_lengths,
          'canonical_target_lengths': batch['canonical_lengths'].to(self.device)
      })
    
    # Perceived phoneme targets
    if 'perceived_labels' in batch:
      loss_inputs.update({
          'perceived_targets': batch['perceived_labels'].to(self.device),
          'perceived_input_lengths': output_lengths,
          'perceived_target_lengths': batch['perceived_lengths'].to(self.device)
      })
    
    # Error classification targets
    if 'error_labels' in batch:
      error_lengths = batch['error_lengths'].to(self.device)
      valid_mask = error_lengths > 0
      
      if valid_mask.any():
        loss_inputs.update({
            'error_targets': batch['error_labels'][valid_mask].to(self.device),
            'error_input_lengths': output_lengths[valid_mask],
            'error_target_lengths': error_lengths[valid_mask]
        })
    
    return loss_inputs
  
  def train_epoch(
      self,
      dataloader: DataLoader,
      criterion,
      epoch: int
  ) -> float:
    """Performs one training epoch.
    
    Args:
      dataloader: Training data loader.
      criterion: Loss function.
      epoch: Current epoch number.
    
    Returns:
      Average training loss for the epoch.
    """
    self.model.train()
    
    # Enable SpecAugment if configured
    if self.config.enable_wav2vec_specaug:
      enable_specaugment(self.model, True)
    
    total_loss = 0.0
    accumulated_loss_components = {}
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
      if batch is None:
        continue
      
      # Prepare inputs
      waveforms = batch['waveforms'].to(self.device)
      audio_lengths = batch['audio_lengths'].to(self.device)
      
      # Compute sequence lengths after Wav2Vec2 encoding
      output_lengths = compute_wav2vec_output_lengths(self.model, audio_lengths)
      
      # Create attention mask
      normalized_lengths = audio_lengths.float() / waveforms.shape[1]
      attention_mask = create_attention_mask(waveforms, normalized_lengths)
      
      # Forward pass with mixed precision
      with torch.amp.autocast('cuda'):
        # Model forward
        outputs = self.model(
            waveforms,
            attention_mask,
            self.config.training_mode
        )
        
        # Prepare loss inputs
        loss_inputs = self._prepare_loss_inputs(batch, output_lengths)
        
        # Compute loss
        loss, loss_components = criterion(outputs, **loss_inputs)
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        # Accumulate loss components for logging
        for key, value in loss_components.items():
          if key not in accumulated_loss_components:
            accumulated_loss_components[key] = []
          accumulated_loss_components[key].append(value)
      
      # Backward pass
      if scaled_loss > 0:
        self.scaler.scale(scaled_loss).backward()
      
      # Optimizer step with gradient accumulation
      if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
        self.scaler.step(self.wav2vec_optimizer)
        self.scaler.step(self.main_optimizer)
        self.scaler.update()
        
        self.wav2vec_optimizer.zero_grad()
        self.main_optimizer.zero_grad()
        
        total_loss += scaled_loss.item() * self.config.gradient_accumulation_steps
      
      # Update progress bar
      if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
        avg_components = {
            k: sum(v) / len(v)
            for k, v in accumulated_loss_components.items()
        }
        progress_bar.set_postfix(avg_components)
      
      # Clear CUDA cache periodically
      if (batch_idx + 1) % 100 == 0:
        torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    num_updates = len(dataloader) // self.config.gradient_accumulation_steps
    return total_loss / num_updates
  
  def validate_epoch(self, dataloader: DataLoader, criterion) -> float:
    """Performs validation.
    
    Args:
      dataloader: Validation data loader.
      criterion: Loss function.
    
    Returns:
      Average validation loss.
    """
    self.model.eval()
    enable_specaugment(self.model, False)
    
    total_loss = 0.0
    
    with torch.no_grad():
      for batch in tqdm(dataloader, desc='Validation'):
        if batch is None:
          continue
        
        # Prepare inputs
        waveforms = batch['waveforms'].to(self.device)
        audio_lengths = batch['audio_lengths'].to(self.device)
        
        # Compute sequence lengths
        output_lengths = compute_wav2vec_output_lengths(self.model, audio_lengths)
        
        # Create attention mask
        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)
        
        # Forward pass
        outputs = self.model(
            waveforms,
            attention_mask,
            self.config.training_mode
        )
        
        # Prepare loss inputs and compute loss
        loss_inputs = self._prepare_loss_inputs(batch, output_lengths)
        loss, _ = criterion(outputs, **loss_inputs)
        
        total_loss += loss.item() if loss > 0 else 0
    
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)
  
  def get_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
    """Returns the optimizers.
    
    Returns:
      Tuple of (wav2vec_optimizer, main_optimizer).
    """
    return self.wav2vec_optimizer, self.main_optimizer
