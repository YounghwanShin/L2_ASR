"""Trainer module for pronunciation assessment model.

This module implements the training loop with multitask learning support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.audio import create_attention_mask, enable_specaugment, compute_output_lengths


class ModelTrainer:
  """Trainer for multitask pronunciation assessment model."""
  
  def __init__(self, model: nn.Module, config, device: str = 'cuda', logger=None):
    """Initialize trainer.
    
    Args:
      model: Model to train.
      config: Configuration object.
      device: Training device.
      logger: Logger instance.
    """
    self.model = model
    self.config = config
    self.device = device
    self.logger = logger
    
    self._setup_optimizers()
    self.scaler = torch.amp.GradScaler('cuda')
  
  def _setup_optimizers(self):
    """Setup optimizers with different learning rates."""
    wav2vec_params = []
    main_params = []
    
    for name, param in self.model.named_parameters():
      if 'encoder.wav2vec2' in name:
        wav2vec_params.append(param)
      else:
        main_params.append(param)
    
    self.wav2vec_optimizer = optim.AdamW(wav2vec_params, lr=self.config.wav2vec_lr)
    self.main_optimizer = optim.AdamW(main_params, lr=self.config.main_lr)
  
  def train_epoch(self, dataloader: DataLoader, criterion, epoch: int) -> float:
    """Perform one training epoch.
    
    Args:
      dataloader: Training data loader.
      criterion: Loss function.
      epoch: Current epoch number.
      
    Returns:
      Average training loss.
    """
    self.model.train()
    if self.config.wav2vec2_specaug:
      enable_specaugment(self.model, True)

    total_loss = 0.0
    loss_components = {}

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch_data in enumerate(progress_bar):
      if batch_data is None:
        continue

      # Prepare inputs
      waveforms = batch_data['waveforms'].to(self.device)
      audio_lengths = batch_data['audio_lengths'].to(self.device)

      input_lengths = compute_output_lengths(self.model, audio_lengths)
      normalized_lengths = audio_lengths.float() / waveforms.shape[1]
      attention_mask = create_attention_mask(waveforms, normalized_lengths)

      # Forward pass
      with torch.amp.autocast('cuda'):
        outputs = self.model(waveforms, attention_mask, self.config.training_mode)

        # Prepare targets
        kwargs = {}
        
        # Canonical targets
        if 'canonical_labels' in batch_data:
          canonical_labels = batch_data['canonical_labels'].to(self.device)
          canonical_lengths = batch_data['canonical_lengths'].to(self.device)
          canonical_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs.get('canonical_logits', outputs['perceived_logits']).size(1)
          )
          kwargs.update({
              'canonical_targets': canonical_labels,
              'canonical_input_lengths': canonical_input_lengths,
              'canonical_target_lengths': canonical_lengths
          })

        # Perceived targets  
        if 'perceived_labels' in batch_data:
          perceived_labels = batch_data['perceived_labels'].to(self.device)
          perceived_lengths = batch_data['perceived_lengths'].to(self.device)
          perceived_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs['perceived_logits'].size(1)
          )
          kwargs.update({
              'perceived_targets': perceived_labels,
              'perceived_input_lengths': perceived_input_lengths,
              'perceived_target_lengths': perceived_lengths
          })

        # Error targets
        if 'error_labels' in batch_data:
          error_labels = batch_data['error_labels'].to(self.device)
          error_lengths = batch_data['error_lengths'].to(self.device)
          error_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs.get('error_logits', outputs['perceived_logits']).size(1)
          )
          
          valid_mask = error_lengths > 0
          if valid_mask.any():
            kwargs.update({
                'error_targets': error_labels[valid_mask],
                'error_input_lengths': error_input_lengths[valid_mask],
                'error_target_lengths': error_lengths[valid_mask]
            })

        loss, loss_dict = criterion(outputs, **kwargs)
        scaled_loss = loss / self.config.gradient_accumulation
        
        # Accumulate loss components
        for key, value in loss_dict.items():
          if key not in loss_components:
            loss_components[key] = []
          loss_components[key].append(value)

      # Backward pass
      if scaled_loss > 0:
        self.scaler.scale(scaled_loss).backward()

      # Optimizer step
      if (batch_idx + 1) % self.config.gradient_accumulation == 0:
        self.scaler.step(self.wav2vec_optimizer)
        self.scaler.step(self.main_optimizer)
        self.scaler.update()
        self.wav2vec_optimizer.zero_grad()
        self.main_optimizer.zero_grad()

        total_loss += scaled_loss.item() * self.config.gradient_accumulation

      # Update progress bar
      if (batch_idx + 1) % self.config.gradient_accumulation == 0:
        avg_components = {k: sum(v) / len(v) for k, v in loss_components.items()}
        progress_bar.set_postfix(avg_components)

      if (batch_idx + 1) % 100 == 0:
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return total_loss / (len(dataloader) // self.config.gradient_accumulation)

  def validate_epoch(self, dataloader: DataLoader, criterion) -> float:
    """Perform validation.
    
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
      for batch_data in tqdm(dataloader, desc='Validation'):
        if batch_data is None:
          continue

        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)

        input_lengths = compute_output_lengths(self.model, audio_lengths)
        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)

        outputs = self.model(waveforms, attention_mask, self.config.training_mode)

        kwargs = {}
        
        if 'canonical_labels' in batch_data:
          canonical_labels = batch_data['canonical_labels'].to(self.device)
          canonical_lengths = batch_data['canonical_lengths'].to(self.device)
          canonical_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs.get('canonical_logits', outputs['perceived_logits']).size(1)
          )
          kwargs.update({
              'canonical_targets': canonical_labels,
              'canonical_input_lengths': canonical_input_lengths,
              'canonical_target_lengths': canonical_lengths
          })

        if 'perceived_labels' in batch_data:
          perceived_labels = batch_data['perceived_labels'].to(self.device)
          perceived_lengths = batch_data['perceived_lengths'].to(self.device)
          perceived_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs['perceived_logits'].size(1)
          )
          kwargs.update({
              'perceived_targets': perceived_labels,
              'perceived_input_lengths': perceived_input_lengths,
              'perceived_target_lengths': perceived_lengths
          })

        if 'error_labels' in batch_data:
          error_labels = batch_data['error_labels'].to(self.device)
          error_lengths = batch_data['error_lengths'].to(self.device)
          error_input_lengths = torch.clamp(
              input_lengths, min=1, max=outputs.get('error_logits', outputs['perceived_logits']).size(1)
          )
          
          valid_mask = error_lengths > 0
          if valid_mask.any():
            kwargs.update({
                'error_targets': error_labels[valid_mask],
                'error_input_lengths': error_input_lengths[valid_mask],
                'error_target_lengths': error_lengths[valid_mask]
            })

        loss, _ = criterion(outputs, **kwargs)
        total_loss += loss.item() if loss > 0 else 0

    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

  def get_optimizers(self):
    """Return optimizers."""
    return self.wav2vec_optimizer, self.main_optimizer
