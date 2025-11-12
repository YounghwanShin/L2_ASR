"""Training utility functions.

This module provides utilities for experiment setup, checkpoint management,
and reproducibility.
"""

import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import pytz
import torch


def set_random_seed(seed: int):
  """Sets random seeds for reproducibility.
  
  Args:
    seed: Random seed value.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def setup_experiment_directories(config, resume: bool = False):
  """Sets up experiment directories and logging.
  
  Args:
    config: Configuration object.
    resume: Whether resuming from checkpoint.
  """
  os.makedirs(config.checkpoint_dir, exist_ok=True)
  os.makedirs(config.log_dir, exist_ok=True)
  os.makedirs(config.result_dir, exist_ok=True)

  config_path = os.path.join(config.experiment_dir, 'config.json')
  if not resume:
    config.save_config(config_path)

  log_file = os.path.join(config.log_dir, 'training.log')
  file_mode = 'a' if resume else 'w'

  # Get root logger
  root_logger = logging.getLogger()
  
  # Remove existing handlers to avoid duplicates
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
  
  # Set logging level
  root_logger.setLevel(logging.INFO)
  
  # Create formatter
  formatter = logging.Formatter(
      '%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
  )
  
  # Add file handler
  file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  root_logger.addHandler(file_handler)
  
  # Add console handler
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)
  root_logger.addHandler(console_handler)
  
  logging.info(f"Logging initialized - Log file: {log_file}")


def save_checkpoint(
    model, 
    wav2vec_opt, 
    main_opt, 
    epoch,
    val_loss, 
    train_loss, 
    metrics, 
    path,
    config=None
):
  """Saves model checkpoint with configuration.
  
  Args:
    model: Model to save.
    wav2vec_opt: Wav2Vec2 optimizer.
    main_opt: Main optimizer.
    epoch: Current epoch number.
    val_loss: Validation loss.
    train_loss: Training loss.
    metrics: Dictionary of evaluation metrics.
    path: Output checkpoint path.
    config: Configuration object to save with checkpoint.
  """
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
      'main_optimizer_state_dict': main_opt.state_dict(),
      'val_loss': val_loss,
      'train_loss': train_loss,
      'metrics': metrics,
      'saved_time': datetime.now(
          pytz.timezone('Asia/Seoul')
      ).strftime('%Y-%m-%d %H:%M:%S')
  }
  
  # Save configuration information
  if config is not None:
    checkpoint['config'] = {
        'pretrained_model': config.pretrained_model,
        'training_mode': config.training_mode,
        'model_type': config.model_type,
        'hidden_dim': config.get_model_config()['hidden_dim'],
        'num_phonemes': config.num_phonemes,
        'num_error_types': config.num_error_types
    }
  
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path, 
    model, 
    wav2vec_optimizer,
    main_optimizer, 
    device,
    config=None
):
  """Loads checkpoint and restores training state.
  
  Args:
    checkpoint_path: Path to checkpoint file.
    model: Model to load state into.
    wav2vec_optimizer: Wav2Vec2 optimizer.
    main_optimizer: Main optimizer.
    device: Device to load model to.
    config: Configuration object for validation.
    
  Returns:
    Tuple of (start_epoch, best_metrics).
  """
  logger = logging.getLogger(__name__)
  logger.info(f"Loading checkpoint from {checkpoint_path}")
  
  checkpoint = torch.load(checkpoint_path, map_location=device)
  
  # Validate configuration if available
  if 'config' in checkpoint and config is not None:
    ckpt_config = checkpoint['config']
    if ckpt_config.get('pretrained_model') != config.pretrained_model:
      logger.warning(
          f"Pretrained model mismatch: "
          f"checkpoint={ckpt_config.get('pretrained_model')}, "
          f"current={config.pretrained_model}"
      )
    if ckpt_config.get('training_mode') != config.training_mode:
      logger.warning(
          f"Training mode mismatch: "
          f"checkpoint={ckpt_config.get('training_mode')}, "
          f"current={config.training_mode}"
      )
  
  model.load_state_dict(checkpoint['model_state_dict'])
  wav2vec_optimizer.load_state_dict(
      checkpoint['wav2vec_optimizer_state_dict']
  )
  main_optimizer.load_state_dict(
      checkpoint['main_optimizer_state_dict']
  )
  
  start_epoch = checkpoint['epoch'] + 1
  best_metrics = checkpoint.get('metrics', {})
  
  logger.info(f"Resumed from epoch {checkpoint['epoch']}")
  logger.info(f"Best metrics: {best_metrics}")
  
  return start_epoch, best_metrics


def detect_model_type_from_checkpoint(checkpoint_path: str) -> str:
  """Auto-detects model architecture from checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint file.
    
  Returns:
    Model type ('simple' or 'transformer').
  """
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  
  # Try to get from saved config first
  if 'config' in checkpoint and 'model_type' in checkpoint['config']:
    return checkpoint['config']['model_type']
  
  # Fallback to checking state dict keys
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  keys = list(state_dict.keys())
  
  if any('feature_encoder.transformer' in key for key in keys):
    return 'transformer'
  return 'simple'


def get_checkpoint_config(checkpoint_path: str) -> dict:
  """Extracts configuration from checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint file.
    
  Returns:
    Dictionary containing configuration information.
  """
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  return checkpoint.get('config', {})