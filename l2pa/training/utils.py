"""Training utility functions.

Provides utilities for:
  - Experiment setup and directory management
  - Checkpoint saving and loading
  - Random seed setting for reproducibility
"""

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pytz
import torch


def set_random_seed(seed: int):
  """Sets random seeds for reproducibility across all libraries.
  
  Args:
    seed: Random seed value.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def setup_experiment_directories(config, is_resuming: bool = False):
  """Sets up experiment directories and configures logging.
  
  Args:
    config: Configuration object with directory paths.
    is_resuming: Whether resuming from a checkpoint.
  """
  # Create necessary directories
  os.makedirs(config.checkpoint_dir, exist_ok=True)
  os.makedirs(config.log_dir, exist_ok=True)
  os.makedirs(config.result_dir, exist_ok=True)
  
  # Save configuration
  config_path = os.path.join(config.experiment_dir, 'config.json')
  if not is_resuming:
    config.save_to_file(config_path)
  
  # Setup logging
  log_file = os.path.join(config.log_dir, 'training.log')
  file_mode = 'a' if is_resuming else 'w'
  
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[
          logging.FileHandler(log_file, mode=file_mode),
          logging.StreamHandler()
      ]
  )


def save_checkpoint(
    model,
    wav2vec_optimizer,
    main_optimizer,
    epoch: int,
    val_loss: float,
    train_loss: float,
    best_metrics: Dict,
    save_path: str
):
  """Saves training checkpoint.
  
  Args:
    model: Model to save.
    wav2vec_optimizer: Wav2Vec2 optimizer.
    main_optimizer: Main optimizer.
    epoch: Current epoch.
    val_loss: Validation loss.
    train_loss: Training loss.
    best_metrics: Dictionary of best metrics achieved.
    save_path: Path where checkpoint will be saved.
  """
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'wav2vec_optimizer_state_dict': wav2vec_optimizer.state_dict(),
      'main_optimizer_state_dict': main_optimizer.state_dict(),
      'val_loss': val_loss,
      'train_loss': train_loss,
      'best_metrics': best_metrics,
      'saved_time': datetime.now(
          pytz.timezone('Asia/Seoul')
      ).strftime('%Y-%m-%d %H:%M:%S')
  }
  
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model,
    wav2vec_optimizer,
    main_optimizer,
    device: str
) -> Tuple[int, Dict]:
  """Loads checkpoint and restores training state.
  
  Args:
    checkpoint_path: Path to checkpoint file.
    model: Model to load state into.
    wav2vec_optimizer: Wav2Vec2 optimizer.
    main_optimizer: Main optimizer.
    device: Device to load tensors to.
  
  Returns:
    Tuple of (start_epoch, best_metrics).
  """
  logger = logging.getLogger(__name__)
  logger.info(f'Loading checkpoint from {checkpoint_path}')
  
  checkpoint = torch.load(checkpoint_path, map_location=device)
  
  # Handle DataParallel state dict
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
  
  model.load_state_dict(state_dict)
  wav2vec_optimizer.load_state_dict(
      checkpoint['wav2vec_optimizer_state_dict']
  )
  main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
  
  start_epoch = checkpoint['epoch'] + 1
  best_metrics = checkpoint.get('best_metrics', {})
  
  logger.info(f'Resumed from epoch {checkpoint["epoch"]}')
  return start_epoch, best_metrics


def detect_model_architecture(checkpoint_path: str) -> str:
  """Auto-detects model architecture from checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint file.
  
  Returns:
    Model type ('simple' or 'transformer').
  """
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  
  keys = list(state_dict.keys())
  if any('feature_encoder.transformer_encoder' in key for key in keys):
    return 'transformer'
  return 'simple'
