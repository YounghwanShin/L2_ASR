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
  """Set random seeds for reproducibility.
  
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
  """Setup experiment directories and logging.
  
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

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[
          logging.FileHandler(log_file, mode=file_mode),
          logging.StreamHandler()
      ]
  )


def save_checkpoint(model, wav2vec_opt, main_opt, epoch,
                   val_loss, train_loss, metrics, path):
  """Save model checkpoint.
  
  Args:
    model: Model to save.
    wav2vec_opt: Wav2Vec2 optimizer.
    main_opt: Main optimizer.
    epoch: Current epoch.
    val_loss: Validation loss.
    train_loss: Training loss.
    metrics: Evaluation metrics dictionary.
    path: Output path.
  """
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
      'main_optimizer_state_dict': main_opt.state_dict(),
      'val_loss': val_loss,
      'train_loss': train_loss,
      'metrics': metrics,
      'saved_time': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
  }
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save(checkpoint, path)


def load_checkpoint(checkpoint_path, model, wav2vec_optimizer,
                   main_optimizer, device):
  """Load checkpoint and restore training state.
  
  Args:
    checkpoint_path: Path to checkpoint.
    model: Model to load state into.
    wav2vec_optimizer: Wav2Vec2 optimizer.
    main_optimizer: Main optimizer.
    device: Device to load to.
    
  Returns:
    Tuple of (start_epoch, best_metrics).
  """
  logger = logging.getLogger(__name__)
  logger.info(f"Loading checkpoint from {checkpoint_path}")
  
  checkpoint = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  wav2vec_optimizer.load_state_dict(checkpoint['wav2vec_optimizer_state_dict'])
  main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
  
  start_epoch = checkpoint['epoch'] + 1
  best_metrics = checkpoint.get('metrics', {})
  
  logger.info(f"Resumed from epoch {checkpoint['epoch']}")
  return start_epoch, best_metrics


def detect_model_type_from_checkpoint(checkpoint_path: str) -> str:
  """Auto-detect model architecture from checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint.
    
  Returns:
    Model type ('simple' or 'transformer').
  """
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  
  keys = list(state_dict.keys())
  if any('feature_encoder.transformer' in key for key in keys):
    return 'transformer'
  return 'simple'
