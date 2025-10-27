"""Training utility functions for pronunciation assessment model.

This module provides utilities for experiment setup, checkpoint management,
model type detection, and reproducibility.
"""

import os
import json
import random
import logging
import torch
import numpy as np
from datetime import datetime
import pytz


def seed_everything(seed: int):
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


def setup_experiment_dirs(config, resume: bool = False):
    """Sets up experiment directories and logging.
    
    Creates necessary directories for checkpoints, logs, and results.
    Configures logging to both file and console.
    
    Args:
        config: Configuration object containing directory paths.
        resume: Whether resuming from a checkpoint.
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path):
    """Saves model checkpoint with training state.
    
    Args:
        model: Model to save.
        wav2vec_opt: Optimizer for Wav2Vec2 parameters.
        main_opt: Optimizer for other parameters.
        epoch: Current epoch number.
        val_loss: Validation loss.
        train_loss: Training loss.
        metrics: Dictionary of evaluation metrics.
        path: Path where checkpoint will be saved.
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


def load_checkpoint(checkpoint_path, model, wav2vec_optimizer, main_optimizer, device):
    """Loads checkpoint and restores training state.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load state into.
        wav2vec_optimizer: Optimizer for Wav2Vec2 parameters.
        main_optimizer: Optimizer for other parameters.
        device: Device to load checkpoint to.
        
    Returns:
        tuple: (start_epoch, best_metrics) where start_epoch is the epoch
            to resume from and best_metrics contains previous best metrics.
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
    if 'saved_time' in checkpoint:
        logger.info(f"Checkpoint saved at: {checkpoint['saved_time']}")
    logger.info(f"Previous metrics: {best_metrics}")
    
    return start_epoch, best_metrics


def get_model_class(model_type: str):
    """Returns model class and loss function based on model type.
    
    Args:
        model_type: Type of model architecture.
        
    Returns:
        tuple: (model_class, loss_class)
    """
    from ..models.unified_model import UnifiedModel
    from ..models.losses import UnifiedLoss
    return UnifiedModel, UnifiedLoss


def detect_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """Auto-detects model architecture type from checkpoint.
    
    Examines the state dict keys to determine whether the model uses
    simple or transformer architecture.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        str: Model type ('simple' or 'transformer').
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())

    # Check for transformer-specific keys
    if any('feature_encoder.transformer' in key for key in keys):
        return 'transformer'
    else:
        return 'simple'


def remove_module_prefix(state_dict):
    """Removes 'module.' prefix from state dict keys.
    
    This prefix is added by DataParallel and needs to be removed
    when loading into a non-DataParallel model.
    
    Args:
        state_dict: Model state dictionary.
        
    Returns:
        dict: State dictionary with 'module.' prefix removed.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
