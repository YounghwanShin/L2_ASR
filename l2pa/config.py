"""Configuration for pronunciation assessment model.

This module defines all hyperparameters, model settings, and file paths
for the L2 pronunciation assessment system. Supports both cross-validation
and single-split training modes.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from pytz import timezone


@dataclass
class Config:
  """Configuration for model training and evaluation.
  
  This class contains all hyperparameters and settings needed for training
  and evaluating pronunciation assessment models. It automatically handles
  path setup, loss weight normalization, and cross-validation configuration.
  
  Attributes:
    pretrained_model: Hugging Face model identifier for Wav2Vec2.
    sampling_rate: Audio sampling rate in Hz.
    max_length: Maximum audio length in samples (longer clips are filtered).
    num_phonemes: Total number of phoneme classes including blank.
    num_error_types: Number of error types (blank, D, I, S, C).
    training_mode: One of 'phoneme_only', 'phoneme_error', or 'multitask'.
    model_type: Model architecture - 'simple' or 'transformer'.
    batch_size: Batch size for training.
    eval_batch_size: Batch size for evaluation.
    num_workers: Number of data loader worker processes.
    num_epochs: Total number of training epochs.
    gradient_accumulation: Number of gradient accumulation steps.
    main_lr: Learning rate for main model parameters.
    wav2vec_lr: Learning rate for Wav2Vec2 parameters.
    canonical_weight: Loss weight for canonical phoneme prediction.
    perceived_weight: Loss weight for perceived phoneme prediction.
    error_weight: Loss weight for error detection.
    focal_alpha: Focal loss alpha parameter for class balancing.
    focal_gamma: Focal loss gamma parameter for focusing on hard examples.
    save_best_metrics: Metrics to track for saving best checkpoints.
    wav2vec2_specaug: Whether to apply SpecAugment during training.
    seed: Random seed for reproducibility.
    use_cross_validation: Whether to use k-fold cross-validation.
    cv_fold: Current cross-validation fold index (0-indexed).
    num_cv_folds: Total number of cross-validation folds.
    test_speakers: Speaker IDs reserved for test set.
    base_experiment_dir: Root directory for all experiments.
    experiment_name: Name of current experiment (auto-generated if None).
    data_dir: Root directory containing dataset files.
    device: Device for training ('cuda' or 'cpu').
    model_configs: Architecture-specific hyperparameters.
  """
  
  # Pretrained model configuration
  pretrained_model: str = "facebook/wav2vec2-large-xlsr-53"
  sampling_rate: int = 16000
  max_length: int = 140000

  # Model output dimensions
  num_phonemes: int = 42
  num_error_types: int = 5

  # Training configuration
  training_mode: str = 'multitask'
  model_type: str = 'transformer'

  # Training hyperparameters
  batch_size: int = 16
  eval_batch_size: int = 16
  num_workers: int = 4
  num_epochs: int = 100
  gradient_accumulation: int = 2

  # Optimizer learning rates
  main_lr: float = 3e-4
  wav2vec_lr: float = 1e-5

  # Multitask loss weights
  canonical_weight: float = 0.3
  perceived_weight: float = 0.3
  error_weight: float = 0.4

  # Focal loss parameters
  focal_alpha: float = 0.25
  focal_gamma: float = 2.0

  # Checkpoint and training settings
  save_best_metrics: List[str] = field(
      default_factory=lambda: ['canonical', 'perceived', 'error', 'loss']
  )
  wav2vec2_specaug: bool = True
  seed: int = 42

  # Cross-validation settings
  use_cross_validation: bool = True
  cv_fold: int = 0
  num_cv_folds: Optional[int] = None
  test_speakers: List[str] = field(
      default_factory=lambda: ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']
  )

  # Directory paths
  base_experiment_dir: str = "experiments"
  experiment_name: Optional[str] = None
  data_dir: str = "data"
  device: str = "cuda"

  # Model architecture configurations
  model_configs: Dict = field(default_factory=lambda: {
      'simple': {
          'hidden_dim': 1024,
          'dropout': 0.1
      },
      'transformer': {
          'hidden_dim': 1024,
          'num_layers': 2,
          'num_heads': 8,
          'dropout': 0.1
      }
  })

  def __post_init__(self):
    """Initializes configuration after dataclass initialization.
    
    Performs the following operations:
      1. Normalizes loss weights based on training mode
      2. Sets up experiment directory structure
      3. Configures data file paths
    """
    self._normalize_loss_weights()
    self._setup_experiment_paths()

  def _normalize_loss_weights(self):
    """Normalizes loss weights to sum to 1.0 based on training mode.
    
    Different training modes use different combinations of losses:
      - phoneme_only: Only perceived phoneme (weight=1.0)
      - phoneme_error: Perceived phoneme + error detection
      - multitask: All three tasks (canonical, perceived, error)
    
    Weights are automatically normalized to sum to 1.0 for the active tasks.
    """
    if self.training_mode == 'phoneme_only':
      self.canonical_weight = 0.0
      self.perceived_weight = 1.0
      self.error_weight = 0.0
    elif self.training_mode == 'phoneme_error':
      self.canonical_weight = 0.0
      total = self.perceived_weight + self.error_weight
      if abs(total - 1.0) > 1e-6:
        self.perceived_weight /= total
        self.error_weight /= total
    elif self.training_mode == 'multitask':
      total = (self.canonical_weight + 
               self.perceived_weight + 
               self.error_weight)
      if abs(total - 1.0) > 1e-6:
        self.canonical_weight /= total
        self.perceived_weight /= total
        self.error_weight /= total

  def _setup_experiment_paths(self):
    """Sets up experiment directory structure and data file paths.
    
    Creates a unique experiment name with timestamp if not provided.
    Configures all necessary directories (checkpoints, logs, results)
    and sets appropriate data split paths based on cross-validation mode.
    """
    if self.experiment_name is None:
      timestamp = datetime.now(
          timezone('Asia/Seoul')
      ).strftime('%Y%m%d_%H%M%S')
      
      if self.use_cross_validation:
        self.experiment_name = (
            f"{self.training_mode}_{self.model_type}_"
            f"cv{self.cv_fold}_{timestamp}"
        )
      else:
        self.experiment_name = (
            f"{self.training_mode}_{self.model_type}_{timestamp}"
        )
    
    self.experiment_dir = os.path.join(
        self.base_experiment_dir, 
        self.experiment_name
    )
    self.checkpoint_dir = os.path.join(
        self.experiment_dir, 
        'checkpoints'
    )
    self.log_dir = os.path.join(self.experiment_dir, 'logs')
    self.result_dir = os.path.join(self.experiment_dir, 'results')
    
    if self.use_cross_validation:
      fold_dir = os.path.join(self.data_dir, f'fold_{self.cv_fold}')
      self.train_data = os.path.join(fold_dir, 'train_labels.json')
      self.val_data = os.path.join(fold_dir, 'val_labels.json')
    else:
      self.train_data = os.path.join(self.data_dir, 'train_labels.json')
      self.val_data = os.path.join(self.data_dir, 'val_labels.json')
    
    self.test_data = os.path.join(self.data_dir, 'test_labels.json')
    self.phoneme_map = os.path.join(self.data_dir, 'phoneme_to_id.json')

  def get_model_config(self) -> Dict:
    """Returns architecture-specific configuration for model initialization.
    
    Returns:
      Dictionary containing model hyperparameters including hidden_dim,
      num_layers, num_heads, dropout, and use_transformer flag.
    """
    config = self.model_configs[self.model_type].copy()
    config['use_transformer'] = (self.model_type == 'transformer')
    return config

  def has_canonical_task(self) -> bool:
    """Checks if current configuration includes canonical phoneme task.
    
    Returns:
      True if canonical phoneme prediction is enabled.
    """
    return self.training_mode == 'multitask'

  def has_perceived_task(self) -> bool:
    """Checks if current configuration includes perceived phoneme task.
    
    Returns:
      True if perceived phoneme prediction is enabled.
    """
    return self.training_mode in [
        'phoneme_only', 
        'phoneme_error', 
        'multitask'
    ]

  def has_error_task(self) -> bool:
    """Checks if current configuration includes error detection task.
    
    Returns:
      True if error detection is enabled.
    """
    return self.training_mode in ['phoneme_error', 'multitask']

  def save_config(self, path: str):
    """Saves configuration to JSON file.
    
    Args:
      path: Output file path for saving configuration.
    """
    config_dict = {
        attr: getattr(self, attr)
        for attr in dir(self)
        if not attr.startswith('_') and not callable(getattr(self, attr))
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
      json.dump(config_dict, f, indent=2)

  @staticmethod
  def get_error_type_names() -> Dict[int, str]:
    """Returns mapping from error type ID to human-readable name.
    
    Returns:
      Dictionary mapping error type IDs to their names:
        0: blank (CTC blank token)
        1: deletion (phoneme missing in learner's speech)
        2: insertion (extra phoneme in learner's speech)
        3: substitution (wrong phoneme produced)
        4: correct (phoneme correctly produced)
    """
    return {
        0: 'blank',
        1: 'deletion',
        2: 'insertion',
        3: 'substitution',
        4: 'correct'
    }