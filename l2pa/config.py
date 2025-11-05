"""Configuration module for pronunciation assessment.

This module defines all hyperparameters, model settings, and paths for
training and evaluation.
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
  
  This class manages all settings for the pronunciation assessment system,
  including model architecture, training hyperparameters, and data paths.
  """
  
  # Model configuration
  pretrained_model: str = 'facebook/wav2vec2-large-xlsr-53'
  sampling_rate: int = 16000
  max_length: int = 140000
  
  # Output dimensions
  num_phonemes: int = 42
  num_error_types: int = 5
  
  # Training mode (main focus: multitask error learning)
  training_mode: str = 'multitask'
  model_type: str = 'transformer'
  
  # Training hyperparameters
  batch_size: int = 16
  eval_batch_size: int = 16
  num_epochs: int = 100
  gradient_accumulation: int = 2
  
  # Learning rates
  main_lr: float = 3e-4
  wav2vec_lr: float = 1e-5
  
  # Loss weights
  canonical_weight: float = 0.3
  perceived_weight: float = 0.3
  error_weight: float = 0.4
  
  # Focal loss parameters
  focal_alpha: float = 0.25
  focal_gamma: float = 2.0
  
  # Checkpoint options
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
  
  # Paths
  base_experiment_dir: str = 'experiments'
  experiment_name: Optional[str] = None
  data_dir: str = 'data'
  device: str = 'cuda'
  
  # Model architecture configs
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
    """Initializes derived attributes and validates configuration."""
    self._normalize_loss_weights()
    self._setup_paths()
  
  def _normalize_loss_weights(self):
    """Normalizes loss weights to sum to 1.0."""
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
  
  def _setup_paths(self):
    """Sets up experiment directories and data paths."""
    # Generate experiment name if not provided
    if self.experiment_name is None:
      timestamp = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
      if self.use_cross_validation:
        self.experiment_name = (
            f'{self.training_mode}_{self.model_type}_cv{self.cv_fold}_{timestamp}'
        )
      else:
        self.experiment_name = (
            f'{self.training_mode}_{self.model_type}_{timestamp}'
        )
    
    # Setup directory paths
    self.experiment_dir = os.path.join(
        self.base_experiment_dir, self.experiment_name
    )
    self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
    self.log_dir = os.path.join(self.experiment_dir, 'logs')
    self.result_dir = os.path.join(self.experiment_dir, 'results')
    
    # Setup data paths
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
    """Gets model architecture configuration.
    
    Returns:
      Dictionary with model architecture parameters.
    """
    config = self.model_configs[self.model_type].copy()
    config['use_transformer'] = (self.model_type == 'transformer')
    return config
  
  def has_canonical_task(self) -> bool:
    """Checks if canonical phoneme prediction is enabled.
    
    Returns:
      True if canonical task is active, False otherwise.
    """
    return self.training_mode == 'multitask'
  
  def has_perceived_task(self) -> bool:
    """Checks if perceived phoneme prediction is enabled.
    
    Returns:
      True if perceived task is active, False otherwise.
    """
    return self.training_mode in ['phoneme_only', 'phoneme_error', 'multitask']
  
  def has_error_task(self) -> bool:
    """Checks if error detection is enabled.
    
    Returns:
      True if error task is active, False otherwise.
    """
    return self.training_mode in ['phoneme_error', 'multitask']
  
  def save_config(self, path: str):
    """Saves configuration to JSON file.
    
    Args:
      path: Output file path.
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
    """Gets error type ID to name mapping.
    
    Returns:
      Dictionary mapping error IDs to names.
    """
    return {
        0: 'blank',
        1: 'deletion',
        2: 'insertion',
        3: 'substitution',
        4: 'correct'
    }
