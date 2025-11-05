"""Configuration module for L2 pronunciation assessment system.

Manages all hyperparameters, model settings, and paths for training and
evaluation of multitask pronunciation assessment models.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from pytz import timezone


@dataclass
class Config:
  """Configuration for multitask pronunciation assessment.
  
  This class centralizes all configuration for the pronunciation assessment
  system, which jointly trains models for:
    - Canonical phoneme recognition (ground truth pronunciation)
    - Perceived phoneme recognition (actual learner pronunciation)
    - Error classification (Deletion, Insertion, Substitution, Correct)
  
  Attributes:
    pretrained_model: Wav2Vec2 model identifier for audio encoding.
    sampling_rate: Target audio sampling rate in Hz.
    max_audio_length: Maximum audio length in samples for filtering.
    num_phonemes: Number of phoneme classes including blank token.
    num_error_types: Number of error types (blank + D/I/S/C).
    training_mode: Mode determining active learning tasks.
    model_type: Encoder architecture type.
    batch_size: Training batch size.
    eval_batch_size: Evaluation batch size.
    num_epochs: Total training epochs.
    gradient_accumulation_steps: Steps for gradient accumulation.
    main_learning_rate: Learning rate for task-specific layers.
    wav2vec_learning_rate: Learning rate for Wav2Vec2 encoder.
    canonical_loss_weight: Weight for canonical phoneme loss.
    perceived_loss_weight: Weight for perceived phoneme loss.
    error_loss_weight: Weight for error classification loss.
    focal_alpha: Alpha parameter for focal loss.
    focal_gamma: Gamma parameter for focal loss.
    save_best_metrics: Metrics to track for saving best checkpoints.
    enable_wav2vec_specaug: Whether to enable SpecAugment during training.
    random_seed: Random seed for reproducibility.
    use_cross_validation: Whether to use cross-validation.
    cv_fold_index: Current cross-validation fold index.
    num_cv_folds: Total number of cross-validation folds.
    test_speaker_ids: Fixed test set speaker identifiers.
    experiment_base_dir: Base directory for experiment outputs.
    experiment_name: Unique experiment identifier.
    data_root_dir: Root directory for dataset files.
    device: Computing device (cuda/cpu).
    model_architecture_configs: Architecture-specific hyperparameters.
  """
  
  # Wav2Vec2 configuration
  pretrained_model: str = 'facebook/wav2vec2-large-xlsr-53'
  sampling_rate: int = 16000
  max_audio_length: int = 140000
  
  # Model output dimensions
  num_phonemes: int = 42  # Including blank token
  num_error_types: int = 5  # Blank + D/I/S/C
  
  # Training mode: Core focus is error detection with multitask learning
  training_mode: str = 'multitask'  # Options: 'phoneme_only', 'phoneme_error', 'multitask'
  model_type: str = 'transformer'  # Options: 'simple', 'transformer'
  
  # Training hyperparameters
  batch_size: int = 16
  eval_batch_size: int = 16
  num_epochs: int = 100
  gradient_accumulation_steps: int = 2
  
  # Optimizer settings
  main_learning_rate: float = 3e-4
  wav2vec_learning_rate: float = 1e-5
  
  # Loss weights for multitask learning
  canonical_loss_weight: float = 0.3
  perceived_loss_weight: float = 0.3
  error_loss_weight: float = 0.4
  
  # Focal loss parameters for handling class imbalance
  focal_alpha: float = 0.25
  focal_gamma: float = 2.0
  
  # Checkpoint configuration
  save_best_metrics: List[str] = field(
      default_factory=lambda: ['canonical', 'perceived', 'error', 'loss']
  )
  enable_wav2vec_specaug: bool = True
  random_seed: int = 42
  
  # Cross-validation settings
  use_cross_validation: bool = True
  cv_fold_index: int = 0
  num_cv_folds: Optional[int] = None
  test_speaker_ids: List[str] = field(
      default_factory=lambda: ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']
  )
  
  # Directory paths
  experiment_base_dir: str = 'experiments'
  experiment_name: Optional[str] = None
  data_root_dir: str = 'data'
  device: str = 'cuda'
  
  # Architecture-specific configurations
  model_architecture_configs: Dict = field(default_factory=lambda: {
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
    self._initialize_paths()
  
  def _normalize_loss_weights(self):
    """Normalizes loss weights to sum to 1.0 based on training mode."""
    if self.training_mode == 'phoneme_only':
      # Only perceived phoneme recognition
      self.canonical_loss_weight = 0.0
      self.perceived_loss_weight = 1.0
      self.error_loss_weight = 0.0
      
    elif self.training_mode == 'phoneme_error':
      # Perceived phoneme recognition + error classification
      self.canonical_loss_weight = 0.0
      weight_sum = self.perceived_loss_weight + self.error_loss_weight
      if abs(weight_sum - 1.0) > 1e-6:
        self.perceived_loss_weight /= weight_sum
        self.error_loss_weight /= weight_sum
        
    elif self.training_mode == 'multitask':
      # All three tasks: canonical, perceived, and error
      weight_sum = (
          self.canonical_loss_weight +
          self.perceived_loss_weight +
          self.error_loss_weight
      )
      if abs(weight_sum - 1.0) > 1e-6:
        self.canonical_loss_weight /= weight_sum
        self.perceived_loss_weight /= weight_sum
        self.error_loss_weight /= weight_sum
  
  def _initialize_paths(self):
    """Initializes directory paths and generates experiment name."""
    # Generate experiment name if not provided
    if self.experiment_name is None:
      timestamp = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
      cv_suffix = f'_cv{self.cv_fold_index}' if self.use_cross_validation else ''
      self.experiment_name = (
          f'{self.training_mode}_{self.model_type}{cv_suffix}_{timestamp}'
      )
    
    # Setup experiment directories
    self.experiment_dir = os.path.join(
        self.experiment_base_dir, self.experiment_name
    )
    self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
    self.log_dir = os.path.join(self.experiment_dir, 'logs')
    self.result_dir = os.path.join(self.experiment_dir, 'results')
    
    # Setup dataset paths
    if self.use_cross_validation:
      fold_dir = os.path.join(self.data_root_dir, f'fold_{self.cv_fold_index}')
      self.train_labels_path = os.path.join(fold_dir, 'train_labels.json')
      self.val_labels_path = os.path.join(fold_dir, 'val_labels.json')
    else:
      self.train_labels_path = os.path.join(
          self.data_root_dir, 'train_labels.json'
      )
      self.val_labels_path = os.path.join(self.data_root_dir, 'val_labels.json')
    
    self.test_labels_path = os.path.join(self.data_root_dir, 'test_labels.json')
    self.phoneme_mapping_path = os.path.join(
        self.data_root_dir, 'phoneme_to_id.json'
    )
  
  def get_model_architecture_config(self) -> Dict:
    """Gets architecture-specific configuration.
    
    Returns:
      Dictionary with model architecture parameters including transformer flag.
    """
    config = self.model_architecture_configs[self.model_type].copy()
    config['use_transformer'] = (self.model_type == 'transformer')
    return config
  
  def has_canonical_task(self) -> bool:
    """Checks if canonical phoneme prediction is enabled.
    
    Returns:
      True if canonical task is active in current training mode.
    """
    return self.training_mode == 'multitask'
  
  def has_perceived_task(self) -> bool:
    """Checks if perceived phoneme prediction is enabled.
    
    Returns:
      True if perceived task is active in current training mode.
    """
    return self.training_mode in ['phoneme_only', 'phoneme_error', 'multitask']
  
  def has_error_task(self) -> bool:
    """Checks if error classification is enabled.
    
    Returns:
      True if error task is active in current training mode.
    """
    return self.training_mode in ['phoneme_error', 'multitask']
  
  def save_to_file(self, output_path: str):
    """Saves configuration to JSON file.
    
    Args:
      output_path: Path where configuration will be saved.
    """
    config_dict = {
        attr: getattr(self, attr)
        for attr in dir(self)
        if not attr.startswith('_') and not callable(getattr(self, attr))
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
      json.dump(config_dict, f, indent=2)
  
  @staticmethod
  def get_error_label_names() -> Dict[int, str]:
    """Gets mapping from error type IDs to human-readable names.
    
    Returns:
      Dictionary mapping error IDs to error type names.
    """
    return {
        0: 'blank',
        1: 'deletion',
        2: 'insertion',
        3: 'substitution',
        4: 'correct'
    }
