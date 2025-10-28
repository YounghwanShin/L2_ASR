"""Configuration module for pronunciation assessment model.

This module defines all hyperparameters, model settings, and paths for the
unified pronunciation assessment system.
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass
from pytz import timezone


@dataclass
class Config:
    """Configuration for model training and evaluation.
    
    Attributes:
        pretrained_model: Name of the pretrained Wav2Vec2 model.
        sampling_rate: Audio sampling rate in Hz.
        max_length: Maximum audio length in samples.
        num_phonemes: Number of phoneme classes.
        num_error_types: Number of error types (blank, D, I, S, C).
        training_mode: Training mode ('phoneme_only' or 'phoneme_error').
        model_type: Model architecture type ('simple' or 'transformer').
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        num_epochs: Number of training epochs.
        gradient_accumulation: Number of gradient accumulation steps.
        main_lr: Learning rate for main model parameters.
        wav2vec_lr: Learning rate for Wav2Vec2 parameters.
        error_weight: Weight for error detection loss.
        phoneme_weight: Weight for phoneme recognition loss.
        focal_alpha: Alpha parameter for Focal Loss.
        focal_gamma: Gamma parameter for Focal Loss.
        save_best_error: Whether to save best error detection checkpoint.
        save_best_phoneme: Whether to save best phoneme recognition checkpoint.
        save_best_loss: Whether to save best validation loss checkpoint.
        wav2vec2_specaug: Whether to use SpecAugment.
        seed: Random seed for reproducibility.
    """
    
    # Model configuration
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 140000

    # Output dimensions
    num_phonemes = 42
    num_error_types = 5  # blank(0), D(1), I(2), S(3), C(4)

    # Training mode: 'phoneme_only' or 'phoneme_error'
    training_mode = 'phoneme_error'

    # Model architecture: 'simple' or 'transformer'
    model_type = 'transformer'

    # Training hyperparameters
    batch_size = 16
    eval_batch_size = 16
    num_epochs = 100
    gradient_accumulation = 2

    # Learning rates
    main_lr = 3e-4
    wav2vec_lr = 1e-5

    # Loss weights (error_weight + phoneme_weight = 1.0 for multitask learning)
    error_weight = 0.4
    phoneme_weight = 0.6

    # Focal Loss parameters
    focal_alpha = 0.25
    focal_gamma = 2.0

    # Checkpoint saving options
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True

    # Other settings
    wav2vec2_specaug = True
    seed = 42

    # Directory and file paths
    base_experiment_dir = "experiments"
    experiment_name = None

    train_data = "data/train_labels.json"
    val_data = "data/val_labels.json"
    eval_data = "data/test_labels.json"
    phoneme_map = "data/phoneme_map.json"

    device = "cuda"

    # Model architecture configurations
    model_configs = {
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
    }

    def __post_init__(self):
        """Post-initialization processing.
        
        Validates weights and generates experiment name based on training mode
        and model type.
        """
        self._validate_weights()

        if self.experiment_name is None or not hasattr(self, '_last_model_type') or self._last_model_type != self.model_type:
            current_date = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d%H%M%S')

            # Generate experiment name based on training mode and model type
            if self.training_mode == 'phoneme_only':
                model_prefix = f'phoneme_{self.model_type}'
                self.experiment_name = f"{model_prefix}_{current_date}"
            elif self.training_mode == 'phoneme_error':
                model_prefix = f'phoneme_error_{self.model_type}'
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.phoneme_weight * 10)).zfill(2)
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}_{current_date}"

            self._last_model_type = self.model_type

        # Set directory paths
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir

    def _validate_weights(self):
        """Validates and normalizes loss weights.
        
        For multitask learning (phoneme_error mode), ensures that phoneme_weight
        and error_weight sum to 1.0.
        """
        if self.training_mode == 'phoneme_only':
            pass
        elif self.training_mode == 'phoneme_error':
            total = self.phoneme_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: phoneme_weight ({self.phoneme_weight}) + error_weight ({self.error_weight}) = {total} != 1.0")
                print("Auto-normalizing weights...")
                self.phoneme_weight = self.phoneme_weight / total
                self.error_weight = self.error_weight / total
                print(f"Normalized weights: phoneme_weight={self.phoneme_weight:.3f}, error_weight={self.error_weight:.3f}")

    def get_model_config(self):
        """Returns model configuration with transformer flag.
        
        Returns:
            Dictionary containing architecture-specific parameters and use_transformer flag.
        """
        config = self.model_configs.get(self.model_type, self.model_configs['simple']).copy()
        config['use_transformer'] = (self.model_type == 'transformer')
        return config

    def has_error_component(self):
        """Checks if current training mode includes error detection.
        
        Returns:
            True if training mode includes error detection, False otherwise.
        """
        return self.training_mode == 'phoneme_error'

    def save_config(self, path):
        """Saves configuration to JSON file.
        
        Args:
            path: Path where configuration will be saved.
        """
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_error_mapping(self):
        """Returns error label mapping.
        
        Returns:
            Mapping from error type strings to integer indices.
        """
        return {
            'blank': 0,
            'D': 1,
            'I': 2,
            'S': 3,
            'C': 4
        }

    def get_error_type_names(self):
        """Returns error type name mapping.
        
        Returns:
            Mapping from integer indices to error type names.
        """
        return {
            0: 'blank',
            1: 'deletion',
            2: 'insertion',
            3: 'substitution',
            4: 'correct'
        }