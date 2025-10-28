"""Configuration module for pronunciation assessment model.

This module defines hyperparameters, model settings, and paths for the
unified pronunciation assessment system with cross-validation support.
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
        pretrained_model: Pretrained Wav2Vec2 model identifier.
        sampling_rate: Audio sampling rate in Hz.
        max_length: Maximum audio length in samples.
        num_phonemes: Number of phoneme classes.
        num_error_types: Number of error types.
        training_mode: Training mode selection.
        model_type: Model architecture type.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        num_epochs: Number of training epochs.
        gradient_accumulation: Gradient accumulation steps.
        main_lr: Learning rate for main parameters.
        wav2vec_lr: Learning rate for Wav2Vec2.
        canonical_weight: Weight for canonical loss.
        perceived_weight: Weight for perceived loss.
        error_weight: Weight for error loss.
        focal_alpha: Focal loss alpha parameter.
        focal_gamma: Focal loss gamma parameter.
        use_cross_validation: Enable cross-validation.
        num_folds: Number of cross-validation folds.
        current_fold: Current fold index.
        save_best_canonical: Save best canonical checkpoint.
        save_best_perceived: Save best perceived checkpoint.
        save_best_error: Save best error checkpoint.
        save_best_loss: Save best validation loss checkpoint.
        wav2vec2_specaug: Enable SpecAugment.
        seed: Random seed for reproducibility.
    """
    
    # Model configuration
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 140000

    # Output dimensions
    num_phonemes = 42
    num_error_types = 5

    # Training mode: 'phoneme_only', 'phoneme_error', 'multitask'
    training_mode = 'multitask'

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

    # Loss weights (should sum to 1.0 for multitask mode)
    canonical_weight = 0.3
    perceived_weight = 0.3
    error_weight = 0.4

    # Focal loss parameters
    focal_alpha = 0.25
    focal_gamma = 2.0

    # Cross-validation settings
    use_cross_validation = False
    num_folds = 5
    current_fold = 0

    # Checkpoint saving options
    save_best_canonical = True
    save_best_perceived = True
    save_best_error = True
    save_best_loss = True

    # Other settings
    wav2vec2_specaug = True
    seed = 42

    # Directory paths
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
        """Performs post-initialization processing.
        
        Validates weights and generates experiment name based on configuration.
        """
        self._validate_weights()

        if self.experiment_name is None or not hasattr(self, '_last_model_type') or self._last_model_type != self.model_type:
            current_date = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d%H%M%S')

            # Generate experiment name
            if self.use_cross_validation:
                cv_suffix = f"_fold{self.current_fold}"
            else:
                cv_suffix = ""

            if self.training_mode == 'phoneme_only':
                model_prefix = f'phoneme_{self.model_type}'
                self.experiment_name = f"{model_prefix}_{current_date}{cv_suffix}"
            elif self.training_mode == 'phoneme_error':
                model_prefix = f'phoneme_error_{self.model_type}'
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.perceived_weight * 10)).zfill(2)
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}_{current_date}{cv_suffix}"
            elif self.training_mode == 'multitask':
                model_prefix = f'multitask_{self.model_type}'
                c_ratio = str(int(self.canonical_weight * 10)).zfill(2)
                p_ratio = str(int(self.perceived_weight * 10)).zfill(2)
                e_ratio = str(int(self.error_weight * 10)).zfill(2)
                self.experiment_name = f"{model_prefix}{c_ratio}{p_ratio}{e_ratio}_{current_date}{cv_suffix}"

            self._last_model_type = self.model_type

        # Set directory paths
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir

        # Update data paths for cross-validation
        if self.use_cross_validation:
            self.train_data = f"data/fold_{self.current_fold}_train.json"
            self.val_data = f"data/fold_{self.current_fold}_val.json"

    def _validate_weights(self):
        """Validates and normalizes loss weights.
        
        Ensures weights sum to 1.0 for multitask learning modes.
        """
        if self.training_mode == 'phoneme_only':
            pass
        elif self.training_mode == 'phoneme_error':
            total = self.perceived_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: Loss weights sum to {total} != 1.0")
                print("Auto-normalizing weights...")
                self.perceived_weight = self.perceived_weight / total
                self.error_weight = self.error_weight / total
                print(f"Normalized: perceived={self.perceived_weight:.3f}, error={self.error_weight:.3f}")
        elif self.training_mode == 'multitask':
            total = self.canonical_weight + self.perceived_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: Loss weights sum to {total} != 1.0")
                print("Auto-normalizing weights...")
                self.canonical_weight = self.canonical_weight / total
                self.perceived_weight = self.perceived_weight / total
                self.error_weight = self.error_weight / total
                print(f"Normalized: canonical={self.canonical_weight:.3f}, "
                      f"perceived={self.perceived_weight:.3f}, error={self.error_weight:.3f}")

    def get_model_config(self):
        """Returns model configuration dictionary.
        
        Returns:
            Dictionary containing architecture parameters and transformer flag.
        """
        config = self.model_configs.get(self.model_type, self.model_configs['simple']).copy()
        config['use_transformer'] = (self.model_type == 'transformer')
        return config

    def has_error_component(self):
        """Checks if training mode includes error detection.
        
        Returns:
            Boolean indicating presence of error detection component.
        """
        return self.training_mode in ['phoneme_error', 'multitask']

    def has_canonical_component(self):
        """Checks if training mode includes canonical phoneme recognition.
        
        Returns:
            Boolean indicating presence of canonical component.
        """
        return self.training_mode == 'multitask'

    def save_config(self, path):
        """Saves configuration to JSON file.
        
        Args:
            path: Output file path for configuration.
        """
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_error_mapping(self):
        """Returns error label to index mapping.
        
        Returns:
            Dictionary mapping error types to indices.
        """
        return {
            'blank': 0,
            'D': 1,
            'I': 2,
            'S': 3,
            'C': 4
        }

    def get_error_type_names(self):
        """Returns error index to name mapping.
        
        Returns:
            Dictionary mapping indices to error type names.
        """
        return {
            0: 'blank',
            1: 'deletion',
            2: 'insertion',
            3: 'substitution',
            4: 'correct'
        }
