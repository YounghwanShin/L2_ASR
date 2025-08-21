import os
import json
from datetime import datetime
from dataclasses import dataclass
from pytz import timezone

@dataclass
class Config:
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 160000

    num_phonemes = 42
    num_error_types = 3

    # Training modes: 'phoneme_only', 'phoneme_error', 'phoneme_error_length'
    training_mode = 'phoneme_error_length'

    # Model architecture: 'simple' or 'transformer'
    model_type = 'simple'

    # Training hyperparameters
    batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation = 2

    main_lr = 3e-4
    wav2vec_lr = 1e-5

    # Loss weights (error + phoneme = 1.0 for multitask, length separate)
    error_weight = 0.35
    phoneme_weight = 0.65
    length_weight = 0.02

    # Focal loss parameters
    focal_alpha = 0.25
    focal_gamma = 2.0

    # Checkpoint saving options
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True

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

    # Model configurations
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
        # Validate weight sums based on training mode
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
            elif self.training_mode == 'phoneme_error_length':
                model_prefix = f'phoneme_error_length_{self.model_type}'
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.phoneme_weight * 10)).zfill(2)
                length_ratio = str(int(self.length_weight * 100)).zfill(2)
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}l{length_ratio}_{current_date}"

            self._last_model_type = self.model_type

        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.length_logs_dir = os.path.join(self.experiment_dir, 'length_logs')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir

    def _validate_weights(self):
        if self.training_mode == 'phoneme_only':
            pass
        elif self.training_mode in ['phoneme_error', 'phoneme_error_length']:
            # phoneme_weight + error_weight should equal 1.0
            total = self.phoneme_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: phoneme_weight ({self.phoneme_weight}) + error_weight ({self.error_weight}) = {total} != 1.0")
                print("Auto-normalizing weights...")
                self.phoneme_weight = self.phoneme_weight / total
                self.error_weight = self.error_weight / total
                print(f"Normalized weights: phoneme_weight={self.phoneme_weight:.3f}, error_weight={self.error_weight:.3f}")
                if self.training_mode == 'phoneme_error_length':
                    print(f"Length weight (separate penalty): {self.length_weight:.3f}")

    def get_model_config(self):
        """Get model configuration and set use_transformer based on model_type"""
        config = self.model_configs.get(self.model_type, self.model_configs['simple']).copy()
        config['use_transformer'] = (self.model_type == 'transformer')
        return config

    def has_error_component(self):
        """Check if current training mode includes error detection"""
        return self.training_mode in ['phoneme_error', 'phoneme_error_length']

    def has_length_component(self):
        """Check if current training mode includes length loss"""
        return self.training_mode == 'phoneme_error_length'

    def save_config(self, path):
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)