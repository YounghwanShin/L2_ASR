import os
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config:
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 320000
    
    num_phonemes = 42
    num_error_types = 3
    
    task_mode = 'both'
    error_task_ratio = 0.5
    simultaneous_training = False
    
    model_type = 'simple'
    
    batch_size = 32
    eval_batch_size = 32
    num_epochs = 50
    gradient_accumulation = 2
    
    main_lr = 3e-4
    wav2vec_lr = 1e-5
    scheduler_factor = 0.5
    scheduler_patience = 3
    
    error_weight = 1.0
    phoneme_weight = 1.0
    
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True
    
    wav2vec2_specaug = True
    
    seed = 42
    
    base_experiment_dir = "experiments"
    experiment_name = None
    
    train_data = "data/train_labels.json"
    val_data = "data/val_labels.json" 
    eval_data = "data/test_labels.json"
    phoneme_map = "data/phoneme_map.json"
    
    device = "cuda"
    
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
        },
        'cross': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'cross_layers': 1,
            'dropout': 0.1
        },
        'hierarchical': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        }
    }
    
    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.experiment_name = f"{self.model_type}_{timestamp}"
        
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir
    
    def get_model_config(self):
        return self.model_configs.get(self.model_type, self.model_configs['simple'])
    
    def save_config(self, path):
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self) 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)