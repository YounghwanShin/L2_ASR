import os
import json
from datetime import datetime
from dataclasses import dataclass
from pytz import timezone

@dataclass
class Config:
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 320000
    
    num_phonemes = 42
    num_error_types = 3
    
    task_mode = {'multi_train' : 'multi_train',
                 'milti_eval' : 'multi_eval',
                 'phoneme_train' : 'phoneme_train',
                 'phoneme_eval' : 'phoneme_eval',
                 'error_train' : 'error_train',
                 'error_eval' : 'error_eval'}
    error_task_ratio = 0.5
    
    model_type = 'simple'
    
    batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation = 2
    
    main_lr = 3e-4
    wav2vec_lr = 1e-5
    
    error_weight = 0.3
    phoneme_weight = 0.4
    length_weight = 0.3
    
    focal_alpha = 0.25
    focal_gamma = 2.0
    
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
        }
    }
    
    def __post_init__(self):
        if self.experiment_name is None or not hasattr(self, '_last_model_type') or self._last_model_type != self.model_type:
            current_date = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d')
            
            if hasattr(self, '_is_phoneme_model') and self._is_phoneme_model:
                model_prefix = 'phoneme_simple' if self.model_type == 'simple' else f'phoneme_{self.model_type}'
                self.experiment_name = f"{model_prefix}_{current_date}"
            else:
                model_prefix = 'simple' if self.model_type == 'simple' else 'trm' if self.model_type == 'transformer' else self.model_type
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.phoneme_weight * 10)).zfill(2)
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}_{current_date}"
            
            self._last_model_type = self.model_type
        
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
