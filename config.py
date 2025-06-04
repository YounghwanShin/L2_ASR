import torch
import os
import json
from datetime import datetime

class Config:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_folder = './data'
    train_data = f'{data_folder}/train_data.json'
    val_data = f'{data_folder}/val_data.json'
    eval_data = f'{data_folder}/eval.json'
    phoneme_map = f'{data_folder}/phoneme_to_id.json'
    
    model_type = 'simple'
    experiment_name = None
    base_experiment_dir = './experiments'
    
    pretrained_model = 'facebook/wav2vec2-large-xlsr-53'
    num_phonemes = 42
    num_error_types = 3
    
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
            'cross_attention_dim': 512,
            'dropout': 0.1
        },
        'hierarchical': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        }
    }
    
    batch_size = 32
    eval_batch_size = 32
    wav2vec_lr = 1e-5
    main_lr = 3e-4
    num_epochs = 30
    gradient_accumulation = 2
    
    error_weight = 1.0
    phoneme_weight = 1.0
    
    scheduler_factor = 0.5
    scheduler_patience = 3
    
    max_length = 320000
    sampling_rate = 16000
    
    task_mode = 'both'
    error_task_ratio = 0.5
    
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True
    
    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"{self.model_type}_{timestamp}"
        
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir
    
    @property
    def hidden_dim(self):
        return self.model_configs[self.model_type]['hidden_dim']
    
    @property
    def dropout(self):
        return self.model_configs[self.model_type]['dropout']
    
    def get_model_config(self):
        return self.model_configs[self.model_type]
    
    def save_config(self, path):
        config_dict = {
            'model_type': self.model_type,
            'experiment_name': self.experiment_name,
            'pretrained_model': self.pretrained_model,
            'batch_size': self.batch_size,
            'wav2vec_lr': self.wav2vec_lr,
            'main_lr': self.main_lr,
            'num_epochs': self.num_epochs,
            'gradient_accumulation': self.gradient_accumulation,
            'model_config': self.get_model_config()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)