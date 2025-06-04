import torch

class Config:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_folder = './data'
    train_data = f'{data_folder}/train_data.json'
    val_data = f'{data_folder}/val_data.json'
    eval_data = f'{data_folder}/eval.json'
    phoneme_map = f'{data_folder}/phoneme_to_id.json'
    
    pretrained_model = 'facebook/wav2vec2-large-xlsr-53'
    hidden_dim = 1024
    num_phonemes = 42
    num_error_types = 3
    
    batch_size = 16
    eval_batch_size = 16
    wav2vec_lr = 1e-5
    main_lr = 1e-4
    num_epochs = 30
    gradient_accumulation = 2
    
    error_weight = 1.0
    phoneme_weight = 1.0
    
    max_length = None
    sampling_rate = 16000
    task_mode = 'both'
    error_task_ratio = 0.5
    dropout = 0.1
    
    output_dir = 'models'
    result_dir = 'results'
    
    scheduler_patience = 3
    scheduler_factor = 0.5
    
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config