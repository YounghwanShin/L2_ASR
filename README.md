# Simplified Multi-task L2 Pronunciation Assessment

## Configuration

All hyperparameters are managed in `config.py`. Modify the Config class to adjust training parameters:

```python
class Config:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data paths
    data_folder = './data'
    train_data = f'{data_folder}/train_data.json'
    val_data = f'{data_folder}/val_data.json'
    eval_data = f'{data_folder}/eval.json'
    phoneme_map = f'{data_folder}/phoneme_to_id.json'
    
    # Model parameters
    pretrained_model = 'facebook/wav2vec2-large-xlsr-53'
    hidden_dim = 512
    num_phonemes = 42
    num_error_types = 3
    
    # Training parameters
    batch_size = 8
    wav2vec_lr = 1e-5
    main_lr = 1e-4
    num_epochs = 30
    gradient_accumulation = 2
```

## Training

Basic training (uses config.py defaults):
```bash
python train.py
```

Override data paths:
```bash
python train.py \
  --train_data data/custom_train.json \
  --val_data data/custom_val.json \
  --eval_data data/custom_eval.json \
  --phoneme_map data/custom_phoneme_map.json
```

Override hyperparameters:
```bash
python train.py \
  --config batch_size=16,num_epochs=50,wav2vec_lr=5e-6
```

## Evaluation

Basic evaluation:
```bash
python eval.py --model_checkpoint models/best_phoneme.pth
```

With custom config:
```bash
python eval.py \
  --model_checkpoint models/best_phoneme.pth \
  --eval_data data/test.json \
  --phoneme_map data/phoneme_to_id.json
```

## Key Features
- Centralized configuration management
- Simplified multi-task architecture
- Separated optimizers for Wav2Vec2 and other modules
- Mixed precision training
- Gradient accumulation

## Files
- `config.py`: All hyperparameters and paths
- `model.py`: Model architecture
- `train.py`: Training script
- `eval.py`: Evaluation script
- `evaluate.py`: Evaluation functions
- `data_prepare.py`: Data loading and preprocessing