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
    hidden_dim = 1024  # For wav2vec2-large
    num_phonemes = 42
    num_error_types = 3
    
    # Training parameters
    batch_size = 8
    wav2vec_lr = 1e-5      # Lower LR for pre-trained Wav2Vec2
    main_lr = 1e-4         # Higher LR for other modules
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

## Training Features

- **Sample Predictions**: Shows actual vs predicted results every 5 epochs
- **Detailed Metrics**: Token accuracy, weighted F1, class-wise precision/recall
- **Best Model Tracking**: Saves best models for error detection, phoneme recognition, and validation loss
- **Progress Monitoring**: Real-time loss tracking and evaluation progress bars

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
- **Centralized Configuration**: All settings in config.py
- **Optimizer Separation**: Different learning rates for Wav2Vec2 vs other modules
- **Mixed Precision Training**: FP16 forward pass with FP32 gradients
- **Gradient Accumulation**: Effective larger batch sizes with limited memory
- **Detailed Logging**: Sample predictions and comprehensive metrics
- **Multi-task Learning**: Joint error detection and phoneme recognition

## Output Examples

```
06/04/2025 08:02:42 - __main__ - INFO - Epoch 5 - Sample Predictions
06/04/2025 08:02:42 - __main__ - INFO - ==================================================

--- Multi-task Sample 1 ---
File: data/l2arctic_dataset/TLV/wav/arctic_a0126.wav
Error Actual:    correct correct incorrect correct correct
Error Predicted: correct correct correct incorrect correct
Phoneme Actual:    sil iy sh sil d ey s iy b iy k ey m
Phoneme Predicted: sil iy ch sil d ey s sh iy b iy k ah m

06/04/2025 08:02:58 - __main__ - INFO - Error Token Accuracy: 0.8499
06/04/2025 08:02:58 - __main__ - INFO - Error Weighted F1: 0.8028
06/04/2025 08:02:58 - __main__ - INFO -   incorrect: Precision=0.1464, Recall=0.0550, F1=0.0800
06/04/2025 08:02:58 - __main__ - INFO -   correct: Precision=0.8716, Recall=0.9523, F1=0.9102
06/04/2025 08:03:13 - __main__ - INFO - Phoneme Error Rate (PER): 0.1718
06/04/2025 08:03:13 - __main__ - INFO - Phoneme Accuracy: 0.8282
06/04/2025 08:03:18 - __main__ - INFO - New best error accuracy: 0.8499
06/04/2025 08:03:22 - __main__ - INFO - New best phoneme accuracy: 0.8282 (PER: 0.1718)
```

## Files
- `config.py`: All hyperparameters and paths
- `model.py`: Simplified multi-task model architecture
- `train.py`: Training script with detailed logging
- `eval.py`: Evaluation script
- `evaluate.py`: Evaluation functions
- `data_prepare.py`: Data loading and preprocessing