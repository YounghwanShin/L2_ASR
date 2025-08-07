## Data Format

Your data should be organized as JSON files with the following structure:

```json
{
  "path/to/audio1.wav": {
    "error_labels": "C I C C I",
    "perceived_train_target": "ae n t iy k",
    "canonical_aligned": "ae n t iy k", 
    "spk_id": "US"
  },
  "path/to/audio2.wav": {
    "error_labels": "C C I C",
    "perceived_train_target": "hh ae p iy",
    "canonical_aligned": "hh ae p iy",
    "spk_id": "CN"
  }
}
```

**Label Descriptions:**
- `error_labels`: C (Correct), I (Incorrect) - space-separated
- `perceived_train_target`: Phonemes as actually pronounced
- `canonical_aligned`: Reference canonical phonemes
- `spk_id`: Speaker country/group identifier

## Quick Start

### 1. Prepare Your Data

Create the required directory structure:
```bash
mkdir -p data
# Place your JSON files in data/
# - train_labels.json
# - val_labels.json  
# - test_labels.json
# - phoneme_map.json
```

Create `phoneme_map.json`:
```json
{
  "sil": 0,
  "aa": 1,
  "ae": 2,
  "ah": 3,
  ...
}
```

### 2. Training

**Simple Model (Recommended for first try):**
```bash
python multitask_train.py
```

**Transformer Model:**
```bash
python multitask_train.py --config model_type=transformer
```

**Custom Configuration:**
```bash
python multitask_train.py \
  --train_data data/my_train.json \
  --val_data data/my_val.json \
  --eval_data data/my_test.json \
  --experiment_name my_experiment \
  --config batch_size=8,num_epochs=30
```

**Resume Training:**
```bash
python multitask_train.py --resume experiments/multi_transformer0304_20250807/checkpoints/latest.pth
```

### 3. Evaluation

```bash
python multitask_eval.py \
  --model_checkpoint experiments/simple0304_20241201/checkpoints/best_phoneme.pth \
  --batch_size 16
```

**With Custom Data:**
```bash
python multitask_eval.py \
  --model_checkpoint path/to/model.pth \
  --eval_data data/custom_test.json \
  --phoneme_map data/custom_phoneme_map.json \
  --save_predictions
```


## Configuration Options

Key parameters in `config.py`:

```python
# Model Architecture
model_type = 'simple'  # or 'transformer'
hidden_dim = 1024
num_layers = 2  # for transformer
num_heads = 8   # for transformer

# Training
batch_size = 16
num_epochs = 50
main_lr = 3e-4
wav2vec_lr = 1e-5

# Loss Weights
error_weight = 0.3
phoneme_weight = 0.4
focal_alpha = 0.25
focal_gamma = 2.0

# Data Augmentation
wav2vec2_specaug = True
```

## Training Output

The system creates organized experiment directories:

```
experiments/
├── simple0304_20241201/           # Auto-generated name
│   ├── checkpoints/
│   │   ├── best_error.pth        # Best error detection model
│   │   ├── best_phoneme.pth      # Best phoneme recognition model
│   │   ├── best_loss.pth         # Best validation loss model
│   │   └── latest.pth            # Latest checkpoint
│   ├── logs/
│   │   └── training.log          # Training logs
│   └── results/
│       └── final_metrics.json    # Final training metrics
```

### Multi-GPU Training
The framework automatically uses DataParallel when multiple GPUs are available.

### Task-Specific Training
```bash
# Error detection only
python multitask_train.py --config "task_mode=error,error_task_ratio=1.0"

# Phoneme recognition only  
python multitask_train.py --config "task_mode=phoneme,error_task_ratio=0.0"
```

### Experiment Naming
```bash
python multitask_train.py --experiment_name my_custom_experiment_v2
```

## Sample Training Output

```
Epoch 10: Train Loss: 2.1456, Val Loss: 2.3421
Error Token Accuracy: 0.7834
Error Weighted F1: 0.7623
Phoneme Error Rate (PER): 0.2341
Phoneme Accuracy: 0.7659
New best phoneme accuracy: 0.7659 (PER: 0.2341)

--- Multi-task Sample 1 ---
File: /path/to/audio.wav
Error Actual:    C I C C
Error Predicted: C I I C
Phoneme Actual:    ae n t iy
Phoneme Predicted: ae n t iy
```

## Evaluation Results

Results are saved to `evaluation_results/` with detailed metrics:

```json
{
  "config": {
    "model_type": "simple",
    "evaluation_date": "2024-12-01 15:30:45"
  },
  "evaluation_results": {
    "error_detection": {
      "token_accuracy": 0.7834,
      "weighted_f1": 0.7623,
      "by_country": {
        "US": {"token_accuracy": 0.8012},
        "CN": {"token_accuracy": 0.7456}
      }
    },
    "phoneme_recognition": {
      "per": 0.2341,
      "mispronunciation_f1": 0.6789,
      "by_country": {...}
    }
  }
}
```