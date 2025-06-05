# Multi-Task L2 Pronunciation Assessment System

A multi-task learning framework for second language pronunciation assessment with progressive model architecture improvements.

## Overview

This repository contains a comprehensive system for evaluating second language pronunciation through simultaneous error detection and phoneme recognition. The framework implements multiple neural architectures with increasing complexity to establish baseline performance and explore architectural improvements. Additionally, it provides phoneme-only models for comparative analysis against multi-task approaches.

## Quick Start

```bash
# Multi-task model training (alternating mode - default)
python train.py

# Multi-task model training (simultaneous mode)
python train.py --config simultaneous_training=True

# Specific multi-task model training
python train.py --config model_type=transformer
python train.py --config model_type=cross
python train.py --config model_type=hierarchical

# Simultaneous training with specific model
python train.py --config model_type=transformer,simultaneous_training=True

# Phoneme-only model training
python phoneme_train.py --config model_type=simple
python phoneme_train.py --config model_type=transformer

# Model evaluation
python eval.py --model_checkpoint experiments/simple_*/checkpoints/best_phoneme.pth
python phoneme_eval.py --model_checkpoint experiments/phoneme_simple_*/checkpoints/best_phoneme.pth
```

## Project Structure

```
project/
├── experiments/                    # Experiment outputs
│   ├── simple_20250604_0802/      # Timestamped experiment directories
│   │   ├── checkpoints/           # Model checkpoints
│   │   ├── logs/                  # Training logs
│   │   ├── results/               # Evaluation results
│   │   └── config.json            # Experiment configuration
│   └── comparison_results/        # Cross-experiment comparisons
├── models/                        # Model implementations
│   ├── model.py                   # Simple multi-task model
│   ├── model_transformer.py       # Transformer multi-task model
│   ├── model_cross.py            # Cross-attention multi-task model
│   ├── model_hierarchical.py     # Hierarchical multi-task model
│   ├── phoneme_model.py          # Simple phoneme-only model
│   └── phoneme_model_transformer.py # Transformer phoneme-only model
├── config.py                      # Configuration management
├── train.py                       # Multi-task training script
├── eval.py                        # Multi-task evaluation script
├── phoneme_train.py               # Phoneme-only training script
├── phoneme_eval.py                # Phoneme-only evaluation script
├── phoneme_data_prepare.py        # Phoneme-only data processing
├── experiment_manager.py          # Experiment management utilities
├── compare_experiments.py         # Performance comparison tools
└── [other data processing modules]
```

## Model Architectures

### Multi-Task Models

| Model | Description | Features |
|-------|-------------|----------|
| `simple` | Baseline multi-task | Wav2Vec2 + Linear encoder |
| `transformer` | Enhanced multi-task | Self-attention mechanisms |
| `cross` | Cross-attention multi-task | Inter-task information exchange |
| `hierarchical` | Multi-level processing | Hierarchical feature extraction |

### Phoneme-Only Models

| Model | Description | Features |
|-------|-------------|----------|
| `phoneme_simple` | Baseline phoneme-only | Wav2Vec2 + Linear encoder |
| `phoneme_transformer` | Enhanced phoneme-only | Self-attention mechanisms |

## Training

### Multi-Task Training Commands

```bash
# Default simple model (alternating training)
python train.py

# Simultaneous training
python train.py --config simultaneous_training=True

# Transformer model
python train.py --config model_type=transformer

# Cross-attention model with simultaneous training
python train.py --config model_type=cross,simultaneous_training=True

# Hierarchical model
python train.py --config model_type=hierarchical
```

### Phoneme-Only Training Commands

```bash
# Simple phoneme-only model
python phoneme_train.py --config model_type=simple

# Transformer phoneme-only model
python phoneme_train.py --config model_type=transformer
```

### Parameter Configuration

```bash
# Training mode selection
python train.py --config simultaneous_training=True

# Batch size and epoch adjustment
python train.py --config model_type=transformer,batch_size=16,num_epochs=50

# Learning rate configuration
python train.py --config model_type=cross,main_lr=2e-4,wav2vec_lr=2e-5

# Combined parameters
python train.py --config model_type=hierarchical,simultaneous_training=True,batch_size=16

# Custom experiment naming
python train.py --config model_type=hierarchical,experiment_name=custom_experiment
```

### Data Path Override

```bash
python train.py \
  --train_data data/custom_train.json \
  --val_data data/custom_val.json \
  --eval_data data/custom_eval.json \
  --config model_type=transformer
```

## Evaluation

### Multi-Task Model Evaluation

```bash
# Automatic model type detection
python eval.py --model_checkpoint experiments/transformer_20250604_0834/checkpoints/best_phoneme.pth

# Explicit model type specification
python eval.py \
  --model_checkpoint path/to/model.pth \
  --model_type cross \
  --save_predictions
```

### Phoneme-Only Model Evaluation

```bash
# Automatic model type detection
python phoneme_eval.py --model_checkpoint experiments/phoneme_transformer_20250604_0834/checkpoints/best_phoneme.pth

# Explicit model type specification
python phoneme_eval.py \
  --model_checkpoint path/to/model.pth \
  --model_type transformer \
  --save_predictions
```

### Comparative Analysis

```bash
# Compare all experiments
python compare_experiments.py

# Compare specific experiments
python compare_experiments.py experiments/simple_* experiments/transformer_*

# Pattern-based comparison
python compare_experiments.py --pattern "experiments/*cross*"
```

## Experiment Management

### List Experiments

```bash
python experiment_manager.py list
```

### Cleanup Operations

```bash
# Remove experiments older than 7 days (preserve high-performance models)
python experiment_manager.py cleanup --days-old 7 --keep-best

# Pattern-based cleanup
python experiment_manager.py cleanup --pattern "test_*"
```

### Archive Important Experiments

```bash
python experiment_manager.py archive transformer_20250604_0834
```

## Configuration

The system uses a centralized configuration approach through `config.py`:

```python
class Config:
    # Model selection
    model_type = 'simple'  # simple, transformer, cross, hierarchical
    
    # Training parameters
    batch_size = 8
    wav2vec_lr = 1e-5      # Wav2Vec2 learning rate
    main_lr = 1e-4         # Other modules learning rate
    num_epochs = 30
    gradient_accumulation = 2
    
    # Architecture-specific configurations
    model_configs = {
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
        }
    }
```

## Performance Metrics

The system evaluates models using comprehensive metrics:

- **Error Detection**: Token accuracy, weighted F1-score, class-specific precision/recall
- **Phoneme Recognition**: Phoneme Error Rate (PER), sequence accuracy
- **Training Metrics**: Validation loss, convergence analysis

## Experimental Results Format

```
================================================================================
EXPERIMENT COMPARISON
================================================================================
                    Experiment     Model Type  Error Acc  Phoneme Acc    PER
      hierarchical_20250604       hierarchical    0.8734      0.8521  0.1479
         cross_20250604            cross          0.8687      0.8493  0.1507
   transformer_20250604        transformer        0.8621      0.8445  0.1555
        simple_20250604           simple          0.8499      0.8282  0.1718

Best Performance Models
==================================================
Error Detection: hierarchical_20250604 (0.8734)
Phoneme Recognition: hierarchical_20250604 (0.8521)
Validation Loss: cross_20250604 (1.1834)
```

## Key Features

- **Automated Experiment Management**: Timestamp-based directory organization
- **Model Type Inference**: Automatic detection from checkpoint paths
- **Mixed Precision Training**: Memory-efficient training with gradient scaling
- **Gradient Accumulation**: Effective large batch size training
- **Dual Learning Rates**: Separate optimization for Wav2Vec2 and task-specific modules
- **Comprehensive Evaluation**: Multi-metric assessment including token-level and sequence-level accuracy
- **Reproducible Experiments**: Complete configuration preservation and random seed control

## Usage Workflow

1. **Baseline Establishment**
   ```bash
   python train.py --config num_epochs=5
   ```

2. **Performance Assessment**
   ```bash
   python compare_experiments.py
   ```

3. **Architecture Exploration**
   ```bash
   python train.py --config model_type=transformer,num_epochs=5
   ```

4. **Full Training**
   ```bash
   python train.py --config model_type=hierarchical,num_epochs=30
   ```

## Requirements

See `requirements.txt` for complete dependency list. Key requirements include:
- PyTorch >= 1.9.0
- Transformers >= 4.11.0
- SpeechBrain
- scikit-learn
- torchaudio

## Data Format

The system expects JSON-formatted data with the following structure:
- Audio file paths as keys
- Error labels in format: 'C' (correct), 'I' (incorrect)
- Phoneme sequences for both perceived and canonical pronunciations

## Citation

If you use this system in your research, please cite the original work and methodology as appropriate for your academic context.