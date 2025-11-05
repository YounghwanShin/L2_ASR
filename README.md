# L2 Pronunciation Assessment with Multitask Learning

Deep learning system for assessing L2 learner pronunciation quality through multitask learning of phoneme recognition and error detection.

## Overview

This system simultaneously learns three related tasks:
- **Canonical Phoneme Recognition**: Predicting correct phonemes
- **Perceived Phoneme Recognition**: Recognizing actual pronunciations  
- **Error Detection**: Classifying errors as Deletion (D), Insertion (I), Substitution (S), or Correct (C)

## Key Features

- **Multitask Learning**: Joint training of related pronunciation assessment tasks
- **Flexible Architecture**: Simple feedforward or Transformer encoders
- **Cross-Validation**: Speaker-based k-fold validation for robust evaluation
- **Comprehensive Metrics**: Phoneme error rates, mispronunciation detection, error classification

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/l2-pronunciation-assessment.git
cd l2-pronunciation-assessment

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python setup_nltk.py
```

### Data Preparation

```bash
# Download L2-ARCTIC dataset
chmod +x download_dataset.sh
./download_dataset.sh

# Preprocess data (extract phonemes, generate error labels, create splits)
python preprocess.py all --data_root data/l2arctic --output_dir data
```

### Training

```bash
# Train with multitask learning (recommended)
python main.py train --training_mode multitask --model_type transformer

# Train specific fold
python main.py train --training_mode multitask --cv_fold 0

# Train without cross-validation
python main.py train --training_mode multitask --no_cv
```

### Evaluation

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## Training Modes

### Multitask (Recommended)
Jointly trains all three tasks for best performance:
```bash
python main.py train --training_mode multitask
```

### Phoneme-Error
Trains perceived phoneme recognition and error detection:
```bash
python main.py train --training_mode phoneme_error
```

### Phoneme Only
Trains only perceived phoneme recognition:
```bash
python main.py train --training_mode phoneme_only
```

## Model Architectures

### Transformer Encoder (Recommended)
Uses multi-head self-attention for contextual modeling:
```bash
python main.py train --model_type transformer
```

### Simple Encoder
Feedforward architecture for faster training:
```bash
python main.py train --model_type simple
```

## Configuration

Key settings in `l2pa/config.py`:

```python
# Training mode (main focus: multitask error learning)
training_mode = 'multitask'

# Model architecture
model_type = 'transformer'

# Loss weights
canonical_weight = 0.3
perceived_weight = 0.3
error_weight = 0.4

# Training hyperparameters
batch_size = 16
num_epochs = 100
gradient_accumulation = 2
main_lr = 3e-4
wav2vec_lr = 1e-5
```

## Cross-Validation

Speaker-based cross-validation ensures model generalization:

- **Test Set**: 6 fixed speakers (TLV, NJS, TNI, TXHC, ZHAA, YKWK)
- **Training Folds**: Each fold uses different speaker for validation
- **Automatic**: Runs all folds sequentially

```bash
# Train all folds
python main.py train --training_mode multitask

# Train specific fold
python main.py train --training_mode multitask --cv_fold 0
```

## Evaluation Metrics

### Phoneme Recognition
- Phoneme Error Rate (PER)
- Mispronunciation detection (Precision, Recall, F1)
- Per-speaker accuracy

### Error Detection
- Token-level accuracy
- Per-class F1 (Deletion, Insertion, Substitution, Correct)
- Weighted/Macro F1 scores

## Advanced Usage

### Custom Configuration
```bash
python main.py train \
    --training_mode multitask \
    --config "batch_size=32,num_epochs=150,main_lr=5e-4" \
    --experiment_name custom_experiment
```

### Resume Training
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

## Project Structure

```
l2-pronunciation-assessment/
├── main.py                     # Training/evaluation entry point
├── preprocess.py               # Data preprocessing entry point
├── requirements.txt
├── README.md
├── data/
│   ├── l2arctic/              # L2-ARCTIC dataset
│   ├── fold_X/                # Cross-validation folds
│   ├── test_labels.json       # Test set
│   └── phoneme_to_id.json     # Phoneme mapping
├── experiments/               # Training outputs
└── l2pa/                      # Main package
    ├── config.py              # Configuration
    ├── train.py               # Training logic
    ├── evaluate.py            # Evaluation logic
    ├── data/                  # Data loading
    ├── models/                # Model architectures
    ├── preprocessing/         # Data preprocessing
    ├── training/              # Training utilities
    ├── evaluation/            # Evaluation metrics
    └── utils/                 # Utility functions
```

## Citation

```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Assessment with Multitask Learning},
  author={L2PA Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/l2-pronunciation-assessment}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Wav2Vec2 model from Hugging Face Transformers
- L2-ARCTIC dataset
- CMU Pronouncing Dictionary
