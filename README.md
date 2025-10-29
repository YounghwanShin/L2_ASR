# L2 Pronunciation Assessment with Cross-Validation

State-of-the-art deep learning system for assessing pronunciation quality of second language learners, featuring multitask learning and speaker-based cross-validation.

## Features

- **Multitask Learning**: Simultaneous canonical phoneme recognition, perceived phoneme recognition, and error detection
- **Cross-Validation**: Robust evaluation with speaker-based k-fold cross-validation
- **Flexible Architecture**: Support for Simple and Transformer encoders
- **Comprehensive Metrics**: Per-speaker analysis, confusion matrices, and detailed error breakdowns

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/l2-pronunciation-assessment.git
cd l2-pronunciation-assessment

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup NLTK
python setup_nltk.py
```

## Quick Start

### 1. Download Dataset

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 2. Preprocess Data

Run all preprocessing steps including cross-validation split creation:

```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

This creates:
- `test_labels.json`: Fixed test set (speakers: TLV, NJS, TNI, TXHC, ZHAA, YKWK)
- `fold_X/train_labels.json`: Training set for fold X
- `fold_X/val_labels.json`: Validation set for fold X (one speaker each)
- `phoneme_map.json`: Phoneme to ID mapping

### 3. Train Model

**Train all cross-validation folds:**

```bash
python main.py train --training_mode multitask --model_type transformer
```

**Train specific fold:**

```bash
python main.py train --training_mode multitask --model_type transformer --cv_fold 0
```

**Train without cross-validation:**

```bash
python main.py train --training_mode multitask --model_type transformer --no_cv
```

### 4. Evaluate Model

```bash
python main.py eval --checkpoint experiments/[experiment_name]/checkpoints/best_perceived.pth
```

## Training Modes

### 1. Phoneme Only Mode
Trains only perceived phoneme recognition:
```bash
python main.py train --training_mode phoneme_only
```

### 2. Phoneme-Error Mode
Trains perceived phoneme recognition + error detection:
```bash
python main.py train --training_mode phoneme_error
```

### 3. Multitask Mode (Recommended)
Trains all three tasks: canonical phonemes, perceived phonemes, and error detection:
```bash
python main.py train --training_mode multitask
```

## Model Architectures

### Simple Encoder
Feed-forward architecture:
```bash
python main.py train --model_type simple
```

### Transformer Encoder (Recommended)
Multi-head self-attention:
```bash
python main.py train --model_type transformer
```

## Configuration

Key parameters in `l2pa/config.py`:

```python
# Training mode
training_mode = 'multitask'  # 'phoneme_only', 'phoneme_error', or 'multitask'

# Model architecture
model_type = 'transformer'  # 'simple' or 'transformer'

# Loss weights (for multitask mode)
canonical_weight = 0.3
perceived_weight = 0.3
error_weight = 0.4

# Training hyperparameters
batch_size = 16
num_epochs = 100
gradient_accumulation = 2
main_lr = 3e-4
wav2vec_lr = 1e-5

# Cross-validation
use_cross_validation = True
```

## Cross-Validation Details

The system implements speaker-based cross-validation:

1. **Test Set**: Fixed 6 speakers (TLV, NJS, TNI, TXHC, ZHAA, YKWK)
2. **Training Speakers**: Remaining speakers split into folds
3. **Validation**: Each fold uses one training speaker for validation
4. **Number of Folds**: Equals number of training speakers

Example with 18 total speakers:
- Test: 6 speakers (fixed)
- Training: 12 speakers
- Folds: 12 (each using 11 for train, 1 for validation)

## Evaluation Metrics

### Canonical Phoneme Recognition
- Phoneme Error Rate (PER)
- Per-speaker accuracy

### Perceived Phoneme Recognition
- Phoneme Error Rate (PER)
- Mispronunciation Detection: Precision, Recall, F1
- Confusion Matrix

### Error Detection
- Token Accuracy
- Per-class F1 scores (Deletion, Insertion, Substitution, Correct)
- Weighted F1 / Macro F1

## Advanced Usage

**Custom configuration:**
```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --config "batch_size=32,num_epochs=150,main_lr=5e-4" \
    --experiment_name my_experiment
```

**Resume training:**
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

**Evaluate specific fold:**
```bash
python main.py eval \
    --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## Project Structure

```
l2-pronunciation-assessment/
├── main.py                      # Main entry point
├── preprocess.py                # Preprocessing entry point
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── data/                        # Data directory
│   ├── l2arctic/                # L2-ARCTIC dataset
│   ├── fold_X/                  # CV fold data
│   ├── test_labels.json         # Test set
│   └── phoneme_map.json         # Phoneme mapping
├── experiments/                 # Experiment outputs
└── l2pa/                        # Main package
    ├── config.py                # Configuration
    ├── train.py                 # Training logic
    ├── evaluate.py              # Evaluation logic
    ├── preprocessing/           # Preprocessing modules
    ├── data/                    # Data loading
    ├── models/                  # Model architectures
    ├── training/                # Training utilities
    ├── evaluation/              # Evaluation metrics
    └── utils/                   # Utility functions
```

## Citation

```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Assessment with Multitask Learning and Cross-Validation},
  author={Research Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/l2-pronunciation-assessment}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Wav2Vec2 model from Hugging Face Transformers
- L2-ARCTIC dataset creators
- CMU Pronouncing Dictionary
