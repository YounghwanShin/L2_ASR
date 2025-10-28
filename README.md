# L2 Pronunciation Assessment

A state-of-the-art deep learning system for assessing pronunciation quality of second language learners. This system uses a Wav2Vec2-based unified architecture to perform phoneme recognition and error detection simultaneously.

## Features

- **Phoneme Recognition**: CTC-based phoneme sequence prediction with 42 phoneme classes
- **Error Detection**: Classification of pronunciation errors (deletion, insertion, substitution, correct)
- **Unified Training**: Multitask learning framework for joint optimization
- **Flexible Architecture**: Support for both Simple and Transformer-based encoders
- **Comprehensive Evaluation**: Per-speaker metrics and detailed error analysis

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU with 8GB+ memory (recommended)
- 16GB+ system RAM
- Linux, macOS, or Windows with WSL2

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/l2-pronunciation-assessment.git
cd l2-pronunciation-assessment
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup NLTK Data
```bash
python setup_nltk.py
```

## Quick Start

### 1. Download Dataset
```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 2. Preprocess Data

Run all preprocessing steps (extraction, error label generation, train/test split):
```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

Or run each step individually:
```bash
# Step 1: Extract phoneme data
python preprocess.py extract --data_root data/l2arctic --output data/preprocessed.json

# Step 2: Generate error labels
python preprocess.py labels --input data/preprocessed.json --output data/processed_with_error.json

# Step 3: Split into train/test sets
python preprocess.py split --input data/processed_with_error.json --output_dir data
```

### 3. Train Model

Train with default settings:
```bash
python main.py train --training_mode phoneme_error --model_type transformer
```

Train with custom configuration:
```bash
python main.py train \
    --training_mode phoneme_error \
    --model_type transformer \
    --config "batch_size=32,num_epochs=100,main_lr=5e-4" \
    --experiment_name my_experiment
```

Resume training from checkpoint:
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

### 4. Evaluate Model
```bash
python main.py eval --checkpoint experiments/my_experiment/checkpoints/best_phoneme.pth
```

With specific configuration:
```bash
python main.py eval \
    --checkpoint experiments/my_experiment/checkpoints/best_phoneme.pth \
    --training_mode phoneme_error \
    --save_predictions
```

## Project Structure
```
l2-pronunciation-assessment/
├── main.py                          # Main entry point for training/evaluation
├── preprocess.py                    # Preprocessing entry point
├── download_dataset.sh              # Dataset download script
├── requirements.txt                 # Python dependencies
├── setup_nltk.py                    # NLTK data setup
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── data/                            # Data directory
│   ├── l2arctic/                    # L2-ARCTIC dataset (downloaded)
│   ├── preprocessed.json            # Extracted phoneme data
│   ├── processed_with_error.json   # Data with error labels
│   ├── train_labels.json            # Training set
│   ├── val_labels.json              # Validation set
│   ├── test_labels.json             # Test set
│   └── phoneme_map.json             # Phoneme to ID mapping
├── experiments/                     # Experiment outputs
│   └── [experiment_name]/
│       ├── checkpoints/             # Model checkpoints
│       ├── logs/                    # Training logs
│       └── results/                 # Final metrics
├── evaluation_results/              # Evaluation outputs
└── l2pa/                           # Main package
    ├── __init__.py
    ├── config.py                    # Configuration
    ├── train.py                     # Training logic
    ├── evaluate.py                  # Evaluation logic
    ├── preprocessing/               # Preprocessing modules
    │   ├── preprocess_dataset.py
    │   ├── generate_labels.py
    │   └── split_data.py
    ├── data/                        # Data loading
    │   └── dataset.py
    ├── models/                      # Neural network models
    │   ├── encoders.py
    │   ├── heads.py
    │   ├── losses.py
    │   └── unified_model.py
    ├── training/                    # Training utilities
    │   ├── trainer.py
    │   └── utils.py
    ├── evaluation/                  # Evaluation metrics
    │   ├── evaluator.py
    │   └── metrics.py
    └── utils/                       # Utility functions
        └── audio.py
```

## Configuration

Key configuration parameters can be modified in `l2pa/config.py`:

### Model Architecture
```python
model_type = 'transformer'  # 'simple' or 'transformer'
hidden_dim = 1024
num_layers = 2              # For transformer
num_heads = 8               # For transformer
dropout = 0.1
```

### Training Hyperparameters
```python
batch_size = 16
num_epochs = 100
gradient_accumulation = 2
main_lr = 3e-4
wav2vec_lr = 1e-5
```

### Loss Configuration
```python
training_mode = 'phoneme_error'  # 'phoneme_only' or 'phoneme_error'
error_weight = 0.4
phoneme_weight = 0.6
focal_alpha = 0.25
focal_gamma = 2.0
```

## Training Modes

### Phoneme-Only Mode

Trains only phoneme recognition:
```bash
python main.py train --training_mode phoneme_only --model_type transformer
```

### Phoneme-Error Mode

Trains both phoneme recognition and error detection:
```bash
python main.py train --training_mode phoneme_error --model_type transformer
```

## Model Architectures

### Simple Encoder

Feed-forward architecture with two linear layers:
```bash
python main.py train --model_type simple
```

### Transformer Encoder

Multi-head self-attention with configurable layers:
```bash
python main.py train --model_type transformer
```

## Evaluation Metrics

### Phoneme Recognition

- **PER (Phoneme Error Rate)**: Overall phoneme prediction accuracy
- **Mispronunciation Detection**: Precision, recall, and F1 for error detection
- **Confusion Matrix**: True/False acceptance and rejection rates

### Error Classification

- **Token Accuracy**: Per-token error classification accuracy
- **Sequence Accuracy**: Percentage of perfectly predicted sequences
- **Per-Class Metrics**: F1, precision, recall for each error type (D, I, S, C)
- **Weighted/Macro F1**: Aggregate performance metrics

### Per-Speaker Analysis

Results are automatically computed for each speaker in the test set.

## Data Format

### Input JSON Structure
```json
{
  "data/l2arctic/SPEAKER/wav/file.wav": {
    "wav": "data/l2arctic/SPEAKER/wav/file.wav",
    "duration": 3.5,
    "spk_id": "SPEAKER",
    "canonical_aligned": "sil hh iy w ih l ...",
    "perceived_aligned": "sil hh iy w ih l ...",
    "canonical_train_target": "hh iy w ih l ...",
    "perceived_train_target": "hh iy w ih l ...",
    "error_labels": "C C C D S ...",
    "wrd": "He will follow us soon"
  }
}
```

### Error Label Types

- **C**: Correct pronunciation
- **D**: Deletion (phoneme missing from perceived speech)
- **I**: Insertion (extra phoneme in perceived speech)
- **S**: Substitution (phoneme replaced with different phoneme)

## Troubleshooting

### GPU Memory Issues

Reduce batch size or increase gradient accumulation:
```bash
python main.py train --config "batch_size=8,gradient_accumulation=4"
```

### Training Instability

Lower learning rate:
```bash
python main.py train --config "main_lr=1e-4,wav2vec_lr=5e-6"
```

### Overfitting

Increase dropout or adjust loss weights:
```bash
python main.py train --config "dropout=0.2,error_weight=0.5,phoneme_weight=0.5"
```

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Assessment with Unified Neural Architecture},
  author={Research Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/l2-pronunciation-assessment}
}
```

## Dataset Citation

This project uses the L2-ARCTIC dataset:
```bibtex
@inproceedings{zhao2018l2arctic,
  title={L2-ARCTIC: A non-native English speech corpus},
  author={Zhao, Guanlong and Sonsaat, Sinem and Silpachai, Alif and Lucic, Ivana and Chukharev-Hudilainen, Evgeny and Levis, John and Gutierrez-Osuna, Ricardo},
  booktitle={Interspeech},
  pages={2783--2787},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wav2Vec2 model from Hugging Face Transformers
- L2-ARCTIC dataset creators
- CMU Pronouncing Dictionary
- g2p-en library for grapheme-to-phoneme conversion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].