# L2 Pronunciation Assessment Model

Deep learning model for assessing pronunciation of second language learners. Uses a Wav2Vec2-based unified architecture to perform phoneme recognition and error detection simultaneously.

## Key Features

- **Phoneme Recognition**: CTC-based phoneme sequence prediction
- **Error Detection**: Classification of pronunciation error types (deletion, insertion, substitution, correct)
- **Unified Training**: Multitask learning for simultaneous optimization of both tasks
- **Flexible Architecture**: Support for both Simple and Transformer encoders

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Minimum 8GB GPU memory

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

Preprocess your data before training:

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

```bash
python data_preprocessing.py
```

This script performs:
- Generation of canonical_train_target from canonical_aligned
- Creation of error labels (D: deletion, I: insertion, S: substitution, C: correct)
- Sequence compression suitable for CTC characteristics

## Training

### Basic Training

```bash
python train.py --training_mode phoneme_error --model_type transformer
```

### Training Modes

- `phoneme_only`: Phoneme recognition only
- `phoneme_error`: Phoneme recognition + error detection

### Model Architectures

- `simple`: Basic feed-forward encoder
- `transformer`: Transformer-based encoder

### Advanced Options

```bash
python train.py \
    --training_mode phoneme_error \
    --model_type transformer \
    --config "batch_size=32,num_epochs=100,main_lr=5e-4" \
    --experiment_name my_experiment
```

### Resume Training

```bash
python train.py --resume ../shared/experiments/phoneme_error_transformer0406_20250915154806/checkpoints/latest.pth
```

## Evaluation

```bash
python eval.py --model_checkpoint experiments/my_experiment/checkpoints/best_phoneme.pth
```

### Evaluation Options

```bash
python eval.py \
    --model_checkpoint path/to/checkpoint.pth \
    --eval_data path/to/test_data.json \
    --batch_size 16 \
    --save_predictions
```

## Configuration

You can modify key settings in `config.py`:

- **Model Settings**: Hidden layer size, dropout, etc.
- **Training Settings**: Learning rate, batch size, number of epochs
- **Loss Weights**: Balance between phoneme and error losses
- **Data Paths**: Location of training/validation/test data

## Project Structure

```
l2_pronunciation_assessment/
├── config.py                    # Configuration file
├── train.py                     # Training script
├── eval.py                      # Evaluation script
├── data_preprocessing.py        # Data preprocessing
├── src/
│   ├── data/                    # Data processing modules
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/                  # Model-related modules
│   │   ├── unified_model.py
│   │   ├── encoders.py
│   │   ├── heads.py
│   │   └── losses.py
│   ├── training/                # Training-related modules
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── evaluation/              # Evaluation-related modules
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/                   # Utility modules
│       └── audio.py
├── data/                        # Data files
├── experiments/                 # Experiment results
└── evaluation_results/          # Evaluation results
```

## Data Format

### Input Data

JSON format with the following fields:

```json
{
  "path/to/audio.wav": {
    "wav": "path/to/audio.wav",
    "duration": 4.5,
    "spk_id": "SPEAKER_ID",
    "canonical_aligned": "sil hh iy w ih l ...",
    "perceived_aligned": "sil hh iy w ih l ...",
    "perceived_train_target": "sil hh iy w ih l ...",
    "canonical_train_target": "sil hh iy w ih l ...",
    "wrd": "He will follow us soon",
    "error_labels": "C C C C C C ..."
  }
}
```

### Error Labels

- `C`: Correct pronunciation
- `D`: Deletion (present in canonical but missing in perceived)
- `I`: Insertion (present in perceived but not in canonical)
- `S`: Substitution (pronounced as different phoneme)

## Evaluation Metrics

### Phoneme Recognition

- **PER (Phoneme Error Rate)**: Phoneme error rate
- **Phoneme Accuracy**: 1 - PER
- **Mispronunciation Detection**: Precision, recall, F1-score

### Error Detection

- **Token Accuracy**: Individual error label accuracy
- **Sequence Accuracy**: Proportion of completely correct sequences
- **Per-Class Metrics**: Performance for each error type

## Experiment Results

Experiment results are saved in:

- **Checkpoints**: `experiments/{experiment_name}/checkpoints/`
- **Logs**: `experiments/{experiment_name}/logs/`
- **Final Metrics**: `experiments/{experiment_name}/results/`
- **Evaluation Results**: `evaluation_results/`

## Model Architecture

### Encoder

1. **Wav2Vec2**: Pretrained speech representation learning
2. **Feature Encoder**: Speech feature enhancement
   - Simple: Feed-forward network
   - Transformer: Multi-head attention

### Output Heads

1. **Phoneme Recognition Head**: 42 phoneme classification
2. **Error Detection Head**: 5 error type classification

### Loss Functions

- **Focal CTC Loss**: Handles class imbalance
- **Weighted Combination**: Loss weighting for multitask learning

## Performance Optimization

### Memory Optimization

- Use gradient accumulation
- Mixed precision training
- Regular GPU memory cleanup

### Training Stability

- Separate learning rates (Wav2Vec2 vs other parts)
- Focus on hard samples with Focal Loss
- Data augmentation with SpecAugment

## Troubleshooting

### Common Issues

1. **GPU Memory Shortage**
   - Reduce batch size
   - Increase gradient accumulation

2. **Training Instability**
   - Lower learning rate
   - Adjust Focal Loss parameters

3. **Overfitting**
   - Increase dropout
   - Adjust regularization weights

### Log Inspection

Check logs during training issues:

```bash
tail -f experiments/{experiment_name}/logs/training.log
```

## Contributing

Please submit bug reports or feature suggestions through issues.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{l2pronunciation2024,
  title={L2 Pronunciation Assessment with Unified Neural Architecture},
  author={Research Team},
  year={2024},
  url={https://github.com/YounghwanShin/L2_ASR.git}
}
```
