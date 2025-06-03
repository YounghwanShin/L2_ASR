# Simple L2Arctic Multi-task Pronunciation Assessment

A simplified multi-task deep learning model for L2 pronunciation assessment.

## Features
- Simple error detection (pronunciation errors)
- Simple phoneme recognition  
- Wav2Vec2-based feature extraction
- CTC-based training
- No complex dependencies

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchaudio transformers pyyaml tqdm numpy
```

### 2. Prepare Data
Ensure your data folder contains:
- `train_data.json`
- `val_data.json` 
- `eval.json`
- `phoneme_to_id.json`

### 3. Train Model
```bash
python train.py multitask.yaml --data_folder ./data --output_folder ./results --device cuda
```

### 4. Evaluate Model
```bash
python evaluate.py multitask.yaml --output_folder ./eval_results --device cuda
```

### 5. Validate Dataset
```bash
python -c "from data_prepare import validate_l2arctic_data; validate_l2arctic_data('./data')"
```

## Configuration
Edit `multitask.yaml` to adjust:
- Model parameters (hidden_dim, dropout)
- Training hyperparameters (lr, epochs, batch_size)
- Data paths
- Task settings (error, phoneme, both)

## Output Structure
```
results/
├── save/
│   └── train.log
└── evaluation/
    └── evaluation_results.json
```