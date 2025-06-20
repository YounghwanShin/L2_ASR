# Multi-task L2 Pronunciation Assessment

다중 과제 학습 기반 L2 발음 평가 모델

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

```
data/
├── train_labels.json
├── val_labels.json  
├── test_labels.json
└── phoneme_map.json
```

## 기본 사용법

### 1. Multi-task 모델 학습

```bash
# 기본 학습 (Simple 모델)
python train.py

# Transformer 모델
python train.py --config model_type=transformer

# Hierarchical 모델  
python train.py --config model_type=hierarchical

# 가중치 조정
python train.py --config error_weight=0.5,phoneme_weight=0.5
```

### 2. Phoneme-only 모델 학습

```bash
# Simple Phoneme 모델
python phoneme_train.py

# Transformer Phoneme 모델
python phoneme_train.py --config model_type=transformer
```

### 3. 학습 이어서 하기

```bash
# Multi-task 모델 resume
python train.py --resume experiments/simple0406/checkpoints/best_phoneme.pth

# Phoneme 모델 resume
python phoneme_train.py --resume experiments/phoneme_simple/checkpoints/latest.pth

# 실험명 직접 지정하여 resume
python train.py --resume path/to/checkpoint.pth --experiment_name custom_name
```

### 4. 모델 평가

```bash
# Multi-task 모델 평가
python eval.py --model_checkpoint experiments/trm0406/checkpoints/best_phoneme.pth

# Phoneme 모델 평가  
python phoneme_eval.py --model_checkpoint experiments/phoneme_transformer/checkpoints/best_phoneme.pth

# 예측 결과 저장
python eval.py --model_checkpoint path/to/model.pth --save_predictions
```

## 실험 결과

학습된 모델은 `experiments/` 에 자동으로 저장됩니다:

- `simple0406/`: Simple 모델, error=0.4, phoneme=0.6
- `trm0505/`: Transformer 모델, error=0.5, phoneme=0.5  
- `phoneme_simple/`: Phoneme-only Simple 모델
- `phoneme_transformer/`: Phoneme-only Transformer 모델

각 폴더 구조:
```
experiments/simple0406/
├── checkpoints/
│   ├── best_error.pth
│   ├── best_phoneme.pth
│   ├── best_loss.pth
│   └── latest.pth
├── logs/training.log
└── results/final_metrics.json
```

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|-------|
| `model_type` | simple, transformer, hierarchical | simple |
| `error_weight` | Error detection 가중치 | 0.4 |
| `phoneme_weight` | Phoneme recognition 가중치 | 0.6 |
| `batch_size` | 배치 크기 | 16 |
| `num_epochs` | 에포크 수 | 50 |
| `--resume` | 체크포인트에서 이어서 학습 | - |
| `--experiment_name` | 실험명 직접 지정 | auto |

## 빠른 시작

```bash
# 1. 기본 모델 학습
python train.py

# 2. 결과 확인
python eval.py --model_checkpoint experiments/simple0406/checkpoints/best_phoneme.pth

# 3. Phoneme 전용 모델 비교
python phoneme_train.py
python phoneme_eval.py --model_checkpoint experiments/phoneme_simple/checkpoints/best_phoneme.pth

# 4. 학습 중단 후 재개
python train.py --resume experiments/simple0406/checkpoints/latest.pth
```