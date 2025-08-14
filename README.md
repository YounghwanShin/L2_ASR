# 통합 L2 발음 평가 시스템

## 프로젝트 구조

```
project/
├── config.py                 # 통합 설정 파일
├── train.py                  # 통합 학습 스크립트
├── eval.py                   # 통합 평가 스크립트
├── data_prepare.py           # 통합 데이터 처리
├── utils.py                  # 통합 유틸리티 함수
├── models/
│   ├── __init__.py
│   ├── model.py              # 통합 모델 (Simple/Transformer)
│   ├── loss_functions.py     # 통합 손실 함수
│   └── utils_models.py       # 모델 구성 요소
└── data/
    ├── train_labels.json
    ├── val_labels.json
    ├── test_labels.json
    └── phoneme_map.json
```

## 사용법

### 1. 음소 인식만 학습 (Phoneme Only)

```bash
# Simple 모델
python train.py --training_mode phoneme_only --model_type simple

# Transformer 모델  
python train.py --training_mode phoneme_only --model_type transformer
```

### 2. 음소 인식 + 오류 탐지 학습 (Phoneme + Error)

```bash
# Simple 모델
python train.py --training_mode phoneme_error --model_type simple

# Transformer 모델
python train.py --training_mode phoneme_error --model_type transformer
```

### 3. 음소 인식 + 오류 탐지 + 길이 손실 학습 (Full Multi-task)

```bash
# Simple 모델
python train.py --training_mode phoneme_error_length --model_type simple

# Transformer 모델
python train.py --training_mode phoneme_error_length --model_type transformer
```

### 4. 하이퍼파라미터 조정

```bash
# 손실 가중치 조정
python train.py \
    --training_mode phoneme_error_length \
    --model_type transformer \
    --config "error_weight=0.4,phoneme_weight=0.4,length_weight=0.2" \
    --experiment_name "custom_experiment"

# 데이터 경로 지정
python train.py \
    --training_mode phoneme_error \
    --train_data "data/custom_train.json" \
    --val_data "data/custom_val.json" \
    --phoneme_map "data/custom_phoneme_map.json"
```

### 5. 학습 재개

```bash
python train.py --resume "experiments/experiment_name/checkpoints/latest.pth"
```

### 6. 모델 평가

```bash
# 기본 평가 (자동으로 모델 타입 감지)
python eval.py --model_checkpoint "experiments/experiment_name/checkpoints/best_phoneme.pth"

# 학습 모드 명시적 지정
python eval.py \
    --model_checkpoint "path/to/checkpoint.pth" \
    --training_mode phoneme_error_length \
    --model_type transformer
```