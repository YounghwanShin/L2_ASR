# L2 Speech Multi-Task Learning (Improved)

이 프로젝트는 L2(제2언어) 화자의 실제 발음을 인식하는 다중 작업 학습 모델입니다. 성능 개선을 위한 여러 고급 기법들이 적용되었습니다.

## 주요 기능

- **Phoneme Recognition**: L2 화자의 실제 발음(perceived_train_target) 인식
- **Error Detection**: 발음 오류(error_labels) 감지
- **Multi-task Learning**: 두 작업을 동시에 학습
- **Wav2Vec2 기반**: Pre-trained Wav2Vec2 인코더 사용
- **자동 체크포인트**: 최고 성능 모델 자동 저장

## 성능 개선 사항

- **Attention Mask**: Wav2Vec2에 attention mask 적용으로 패딩 처리 개선
- **Audio Pipeline**: Wav2Vec2 feature extractor 사용한 고품질 전처리
- **CTC Decoder**: SpeechBrain의 최적화된 CTC greedy decoder 사용
- **Data Sorting**: Duration 기반 정렬로 효율적인 배치 처리
- **Spec Augmentation**: 훈련/검증 시 명시적 제어
- **Gradient Accumulation**: 메모리 효율적인 대배치 시뮬레이션
- **Mixed Precision**: 선택적 mixed precision training 지원
- **Optimizer 분리**: Wav2Vec2와 다른 파라미터에 별도 학습률 적용

## 데이터 형식

### Train/Validation 데이터
```json
{
  "audio_file_path": {
    "wav": "path/to/audio.wav",
    "duration": 4.135,
    "spk_id": "SPEAKER_ID",
    "canonical_aligned": "정답 음소 시퀀스",
    "perceived_aligned": "인식된 음소 시퀀스",
    "perceived_train_target": "훈련 타겟 음소 시퀀스",
    "wrd": "전사 텍스트",
    "error_labels": "C C I C C ..."
  }
}
```

### Test 데이터
```json
{
  "audio_file_path": {
    "wav": "path/to/audio.wav",
    "duration": 3.39,
    "spk_id": "SPEAKER_ID", 
    "canonical_aligned": "정답 음소 시퀀스",
    "perceived_aligned": "인식된 음소 시퀀스",
    "perceived_train_target": "훈련 타겟 음소 시퀀스",
    "wrd": "전사 텍스트",
    "error_labels": "C C I C C ..."
  }
}
```

## 설치

```bash
# 가상환경 생성
python -m venv env
source env/bin/activate  # Linux/Mac
# 또는 env\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 준비
데이터를 다음 구조로 배치:
```
data/
├── train_data.json
├── val_data.json
└── eval.json
```

### 2. 훈련 실행
```bash
# 전체 훈련 (phoneme + error detection)
python train.py multitask.yaml --data_folder ./data --output_folder ./results --device cuda

# Phoneme recognition만
python train.py multitask.yaml --data_folder ./data --output_folder ./results --device cuda task=phoneme

# Error detection만  
python train.py multitask.yaml --data_folder ./data --output_folder ./results --device cuda task=error

# 데이터 정렬 방식 변경
python train.py multitask.yaml --device cuda sorting=descending

# Gradient accumulation 사용
python train.py multitask.yaml --device cuda gradient_accumulation=4

# Mixed precision training
python train.py multitask.yaml --device cuda auto_mix_prec=True
```

### 3. 평가 실행
```bash
# 최고 PER 모델로 평가
python evaluate.py multitask.yaml ./results/save/best_phoneme_per.ckpt

# 최고 정확도 모델로 평가
python evaluate.py multitask.yaml ./results/save/best_error_acc.ckpt

# 최종 에포크 모델로 평가
python evaluate.py multitask.yaml ./results/save/epoch_30.ckpt
```

## 모델 구조

1. **Wav2Vec2 Encoder**: Pre-trained된 facebook/wav2vec2-base 사용
   - Attention mask 지원
   - Feature extractor를 통한 정규화
2. **Multi-Task Head**: 
   - Phoneme Recognition Head (CTC loss)
   - Error Detection Head (CTC loss)

## 평가 지표

- **Phoneme Error Rate (PER)**: 실제 발음 대비 인식 오류율
- **Error Detection Accuracy**: 발음 오류 감지 정확도

## 출력 파일

훈련 후 `./results/save/` 폴더에 다음 체크포인트들이 저장됩니다:

- `best_loss.ckpt`: 최고 validation loss
- `best_phoneme_per.ckpt`: 최고 phoneme PER
- `best_error_acc.ckpt`: 최고 error detection accuracy
- `epoch_N.ckpt`: 각 에포크별 모델

## 설정 변경

`multitask.yaml` 파일에서 하이퍼파라미터 조정 가능:

```yaml
# 작업 선택
task: both  # "phoneme", "error", "both"

# 학습률
lr: 0.0003
lr_wav2vec: 0.00001

# 손실 가중치
phoneme_weight: 1.0
error_weight: 1.0

# 배치 크기
batch_size: 4

# 에포크 수
number_of_epochs: 30

# 데이터 정렬
sorting: ascending  # "ascending", "descending", "random"

# Gradient accumulation
gradient_accumulation: 1

# Mixed precision
auto_mix_prec: False

# Spec augmentation
wav2vec2_specaug: True
```

## 특징

- ✅ 전체 데이터셋 사용 (샘플링 없음)
- ✅ 실시간 성능 모니터링
- ✅ 자동 베스트 모델 저장
- ✅ CTC loss를 통한 시퀀스 학습
- ✅ 실제 L2 발음 인식에 최적화
- ✅ Attention mask 기반 개선된 처리
- ✅ 고품질 오디오 전처리
- ✅ 효율적인 데이터 로딩
- ✅ Mixed precision training 지원
- ✅ Gradient accumulation 지원

## 문제 해결

### CUDA 메모리 부족
```yaml
batch_size: 2  # 배치 크기 줄이기
gradient_accumulation: 4  # Gradient accumulation 늘리기
auto_mix_prec: True  # Mixed precision 사용
```

### 느린 훈련
```yaml
num_workers: 0  # 데이터 로더 워커 수 줄이기
sorting: ascending  # 데이터 정렬로 효율성 향상
```

### 과적합
```yaml
weight_decay: 0.01  # 정규화 강화
wav2vec2_specaug: True  # Spec augmentation 사용
```

### 성능 향상 팁
- `sorting: ascending` 사용으로 훈련 효율성 향상
- `gradient_accumulation > 1` 사용으로 효과적인 큰 배치 크기 시뮬레이션  
- `auto_mix_prec: True` 사용으로 메모리 절약 및 속도 향상
- `wav2vec2_specaug: True` 사용으로 일반화 성능 향상