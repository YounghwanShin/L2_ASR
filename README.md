# L2 Speech Multi-Task Learning

이 프로젝트는 L2(제2언어) 화자의 실제 발음을 인식하는 다중 작업 학습 모델입니다.

## 주요 기능

- **Phoneme Recognition**: L2 화자의 실제 발음(perceived_train_target) 인식
- **Error Detection**: 발음 오류(error_labels) 감지
- **Multi-task Learning**: 두 작업을 동시에 학습
- **Wav2Vec2 기반**: Pre-trained Wav2Vec2 인코더 사용
- **자동 체크포인트**: 최고 성능 모델 자동 저장

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
pip install torch torchaudio speechbrain transformers hyperpyyaml
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
lr: 0.0001
lr_wav2vec: 0.00001

# 손실 가중치
phoneme_weight: 1.0
error_weight: 1.0

# 배치 크기
batch_size: 8

# 에포크 수
number_of_epochs: 30
```

## 특징

- ✅ 전체 데이터셋 사용 (샘플링 없음)
- ✅ 실시간 성능 모니터링
- ✅ 자동 베스트 모델 저장
- ✅ CTC loss를 통한 시퀀스 학습
- ✅ 실제 L2 발음 인식에 최적화

## 문제 해결

### CUDA 메모리 부족
```yaml
batch_size: 4  # 배치 크기 줄이기
```

### 느린 훈련
```yaml
num_workers: 0  # 데이터 로더 워커 수 줄이기
```

### 과적합
```yaml
weight_decay: 0.001  # 정규화 강화
```