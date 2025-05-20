# L2 발음 오류 탐지 및 음소 인식 시스템

L2(제2언어) 학습자의 발음 오류를 탐지하고 정확한 음소 인식을 수행하는 이중 Wav2Vec 2.0 기반 시스템입니다. 
이 프로젝트는 두 개의 독립적인 모델을 제공합니다:
1. 발음 오류 유형(정확, 삭제, 대체, 추가)을 탐지하는 오류 탐지 모델
2. 음소 시퀀스를 인식하는 음소 인식 모델

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-pronunciation-error-detection.git
cd l2-pronunciation-error-detection
```

### 2. 가상 환경 설정
```bash
# 가상 환경 생성
python3 -m venv .env

# 가상 환경 활성화 (Linux/Mac)
source .env/bin/activate

# 가상 환경 활성화 (Windows)
.env\Scripts\activate
```

### 3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

## 데이터 준비

프로젝트에 필요한 데이터를 다운로드하기 위해 다음 스크립트를 실행하세요.

```bash
python download.py
```

## 모델 학습

두 모델은 독립적으로 학습됩니다. 각 모델의 학습에 대한 설명은 아래와 같습니다.

### 오류 탐지 모델 학습

```bash
python train.py --model_type error_detection --train_data data/errors_train.json --val_data data/errors_val.json --num_epochs 100 --batch_size 32 --learning_rate 4e-4 --use_scheduler
```

### 음소 인식 모델 학습

```bash
python train.py --model_type phoneme_recognition --train_data data/perceived_train.json --val_data data/perceived_val.json --phoneme_map data/phoneme_to_id.json --num_epochs 100 --batch_size 32 --learning_rate 4e-4 --use_scheduler
```

### 학습 인자 설명

#### 필수 인자:
- `--model_type`: 학습할 모델 유형 (`error_detection` 또는 `phoneme_recognition`)
- `--train_data`: 학습 데이터 JSON 파일 경로
- `--val_data`: 검증 데이터 JSON 파일 경로

#### 데이터 관련 인자:
- `--phoneme_map`: 음소-ID 매핑 파일 경로 (음소 인식 모델에 필요)
- `--max_audio_length`: 최대 오디오 길이(샘플 단위) 제한 (기본값: None)

#### 모델 관련 인자:
- `--pretrained_model`: 사전학습된 wav2vec2 모델 이름 (기본값: facebook/wav2vec2-base-960h)
- `--hidden_dim`: 은닉층 차원 크기 (기본값: 768)
- `--num_phonemes`: 음소 수 (음소 인식 모델에 필요, 기본값: 42)
- `--num_error_types`: 오류 유형 수 (오류 탐지 모델에 필요, 기본값: 5)
- `--adapter_dim_ratio`: 어댑터 차원 비율 (기본값: 0.25)

#### 학습 관련 인자:
- `--batch_size`: 배치 크기 (기본값: 8)
- `--learning_rate`: 학습률 (기본값: 5e-5)
- `--num_epochs`: 학습 에폭 수 (기본값: 10)
- `--seed`: 랜덤 시드 (기본값: 42)
- `--device`: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)
- `--max_grad_norm`: 그라디언트 클리핑을 위한 최대 노름값 (기본값: 0.5)

#### 학습률 스케줄러 관련 인자:
- `--use_scheduler`: 학습률 스케줄러(ReduceLROnPlateau) 사용 여부 (플래그)
- `--scheduler_patience`: 학습률 감소 전 기다릴 에폭 수 (기본값: 2)
- `--scheduler_factor`: 학습률 감소 비율 (기본값: 0.5, 즉 50% 감소)

#### 출력 관련 인자:
- `--output_dir`: 모델 체크포인트 저장 디렉토리 (기본값: models/)
- `--result_dir`: 결과 로그 저장 디렉토리 (기본값: results/)
- `--model_checkpoint`: 로드할 사전 학습된 모델 체크포인트 경로 (기본값: None)

## 모델 평가

각 모델은 독립적으로 평가할 수 있습니다.

### 오류 탐지 모델 평가

```bash
python evaluate.py \
  --model_type error_detection \
  --eval_data data/eval.json \
  --model_checkpoint models/best_error_detection.pth \
  --output_dir evaluation_results \
  --detailed
```

### 음소 인식 모델 평가

```bash
python evaluate.py \
  --model_type phoneme_recognition \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --model_checkpoint models/best_phoneme_recognition.pth \
  --output_dir evaluation_results \
  --detailed
```

### 평가 인자 설명

- `--model_type`: 평가할 모델 유형 (`error_detection` 또는 `phoneme_recognition`) (필수)
- `--eval_data`: 평가 데이터 JSON 파일 경로 (필수)
- `--phoneme_map`: 음소-ID 매핑 JSON 파일 경로 (음소 인식 모델에 필요)
- `--model_checkpoint`: 평가할 모델 체크포인트 경로 (필수)
- `--num_error_types`: 오류 유형 수 (오류 탐지 모델에 필요, 기본값: 5)
- `--num_phonemes`: 음소 수 (음소 인식 모델에 필요, 기본값: 42)
- `--output_dir`: 평가 결과 저장 디렉토리 (기본값: evaluation_results)
- `--detailed`: 상세 결과 출력 (샘플별 결과, 혼동 행렬 등)
- `--batch_size`: 배치 크기 (기본값: 8)
- `--device`: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)

## 평가 결과 해석

평가 결과는 다음 지표를 포함합니다:

### 오류 탐지 평가
- 전체 정확도
- 오류 유형별 정밀도, 재현율, F1 점수
- 오류 유형별 혼동 행렬

### 음소 인식 평가
- 음소 오류율(PER)
- 총 오류 수 및 유형별 오류 수(삽입, 삭제, 대체)
- 샘플별 상세 인식 결과

## 모델 구조

이 시스템은 두 개의 독립적인 모델로 구성됩니다:

### 1. 오류 탐지 모델
- **이중 Wav2Vec 2.0 인코더**:
  - 고정된 Wav2Vec + 어댑터: 효율적인 파라미터 학습
  - 학습 가능한 Wav2Vec: 도메인 적응력 향상
- **특징 융합 모듈**: 두 인코더의 특징을 결합
- **오류 탐지 헤드**: 음소 단위로 오류 유형 분류 (정확, 삭제, 대체, 추가)

### 2. 음소 인식 모델
- **이중 Wav2Vec 2.0 인코더**:
  - 고정된 Wav2Vec + 어댑터: 효율적인 파라미터 학습
  - 학습 가능한 Wav2Vec: 도메인 적응력 향상
- **특징 융합 모듈**: 두 인코더의 특징을 결합
- **음소 인식 헤드**: CTC 손실을 사용한 음소 시퀀스 인식

## 데이터 형식

시스템이 기대하는 JSON 데이터 형식은 다음과 같습니다:

```json
{
  "wav_file_path.wav": {
    "wav": "wav_file_path.wav",
    "duration": 3.39,
    "spk_id": "SPEAKER_ID",
    "canonical_aligned": "sil iy ch sil d ey ...",
    "perceived_aligned": "sil iy sh sil d ey ...",
    "perceived_train_target": "sil iy sh sil d ey ...",
    "wrd": "Each day she became a more vital part of him",
    "error_labels": "C C S C C C S C C C C C C D ..."
  }
}
```

- `canonical_aligned`: 정규 발음 음소 시퀀스
- `perceived_aligned`: 실제 인식된 음소 시퀀스
- `perceived_train_target`: 학습에 사용할 인식된 음소 시퀀스
- `error_labels`: 음소 단위 오류 레이블 (C: 정확, D: 삭제, S: 대체, A/I: 추가/삽입)