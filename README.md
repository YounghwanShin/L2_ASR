# L2 발음 오류 탐지 및 음소 인식 시스템

L2(제2언어) 학습자의 발음 오류를 탐지하고 정확한 음소 인식을 수행하는 이중 Wav2Vec 2.0 기반 모델입니다. 
이 시스템은 발음 오류 유형(정확, 삭제, 대체, 추가)을 탐지하고, 오류 정보를 활용하여 더 정확한 음소 인식을 수행합니다.

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

### 1단계: 오류 탐지 학습

```bash
python train.py --stage 1 --num_epochs 100 --batch_size 32 --learning_rate 4e-4 --use_scheduler
python train.py --stage 1 --num_epochs 100 --batch_size 32 --learning_rate 5e-5
```

### 2단계: 음소 인식 학습

```bash
python train.py --stage 2 --num_epochs 100 --batch_size 32 --learning_rate 4e-4 --model_checkpoint models/best_error_detection.pth --use_scheduler
python train.py --stage 2 --num_epochs 100 --batch_size 32 --learning_rate 5e-5 --model_checkpoint models/best_error_detection.pth
```

### 학습 인자 설명

#### 필수 인자:
- `--stage`: 학습 단계 (1: 오류 탐지, 2: 음소 인식)

#### 데이터 관련 인자:
- `--error_train_data`: 오류 탐지 학습 데이터 경로 (기본값: data/errors_train.json)
- `--error_val_data`: 오류 탐지 검증 데이터 경로 (기본값: data/errors_val.json)
- `--phoneme_train_data`: 음소 인식 학습 데이터 경로 (기본값: data/perceived_train.json)
- `--phoneme_val_data`: 음소 인식 검증 데이터 경로 (기본값: data/perceived_val.json)
- `--phoneme_map`: 음소-ID 매핑 파일 경로 (기본값: data/phoneme_to_id.json)
- `--max_audio_length`: 최대 오디오 길이(샘플 단위) 제한 (기본값: None)

#### 모델 관련 인자:
- `--pretrained_model`: 사전학습된 wav2vec2 모델 이름 (기본값: facebook/wav2vec2-base-960h)
- `--hidden_dim`: 은닉층 차원 크기 (기본값: 768)
- `--num_phonemes`: 음소 수 (기본값: 42)
- `--adapter_dim_ratio`: 어댑터 차원 비율 (기본값: 0.25)
- `--error_influence_weight`: 오류 영향 가중치 (기본값: 0.2)

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
- `--scheduler_threshold`: 개선으로 간주할 최소 변화량 (기본값: 0.001)
- `--scheduler_cooldown`: 감소 후 감시 재개 전 대기 에폭 수 (기본값: 1)
- `--scheduler_min_lr`: 최소 학습률 (기본값: 1e-6)

#### 출력 관련 인자:
- `--output_dir`: 모델 체크포인트 저장 디렉토리 (기본값: models/)
- `--result_dir`: 결과 로그 저장 디렉토리 (기본값: results/)
- `--model_checkpoint`: 로드할 사전 학습된 모델 체크포인트 경로 (기본값: None)
- `--evaluate_every_epoch`: 각 에폭마다 평가 진행 (플래그)

## 모델 평가

학습된 모델의 성능을 평가하려면 다음 명령어를 실행하세요:

```bash
python evaluate.py \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --model_checkpoint models/best_phoneme_recognition.pth \
  --output_dir evaluation_results \
  --detailed
```

### 평가 인자 설명

- `--eval_data`: 평가 데이터 JSON 파일 경로 (필수)
- `--phoneme_map`: 음소-ID 매핑 JSON 파일 경로 (필수)
- `--model_checkpoint`: 평가할 모델 체크포인트 경로 (필수)
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

### 이중 Wav2Vec 2.0
- **고정 Wav2Vec + 어댑터**: 기본 특징 추출
- **학습 가능 Wav2Vec**: 맥락별 특징 추출

### 오류 탐지 헤드
- 음소 단위로 오류 유형 분류 (정확, 삭제, 대체, 추가)

### 음소 인식 헤드
- 오류 정보를 활용한 향상된 음소 인식

### 오류 인식 기반 디코더
- 오류 유형에 따라 음소 확률 조정

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