# L2 발음 오류 탐지 및 음소 인식

이 프로젝트는 L2(제2언어) 학습자의 발음 오류를 탐지하고 음소를 인식하는 이중 wav2vec2 모델을 구현합니다. 모델은 2단계에 걸쳐 학습되며, 첫 번째 단계에서는 오류 탐지를, 두 번째 단계에서는 음소 인식을 수행합니다.

## 주요 특징

- **이중 wav2vec2 아키텍처**: 오류 탐지와 음소 인식을 위한 병렬 처리
- **2단계 학습 프로세스**: 
  - 1단계: 발음 오류 탐지 (deletion, substitution, insertion, correct)
  - 2단계: 오류 정보를 활용한 정확한 음소 인식
- **CTC 손실 함수**: 시퀀스 정렬 문제 해결
- **어댑터 기반 아키텍처**: 효율적인 모델 튜닝

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-pronunciation-error-detection.git
cd l2-pronunciation-error-detection
```

### 2. 가상 환경 설정

#### venv 사용 (Python 기본)
```bash
# 가상 환경 생성
python3 -m venv l2-pronunciation-env

# 가상 환경 활성화 (Linux/Mac)
source l2-pronunciation-env/bin/activate

# 가상 환경 활성화 (Windows)
l2-pronunciation-env\Scripts\activate
```

#### Conda 사용 (선택사항)
```bash
# 새로운 conda 환경 생성
conda create -n l2-pronunciation python=3.9

# conda 환경 활성화
conda activate l2-pronunciation
```

### 3. 필요 패키지 설치
```bash
# 가상 환경이 활성화된 상태에서 실행
pip install -r requirements.txt
```

### 4. 데이터 다운로드

프로젝트에 필요한 L2Arctic 데이터셋과 기타 데이터 파일을 다운로드합니다:

```bash
python download.py
```

이 스크립트는 다음을 다운로드합니다:
- L2Arctic 데이터셋 (24개 폴더)
- 오류 레이블 파일들 (7개 파일)
- 음소 인식 관련 데이터

## 모델 구조

```
DualWav2VecWithErrorAwarePhonemeRecognition
├── FrozenWav2VecWithAdapter (첫 번째 wav2vec2)
│   ├── Frozen Wav2Vec2 Model
│   └── Bottleneck Adapter
├── LearnableWav2Vec (두 번째 wav2vec2)
├── FeatureFusion (특징 융합)
├── ErrorDetectionHead (오류 탐지)
├── PhonemeRecognitionHead (음소 인식)
└── ErrorAwarePhonemeDecoder (오류 인식 결합)
```

## 학습 방법

### 1단계: 오류 탐지 학습

```bash
python train.py --stage 1 --num_epochs 10 --batch_size 8 --learning_rate 5e-5
```

### 2단계: 음소 인식 학습

```bash
python train.py --stage 2 --num_epochs 10 --batch_size 8 --learning_rate 5e-5 \
    --model_checkpoint models/best_error_detection.pth
```

### 주요 학습 인자

#### 필수 인자:
- `--stage`: 학습 단계 (1: 오류 탐지, 2: 음소 인식)

#### 데이터 관련 인자:
- `--error_train_data`: 오류 탐지 학습 데이터 경로 (기본값: `data/errors_train.json`)
- `--error_val_data`: 오류 탐지 검증 데이터 경로 (기본값: `data/errors_val.json`)
- `--phoneme_train_data`: 음소 인식 학습 데이터 경로 (기본값: `data/perceived_train.json`)
- `--phoneme_val_data`: 음소 인식 검증 데이터 경로 (기본값: `data/perceived_val.json`)
- `--phoneme_map`: 음소-ID 매핑 파일 경로 (기본값: `data/phoneme_to_id.json`)
- `--max_audio_length`: 최대 오디오 길이(샘플 단위) 제한 (기본값: None)

#### 모델 관련 인자:
- `--pretrained_model`: 사전학습된 wav2vec2 모델 이름 (기본값: `facebook/wav2vec2-base-960h`)
- `--hidden_dim`: 은닉층 차원 크기 (기본값: 768)
- `--num_phonemes`: 음소 수 (기본값: 42)
- `--adapter_dim_ratio`: 어댑터 차원 비율 (기본값: 0.25)
- `--unfreeze_top_percent`: 상위 레이어 언프리징 비율 (기본값: 0.5)
- `--error_influence_weight`: 오류 영향 가중치 (기본값: 0.2)

#### 학습 관련 인자:
- `--batch_size`: 배치 크기 (기본값: 8)
- `--learning_rate`: 학습률 (기본값: 5e-5)
- `--num_epochs`: 학습 에폭 수 (기본값: 10)
- `--seed`: 랜덤 시드 (기본값: 42)
- `--device`: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)
- `--max_grad_norm`: 그라디언트 클리핑을 위한 최대 노름값 (기본값: 1.0)

#### 출력 관련 인자:
- `--output_dir`: 모델 체크포인트 저장 디렉토리 (기본값: `models/`)
- `--result_dir`: 결과 로그 저장 디렉토리 (기본값: `results/`)
- `--model_checkpoint`: 로드할 사전 학습된 모델 체크포인트 경로 (기본값: None)

## 프로젝트 구조

```
l2-pronunciation-error-detection/
├── model.py                  # 모델 아키텍처 정의
├── train.py                  # 학습 스크립트
├── download.py              # 데이터 다운로드 스크립트
├── requirements.txt         # 필요 패키지 목록
├── .gitignore              # Git 제외 파일 목록
├── README.md               # 프로젝트 설명서
│
├── data/                   # 데이터 디렉토리
│   ├── l2arctic_dataset/   # L2Arctic 데이터셋
│   ├── errors_train.json   # 오류 탐지 학습 데이터
│   ├── errors_val.json     # 오류 탐지 검증 데이터
│   ├── perceived_train.json # 음소 인식 학습 데이터
│   ├── perceived_val.json  # 음소 인식 검증 데이터
│   └── phoneme_to_id.json  # 음소-ID 매핑
│
├── models/                 # 모델 체크포인트
│   ├── best_error_detection.pth
│   ├── best_phoneme_recognition.pth
│   ├── last_error_detection.pth
│   └── last_phoneme_recognition.pth
│
└── results/                # 학습 결과 로그
    ├── train_stage1.log
    ├── train_stage2.log
    ├── hyperparams_stage1.json
    ├── hyperparams_stage2.json
    ├── error_detection_epoch*.json
    └── phoneme_recognition_epoch*.json
```

## 오류 유형

모델은 다음 4가지 오류 유형을 탐지합니다:

1. **Deletion (D)**: 발음해야 할 음소가 누락된 경우
2. **Substitution (S)**: 의도된 음소와 다른 음소로 발음된 경우
3. **Insertion (A)**: 추가적인 음소가 삽입된 경우
4. **Correct (C)**: 정확하게 발음된 경우

## GPU/CPU 설정

```bash
# GPU 사용 (CUDA가 설치된 경우)
python train.py --stage 1 --device cuda

# CPU 사용
python train.py --stage 1 --device cpu

# 자동 선택 (기본값)
python train.py --stage 1
```

## 학습 모니터링

학습 진행 상황은 다음 위치에서 확인할 수 있습니다

- **콘솔 출력**: 각 에폭의 손실값과 진행률
- **로그 파일**: `results/train_stage*.log`
- **에폭별 결과**: `results/*_epoch*.json`