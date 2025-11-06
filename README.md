# L2 발음 오류 탐지 시스템

제2 언어(L2) 학습자의 발음 오류를 자동으로 탐지하는 딥러닝 기반 시스템입니다. Wav2Vec2와 Transformer를 활용하여 학습자의 발음에서 삭제(Deletion), 삽입(Insertion), 치환(Substitution) 오류를 검출합니다.

## 주요 기능

- 발음 오류 자동 탐지 (D, I, S, C 분류)
- 학습자가 실제로 발음한 음소 인식
- 화자별 상세 성능 분석
- L2-ARCTIC 데이터셋 지원

## 설치

### 요구사항
- Python 3.8 이상
- CUDA GPU 권장
- 디스크 공간 20GB 이상

### 설치 과정

```bash
# 저장소 클론
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python setup_nltk.py
```

## 빠른 시작

### 1. 데이터셋 다운로드

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 2. 데이터 전처리

```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

이 명령어는 다음을 자동으로 수행합니다:
- TextGrid 파일에서 음소 정렬 추출
- 오류 레이블 생성
- 교차 검증 폴드 생성
- 텍스트 분리 스플릿 생성

### 3. 모델 훈련

**기본 훈련 (교차 검증):**
```bash
python main.py train --training_mode multitask --model_type transformer
```

**특정 폴드만 훈련:**
```bash
python main.py train --training_mode multitask --model_type transformer --cv_fold 0
```

**텍스트 분리 스플릿 훈련:**
```bash
python main.py train --training_mode multitask --model_type transformer --no_cv --data_split_mode disjoint
```

### 4. 모델 평가

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## 상세 사용법

### 훈련 모드

시스템은 세 가지 훈련 모드를 지원합니다:

**1. phoneme_only** - 음소 인식만 수행
```bash
python main.py train --training_mode phoneme_only --model_type transformer
```

**2. phoneme_error** - 음소 인식 + 오류 탐지 (권장)
```bash
python main.py train --training_mode phoneme_error --model_type transformer
```

**3. multitask** - 정규 음소, 인지 음소, 오류 탐지 모두 수행
```bash
python main.py train --training_mode multitask --model_type transformer
```

### 데이터 스플릿 옵션

**교차 검증 (기본):**
- 화자 기반 k-fold 교차 검증
- 모든 폴드 훈련: `--training_mode multitask`
- 특정 폴드 훈련: `--cv_fold 0`

**텍스트 분리 스플릿:**
- 훈련/검증/테스트 세트 간 텍스트 중복 없음
- 사용: `--no_cv --data_split_mode disjoint`

**표준 스플릿:**
- 단순 훈련/검증/테스트 분할
- 사용: `--no_cv --data_split_mode standard`

### 하이퍼파라미터 조정

```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --config "batch_size=32,num_epochs=150,main_lr=5e-4"
```

사용 가능한 주요 파라미터:
- `batch_size`: 배치 크기 (기본값: 16)
- `num_epochs`: 훈련 에폭 수 (기본값: 100)
- `main_lr`: 메인 모델 학습률 (기본값: 3e-4)
- `wav2vec_lr`: Wav2Vec2 학습률 (기본값: 1e-5)

### 훈련 재개

```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

### 실험 이름 지정

```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --experiment_name my_experiment
```

## 평가 지표

### 오류 탐지
- Token Accuracy: 전체 토큰 정확도
- Weighted F1 / Macro F1
- 클래스별 성능 (Deletion, Insertion, Substitution, Correct)

### 음소 인식
- Phoneme Error Rate (PER)
- 발음 오류 탐지: Precision, Recall, F1
- 화자별 성능

## 프로젝트 구조

```
L2_ASR/
├── main.py                      # 훈련/평가 진입점
├── preprocess.py                # 전처리 진입점
├── requirements.txt
├── download_dataset.sh
├── data/                        # 데이터 디렉토리
│   ├── l2arctic/               # 원본 데이터
│   ├── fold_X/                 # 교차 검증 폴드
│   ├── disjoint_wrd_split/     # 텍스트 분리 스플릿
│   └── test_labels.json
├── experiments/                 # 실험 결과
│   └── [실험명]/
│       ├── checkpoints/
│       ├── logs/
│       └── results/
└── l2pa/                       # 메인 패키지
    ├── config.py              # 설정
    ├── train.py               # 훈련 로직
    ├── evaluate.py            # 평가 로직
    ├── preprocessing/         # 전처리 모듈
    ├── data/                  # 데이터 로더
    ├── models/                # 모델 구조
    ├── training/              # 훈련 유틸
    ├── evaluation/            # 평가 메트릭
    └── utils/                 # 헬퍼 함수
```

## 출력 파일

### 체크포인트
`experiments/[실험명]/checkpoints/`
- `best_perceived.pth` - 최고 음소 인식 성능
- `best_error.pth` - 최고 오류 탐지 성능
- `best_canonical.pth` - 최고 정규 음소 성능
- `latest.pth` - 최신 체크포인트

### 로그 및 결과
- `experiments/[실험명]/logs/training.log` - 훈련 로그
- `evaluation_results/[실험명]_results.json` - 평가 결과

## 문제 해결

### CUDA 메모리 부족
```bash
python main.py train --config "batch_size=8,gradient_accumulation=4"
```

### 전처리 오류
단계별로 실행:
```bash
python preprocess.py extract --data_root data/l2arctic
python preprocess.py labels --input data/preprocessed.json
python preprocess.py split --input data/processed_with_error.json
```

### 데이터셋 찾을 수 없음
```bash
./download_dataset.sh
```

## 고급 사용법

### 개별 전처리 단계 실행

```bash
# 음소 추출만
python preprocess.py extract --data_root data/l2arctic --output data/preprocessed.json

# 오류 레이블 생성만
python preprocess.py labels --input data/preprocessed.json --output data/processed_with_error.json

# 교차 검증 스플릿만
python preprocess.py split --input data/processed_with_error.json --output_dir data

# 텍스트 분리 스플릿만
python preprocess.py split_disjoint --input data/processed_with_error.json --output_dir data
```

### 특정 체크포인트 평가

```bash
python main.py eval \
    --checkpoint experiments/my_experiment/checkpoints/best_error.pth \
    --training_mode multitask
```

### 모델 타입 지정

```bash
# Simple Encoder
python main.py train --model_type simple

# Transformer Encoder (권장)
python main.py train --model_type transformer
```

## 인용

이 코드를 연구에 사용하는 경우:

```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Error Detection with Deep Learning},
  author={Younghwan Shin},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YounghwanShin/L2_ASR}
}
```

## 라이선스

MIT License

## 참고

- L2-ARCTIC 데이터셋
- Hugging Face Transformers (Wav2Vec2)
- CMU Pronouncing Dictionary

---

문의사항은 [GitHub Issues](https://github.com/YounghwanShin/L2_ASR/issues)에 등록해주세요.