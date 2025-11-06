# L2 발음 오류 탐지 시스템

제2 언어(L2) 학습자의 발음 오류를 자동으로 탐지하고 분석하는 딥러닝 기반 시스템입니다. L2-ARCTIC 데이터셋을 사용하여 발음 오류(삭제, 삽입, 치환)를 검출합니다.

## 주요 기능

- **발음 오류 탐지**: 삭제(D), 삽입(I), 치환(S), 정답(C) 분류
- **음소 인식**: 학습자가 실제로 발음한 음소 인식
- **화자별 분석**: 화자 단위 상세 평가 제공
- **교차 검증**: 화자 기반 k-fold 교차 검증 지원

## 설치

### 요구사항
- Python 3.8 이상
- CUDA GPU (권장)
- 디스크 공간 20GB 이상

### 설치 과정

```bash
# 1. 저장소 클론
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. NLTK 데이터 다운로드
python setup_nltk.py
```

## 사용 방법

### 1단계: 데이터셋 다운로드

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

생성되는 파일:
- `data/l2arctic/` - L2-ARCTIC 데이터셋
- `data/phoneme_to_id.json` - 음소 매핑 파일

### 2단계: 데이터 전처리

**전체 전처리 실행 (권장):**
```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

이 명령어는 다음을 수행합니다:
1. TextGrid 파일에서 음소 정렬 추출
2. 오류 레이블(D, I, S, C) 생성
3. 교차 검증용 폴드 생성
4. 텍스트 분리 스플릿 생성

**생성되는 데이터 구조:**
```
data/
├── fold_0/
│   ├── train_labels.json
│   └── val_labels.json
├── fold_1/
│   └── ...
├── test_labels.json           # 고정 테스트 세트 (6명 화자)
└── disjoint_wrd_split/        # 텍스트 분리 스플릿
    ├── train_labels.json
    ├── val_labels.json
    └── test_labels.json
```

**개별 전처리 단계 실행:**
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

### 3단계: 모델 훈련

**기본 훈련 (모든 교차 검증 폴드):**
```bash
python main.py train --training_mode multitask --model_type transformer
```

**특정 폴드만 훈련:**
```bash
python main.py train --training_mode multitask --model_type transformer --cv_fold 0
```

**교차 검증 없이 훈련:**
```bash
python main.py train --training_mode multitask --model_type transformer --no_cv
```

**텍스트 분리 스플릿으로 훈련:**

`l2pa/config.py` 파일을 수정:
```python
# 교차 검증 비활성화
use_cross_validation = False

# 데이터 경로 수정
train_data = 'data/disjoint_wrd_split/train_labels.json'
val_data = 'data/disjoint_wrd_split/val_labels.json'
test_data = 'data/disjoint_wrd_split/test_labels.json'
```

그 후 훈련:
```bash
python main.py train --training_mode multitask --model_type transformer
```

### 4단계: 모델 평가

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## 훈련 모드

### 1. phoneme_only
인지된 음소 인식만 수행:
```bash
python main.py train --training_mode phoneme_only
```

### 2. phoneme_error (오류 탐지 권장)
인지된 음소 인식 + 오류 탐지:
```bash
python main.py train --training_mode phoneme_error
```

### 3. multitask
정규 음소, 인지 음소, 오류 탐지 모두 수행:
```bash
python main.py train --training_mode multitask
```

## 모델 아키텍처

### Simple Encoder
피드포워드 구조:
```bash
python main.py train --model_type simple
```

### Transformer Encoder (권장)
멀티헤드 셀프 어텐션:
```bash
python main.py train --model_type transformer
```

## 설정

`l2pa/config.py`에서 주요 파라미터 조정:

```python
# 훈련 모드
training_mode = 'multitask'  # 'phoneme_only', 'phoneme_error', 'multitask'

# 모델 타입
model_type = 'transformer'  # 'simple' 또는 'transformer'

# 하이퍼파라미터
batch_size = 16
num_epochs = 100
main_lr = 3e-4
wav2vec_lr = 1e-5

# 손실 가중치 (multitask 모드)
canonical_weight = 0.3
perceived_weight = 0.3
error_weight = 0.4
```

**커맨드라인에서 설정 오버라이드:**
```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --config "batch_size=32,num_epochs=150,main_lr=5e-4"
```

## 데이터 스플릿 종류

### 1. 교차 검증 스플릿 (기본)
- 화자 기반 k-fold 교차 검증
- 테스트: 고정 6명 화자
- 폴드별로 1명씩 검증용으로 사용
- 위치: `data/fold_0/`, `data/fold_1/`, ...

### 2. 텍스트 분리 스플릿
- 훈련/검증/테스트 세트 간 텍스트 중복 없음
- 정규 음소 정보 누출 방지
- 위치: `data/disjoint_wrd_split/`
- 사용법: `config.py`에서 경로 수정 후 훈련

## 평가 지표

### 오류 탐지
- Token Accuracy: 전체 토큰 정확도
- Weighted F1 / Macro F1
- 클래스별 성능: Deletion, Insertion, Substitution, Correct

### 음소 인식
- Phoneme Error Rate (PER)
- 발음 오류 탐지: Precision, Recall, F1
- 화자별 정확도

### 출력 예시
```json
{
  "error": {
    "token_accuracy": 0.8542,
    "weighted_f1": 0.8321,
    "class_metrics": {
      "deletion": {"f1": 0.7234},
      "insertion": {"f1": 0.6891},
      "substitution": {"f1": 0.7456},
      "correct": {"f1": 0.9123}
    }
  }
}
```

## 고급 사용법

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

### 특정 체크포인트 평가
```bash
python main.py eval \
    --checkpoint experiments/my_experiment/checkpoints/best_error.pth \
    --training_mode multitask
```

## 프로젝트 구조

```
L2_ASR/
├── main.py                      # 메인 진입점
├── preprocess.py                # 전처리 진입점
├── requirements.txt             # 의존성
├── download_dataset.sh          # 데이터셋 다운로드
├── data/                        # 데이터 디렉토리
│   ├── l2arctic/               # 원본 데이터
│   ├── fold_X/                 # CV 폴드
│   ├── disjoint_wrd_split/     # 텍스트 분리 스플릿
│   └── test_labels.json        # 테스트 세트
├── experiments/                 # 실험 결과
│   └── [실험명]/
│       ├── checkpoints/        # 모델 체크포인트
│       ├── logs/              # 훈련 로그
│       └── results/           # 평가 결과
└── l2pa/                       # 메인 패키지
    ├── config.py              # 설정
    ├── train.py               # 훈련 로직
    ├── evaluate.py            # 평가 로직
    ├── preprocessing/         # 전처리
    ├── data/                  # 데이터 로딩
    ├── models/                # 모델 구조
    ├── training/              # 훈련 유틸
    ├── evaluation/            # 평가 메트릭
    └── utils/                 # 헬퍼 함수
```

## 출력 파일

훈련 및 평가 후 생성되는 파일:

### 체크포인트
`experiments/[실험명]/checkpoints/`
- `best_perceived.pth` - 최고 음소 인식 모델
- `best_error.pth` - 최고 오류 탐지 모델
- `best_canonical.pth` - 최고 정규 음소 모델
- `latest.pth` - 최신 체크포인트

### 로그
`experiments/[실험명]/logs/training.log`

### 평가 결과
`evaluation_results/[실험명]_results.json`

## 문제 해결

### CUDA 메모리 부족
```bash
python main.py train --config "batch_size=8,gradient_accumulation=4"
```

### NLTK 데이터 누락
```bash
python setup_nltk.py
```

### 데이터셋 찾을 수 없음
```bash
./download_dataset.sh
```

### 전처리 오류
```bash
# 단계별로 실행
python preprocess.py extract --data_root data/l2arctic
python preprocess.py labels --input data/preprocessed.json
python preprocess.py split --input data/processed_with_error.json
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

## 감사의 글

- Hugging Face Transformers (Wav2Vec2)
- L2-ARCTIC 데이터셋
- CMU Pronouncing Dictionary

---

문의사항은 [GitHub Issues](https://github.com/YounghwanShin/L2_ASR/issues)에 등록해주세요.