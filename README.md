# L2 발음 오류 탐지 시스템

제2 언어 학습자의 발음 오류를 자동으로 탐지하고 분석하는 딥러닝 기반 시스템입니다.

## 개요

이 시스템은 L2-ARCTIC 데이터셋을 사용하여 다음을 수행합니다:
- 발음 오류 탐지 및 분류 (Deletion, Insertion, Substitution)
- 인지된 음소 인식
- 화자별 상세 오류 분석

## 설치

### 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 20GB 이상 디스크 공간

### 설치 과정

```bash
# 저장소 클론
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python setup_nltk.py
```

## 사용법

### 1단계: 데이터셋 다운로드

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

생성되는 파일:
- `data/l2arctic/` - L2-ARCTIC 데이터셋
- `data/phoneme_to_id.json` - 음소 매핑

### 2단계: 데이터 전처리

```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

이 명령은 다음을 수행합니다:
1. TextGrid 파일에서 음소 정렬 추출
2. 오류 레이블 (D, I, S, C) 생성
3. 교차 검증 분할 생성

생성되는 파일:
- `data/fold_0/`, `data/fold_1/`, ... - 교차 검증 폴드
- `data/test_labels.json` - 테스트 세트 (고정 6명의 화자)

### 3단계: 모델 학습

**모든 교차 검증 폴드 학습:**
```bash
python main.py train --training_mode multitask --model_type transformer
```

**특정 폴드만 학습:**
```bash
python main.py train --training_mode multitask --model_type transformer --cv_fold 0
```

**교차 검증 없이 학습:**
```bash
python main.py train --training_mode multitask --model_type transformer --no_cv
```

### 4단계: 모델 평가

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## 훈련 모드

### 1. phoneme_only
인지된 음소 인식만 학습
```bash
python main.py train --training_mode phoneme_only
```

### 2. phoneme_error (오류 탐지 추천)
인지된 음소 인식 + 오류 탐지 학습
```bash
python main.py train --training_mode phoneme_error
```

### 3. multitask
정규 음소, 인지된 음소, 오류 탐지 모두 학습
```bash
python main.py train --training_mode multitask
```

## 모델 아키텍처

### Simple
피드포워드 네트워크:
```bash
python main.py train --model_type simple
```

### Transformer (권장)
멀티헤드 셀프 어텐션:
```bash
python main.py train --model_type transformer
```

## 주요 설정

`l2pa/config.py`에서 주요 파라미터 설정:

```python
training_mode = 'multitask'  # 훈련 모드
model_type = 'transformer'    # 모델 구조
batch_size = 16               # 배치 크기
num_epochs = 100              # 에포크 수
main_lr = 3e-4                # 학습률
```

**커맨드라인에서 설정 변경:**
```bash
python main.py train --config "batch_size=32,num_epochs=150"
```

## 교차 검증

화자 기반 교차 검증을 사용합니다:

- **테스트 세트**: 6명의 화자 (고정)
- **훈련 세트**: 나머지 화자들
- **검증 세트**: 각 폴드마다 훈련 화자 중 1명

예시 (총 18명):
- 테스트: 6명
- 훈련 가능: 12명
- 폴드 수: 12개 (각각 11명 훈련, 1명 검증)

## 평가 지표

### 오류 탐지
- Token Accuracy
- Weighted F1 / Macro F1
- 클래스별 F1 (D, I, S, C)

### 음소 인식
- Phoneme Error Rate (PER)
- 발음 오류 탐지: Precision, Recall, F1
- 화자별 정확도

## 프로젝트 구조

```
L2_ASR/
├── main.py                 # 메인 실행 파일
├── preprocess.py           # 전처리 실행 파일
├── requirements.txt        # 의존성
├── data/                   # 데이터
│   ├── l2arctic/          # 원본 데이터셋
│   ├── fold_X/            # 교차 검증 폴드
│   └── test_labels.json   # 테스트 세트
├── experiments/           # 실험 결과
│   └── [실험명]/
│       ├── checkpoints/   # 모델 체크포인트
│       ├── logs/          # 학습 로그
│       └── results/       # 평가 결과
└── l2pa/                  # 메인 패키지
    ├── config.py          # 설정
    ├── train.py           # 학습 로직
    ├── evaluate.py        # 평가 로직
    ├── preprocessing/     # 전처리
    ├── data/              # 데이터셋
    ├── models/            # 모델
    ├── training/          # 학습 유틸
    ├── evaluation/        # 평가 메트릭
    └── utils/             # 유틸리티
```

## 추가 기능

### 학습 재개
```bash
python main.py train --resume experiments/실험명/checkpoints/latest.pth
```

### 실험 이름 지정
```bash
python main.py train --experiment_name my_experiment
```

### 개별 전처리 단계 실행
```bash
# 1. 음소 추출
python preprocess.py extract --data_root data/l2arctic --output data/preprocessed.json

# 2. 오류 레이블 생성
python preprocess.py labels --input data/preprocessed.json --output data/processed_with_error.json

# 3. 데이터 분할
python preprocess.py split --input data/processed_with_error.json --output_dir data
```

## 문제 해결

### CUDA 메모리 부족
```bash
python main.py train --config "batch_size=8,gradient_accumulation=4"
```

### NLTK 데이터 누락
```bash
python setup_nltk.py
```

### 데이터셋 경로 오류
```bash
./download_dataset.sh
```

## 결과 저장 위치

- **체크포인트**: `experiments/[실험명]/checkpoints/`
  - `best_perceived.pth` - 최고 성능 인지된 음소 모델
  - `best_error.pth` - 최고 성능 오류 탐지 모델
  - `latest.pth` - 최신 체크포인트

- **학습 로그**: `experiments/[실험명]/logs/training.log`

- **평가 결과**: `evaluation_results/[실험명]_results.json`

## 인용

```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Error Detection with Deep Learning},
  author={Younghwan Shin},
  year={2025},
  url={https://github.com/YounghwanShin/L2_ASR}
}
```

## 라이선스

MIT License

---

문의사항은 [GitHub Issues](https://github.com/YounghwanShin/L2_ASR/issues)에 남겨주세요.