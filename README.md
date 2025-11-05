# L2 발음 평가 시스템 with 멀티태스크 학습

딥러닝 기반의 제2언어(L2) 학습자 발음 품질 평가 시스템입니다. 발음 오류 검출을 위한 멀티태스크 학습을 핵심 아이디어로 합니다.

## 개요

본 시스템은 세 가지 관련된 태스크를 동시에 학습합니다:
- **표준 음소 인식**: 올바른 발음 예측
- **실제 음소 인식**: 학습자의 실제 발음 인식
- **오류 검출**: 발음 오류를 Deletion(D), Insertion(I), Substitution(S), Correct(C)로 분류

**핵심 아이디어**: 발음 오류 유형을 명시적으로 학습하여 더 정확한 발음 평가를 수행합니다.

## 주요 기능

- **멀티태스크 학습**: 관련 태스크들의 동시 학습으로 성능 향상
- **유연한 아키텍처**: Simple Feedforward 또는 Transformer 인코더 선택 가능
- **교차 검증**: 화자 기반 k-fold 검증으로 robust한 평가
- **종합적 메트릭**: 음소 오류율(PER), 오발음 검출, 오류 분류 성능 측정

## 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 데이터 준비

L2-ARCTIC 데이터셋을 다운로드하고 전처리합니다:

```bash
# 1. 데이터셋 다운로드
# L2-ARCTIC 웹사이트에서 수동 다운로드: https://psi.engr.tamu.edu/l2-arctic-corpus/
# 다운로드 후 data/l2arctic 경로에 압축 해제

# 2. NLTK 데이터 설치 (선택사항, 음소 변환에 필요)
python setup_nltk.py

# 3-A. 전체 전처리 파이프라인 실행 (권장)
python preprocess.py all --data_root data/l2arctic --output_dir data

# 3-B. 또는 단계별 실행
# Step 1: 데이터셋에서 음소 추출
python preprocess.py dataset --data_root data/l2arctic --output data/preprocessed.json

# Step 2: 오류 라벨 생성
python preprocess.py labels --input data/preprocessed.json --output data/processed_with_error.json

# Step 3: 교차 검증을 위한 데이터 분할
python preprocess.py split --input data/processed_with_error.json --output_dir data
```

**참고**: TextGrid 파일이 없는 경우, `preprocess_dataset.py`가 기본 처리를 수행합니다. 완전한 기능을 위해서는 L2-ARCTIC 데이터셋의 TextGrid 파일이 필요합니다.

### 학습

```bash
# 멀티태스크 학습 (권장)
python main.py train --training_mode multitask --model_type transformer

# 특정 fold 학습
python main.py train --training_mode multitask --cv_fold 0

# 교차 검증 없이 학습
python main.py train --training_mode multitask --no_cv
```

### 평가

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## 학습 모드

### Multitask (권장)
세 가지 태스크를 모두 학습하여 최고 성능 달성:
```bash
python main.py train --training_mode multitask
```

### Phoneme-Error
실제 음소 인식과 오류 검출만 학습:
```bash
python main.py train --training_mode phoneme_error
```

### Phoneme Only
실제 음소 인식만 학습:
```bash
python main.py train --training_mode phoneme_only
```

## 모델 아키텍처

### Transformer Encoder (권장)
Multi-head self-attention을 사용한 컨텍스트 모델링:
```bash
python main.py train --model_type transformer
```

### Simple Encoder
빠른 학습을 위한 Feedforward 아키텍처:
```bash
python main.py train --model_type simple
```

## 설정

주요 설정은 `l2pa/config.py`에서 수정 가능합니다:

```python
# 학습 모드 (핵심: 멀티태스크 오류 학습)
training_mode = 'multitask'

# 모델 아키텍처
model_type = 'transformer'

# 손실 가중치
canonical_loss_weight = 0.3
perceived_loss_weight = 0.3
error_loss_weight = 0.4  # 오류 검출에 더 높은 가중치

# 학습 하이퍼파라미터
batch_size = 16
num_epochs = 100
gradient_accumulation_steps = 2
main_learning_rate = 3e-4
wav2vec_learning_rate = 1e-5
```

## 교차 검증

화자 기반 교차 검증으로 모델의 일반화 성능을 확보합니다:

- **테스트 세트**: 6명의 고정 화자 (TLV, NJS, TNI, TXHC, ZHAA, YKWK)
- **학습 Fold**: 각 fold마다 다른 화자를 검증 세트로 사용
- **자동 실행**: 모든 fold를 순차적으로 학습

```bash
# 모든 fold 학습
python main.py train --training_mode multitask

# 특정 fold 학습
python main.py train --training_mode multitask --cv_fold 0
```

## 전처리 상세 설명

### 전처리 단계

1. **데이터셋 추출** (`preprocess.py dataset`)
   - TextGrid 파일에서 표준(canonical) 음소와 실제(perceived) 음소를 추출
   - 화자별로 오디오 파일과 전사본 매핑
   - 출력: `preprocessed.json`

2. **오류 라벨 생성** (`preprocess.py labels`)
   - Needleman-Wunsch 알고리즘으로 표준 음소와 실제 음소를 정렬
   - 각 음소에 대해 D(Deletion), I(Insertion), S(Substitution), C(Correct) 라벨 할당
   - 출력: `processed_with_error.json`

3. **데이터 분할** (`preprocess.py split`)
   - 테스트 세트 분리 (6명의 고정 화자)
   - 나머지 화자들로 K-fold 교차 검증 세트 생성
   - 출력: `fold_0/`, `fold_1/`, ..., `test_labels.json`

### 전처리 옵션

```bash
# 전체 파이프라인 (권장)
python preprocess.py all \
    --data_root data/l2arctic \
    --output_dir data \
    --test_speakers TLV NJS TNI TXHC ZHAA YKWK

# 데이터셋만 처리
python preprocess.py dataset \
    --data_root data/l2arctic \
    --output data/preprocessed.json

# 오류 라벨만 생성
python preprocess.py labels \
    --input data/preprocessed.json \
    --output data/processed_with_error.json

# 데이터 분할만 수행
python preprocess.py split \
    --input data/processed_with_error.json \
    --output_dir data \
    --test_speakers TLV NJS TNI TXHC ZHAA YKWK
```

## 평가 메트릭

### 음소 인식
- 음소 오류율(PER)
- 오발음 검출 (Precision, Recall, F1)
- 화자별 정확도

### 오류 검출
- 토큰 레벨 정확도
- 클래스별 F1 (Deletion, Insertion, Substitution, Correct)
- Weighted/Macro F1 점수

## 고급 사용법

### 사용자 정의 설정
```bash
python main.py train \
    --training_mode multitask \
    --config "batch_size=32,num_epochs=150,main_learning_rate=5e-4" \
    --experiment_name custom_experiment
```

### 학습 재개
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

## 프로젝트 구조

```
l2_pronunciation_assessment/
├── main.py                     # 학습/평가 진입점
├── preprocess.py               # 데이터 전처리 진입점
├── requirements.txt
├── README.md
├── data/
│   ├── l2arctic/              # L2-ARCTIC 데이터셋
│   ├── fold_X/                # 교차 검증 fold
│   ├── test_labels.json       # 테스트 세트
│   └── phoneme_to_id.json     # 음소 매핑
├── experiments/               # 학습 출력
└── l2pa/                      # 메인 패키지
    ├── config.py              # 설정
    ├── train.py               # 학습 로직
    ├── evaluate.py            # 평가 로직
    ├── data/                  # 데이터 로딩
    ├── models/                # 모델 아키텍처
    ├── preprocessing/         # 데이터 전처리
    ├── training/              # 학습 유틸리티
    ├── evaluation/            # 평가 메트릭
    └── utils/                 # 유틸리티 함수
```

## 주요 개선사항

### 1. 효율적인 데이터 로딩
- Waveform을 한 번만 로드하여 처리 속도 향상
- `__getitem__`에서 길이 필터링 수행

### 2. 코드 품질
- Google 스타일 가이드 준수
- 명확한 docstring과 타입 힌트
- 중복 코드 제거 및 가독성 향상

### 3. 멀티태스크 학습 강조
- 오류 검출이 핵심 아이디어임을 명확히 표현
- 교차 검증은 평가 방법일 뿐

## 인용

```bibtex
@misc{l2pronunciation2025,
  title={L2 발음 평가 시스템 with 멀티태스크 학습},
  author={L2PA Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/l2_pronunciation_assessment}
}
```

## 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조

## 감사의 말

- Hugging Face Transformers의 Wav2Vec2 모델
- L2-ARCTIC 데이터셋
- CMU Pronouncing Dictionary

## 문제 해결

### 메모리 부족
- `batch_size`를 줄이거나 `gradient_accumulation_steps`를 늘리세요
- `max_audio_length`를 줄여보세요

### CUDA 오류
- PyTorch와 CUDA 버전 호환성을 확인하세요
- `torch.cuda.empty_cache()`가 주기적으로 호출되는지 확인하세요

### 학습 속도 저하
- `model_type='simple'`을 사용해보세요
- SpecAugment를 비활성화하려면 `enable_wav2vec_specaug=False` 설정

## 연락처

문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해주세요.
