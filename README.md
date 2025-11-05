# L2 발음 평가 시스템

제2 언어 학습자의 발음 품질을 평가하는 최첨단 딥러닝 시스템으로, 멀티태스크 학습과 화자 기반 교차 검증을 특징으로 합니다.

## 주요 기능

- **멀티태스크 학습**: 정규 음소 인식, 인지된 음소 인식, 오류 탐지를 동시에 수행
- **교차 검증**: 화자 기반 k-fold 교차 검증으로 강건한 평가
- **유연한 아키텍처**: Simple 및 Transformer 인코더 지원
- **포괄적인 메트릭**: 화자별 분석, 혼동 행렬, 상세한 오류 분석

## 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/l2-pronunciation-assessment.git
cd l2-pronunciation-assessment

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 설정
python setup_nltk.py
```

## 빠른 시작

### 1. 데이터셋 다운로드

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 2. 데이터 전처리

모든 전처리 단계(교차 검증 분할 생성 포함)를 실행:

```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

생성되는 파일:
- `test_labels.json`: 고정 테스트 세트 (화자: TLV, NJS, TNI, TXHC, ZHAA, YKWK)
- `fold_X/train_labels.json`: X번 폴드의 훈련 세트
- `fold_X/val_labels.json`: X번 폴드의 검증 세트 (화자 1명)

### 3. 모델 훈련

**모든 교차 검증 폴드 훈련:**

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

### 4. 모델 평가

```bash
python main.py eval --checkpoint experiments/multitask_transformer_cv0_20251029_141426/checkpoints/latest.pth
```

## 훈련 모드

### 1. Phoneme Only 모드
인지된 음소 인식만 훈련:
```bash
python main.py train --training_mode phoneme_only
```

### 2. Phoneme-Error 모드
인지된 음소 인식 + 오류 탐지 훈련:
```bash
python main.py train --training_mode phoneme_error
```

### 3. Multitask 모드 (권장)
세 가지 작업 모두 훈련: 정규 음소, 인지된 음소, 오류 탐지:
```bash
python main.py train --training_mode multitask
```

## 모델 아키텍처

### Simple Encoder
피드포워드 아키텍처:
```bash
python main.py train --model_type simple
```

### Transformer Encoder (권장)
멀티헤드 셀프 어텐션:
```bash
python main.py train --model_type transformer
```

## 설정

`l2pa/config.py`의 주요 파라미터:

```python
# 훈련 모드
training_mode = 'multitask'  # 'phoneme_only', 'phoneme_error', 또는 'multitask'

# 모델 아키텍처
model_type = 'transformer'  # 'simple' 또는 'transformer'

# 손실 가중치 (multitask 모드용)
canonical_weight = 0.3
perceived_weight = 0.3
error_weight = 0.4

# 훈련 하이퍼파라미터
batch_size = 16
num_epochs = 100
gradient_accumulation = 2
main_lr = 3e-4
wav2vec_lr = 1e-5

# 교차 검증
use_cross_validation = True
```

## 교차 검증 상세

시스템은 화자 기반 교차 검증을 구현합니다:

1. **테스트 세트**: 고정 6명의 화자 (TLV, NJS, TNI, TXHC, ZHAA, YKWK)
2. **훈련 화자**: 나머지 화자들을 폴드로 분할
3. **검증**: 각 폴드는 1명의 훈련 화자를 검증용으로 사용
4. **폴드 수**: 훈련 화자 수와 동일

예시 (총 18명의 화자):
- 테스트: 6명 (고정)
- 훈련: 12명
- 폴드: 12개 (각각 11명 훈련, 1명 검증)

## 평가 메트릭

### 정규 음소 인식
- 음소 오류율 (PER)
- 화자별 정확도

### 인지된 음소 인식
- 음소 오류율 (PER)
- 발음 오류 탐지: Precision, Recall, F1
- 혼동 행렬

### 오류 탐지
- 토큰 정확도
- 클래스별 F1 점수 (Deletion, Insertion, Substitution, Correct)
- Weighted F1 / Macro F1

## 고급 사용법

**커스텀 설정:**
```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --config "batch_size=32,num_epochs=150,main_lr=5e-4" \
    --experiment_name my_experiment
```

**훈련 재개:**
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

**특정 폴드 평가:**
```bash
python main.py eval \
    --checkpoint experiments/multitask_transformer_cv0_*/checkpoints/best_perceived.pth
```

## 프로젝트 구조

```
l2-pronunciation-assessment/
├── main.py                      # 메인 진입점
├── preprocess.py                # 전처리 진입점
├── requirements.txt             # 의존성
├── README.md                    # 이 파일
├── data/                        # 데이터 디렉토리
│   ├── l2arctic/                # L2-ARCTIC 데이터셋
│   ├── fold_X/                  # 교차 검증 폴드 데이터
│   ├── test_labels.json         # 테스트 세트
│   └── phoneme_to_id.json       # 음소 매핑
├── experiments/                 # 실험 결과
└── l2pa/                        # 메인 패키지
    ├── config.py                # 설정
    ├── train.py                 # 훈련 로직
    ├── evaluate.py              # 평가 로직
    ├── preprocessing/           # 전처리 모듈
    ├── data/                    # 데이터 로딩
    ├── models/                  # 모델 아키텍처
    ├── training/                # 훈련 유틸리티
    ├── evaluation/              # 평가 메트릭
    └── utils/                   # 유틸리티 함수
```

## 인용

```bibtex
@misc{l2pronunciation2025,
  title={L2 Pronunciation Assessment with Multitask Learning and Cross-Validation},
  author={Research Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/l2-pronunciation-assessment}
}
```

## 라이선스

MIT License - LICENSE 파일 참조

## 감사의 글

- Hugging Face Transformers의 Wav2Vec2 모델
- L2-ARCTIC 데이터셋 제작자
- CMU Pronouncing Dictionary
