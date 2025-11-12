# L2 발음 평가 시스템

제2 언어(L2) 학습자의 발음 오류를 자동으로 탐지하는 딥러닝 기반 시스템입니다. Wav2Vec2와 Transformer를 활용하여 학습자의 발음에서 삭제(Deletion), 삽입(Insertion), 치환(Substitution) 오류를 검출하고 실제 발음된 음소를 인식합니다.

## 주요 기능

- **다중 작업 학습**: 정규 음소, 인지 음소, 오류 유형을 동시에 학습
- **자동 아키텍처 적응**: Wav2Vec2 모델 크기에 따라 자동으로 차원 조정
- **유연한 데이터 분할**: 교차 검증, 텍스트 분리, 표준 분할 지원
- **화자별 성능 분석**: 개별 화자에 대한 상세 평가 제공
- **L2-ARCTIC 데이터셋**: 다양한 모국어 화자의 영어 발음 데이터 지원

## 시스템 요구사항

### 하드웨어
- **GPU**: CUDA 지원 GPU 권장 (최소 8GB VRAM)
- **RAM**: 최소 16GB
- **디스크**: 20GB 이상의 여유 공간

### 소프트웨어
- Python 3.8 이상
- CUDA 11.0 이상 (GPU 사용 시)

## 설치 방법

### 1. 저장소 복제
```bash
git clone https://github.com/YounghwanShin/L2_ASR.git
cd L2_ASR
```

### 2. 가상환경 생성
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. NLTK 데이터 다운로드
```bash
python setup_nltk.py
```

## 빠른 시작

### 1단계: 데이터셋 다운로드
```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- L2-ARCTIC 데이터셋 다운로드
- 음소 매핑 파일 다운로드
- 디렉토리 구조 설정

### 2단계: 데이터 전처리
```bash
python preprocess.py all --data_root data/l2arctic --output_dir data
```

전처리 단계:
1. TextGrid에서 음소 정렬 추출
2. 오류 레이블 자동 생성 (D, I, S, C)
3. 교차 검증 폴드 생성 (화자 기반)
4. 텍스트 분리 스플릿 생성

### 3단계: 모델 훈련

**모든 폴드 교차 검증 훈련**:
```bash
python main.py train --training_mode multitask --model_type transformer
```

**특정 폴드만 훈련**:
```bash
python main.py train --training_mode multitask --model_type transformer --cv_fold 0
```

**텍스트 분리 스플릿으로 훈련**:
```bash
python main.py train --training_mode multitask --model_type transformer --no_cv --data_split_mode disjoint
```

### 4단계: 모델 평가
```bash
python main.py eval --checkpoint experiments/multitask_transformer_fold0_*/checkpoints/best_perceived.pth
```

## 상세 사용 가이드

### 훈련 모드

시스템은 세 가지 훈련 모드를 지원합니다:

#### 1. phoneme_only
인지된 음소만 예측합니다.
```bash
python main.py train --training_mode phoneme_only --model_type transformer
```

**사용 사례**: 기본적인 음소 인식 시스템

#### 2. phoneme_error
인지된 음소와 오류 유형을 동시에 예측합니다.
```bash
python main.py train --training_mode phoneme_error --model_type transformer
```

**사용 사례**: 오류 탐지에 집중하면서 음소 정보 활용

#### 3. multitask (권장)
정규 음소, 인지 음소, 오류 유형을 모두 예측합니다.
```bash
python main.py train --training_mode multitask --model_type transformer
```

**사용 사례**: 최고 성능을 위한 전체 작업 학습

### Wav2Vec2 모델 선택

다양한 Wav2Vec2 모델을 사용할 수 있으며, 시스템이 자동으로 아키텍처를 조정합니다:
```bash
# Large 모델 (기본값)
python main.py train --pretrained_model facebook/wav2vec2-large-xlsr-53

# Large 960h 모델
python main.py train --pretrained_model facebook/wav2vec2-large-960h

# Base 모델 (빠른 실험용)
python main.py train --pretrained_model facebook/wav2vec2-base
```

**모델 크기 비교**:
- **Base**: ~94M 파라미터, 빠른 훈련
- **Large**: ~317M 파라미터, 높은 성능

### 데이터 분할 방식

#### 교차 검증 (기본값)
화자 기반 k-fold 교차 검증을 수행합니다.
```bash
# 모든 폴드 훈련
python main.py train --training_mode multitask

# 특정 폴드만
python main.py train --training_mode multitask --cv_fold 2
```

**장점**: 모든 데이터 활용, 로버스트한 평가

#### 텍스트 분리 스플릿
훈련/검증/테스트 세트 간 텍스트 중복이 없습니다.
```bash
python main.py train --training_mode multitask --no_cv --data_split_mode disjoint
```

**장점**: 정규 음소 정보 누출 방지, 일반화 능력 평가

#### 표준 스플릿
단순한 훈련/검증/테스트 분할입니다.
```bash
python main.py train --training_mode multitask --no_cv --data_split_mode standard
```

**장점**: 간단한 설정, 빠른 실험

### 하이퍼파라미터 조정

명령줄에서 주요 하이퍼파라미터를 조정할 수 있습니다:
```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --config "batch_size=32,num_epochs=50,main_lr=5e-4,wav2vec_lr=2e-5"
```

**주요 파라미터**:
- `batch_size`: 배치 크기 (기본값: 16)
- `num_epochs`: 훈련 에폭 수 (기본값: 30)
- `main_lr`: 메인 모델 학습률 (기본값: 3e-4)
- `wav2vec_lr`: Wav2Vec2 학습률 (기본값: 1e-5)
- `gradient_accumulation`: 그래디언트 누적 단계 (기본값: 2)

### 모델 아키텍처

#### Simple Encoder
2층 피드포워드 네트워크로 빠르고 가볍습니다.
```bash
python main.py train --model_type simple
```

#### Transformer Encoder (권장)
멀티헤드 어텐션으로 장거리 의존성을 포착합니다.
```bash
python main.py train --model_type transformer
```

### 훈련 재개

중단된 훈련을 재개할 수 있습니다:
```bash
python main.py train --resume experiments/my_experiment/checkpoints/latest.pth
```

시스템이 자동으로:
- 모델 가중치 로드
- 옵티마이저 상태 복원
- 에폭 카운터 재설정
- 최고 성능 지표 복원

### 실험 관리

#### 자동 실험 이름 생성
시스템이 자동으로 설명적인 실험 이름을 생성합니다:
```
multitask_transformer_fold0_bs32_lr1e-03_20250512_143022
│         │           │      │     │        └─ 타임스탬프
│         │           │      │     └─ 학습률 (비기본값인 경우)
│         │           │      └─ 배치 크기 (비기본값인 경우)
│         │           └─ 폴드 번호
│         └─ 모델 타입
└─ 훈련 모드
```

#### 수동 실험 이름 지정
```bash
python main.py train \
    --training_mode multitask \
    --model_type transformer \
    --experiment_name my_custom_experiment
```

## 평가 지표

### 음소 인식 메트릭

**Phoneme Error Rate (PER)**:
```
PER = (삽입 + 삭제 + 치환) / 전체 음소 수
```

**발음 오류 탐지**:
- **Precision**: 탐지된 오류 중 실제 오류 비율
- **Recall**: 실제 오류 중 탐지된 오류 비율
- **F1 Score**: Precision과 Recall의 조화 평균

### 오류 분류 메트릭

**클래스별 성능**:
- Deletion (D): 음소 누락
- Insertion (I): 불필요한 음소 삽입
- Substitution (S): 잘못된 음소로 대체
- Correct (C): 올바른 발음

**전체 성능**:
- **Token Accuracy**: 전체 토큰 정확도
- **Weighted F1**: 클래스 빈도를 고려한 F1
- **Macro F1**: 모든 클래스의 평균 F1

## 출력 구조
```
experiments/
└── multitask_transformer_fold0_20250512_143022/
    ├── checkpoints/
    │   ├── best_canonical.pth      # 최고 정규 음소 성능
    │   ├── best_perceived.pth      # 최고 인지 음소 성능
    │   ├── best_error.pth          # 최고 오류 탐지 성능
    │   ├── best_loss.pth           # 최저 검증 손실
    │   └── latest.pth              # 최신 체크포인트
    ├── logs/
    │   └── training.log            # 상세 훈련 로그
    ├── results/
    │   └── final_metrics.json      # 최종 평가 지표
    └── config.json                 # 사용된 설정
```

## 전처리 상세

### 개별 단계 실행

필요시 전처리를 단계별로 수행할 수 있습니다:
```bash
# 1단계: 음소 추출
python preprocess.py extract \
    --data_root data/l2arctic \
    --output data/preprocessed.json

# 2단계: 오류 레이블 생성
python preprocess.py labels \
    --input data/preprocessed.json \
    --output data/processed_with_error.json

# 3단계: 교차 검증 스플릿
python preprocess.py split \
    --input data/processed_with_error.json \
    --output_dir data

# 4단계: 텍스트 분리 스플릿
python preprocess.py split_disjoint \
    --input data/processed_with_error.json \
    --output_dir data
```

### 데이터 검증

전처리 후 데이터를 검증하세요:
```bash
# 폴드 통계 확인
cat data/split_statistics.json

# 샘플 데이터 확인
head -n 50 data/fold_0/train_labels.json
```

## 문제 해결

### CUDA 메모리 부족

**증상**: `RuntimeError: CUDA out of memory`

**해결 방법 1** - 배치 크기 줄이기:
```bash
python main.py train --config "batch_size=8"
```

**해결 방법 2** - 그래디언트 누적 증가:
```bash
python main.py train --config "batch_size=8,gradient_accumulation=4"
```

**해결 방법 3** - Base 모델 사용:
```bash
python main.py train --pretrained_model facebook/wav2vec2-base
```

### 전처리 오류

**증상**: TextGrid 파일 읽기 오류

**해결책**:
```bash
# TextGrid 파일 확인
ls -la data/l2arctic/*/annotation/*.TextGrid

# 단계별 실행으로 문제 격리
python preprocess.py extract --data_root data/l2arctic
```

### 데이터셋 누락

**증상**: `FileNotFoundError: data/l2arctic not found`

**해결책**:
```bash
# 데이터셋 재다운로드
./download_dataset.sh

# 수동 다운로드
mkdir -p data
cd data
# Google Drive에서 다운로드 후 압축 해제
```

### 학습 불안정

**증상**: 손실이 발산하거나 NaN 발생

**해결책**:
```bash
# 학습률 낮추기
python main.py train --config "main_lr=1e-4,wav2vec_lr=5e-6"

# Focal loss 파라미터 조정
python main.py train --config "focal_alpha=0.5,focal_gamma=1.5"
```

## 고급 기능

### 손실 가중치 조정

다중 작업 학습 시 각 작업의 중요도를 조정할 수 있습니다:
```bash
python main.py train \
    --training_mode multitask \
    --config "canonical_weight=0.2,perceived_weight=0.4,error_weight=0.4"
```

### 커스텀 테스트 화자

테스트 세트 화자를 변경할 수 있습니다:
```bash
python preprocess.py split \
    --input data/processed_with_error.json \
    --output_dir data \
    --test_speakers YBAA YKWK ASI
```

### 분산 학습

여러 GPU를 사용하여 학습 속도를 향상시킬 수 있습니다:
```python
# 시스템이 자동으로 사용 가능한 모든 GPU 활용
# torch.nn.DataParallel 자동 적용
```

## 성능 벤치마크

### 예상 훈련 시간

**단일 폴드** (Tesla V100, 배치 크기 16):
- Simple Encoder: ~2시간
- Transformer Encoder: ~3시간

**전체 교차 검증** (22개 폴드):
- Simple Encoder: ~44시간
- Transformer Encoder: ~66시간

### 예상 성능

L2-ARCTIC 테스트 세트 기준:

| 모델 | PER | 오류 정확도 | F1 Score |
|------|-----|------------|----------|
| Multitask + Transformer | ~15% | ~85% | ~0.82 |
| Phoneme + Error | ~17% | ~83% | ~0.80 |
| Phoneme Only | ~18% | - | - |

## API 참조

주요 클래스와 함수의 상세 문서는 코드의 docstring을 참조하세요:
```python
from l2pa.config import Config
from l2pa.models.unified_model import UnifiedModel
from l2pa.train import train_model
from l2pa.evaluate import evaluate_model

# 설정 생성
config = Config()
config.training_mode = 'multitask'

# 모델 초기화
model = UnifiedModel(
    pretrained_model_name=config.pretrained_model,
    **config.get_model_config()
)
```

## 기여하기

프로젝트 개선에 기여를 환영합니다:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 인용

이 코드를 연구에 사용하시는 경우 다음과 같이 인용해주세요:
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

MIT License - 자유롭게 사용, 수정, 배포할 수 있습니다.

## 참고 자료

- [L2-ARCTIC 데이터셋](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- [Wav2Vec2 논문](https://arxiv.org/abs/2006.11477)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)

## 문의

- **Issues**: [GitHub Issues](https://github.com/YounghwanShin/L2_ASR/issues)
- **Email**: [이메일 주소]
- **Discussion**: [GitHub Discussions](https://github.com/YounghwanShin/L2_ASR/discussions)

---

**마지막 업데이트**: 2025년 5월