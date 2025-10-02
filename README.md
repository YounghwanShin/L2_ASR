# L2 발음 평가 모델

제2언어 학습자의 발음을 평가하는 딥러닝 모델입니다. Wav2Vec2 기반의 통합 아키텍처를 사용하여 음소 인식과 에러 탐지를 동시에 수행합니다.

## 주요 기능

- **음소 인식**: CTC 기반 음소 시퀀스 예측
- **에러 탐지**: 발음 에러 타입 분류 (삭제, 삽입, 대체, 정확)
- **통합 훈련**: 멀티태스크 학습으로 두 작업을 동시에 최적화
- **길이 정규화**: 예측 시퀀스 길이 제어
- **유연한 아키텍처**: Simple 및 Transformer 인코더 지원

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 8GB GPU 메모리

## 설치

1. 저장소 복제:
```bash
git clone https://github.com/your-repo/l2-pronunciation-assessment.git
cd l2-pronunciation-assessment
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 데이터 전처리

훈련 전 데이터를 전처리해야 합니다:

```bash
python data_preprocessing.py
```

이 스크립트는 다음을 수행합니다:
- canonical_aligned에서 canonical_train_target 생성
- 새로운 에러 라벨 생성 (D: 삭제, I: 삽입, S: 대체, C: 정확)
- CTC 특성에 맞는 시퀀스 압축

## 훈련

### 기본 훈련

```bash
python train.py --training_mode phoneme_error_length --model_type transformer
```

### 훈련 모드

- `phoneme_only`: 음소 인식만 수행
- `phoneme_error`: 음소 인식 + 에러 탐지
- `phoneme_error_length`: 음소 인식 + 에러 탐지 + 길이 정규화

### 모델 아키텍처

- `simple`: 기본 피드포워드 인코더
- `transformer`: Transformer 기반 인코더

### 고급 옵션

```bash
python train.py \
    --training_mode phoneme_error_length \
    --model_type transformer \
    --config "batch_size=32,num_epochs=100,main_lr=5e-4" \
    --experiment_name my_experiment
```

### 훈련 재개

```bash
python train.py --resume ../shared/experiments/phoneme_error_transformer0406_20250915154806/checkpoints/latest.pth
```

## 평가

```bash
python eval.py --model_checkpoint experiments/my_experiment/checkpoints/best_phoneme.pth
```

### 평가 옵션

```bash
python eval.py \
    --model_checkpoint path/to/checkpoint.pth \
    --eval_data path/to/test_data.json \
    --batch_size 16 \
    --save_predictions
```

## 설정

`config.py`에서 주요 설정을 수정할 수 있습니다:

- **모델 설정**: 은닉층 크기, 드롭아웃 등
- **훈련 설정**: 학습률, 배치 크기, 에포크 수
- **손실 가중치**: 음소/에러 손실 균형
- **데이터 경로**: 훈련/검증/테스트 데이터 위치

## 프로젝트 구조

```
l2_pronunciation_assessment/
├── config.py                    # 설정 파일
├── train.py                     # 훈련 스크립트
├── eval.py                      # 평가 스크립트
├── data_preprocessing.py        # 데이터 전처리
├── src/
│   ├── data/                    # 데이터 처리 모듈
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/                  # 모델 관련 모듈
│   │   ├── unified_model.py
│   │   ├── encoders.py
│   │   ├── heads.py
│   │   └── losses.py
│   ├── training/                # 훈련 관련 모듈
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── evaluation/              # 평가 관련 모듈
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/                   # 유틸리티 모듈
│       └── audio.py
├── shared/data/                 # 데이터 파일
├── experiments/                 # 실험 결과
└── evaluation_results/          # 평가 결과
```

## 데이터 형식

### 입력 데이터

JSON 형식으로 다음 필드를 포함해야 합니다:

```json
{
  "path/to/audio.wav": {
    "wav": "path/to/audio.wav",
    "duration": 4.5,
    "spk_id": "SPEAKER_ID",
    "canonical_aligned": "sil hh iy w ih l ...",
    "perceived_aligned": "sil hh iy w ih l ...",
    "perceived_train_target": "sil hh iy w ih l ...",
    "canonical_train_target": "sil hh iy w ih l ...",
    "wrd": "He will follow us soon",
    "error_labels": "C C C C C C ..."
  }
}
```

### 에러 라벨

- `C`: 정확한 발음
- `D`: 삭제 (canonical에는 있지만 perceived에서 누락)
- `I`: 삽입 (perceived에는 있지만 canonical에는 없음)
- `S`: 대체 (다른 음소로 발음)

## 평가 메트릭

### 음소 인식

- **PER (Phoneme Error Rate)**: 음소 에러율
- **음소 정확도**: 1 - PER
- **잘못된 발음 탐지**: 정밀도, 재현율, F1-점수

### 에러 탐지

- **토큰 정확도**: 개별 에러 라벨 정확도
- **시퀀스 정확도**: 전체 시퀀스가 정확한 비율
- **클래스별 메트릭**: 각 에러 타입별 성능

## 실험 결과

실험 결과는 다음 위치에 저장됩니다:

- **체크포인트**: `experiments/{experiment_name}/checkpoints/`
- **로그**: `experiments/{experiment_name}/logs/`
- **최종 메트릭**: `experiments/{experiment_name}/results/`
- **평가 결과**: `evaluation_results/`

## 모델 아키텍처

### 인코더

1. **Wav2Vec2**: 사전훈련된 음성 표현 학습
2. **Feature Encoder**: 음성 특성 강화
   - Simple: 피드포워드 네트워크
   - Transformer: 멀티헤드 어텐션

### 출력 헤드

1. **음소 인식 헤드**: 42개 음소 분류
2. **에러 탐지 헤드**: 5개 에러 타입 분류

### 손실 함수

- **Focal CTC Loss**: 클래스 불균형 처리
- **길이 정규화**: Smooth L1 Loss로 시퀀스 길이 제어
- **가중 조합**: 멀티태스크 학습을 위한 손실 가중치 조절

## 성능 최적화

### 메모리 최적화

- Gradient accumulation 사용
- Mixed precision training
- 정기적인 GPU 메모리 정리

### 훈련 안정성

- 학습률 분리 (Wav2Vec2 vs 다른 부분)
- Focal Loss로 어려운 샘플에 집중
- SpecAugment로 데이터 증강

## 문제 해결

### 일반적인 문제

1. **GPU 메모리 부족**
   - 배치 크기 줄이기
   - Gradient accumulation 늘리기

2. **훈련 불안정**
   - 학습률 낮추기
   - Focal Loss 파라미터 조정

3. **과적합**
   - 드롭아웃 늘리기
   - 정규화 가중치 조정

### 로그 확인

훈련 중 문제가 발생하면 다음 로그를 확인하세요:

```bash
tail -f experiments/{experiment_name}/logs/training.log
```

## 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 인용

이 코드를 연구에 사용하시는 경우 다음과 같이 인용해 주세요:

```bibtex
@misc{l2pronunciation2024,
  title={L2 Pronunciation Assessment with Unified Neural Architecture},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/l2-pronunciation-assessment}
}
```