# 다중 과제 L2 발음 평가 시스템

점진적 모델 아키텍처 개선을 통한 다중 과제 학습 프레임워크

## 🚀 빠른 시작

```bash
# 기본 모델 학습
python train.py
# 특정 모델로 학습
python train.py --config model_type=transformer
python train.py --config model_type=cross
python train.py --config model_type=hierarchical

# 모델 평가
python eval.py --model_checkpoint experiments/simple_*/checkpoints/best_phoneme.pth
```

## 📁 프로젝트 구조

```
project/
├── experiments/                    # 실험 결과 저장소
│   ├── simple_20250604_0802/      # 실험별 디렉토리
│   │   ├── checkpoints/           # 모델 체크포인트
│   │   ├── logs/                  # 학습 로그
│   │   ├── results/               # 평가 결과
│   │   └── config.json            # 실험 설정
│   └── comparison_results/        # 실험 비교 결과
├── 
├── model.py                       # 기본 모델
├── model_transformer.py           # 트랜스포머 모델
├── model_cross.py                # 교차 어텐션 모델
├── model_hierarchical.py         # 계층적 모델
├── 
├── config.py                      # 설정 관리
├── train.py                       # 학습 스크립트
├── eval.py                        # 평가 스크립트
├── experiment_manager.py          # 실험 관리 도구
├── compare_experiments.py         # 실험 비교 도구
└── [데이터 처리 파일들]
```

## 🎯 모델 종류

| 모델 | 설명 | 특징 |
|------|------|------|
| `simple` | 기본 모델 | Wav2Vec2 + Linear 인코더 |
| `transformer` | 트랜스포머 강화 | Self-attention 메커니즘 추가 |
| `cross` | 교차 어텐션 | 태스크 간 정보 교환 |
| `hierarchical` | 계층적 구조 | 다단계 특성 추출 |

## 🏃‍♂️ 실험 실행 방법

### 1. 기본 학습

```bash
# 기본 모델 (simple)
python train.py

# 특정 모델 선택
python train.py --config model_type=transformer
python train.py --config model_type=cross
python train.py --config model_type=hierarchical
```

### 2. 하이퍼파라미터 조정

```bash
# 배치 크기 및 에폭 수 조정
python train.py --config model_type=transformer,batch_size=16,num_epochs=50

# 학습률 조정
python train.py --config model_type=cross,main_lr=2e-4,wav2vec_lr=2e-5

# 실험 이름 지정
python train.py --config model_type=hierarchical,experiment_name=my_experiment
```

### 3. 데이터 경로 변경

```bash
python train.py \
  --train_data data/my_train.json \
  --val_data data/my_val.json \
  --eval_data data/my_eval.json \
  --config model_type=transformer
```

## 📊 모델 평가

### 단일 모델 평가

```bash
# 자동 모델 타입 감지
python eval.py --model_checkpoint experiments/transformer_20250604_0834/checkpoints/best_phoneme.pth

# 모델 타입 명시
python eval.py \
  --model_checkpoint path/to/model.pth \
  --model_type cross \
  --save_predictions
```

### 실험 결과 비교

```bash
# 모든 실험 비교
python compare_experiments.py

# 특정 실험들만 비교
python compare_experiments.py experiments/simple_* experiments/transformer_*

# 패턴으로 비교
python compare_experiments.py --pattern "experiments/*cross*"
```

## 🛠️ 실험 관리

### 실험 목록 확인

```bash
python experiment_manager.py list
```

### 오래된 실험 정리

```bash
# 7일 이상 된 실험 정리 (성능 좋은 것은 보존)
python experiment_manager.py cleanup --days-old 7 --keep-best

# 특정 패턴의 실험 정리
python experiment_manager.py cleanup --pattern "test_*"
```

### 실험 아카이브

```bash
# 중요한 실험 아카이브
python experiment_manager.py archive transformer_20250604_0834
```

## ⚙️ 설정 파일 (config.py)

```python
class Config:
    # 모델 선택
    model_type = 'simple'  # simple, transformer, cross, hierarchical
    
    # 학습 파라미터
    batch_size = 8
    wav2vec_lr = 1e-5      # Wav2Vec2 학습률 (낮게)
    main_lr = 1e-4         # 다른 모듈 학습률 (높게)
    num_epochs = 30
    gradient_accumulation = 2
    
    # 모델별 세부 설정
    model_configs = {
        'transformer': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        },
        'cross': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'cross_attention_dim': 512,
            'dropout': 0.1
        }
    }
```

## 📈 학습 결과 예시

```
06/04/2025 08:02:42 - 실험 시작: transformer_20250604_0802

--- 샘플 예측 결과 ---
파일: data/l2arctic_dataset/TLV/wav/arctic_a0126.wav
오류 실제:    correct correct incorrect correct correct
오류 예측:    correct correct correct incorrect correct
음소 실제:    sil iy sh sil d ey s iy b iy k ey m
음소 예측:    sil iy ch sil d ey s sh iy b iy k ah m

오류 탐지 정확도: 0.8621
오류 가중 F1: 0.8156
음소 인식 정확도: 0.8445 (PER: 0.1555)

✓ 최고 성능 갱신!
```

## 📊 실험 비교 결과

```
================================================================================
실험 비교 결과
================================================================================
                    실험명     모델타입  오류정확도  음소정확도    PER
      hierarchical_20250604  hierarchical    0.8734      0.8521  0.1479
         cross_20250604       cross          0.8687      0.8493  0.1507
   transformer_20250604   transformer        0.8621      0.8445  0.1555
        simple_20250604      simple          0.8499      0.8282  0.1718

최고 성능 모델
==================================================
오류 탐지: hierarchical_20250604 (0.8734)
음소 인식: hierarchical_20250604 (0.8521)
검증 손실: cross_20250604 (1.1834)
```

## 💡 실험 팁

1. **단계별 접근**: `simple` → `transformer` → `cross` → `hierarchical` 순서로 실험
2. **짧은 테스트**: 먼저 `--config num_epochs=5`로 빠른 테스트
3. **하이퍼파라미터**: 배치 크기 조정 후 학습률 조정
4. **정기 정리**: `experiment_manager.py cleanup`으로 디스크 공간 관리
5. **성능 추적**: `compare_experiments.py`로 개선사항 확인

## 🔧 주요 기능

- **자동 실험 관리**: 타임스탬프 기반 디렉토리 생성
- **모델 타입 자동 감지**: 경로에서 모델 종류 추론
- **혼합 정밀도 학습**: 메모리 효율적 학습
- **그래디언트 누적**: 큰 배치 크기 효과
- **옵티마이저 분리**: Wav2Vec2와 다른 모듈 별도 학습률
- **포괄적 평가**: 토큰 정확도, F1 점수, 클래스별 메트릭
- **재현 가능**: 완전한 설정 저장 및 시드 고정

## 🎯 시작하기

```bash
# 1. 기본 모델로 시작
python train.py --config num_epochs=5

# 2. 성능 확인
python compare_experiments.py

# 3. 다른 모델 시도
python train.py --config model_type=transformer,num_epochs=5

# 4. 최종 학습
python train.py --config model_type=hierarchical,num_epochs=30
```