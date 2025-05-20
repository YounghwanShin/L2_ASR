# L2 음소 인식 시스템

L2(제2언어) 학습자의 음소 인식을 수행하는 딥러닝 기반 모델입니다. 
이 시스템은 음성 및 텍스트 데이터를 활용하여 L2 화자의 발음을 분석하고 음소 인식 성능을 향상시킵니다.

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-phoneme-recognition.git
cd l2-phoneme-recognition
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

```bash
python train.py \
  --train_data data/perceived_train.json \
  --val_data data/perceived_val.json \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --num_epochs 50 \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --use_scheduler \
  --warmup_steps 2000 \
  --end_factor 0.05 \
  --dropout 0.2 \
  --save_steps 1000
```

### 학습 인자 설명

#### 필수 인자:
- `--train_data`: 학습 데이터 JSON 파일 경로
- `--val_data`: 검증 데이터 JSON 파일 경로

#### 데이터 관련 인자:
- `--eval_data`: 평가 데이터 JSON 파일 경로 (제공하면 매 에폭마다 평가 수행)
- `--phoneme_map`: 음소-ID 매핑 파일 경로 (기본값: data/phoneme_to_id.json)
- `--max_audio_length`: 최대 오디오 길이(샘플 단위) 제한 (기본값: None)
- `--max_text_length`: 최대 텍스트 길이(토큰 단위) 제한 (기본값: 128)

#### 모델 관련 인자:
- `--pretrained_audio_model`: 사전학습된 wav2vec2 모델 이름 (기본값: facebook/wav2vec2-base-960h)
- `--pretrained_text_model`: 사전학습된 텍스트 모델 이름 (기본값: bert-base-uncased)
- `--hidden_dim`: 은닉층 차원 크기 (기본값: 768)
- `--num_phonemes`: 음소 수 (기본값: 42)
- `--num_attention_heads`: 어텐션 헤드 수 (기본값: 8)

#### 학습 관련 인자:
- `--batch_size`: 배치 크기 (기본값: 8)
- `--learning_rate`: 학습률 (기본값: 5e-5)
- `--num_epochs`: 학습 에폭 수 (기본값: 10)
- `--seed`: 랜덤 시드 (기본값: 42)
- `--device`: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)
- `--max_grad_norm`: 그라디언트 클리핑을 위한 최대 노름값 (기본값: 0.5)

#### 학습률 스케줄러 관련 인자:
- `--use_scheduler`: 학습률 스케줄러(LinearLR) 사용 여부 (플래그)
- `--end_factor`: 선형 스케줄러 최종 학습률 비율 (기본값: 0.1)

#### 출력 관련 인자:
- `--output_dir`: 모델 체크포인트 저장 디렉토리 (기본값: models/)
- `--result_dir`: 결과 로그 저장 디렉토리 (기본값: results/)
- `--model_checkpoint`: 로드할 사전 학습된 모델 체크포인트 경로 (기본값: None)

## 모델 평가

학습된 모델의 성능을 평가하려면 다음 명령어를 실행하세요:

```bash
python evaluate.py \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --model_checkpoint models/best_val_phoneme_recognition.pth \
  --output_dir evaluation_results \
  --detailed
```

### 평가 인자 설명

- `--eval_data`: 평가 데이터 JSON 파일 경로 (필수)
- `--phoneme_map`: 음소-ID 매핑 JSON 파일 경로 (필수)
- `--model_checkpoint`: 평가할 모델 체크포인트 경로 (필수)
- `--pretrained_audio_model`: 사전학습된 wav2vec2 모델 이름
- `--pretrained_text_model`: 사전학습된 텍스트 모델 이름
- `--hidden_dim`: 은닉층 차원 크기
- `--num_phonemes`: 음소 수
- `--output_dir`: 평가 결과 저장 디렉토리 (기본값: evaluation_results)
- `--detailed`: 상세 결과 출력 (샘플별 결과 등)
- `--batch_size`: 배치 크기 (기본값: 8)
- `--device`: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)

## 평가 결과 해석

평가 결과는 다음 지표를 포함합니다:

- 음소 오류율(PER)
- 총 오류 수 및 유형별 오류 수(삽입, 삭제, 대체)
- 샘플별 상세 인식 결과

## 모델 구조

### 오디오 인코더
- 사전학습된 음성 특징 추출 모델을 활용하여 오디오 데이터에서 표현 추출

### 텍스트 인코더
- 사전학습된 언어 모델을 사용하여 텍스트 데이터에서 표현 추출

### 특징 융합 모듈
- 다양한 모달리티(오디오, 텍스트)의 특징을 결합하여 강화된 표현 생성

### 음소 인식 헤드
- 통합된 특징으로부터 음소 시퀀스 인식

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
    "wrd": "Each day she became a more vital part of him"
  }
}
```

- `canonical_aligned`: 정규 발음 음소 시퀀스
- `perceived_aligned`: 실제 인식된 음소 시퀀스
- `perceived_train_target`: 학습에 사용할 인식된 음소 시퀀스
- `wrd`: 원문 텍스트

## 모델 저장 형식

학습 과정에서 다음과 같은 모델 체크포인트가 저장됩니다:

- `best_val_phoneme_recognition.pth`: 검증 손실이 가장 낮은 모델
- `best_per_phoneme_recognition.pth`: 음소 오류율(PER)이 가장 낮은 모델 (평가 데이터 제공 시)
- `last_phoneme_recognition.pth`: 마지막 에폭의 모델

## 요구사항

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.15+
- Torchaudio 0.10+
- NumPy
- tqdm

## 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다.