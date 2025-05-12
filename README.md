### 설치 방법

#### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-pronunciation-error-detection.git
cd l2-pronunciation-error-detection
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

## 데이터 다운로드

프로젝트에 필요한 데이터를 다운로드하기 위해 다음 스크립트를 실행하세요:

```bash
python download_drive.py
```


python train.py --stage 1 --num_epochs 10 --batch_size 8 --learning_rate 5e-5
python train.py --stage 2 --num_epochs 10 --batch_size 8 --learning_rate 5e-5 --model_checkpoint models/best_error_detection.pth

필수 인자:

--stage: 학습 단계 (1: 오류 탐지, 2: 음소 인식)

데이터 관련 인자:

--error_train_data: 오류 탐지 학습 데이터 경로 (기본값: data/errors_train.json)
--error_val_data: 오류 탐지 검증 데이터 경로 (기본값: data/errors_val.json)
--phoneme_train_data: 음소 인식 학습 데이터 경로 (기본값: data/perceived_train.json)
--phoneme_val_data: 음소 인식 검증 데이터 경로 (기본값: data/perceived_val.json)
--phoneme_map: 음소-ID 매핑 파일 경로 (기본값: data/phoneme_to_id.json)
--max_audio_length: 최대 오디오 길이(샘플 단위) 제한 (기본값: None)

모델 관련 인자:

--pretrained_model: 사전학습된 wav2vec2 모델 이름 (기본값: facebook/wav2vec2-base-960h)
--hidden_dim: 은닉층 차원 크기 (기본값: 768)
--num_phonemes: 음소 수 (기본값: 42)
--adapter_dim_ratio: 어댑터 차원 비율 (기본값: 0.25)
--unfreeze_top_percent: 상위 레이어 언프리징 비율 (기본값: 0.5)
--error_influence_weight: 오류 영향 가중치 (기본값: 0.2)

학습 관련 인자:

--batch_size: 배치 크기 (기본값: 8)
--learning_rate: 학습률 (기본값: 5e-5)
--num_epochs: 학습 에폭 수 (기본값: 10)
--seed: 랜덤 시드 (기본값: 42)
--device: 사용할 장치 (기본값: cuda 사용 가능시 cuda, 아니면 cpu)

출력 관련 인자:

--output_dir: 모델 체크포인트 저장 디렉토리 (기본값: models/)
--result_dir: 결과 로그 저장 디렉토리 (기본값: results/)
--model_checkpoint: 로드할 사전 학습된 모델 체크포인트 경로 (기본값: None)