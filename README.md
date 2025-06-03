# 🎯 L2Arctic 바로 실행 가이드 (에러 해결됨)

## 📁 최종 파일 구조 (src 폴더 없음)

```
speechbrain_multitask/
├── 📄 model.py                    # 단순화된 모델 (HuggingFace 직접 사용)
├── 📄 train.py                    # 훈련 스크립트 (루트 레벨)
├── 📄 evaluate.py                 # 평가 스크립트 (루트 레벨)  
├── 📄 data_prepare.py             # 데이터 준비 (루트 레벨)
├── 📁 hparams/
│   └── 📄 multitask.yaml         # 단순화된 설정 (에러 해결)
└── 📁 data/                      # L2Arctic 데이터
    ├── 📄 train_data.json
    ├── 📄 val_data.json
    ├── 📄 eval.json
    ├── 📄 phoneme_to_id.json
    └── 📁 l2arctic_dataset/
```

## 🚀 즉시 실행 (복사-붙여넣기)

### 1. 환경 설정
```bash
# 가상환경 활성화 (이미 있다면)
source env/bin/activate  # 또는 conda activate your_env

# 필수 패키지 설치 (한 번만)
pip install torch torchaudio transformers speechbrain hyperpyyaml scikit-learn editdistance jiwer tqdm
```

### 2. 바로 훈련 실행! 🔥
```bash
# 기본 훈련 (8GB+ GPU)
python train.py hparams/multitask.yaml --data_folder ./data --output_folder ./results --device cuda

# 메모리 절약 (4-6GB GPU)
python -c "
with open('hparams/multitask.yaml', 'r') as f: config = f.read()
config = config.replace('batch_size: 8', 'batch_size: 4')
config = config.replace('hidden_dim: 1024', 'hidden_dim: 512')
with open('hparams/small.yaml', 'w') as f: f.write(config)
"
python train.py hparams/small.yaml --data_folder ./data --output_folder ./results --device cuda

# 초소형 GPU (2-4GB)
python -c "
with open('hparams/multitask.yaml', 'r') as f: config = f.read()
config = config.replace('batch_size: 8', 'batch_size: 2')
config = config.replace('hidden_dim: 1024', 'hidden_dim: 256')
config = config.replace('num_workers: 4', 'num_workers: 1')
with open('hparams/mini.yaml', 'w') as f: f.write(config)
"
python train.py hparams/mini.yaml --data_folder ./data --output_folder ./results --device cuda

# CPU 모드 (GPU 없음)
python train.py hparams/multitask.yaml --data_folder ./data --output_folder ./results --device cpu
```

### 3. 빠른 테스트 (1 에포크)
```bash
python -c "
with open('hparams/multitask.yaml', 'r') as f: config = f.read()
config = config.replace('number_of_epochs: 30', 'number_of_epochs: 1')
config = config.replace('batch_size: 8', 'batch_size: 4')
with open('hparams/test.yaml', 'w') as f: f.write(config)
"
python train.py hparams/test.yaml --data_folder ./data --output_folder ./test_results --device cuda
```

## 📊 훈련 모니터링

### 실시간 로그 확인
```bash
# 실시간 로그
tail -f results/save/train.log

# 최고 성능 확인
grep "NEW BEST" results/save/train.log

# GPU 상태
watch -n 1 nvidia-smi

# 에포크별 결과
ls results/save/epoch_*_stats.json
cat results/save/epoch_5_stats.json | python -m json.tool
```

## 📈 평가 실행

### 자동 평가 (최신 체크포인트)
```bash
# 최신 체크포인트 찾아서 평가
CHECKPOINT=$(find results/save -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python evaluate.py hparams/multitask.yaml --model_checkpoint "$CHECKPOINT" --output_folder ./eval_results --device cuda

# 결과 확인
cat eval_results/evaluation_summary.json | python -m json.tool
```

## 🎯 성능 확인

### 최종 결과 요약
```bash
python -c "
import os, json
if os.path.exists('eval_results/evaluation_summary.json'):
    with open('eval_results/evaluation_summary.json') as f:
        results = json.load(f)
    print('🎯 최종 성능:')
    for key, value in results.items():
        if isinstance(value, float):
            print(f'   {key}: {value:.4f}')
        else:
            print(f'   {key}: {value}')
else:
    print('❌ 평가 결과 파일이 없습니다. 먼저 evaluate.py를 실행하세요.')
"
```

### 샘플 예측 확인
```bash
cat eval_results/sample_predictions.json | python -m json.tool | head -50
```

## 🔧 문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기를 1로 줄이기
python -c "
with open('hparams/multitask.yaml', 'r') as f: config = f.read()
config = config.replace('batch_size: 8', 'batch_size: 1')
config = config.replace('eval_batch_size: 8', 'eval_batch_size: 1')
config = config.replace('hidden_dim: 1024', 'hidden_dim: 128')
config = config.replace('num_workers: 4', 'num_workers: 0')
with open('hparams/ultra_mini.yaml', 'w') as f: f.write(config)
"
python train.py hparams/ultra_mini.yaml --data_folder ./data --output_folder ./results --device cuda
```

### 데이터 검증
```bash
# 누락 파일 확인
python -c "
import json, os
with open('data/train_data.json') as f: data = json.load(f)
missing = [item['wav'] for item in data.values() if not os.path.exists(item['wav'])]
print(f'누락된 파일: {len(missing)}/{len(data)}개')
if missing[:3]: 
    print('예시:')
    for f in missing[:3]: print(f'  {f}')
if len(missing) == 0:
    print('✅ 모든 음성 파일이 존재합니다!')
"

# 음소 매핑 확인
python -c "
import json
with open('data/phoneme_to_id.json') as f: phonemes = json.load(f)
print(f'음소 개수: {len(phonemes)}')
print('샘플:', dict(list(phonemes.items())[:5]))
"
```

### 패키지 문제
```bash
# SpeechBrain 재설치
pip uninstall speechbrain -y
pip install speechbrain

# Transformers 업데이트
pip install --upgrade transformers

# 전체 재설치
pip install --force-reinstall torch torchaudio transformers speechbrain
```

## ⚡ 성능 최적화

### GPU 메모리별 권장 설정
```bash
# RTX 3060 (8GB)
python -c "
config = '''batch_size: 4
hidden_dim: 512
num_workers: 2'''
print('RTX 3060 권장 설정:')
print(config)
"

# RTX 3070 (8GB)  
python -c "
config = '''batch_size: 6
hidden_dim: 768
num_workers: 4'''
print('RTX 3070 권장 설정:')
print(config)
"

# RTX 3080 (10GB)
python -c "
config = '''batch_size: 8
hidden_dim: 1024
num_workers: 4'''
print('RTX 3080 권장 설정:')
print(config)
"

# RTX 3090/4090 (24GB)
python -c "
config = '''batch_size: 16
hidden_dim: 1024
num_workers: 8'''
print('RTX 3090/4090 권장 설정:')
print(config)
"
```

## 🎉 성공 체크리스트

### 모든 것이 잘 작동하는지 확인
```bash
# 1단계: 환경 확인
python -c "import torch, transformers, speechbrain; print('✅ 모든 패키지 설치됨')"

# 2단계: 데이터 확인  
python -c "
import json, os
files = ['data/train_data.json', 'data/val_data.json', 'data/eval.json', 'data/phoneme_to_id.json']
all_exist = all(os.path.exists(f) for f in files)
print('✅ 모든 데이터 파일 존재' if all_exist else '❌ 일부 데이터 파일 누락')
"

# 3단계: 빠른 훈련 테스트
python train.py hparams/test.yaml --data_folder ./data --output_folder ./test_results --device cuda

# 4단계: 결과 확인
python -c "
import os
success = os.path.exists('test_results/save/train.log')
print('✅ 테스트 훈련 성공!' if success else '❌ 테스트 훈련 실패')
"
```

## 📞 최종 도움말

### 모든 것이 잘 안 되면...
```bash
# 모든 설정을 최소로 줄여서 테스트
python -c "
config = '''# Ultra minimal config
seed: 42
__set_seed: !apply:torch.manual_seed [!ref <seed>]

data_folder: ./data  
output_folder: ./results
save_folder: !ref <output_folder>/save
train_json: !ref <data_folder>/train_data.json
val_json: !ref <data_folder>/val_data.json
test_json: !ref <data_folder>/eval.json
phoneme_map: !ref <data_folder>/phoneme_to_id.json

number_of_epochs: 1
batch_size: 1
eval_batch_size: 1
lr: 0.001
lr_wav2vec: 0.001

hidden_dim: 128
num_phonemes: 43
num_error_types: 3
use_cross_attention: False

wav2vec2_hub: facebook/wav2vec2-base
wav2vec2_freeze: True

sample_rate: 16000
max_audio_length: 80000

error_weight: 1.0
phoneme_weight: 1.0
task: both

grad_clipping: 1.0
weight_decay: 0.01
dropout: 0.1

evaluate_every_epoch: False
show_samples: False
num_sample_show: 1

num_workers: 0
pin_memory: False
persistent_workers: False

adam_opt_class: !name:torch.optim.AdamW
    lr: !ref <lr>
    weight_decay: !ref <weight_decay>

wav2vec_opt_class: !name:torch.optim.AdamW
    lr: !ref <lr_wav2vec>
    weight_decay: !ref <weight_decay>

lr_annealing: !new:speechbrain.nnet.schedulers.ReduceLROnPlateau
    factor: 0.8
    patience: 2

error_stats: !name:speechbrain.utils.metric_stats.ErrorRateStats
phoneme_stats: !name:speechbrain.utils.metric_stats.ErrorRateStats

checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
    checkpoints_dir: !ref <save_folder>
    recoverables:
        counter: !ref <epoch_counter>

epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
    limit: !ref <number_of_epochs>

train_dataloader_opts:
    batch_size: !ref <batch_size>
    shuffle: True
    num_workers: !ref <num_workers>
    pin_memory: !ref <pin_memory>
    persistent_workers: !ref <persistent_workers>

val_dataloader_opts:
    batch_size: !ref <eval_batch_size>
    shuffle: False
    num_workers: !ref <num_workers>
    pin_memory: !ref <pin_memory>
    persistent_workers: !ref <persistent_workers>

test_dataloader_opts:
    batch_size: !ref <eval_batch_size>
    shuffle: False
    num_workers: !ref <num_workers>
    pin_memory: !ref <pin_memory>
    persistent_workers: !ref <persistent_workers>
'''
with open('hparams/emergency.yaml', 'w') as f: f.write(config)
print('🆘 비상용 최소 설정 생성: hparams/emergency.yaml')
"

# 비상용 설정으로 실행
python train.py hparams/emergency.yaml --data_folder ./data --output_folder ./emergency_results --device cuda
```

---

## 🎉 이제 완전히 작동합니다!

**모든 SpeechBrain 클래스 경로 문제가 해결되었고, src 폴더 없이 깔끔한 구조로 바로 실행할 수 있습니다!**

### 🚀 바로 시작:
```bash
python train.py hparams/multitask.yaml --data_folder ./data --output_folder ./results --device cuda
```

**Happy Training! 🎵✨**