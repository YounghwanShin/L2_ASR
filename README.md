#!/bin/bash

# 디렉토리 생성
mkdir -p models results evaluation_results

# 1. 오류 탐지 모델 학습
echo "=== 오류 탐지 모델 학습 ==="
python train.py \
  --mode error \
  --error_train_data data/errors_train.json \
  --error_val_data data/errors_val.json \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --pretrained_model facebook/wav2vec2-large-xlsr-53 \
  --hidden_dim 1024 \
  --num_error_types 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 10 \
  --max_grad_norm 0.5 \
  --use_scheduler \
  --evaluate_every_epoch \
  --show_samples \
  --num_sample_show 5 \
  --output_dir models \
  --result_dir results

# 2. 음소 인식 모델 학습
echo "=== 음소 인식 모델 학습 ==="
python train.py \
  --mode phoneme \
  --phoneme_train_data data/perceived_train.json \
  --phoneme_val_data data/perceived_val.json \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --pretrained_model facebook/wav2vec2-large-xlsr-53 \
  --error_model_checkpoint models/best_error_detection.pth \
  --hidden_dim 1024 \
  --num_phonemes 42 \
  --num_error_types 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 15 \
  --max_grad_norm 0.5 \
  --use_scheduler \
  --evaluate_every_epoch \
  --show_samples \
  --num_sample_show 5 \
  --output_dir models \
  --result_dir results

# 3. 통합 모델 평가
echo "=== 최종 모델 평가 ==="
python evaluate.py \
  --mode both \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --error_model_checkpoint models/best_error_detection.pth \
  --phoneme_model_checkpoint models/best_phoneme_recognition.pth \
  --pretrained_model facebook/wav2vec2-large-xlsr-53 \
  --hidden_dim 1024 \
  --num_phonemes 42 \
  --num_error_types 3 \
  --batch_size 8 \
  --output_dir evaluation_results \
  --detailed

echo "=== 학습 및 평가 완료 ==="
echo "결과 확인:"
echo "  - 학습 로그: results/"
echo "  - 모델 체크포인트: models/"
echo "  - 평가 결과: evaluation_results/"