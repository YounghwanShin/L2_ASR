#!/bin/bash

mkdir -p models results evaluation_results

echo "=== Training Error Detection Model ==="
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
  --use_entropy_reg \
  --initial_beta 0.2 \
  --target_entropy_factor 1.1 \
  --use_scheduler \
  --evaluate_every_epoch \
  --show_samples \
  --output_dir models \
  --result_dir results

echo "=== Training Phoneme Recognition Model ==="
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

echo "=== Final Model Evaluation ==="
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

echo "=== Training and Evaluation Complete ==="
echo "Check results in:"
echo "  - Training logs: results/"
echo "  - Model checkpoints: models/"
echo "  - Evaluation results: evaluation_results/"