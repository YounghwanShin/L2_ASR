#!/bin/bash

mkdir -p models results evaluation_results

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
  --num_epochs 100 \
  --max_grad_norm 0.5 \
  --use_scheduler \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --evaluate_every_epoch \
  --show_samples \
  --num_sample_show 3 \
  --use_entropy_reg \
  --initial_beta 0.02 \
  --target_entropy_factor 0.6 \
  --output_dir models \
  --result_dir results

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
  --num_epochs 20 \
  --max_grad_norm 0.5 \
  --use_scheduler \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --evaluate_every_epoch \
  --show_samples \
  --num_sample_show 3 \
  --output_dir models \
  --result_dir results

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