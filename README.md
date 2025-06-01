# 1. 데이터 증강
python data_augmentation.py \
  --phoneme_data data/perceived_train.json \
  --output_path data/augmented_train.json \
  --num_workers 8

python data_augmentation.py \
  --phoneme_data data/perceived_val.json \
  --output_path data/augmented_val.json \
  --num_workers 8

# 2. 데이터셋 병합
python merge_datasets.py \
  --error_data data/errors_train.json \
  --phoneme_data data/augmented_train.json \
  --output_path data/complete_train.json

python merge_datasets.py \
  --error_data data/errors_val.json \
  --phoneme_data data/augmented_val.json \
  --output_path data/complete_val.json

# 3. Multi-task 훈련
python train.py \
  --train_data data/complete_train.json \
  --val_data data/complete_val.json \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --hidden_dim 1024 \
  --num_phonemes 42 \
  --num_error_types 3 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --num_epochs 30 \
  --use_cross_attention \
  --adaptive_weights \
  --evaluate_every_epoch \
  --show_samples

# 4. 최종 평가
python multitask_evaluate_final.py \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --model_checkpoint models/best_multitask_phoneme.pth \
  --detailed