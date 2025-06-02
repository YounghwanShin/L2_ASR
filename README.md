python train.py \
  --train_data data/complete_train.json \
  --val_data data/complete_val.json \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --hidden_dim 1024 \
  --num_phonemes 42 \
  --num_error_types 3 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --num_epochs 30 \
  --use_cross_attention \
  --adaptive_weights \
  --evaluate_every_epoch \
  --show_samples

# 평가
python eval.py \
  --eval_data data/eval.json \
  --phoneme_map data/phoneme_to_id.json \
  --model_checkpoint models/best_multitask_phoneme.pth \
  --detailed