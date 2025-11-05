"""Main entry point for training and evaluation."""
import argparse
import json
import logging
import os

from l2pa.config import Config
from l2pa.evaluate import evaluate_model
from l2pa.train import train_model
from l2pa.training.utils import detect_model_architecture


def run_cross_validation(config):
  stats_path = os.path.join(config.data_root_dir, 'split_statistics.json')
  if not os.path.exists(stats_path):
    raise FileNotFoundError(f'Split statistics not found. Run preprocessing first.')
  
  with open(stats_path, 'r') as f:
    stats = json.load(f)
  
  num_folds = stats['num_folds']
  logging.info(f'Starting CV with {num_folds} folds')
  
  for fold_idx in range(num_folds):
    logging.info(f'\n{"="*80}')
    logging.info(f'Training Fold {fold_idx}/{num_folds-1}')
    logging.info(f'{"="*80}\n')
    
    fold_config = Config()
    for attr in dir(config):
      if not attr.startswith('_') and not callable(getattr(config, attr)):
        setattr(fold_config, attr, getattr(config, attr))
    
    fold_config.cv_fold_index = fold_idx
    fold_config.use_cross_validation = True
    fold_config.__post_init__()
    
    train_model(fold_config)
    logging.info(f'Fold {fold_idx} completed\n')
  
  logging.info(f'Cross-validation completed!')


def main():
  parser = argparse.ArgumentParser(description='L2 Pronunciation Assessment')
  subparsers = parser.add_subparsers(dest='command', help='Command')
  
  train_parser = subparsers.add_parser('train', help='Train model')
  train_parser.add_argument('--training_mode', type=str, choices=['phoneme_only', 'phoneme_error', 'multitask'], help='Training mode')
  train_parser.add_argument('--model_type', type=str, choices=['simple', 'transformer'], help='Model architecture')
  train_parser.add_argument('--cv_fold', type=int, help='Cross-validation fold')
  train_parser.add_argument('--no_cv', action='store_true', help='Disable cross-validation')
  train_parser.add_argument('--config', type=str, help='Config overrides (key=value,key=value)')
  train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
  train_parser.add_argument('--experiment_name', type=str, help='Experiment name')
  
  eval_parser = subparsers.add_parser('eval', help='Evaluate model')
  eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
  eval_parser.add_argument('--training_mode', type=str, choices=['phoneme_only', 'phoneme_error', 'multitask'], help='Training mode')
  eval_parser.add_argument('--model_type', type=str, help='Model type')
  
  args = parser.parse_args()
  
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
  if args.command == 'train':
    config = Config()
    if args.training_mode:
      config.training_mode = args.training_mode
    if args.model_type:
      config.model_type = args.model_type
    if args.experiment_name:
      config.experiment_name = args.experiment_name
    if args.no_cv:
      config.use_cross_validation = False
    if args.resume:
      config.model_type = detect_model_architecture(args.resume)
    if args.config:
      for override in args.config.split(','):
        key, value = override.split('=')
        if hasattr(config, key):
          attr_type = type(getattr(config, key))
          if attr_type == bool:
            setattr(config, key, value.lower() == 'true')
          else:
            try:
              setattr(config, key, attr_type(value))
            except:
              setattr(config, key, value)
    
    config.__post_init__()
    
    if config.use_cross_validation and args.cv_fold is None:
      run_cross_validation(config)
    else:
      if args.cv_fold is not None:
        config.cv_fold_index = args.cv_fold
        config.use_cross_validation = True
        config.__post_init__()
      train_model(config, resume_checkpoint=args.resume)
  
  elif args.command == 'eval':
    config = Config()
    if args.training_mode:
      config.training_mode = args.training_mode
    if args.model_type:
      config.model_type = args.model_type
    else:
      config.model_type = detect_model_architecture(args.checkpoint)
      logging.info(f'Auto-detected model type: {config.model_type}')
    
    config.__post_init__()
    evaluate_model(args.checkpoint, config)
  else:
    parser.print_help()


if __name__ == '__main__':
  main()
