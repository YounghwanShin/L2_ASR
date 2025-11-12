"""Main entry point for training and evaluation with cross-validation support."""

import argparse
import logging
import os
import copy

from l2pa.config import Config
from l2pa.evaluate import evaluate_model
from l2pa.train import train_model
from l2pa.training.utils import detect_model_type_from_checkpoint


def create_fold_config(base_config, fold_idx):
  """Creates a configuration for a specific fold.
  
  Args:
    base_config: Base configuration object.
    fold_idx: Fold index for cross-validation.
    
  Returns:
    Configuration object for the specified fold.
  """
  fold_config = Config()
  
  # Copy all attributes from base config
  for attr in dir(base_config):
    if not attr.startswith('_') and not callable(getattr(base_config, attr)):
      try:
        setattr(fold_config, attr, getattr(base_config, attr))
      except AttributeError:
        pass
  
  # Set fold-specific attributes
  fold_config.cv_fold = fold_idx
  fold_config.use_cross_validation = True
  fold_config.experiment_name = None  # Force regeneration with fold info
  
  # Reinitialize to set up paths correctly
  fold_config.__post_init__()
  
  return fold_config


def run_cross_validation(config):
  """Runs cross-validation training for all folds.
  
  Args:
    config: Base configuration object.
  """
  import json
  stats_path = os.path.join(config.data_dir, 'split_statistics.json')
  
  if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"Split statistics not found at {stats_path}. "
        "Please run preprocessing with cross-validation first."
    )
  
  with open(stats_path, 'r') as f:
    stats = json.load(f)
  
  num_folds = stats['num_folds']
  logging.info(f"Starting cross-validation with {num_folds} folds")
  
  for fold_idx in range(num_folds):
    logging.info(f"\n{'='*80}")
    logging.info(f"Training Fold {fold_idx}/{num_folds-1}")
    logging.info(f"{'='*80}\n")
    
    # Create fold-specific configuration
    fold_config = create_fold_config(config, fold_idx)
    
    logging.info(f"Experiment directory: {fold_config.experiment_dir}")
    
    # Train the fold
    train_model(fold_config)
    
    logging.info(f"Fold {fold_idx} training completed\n")
  
  logging.info(f"Cross-validation completed for all {num_folds} folds!")


def main():
  """Main function."""
  parser = argparse.ArgumentParser(
      description='L2 Pronunciation Assessment with Cross-Validation'
  )
  subparsers = parser.add_subparsers(dest='command', help='Command to run')
  
  # Training arguments
  train_parser = subparsers.add_parser('train', help='Train model')
  train_parser.add_argument(
      '--training_mode', type=str,
      choices=['phoneme_only', 'phoneme_error', 'multitask'],
      help='Training mode'
  )
  train_parser.add_argument(
      '--model_type', type=str, choices=['simple', 'transformer'],
      help='Model architecture'
  )
  train_parser.add_argument(
      '--pretrained_model', type=str,
      help='Pretrained Wav2Vec2 model name'
  )
  train_parser.add_argument(
      '--cv_fold', type=int,
      help='Cross-validation fold (0-indexed). If not specified, trains all folds.'
  )
  train_parser.add_argument(
      '--no_cv', action='store_true',
      help='Disable cross-validation (use single train/val split)'
  )
  train_parser.add_argument(
      '--data_split_mode', type=str,
      choices=['cv', 'disjoint', 'standard'],
      default='standard',
      help='Data split mode: cv (cross-validation), disjoint (no transcript overlap), or standard (simple split)'
  )
  train_parser.add_argument(
      '--config', type=str,
      help='Config overrides (key=value format, comma-separated)'
  )
  train_parser.add_argument(
      '--resume', type=str,
      help='Resume from checkpoint'
  )
  train_parser.add_argument(
      '--experiment_name', type=str,
      help='Experiment name'
  )
  
  # Evaluation arguments
  eval_parser = subparsers.add_parser('eval', help='Evaluate model')
  eval_parser.add_argument(
      '--checkpoint', type=str, required=True,
      help='Model checkpoint path'
  )
  eval_parser.add_argument(
      '--training_mode', type=str,
      choices=['phoneme_only', 'phoneme_error', 'multitask'],
      help='Training mode'
  )
  eval_parser.add_argument(
      '--model_type', type=str,
      help='Model type (auto-detected if not specified)'
  )
  eval_parser.add_argument(
      '--pretrained_model', type=str,
      help='Pretrained Wav2Vec2 model name'
  )
  eval_parser.add_argument(
      '--data_split_mode', type=str,
      choices=['cv', 'disjoint', 'standard'],
      default='standard',
      help='Data split mode for test set selection'
  )
  
  args = parser.parse_args()
  
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
  
  if args.command == 'train':
    config = Config()
    
    # Apply command-line arguments
    if args.training_mode:
      config.training_mode = args.training_mode
    if args.model_type:
      config.model_type = args.model_type
    if args.pretrained_model:
      config.pretrained_model = args.pretrained_model
    if args.experiment_name:
      config.experiment_name = args.experiment_name
    if args.no_cv:
      config.use_cross_validation = False
      config.data_split_mode = args.data_split_mode
    
    # Auto-detect model type if resuming
    if args.resume:
      detected_type = detect_model_type_from_checkpoint(args.resume)
      config.model_type = detected_type
    
    # Apply additional config overrides
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
    
    # Reinitialize to apply all changes
    config.__post_init__()
    
    # Run training based on mode
    if config.use_cross_validation and args.cv_fold is None:
      run_cross_validation(config)
    else:
      if args.cv_fold is not None:
        config.cv_fold = args.cv_fold
        config.use_cross_validation = True
        config.experiment_name = None  # Force regeneration
        config.__post_init__()
      train_model(config, resume_checkpoint=args.resume)
  
  elif args.command == 'eval':
    config = Config()
    
    # Apply evaluation arguments
    if args.training_mode:
      config.training_mode = args.training_mode
    
    if args.pretrained_model:
      config.pretrained_model = args.pretrained_model
    
    if args.model_type:
      config.model_type = args.model_type
    else:
      config.model_type = detect_model_type_from_checkpoint(args.checkpoint)
      logging.info(f"Auto-detected model type: {config.model_type}")
    
    config.use_cross_validation = False
    config.data_split_mode = args.data_split_mode
    config.__post_init__()
    
    evaluate_model(args.checkpoint, config)
  
  else:
    parser.print_help()


if __name__ == "__main__":
  main()