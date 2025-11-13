"""Main entry point for training and evaluation with speaker-based splits."""

import argparse
import logging
import os
import copy

from l2pa.config import Config
from l2pa.evaluate import evaluate_model
from l2pa.train import train_model
from l2pa.training.utils import detect_model_type_from_checkpoint


def create_split_config(base_config, split_idx):
  """Create a configuration for a specific data split.
  
  Args:
    base_config: Base configuration object
    split_idx: Split index for speaker-based evaluation
    
  Returns:
    Configuration object for the specified split
  """
  split_config = Config()
  
  # Copy all attributes from base config
  for attr in dir(base_config):
    if not attr.startswith('_') and not callable(getattr(base_config, attr)):
      try:
        setattr(split_config, attr, getattr(base_config, attr))
      except AttributeError:
        pass
  
  # Set split-specific attributes
  split_config.split_index = split_idx
  split_config.use_speaker_splits = True
  split_config.experiment_name = None  # Force regeneration with split info
  
  # Reinitialize to set up paths correctly
  split_config.__post_init__()
  
  return split_config


def run_all_splits(config):
  """Run training for all speaker-based splits.
  
  Args:
    config: Base configuration object
  """
  import json
  stats_path = os.path.join(config.data_dir, 'split_statistics.json')
  
  if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"Split statistics not found at {stats_path}. "
        "Please run preprocessing with speaker splits first."
    )
  
  with open(stats_path, 'r') as f:
    stats = json.load(f)
  
  num_splits = stats['num_folds']
  logging.info(f"Starting training with {num_splits} speaker-based splits")
  
  for split_idx in range(num_splits):
    logging.info(f"\n{'='*80}")
    logging.info(f"Training Split {split_idx}/{num_splits-1}")
    logging.info(f"{'='*80}\n")
    
    # Create split-specific configuration
    split_config = create_split_config(config, split_idx)
    
    logging.info(f"Experiment directory: {split_config.experiment_dir}")
    
    # Train the split
    train_model(split_config)
    
    logging.info(f"Split {split_idx} training completed\n")
  
  logging.info(f"Training completed for all {num_splits} splits!")


def main():
  """Main function for training and evaluation."""
  parser = argparse.ArgumentParser(
      description='L2 Pronunciation Error Detection System'
  )
  subparsers = parser.add_subparsers(dest='command', help='Command to run')
  
  # Training arguments
  train_parser = subparsers.add_parser('train', help='Train model')
  train_parser.add_argument(
      '--training_mode', type=str,
      choices=['phoneme_only', 'phoneme_error', 'multitask'],
      help='Training objective: phoneme recognition and/or error detection'
  )
  train_parser.add_argument(
      '--model_type', type=str, choices=['simple', 'transformer'],
      help='Model architecture type'
  )
  train_parser.add_argument(
      '--pretrained_model', type=str,
      help='Pretrained Wav2Vec2 model name from HuggingFace'
  )
  train_parser.add_argument(
      '--split_index', type=int,
      help='Speaker-based split index (0-indexed). If not specified, trains all splits.'
  )
  train_parser.add_argument(
      '--no_speaker_splits', action='store_true',
      help='Disable speaker-based splits (use single train/val/test split)'
  )
  train_parser.add_argument(
      '--data_split_mode', type=str,
      choices=['speaker', 'disjoint', 'standard'],
      default='standard',
      help='Data split mode: speaker (speaker-based), disjoint (no transcript overlap), or standard'
  )
  train_parser.add_argument(
      '--config', type=str,
      help='Config overrides (key=value format, comma-separated)'
  )
  train_parser.add_argument(
      '--resume', type=str,
      help='Resume training from checkpoint'
  )
  train_parser.add_argument(
      '--experiment_name', type=str,
      help='Custom experiment name'
  )
  
  # Evaluation arguments
  eval_parser = subparsers.add_parser('eval', help='Evaluate model')
  eval_parser.add_argument(
      '--checkpoint', type=str, required=True,
      help='Path to model checkpoint'
  )
  eval_parser.add_argument(
      '--training_mode', type=str,
      choices=['phoneme_only', 'phoneme_error', 'multitask'],
      help='Training mode used for the model'
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
      choices=['speaker', 'disjoint', 'standard'],
      default='standard',
      help='Data split mode for test set selection'
  )
  
  args = parser.parse_args()
  
  # Set up logging
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
    if args.no_speaker_splits:
      config.use_speaker_splits = False
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
    if config.use_speaker_splits and args.split_index is None:
      # Train all splits
      run_all_splits(config)
    else:
      # Train single split
      if args.split_index is not None:
        config.split_index = args.split_index
        config.use_speaker_splits = True
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
    
    config.use_speaker_splits = False
    config.data_split_mode = args.data_split_mode
    config.__post_init__()
    
    evaluate_model(args.checkpoint, config)
  
  else:
    parser.print_help()


if __name__ == "__main__":
  main()