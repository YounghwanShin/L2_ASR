"""Main entry point for training and evaluation.

This script provides a unified interface for training and evaluating
pronunciation assessment models.
"""

import argparse
import logging

from l2pa.config import Config
from l2pa.train import train_model
from l2pa.evaluate import evaluate_model
from l2pa.training.utils import detect_model_type_from_checkpoint


def main():
    """Main function for training and evaluation."""
    parser = argparse.ArgumentParser(description='L2 Pronunciation Assessment')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--training_mode', type=str, 
                             choices=['phoneme_only', 'phoneme_error'],
                             help='Training mode')
    train_parser.add_argument('--model_type', type=str,
                             choices=['simple', 'transformer'],
                             help='Model architecture')
    train_parser.add_argument('--config', type=str,
                             help='Config overrides (key=value format)')
    train_parser.add_argument('--resume', type=str,
                             help='Resume from checkpoint')
    train_parser.add_argument('--experiment_name', type=str,
                             help='Experiment name')
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Model checkpoint path')
    eval_parser.add_argument('--training_mode', type=str,
                            choices=['phoneme_only', 'phoneme_error'],
                            help='Training mode')
    eval_parser.add_argument('--model_type', type=str,
                            help='Model type (auto-detected if not specified)')
    eval_parser.add_argument('--save_predictions', action='store_true',
                            help='Save detailed predictions')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == 'train':
        # Load config
        config = Config()
        
        # Override config
        if args.training_mode:
            config.training_mode = args.training_mode
        if args.model_type:
            config.model_type = args.model_type
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        
        # Auto-detect model type if resuming
        if args.resume:
            detected_type = detect_model_type_from_checkpoint(args.resume)
            config.model_type = detected_type
        
        # Apply config overrides
        if args.config:
            for override in args.config.split(','):
                key, value = override.split('=')
                if hasattr(config, key):
                    attr_type = type(getattr(config, key))
                    if attr_type == bool:
                        setattr(config, key, value.lower() == 'true')
                    else:
                        setattr(config, key, attr_type(value))
        
        config.__post_init__()
        
        # Train model
        train_model(config, resume_checkpoint=args.resume)
        
    elif args.command == 'eval':
        # Load config
        config = Config()
        
        # Override config
        if args.training_mode:
            config.training_mode = args.training_mode
        
        # Auto-detect or use specified model type
        if args.model_type:
            config.model_type = args.model_type
        else:
            config.model_type = detect_model_type_from_checkpoint(args.checkpoint)
            logging.info(f"Auto-detected model type: {config.model_type}")
        
        config.__post_init__()
        
        # Evaluate model
        evaluate_model(args.checkpoint, config, args.save_predictions)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()