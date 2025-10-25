"""Main training script for L2 pronunciation assessment model.

This script handles model training with support for two training modes:
- phoneme_only: Phoneme recognition only
- phoneme_error: Phoneme recognition + error detection

The script supports checkpoint resumption, hyperparameter overriding,
and automatic model architecture detection.
"""

import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from src.data.dataset import UnifiedDataset, collate_fn
from src.models.unified_model import UnifiedModel
from src.models.losses import UnifiedLoss
from src.training.trainer import UnifiedTrainer
from src.training.utils import (
    seed_everything, setup_experiment_dirs, save_checkpoint, 
    load_checkpoint, get_model_class, detect_model_type_from_checkpoint
)
from src.evaluation.evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='L2 Pronunciation Assessment Model Training')
    parser.add_argument('--training_mode', type=str, 
                       choices=['phoneme_only', 'phoneme_error'], 
                       help='Training mode')
    parser.add_argument('--model_type', type=str, 
                       choices=['simple', 'transformer'], 
                       help='Model architecture')
    parser.add_argument('--config', type=str, 
                       help='Configuration overrides (key=value format)')
    parser.add_argument('--train_data', type=str, 
                       help='Training data path override')
    parser.add_argument('--val_data', type=str, 
                       help='Validation data path override')
    parser.add_argument('--eval_data', type=str, 
                       help='Evaluation data path override')
    parser.add_argument('--phoneme_map', type=str, 
                       help='Phoneme map path override')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory override')
    parser.add_argument('--resume', type=str, 
                       help='Resume training from checkpoint')
    parser.add_argument('--experiment_name', type=str, 
                       help='Experiment name override')
    
    args = parser.parse_args()

    # Load and override configuration
    config = Config()
    
    if args.training_mode:
        config.training_mode = args.training_mode
    if args.model_type:
        config.model_type = args.model_type
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.output_dir:
        config.output_dir = args.output_dir

    # Auto-detect model type from checkpoint if resuming
    if args.resume:
        detected_model_type = detect_model_type_from_checkpoint(args.resume)
        config.model_type = detected_model_type
        logger.info(f"Auto-detected model type from checkpoint: {detected_model_type}")

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)

    # Apply individual configuration overrides
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

    # Set random seed for reproducibility
    seed_everything(config.seed)
    
    # Setup experiment directories
    setup_experiment_dirs(config, resume=bool(args.resume))

    # Load phoneme mapping
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = config.get_error_type_names()

    # Create model
    model = UnifiedModel(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **config.get_model_config()
    )

    # Enable multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(config.device)

    # Create loss function
    criterion = UnifiedLoss(
        training_mode=config.training_mode,
        error_weight=config.error_weight,
        phoneme_weight=config.phoneme_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma
    )

    # Create trainer
    trainer = UnifiedTrainer(model, config, config.device, logger)
    wav2vec_optimizer, main_optimizer = trainer.get_optimizers()

    # Create datasets and dataloaders
    train_dataset = UnifiedDataset(
        config.train_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    val_dataset = UnifiedDataset(
        config.val_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    eval_dataset = UnifiedDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )

    # Create evaluator
    evaluator = UnifiedEvaluator(config.device)

    # Initialize training state
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(
            args.resume, model, wav2vec_optimizer, main_optimizer, config.device
        )
        if 'error_accuracy' in resume_metrics:
            best_error_accuracy = resume_metrics['error_accuracy']
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        checkpoint = torch.load(args.resume, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        logger.info("=" * 50)
        logger.info("Resuming Training")
        logger.info("=" * 50)
        logger.info(f"Training mode: {config.training_mode}")
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Current best error accuracy: {best_error_accuracy:.4f}")
        logger.info(f"Current best phoneme accuracy: {best_phoneme_accuracy:.4f}")
        logger.info(f"Current best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting training with mode: {config.training_mode}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Experiment name: {config.experiment_name}")
        logger.info(f"Training for {config.num_epochs} epochs")
        logger.info(f"SpecAugment enabled: {config.wav2vec2_specaug}")
        logger.info(f"Error detection enabled: {config.has_error_component()}")
        logger.info(f"Using Focal Loss with default parameters")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Training
        train_loss = trainer.train_epoch(train_dataloader, criterion, epoch)
        
        # Validation
        val_loss = trainer.validate_epoch(val_dataloader, criterion)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Show sample predictions
        logger.info(f"Epoch {epoch} - Sample Predictions")
        logger.info("=" * 50)
        evaluator.show_sample_predictions(
            model=model,
            eval_dataloader=eval_dataloader,
            id_to_phoneme=id_to_phoneme,
            logger=logger,
            training_mode=config.training_mode,
            error_type_names=error_type_names
        )

        # Evaluate phoneme recognition
        logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
        phoneme_recognition_results = evaluator.evaluate_phoneme_recognition(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")

        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        current_error_accuracy = 0.0

        # Evaluate error detection if applicable
        if config.has_error_component():
            logger.info(f"Epoch {epoch}: Evaluating error detection...")
            error_detection_results = evaluator.evaluate_error_detection(
                model=model,
                dataloader=eval_dataloader,
                training_mode=config.training_mode,
                error_type_names=error_type_names
            )
            logger.info(f"Error Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"Error Weighted F1: {error_detection_results['weighted_f1']:.4f}")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                if error_type != 'blank':
                    logger.info(f"  {error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            current_error_accuracy = error_detection_results['token_accuracy']

        # Save checkpoints
        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_error.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best error accuracy: {best_error_accuracy:.4f}")

        if config.save_best_phoneme and current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best phoneme accuracy: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")

        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Save latest checkpoint
        latest_metrics = {
            'error_accuracy': best_error_accuracy,
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                      epoch, val_loss, train_loss, latest_metrics, latest_path)

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final metrics
    final_metrics = {
        'best_error_accuracy': best_error_accuracy,
        'best_phoneme_accuracy': best_phoneme_accuracy,
        'best_val_loss': best_val_loss,
        'completed_epochs': config.num_epochs,
        'training_mode': config.training_mode,
        'model_type': config.model_type,
        'experiment_name': config.experiment_name
    }
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training completed!")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Best error accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best phoneme accuracy: {best_phoneme_accuracy:.4f}")
    logger.info(f"Final metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
