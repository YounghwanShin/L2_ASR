"""Training script for pronunciation assessment model.

This script handles model training with support for two training modes:
- phoneme_only: Phoneme recognition only
- phoneme_error: Phoneme recognition + error detection
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import PronunciationDataset, collate_batch
from .models.unified_model import UnifiedModel
from .models.losses import UnifiedLoss
from .training.trainer import ModelTrainer
from .training.utils import (
    set_random_seed, setup_experiment_directories, save_checkpoint, 
    load_checkpoint, detect_model_type_from_checkpoint
)
from .evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def train_model(config, resume_checkpoint=None):
    """Main training function.
    
    Args:
        config: Configuration object.
        resume_checkpoint: Path to checkpoint for resuming training.
    """
    # Set random seed
    set_random_seed(config.seed)
    
    # Setup directories
    setup_experiment_directories(config, resume=bool(resume_checkpoint))

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
    trainer = ModelTrainer(model, config, config.device, logger)
    wav2vec_optimizer, main_optimizer = trainer.get_optimizers()

    # Create datasets
    train_dataset = PronunciationDataset(
        config.train_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    val_dataset = PronunciationDataset(
        config.val_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    eval_dataset = PronunciationDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, training_mode=config.training_mode)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, training_mode=config.training_mode)
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, training_mode=config.training_mode)
    )

    # Create evaluator
    evaluator = ModelEvaluator(config.device)

    # Initialize training state
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

    # Resume from checkpoint if specified
    if resume_checkpoint:
        start_epoch, resume_metrics = load_checkpoint(
            resume_checkpoint, model, wav2vec_optimizer, main_optimizer, config.device
        )
        if 'error_accuracy' in resume_metrics:
            best_error_accuracy = resume_metrics['error_accuracy']
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        checkpoint = torch.load(resume_checkpoint, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        logger.info("=" * 50)
        logger.info("Resuming Training")
        logger.info("=" * 50)
        logger.info(f"Training mode: {config.training_mode}")
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best error accuracy: {best_error_accuracy:.4f}")
        logger.info(f"Best phoneme accuracy: {best_phoneme_accuracy:.4f}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting training with mode: {config.training_mode}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Experiment name: {config.experiment_name}")
        logger.info(f"Training for {config.num_epochs} epochs")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Training
        train_loss = trainer.train_epoch(train_dataloader, criterion, epoch)
        
        # Validation
        val_loss = trainer.validate_epoch(val_dataloader, criterion)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Evaluate phoneme recognition
        logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
        phoneme_results = evaluator.evaluate_phoneme_recognition(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"Phoneme Error Rate (PER): {phoneme_results['per']:.4f}")
        logger.info(f"Phoneme Accuracy: {1.0 - phoneme_results['per']:.4f}")

        current_phoneme_accuracy = 1.0 - phoneme_results['per']
        current_error_accuracy = 0.0

        # Evaluate error detection if applicable
        if config.has_error_component():
            logger.info(f"Epoch {epoch}: Evaluating error detection...")
            error_results = evaluator.evaluate_error_detection(
                model=model,
                dataloader=eval_dataloader,
                training_mode=config.training_mode,
                error_type_names=error_type_names
            )
            logger.info(f"Error Token Accuracy: {error_results['token_accuracy']:.4f}")
            current_error_accuracy = error_results['token_accuracy']

        # Save checkpoints
        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_results['per']
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
                'per': phoneme_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best phoneme accuracy: {best_phoneme_accuracy:.4f}")

        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Save latest checkpoint
        latest_metrics = {
            'error_accuracy': best_error_accuracy,
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_results['per']
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
    logger.info(f"Best error accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best phoneme accuracy: {best_phoneme_accuracy:.4f}")