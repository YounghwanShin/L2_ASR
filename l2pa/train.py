"""Training script for pronunciation assessment model.

This script handles model training with support for three training modes:
- phoneme_only: Perceived phoneme recognition only
- phoneme_error: Perceived phoneme recognition + error detection
- multitask: Canonical + perceived phoneme recognition + error detection

Also supports K-fold cross-validation for robust model evaluation.
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
    load_checkpoint
)
from .evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def train_single_fold(config, fold_idx=None, resume_checkpoint=None):
    """Trains model for a single fold or standard training.
    
    Args:
        config: Configuration object.
        fold_idx: Fold index for cross-validation (None for standard training).
        resume_checkpoint: Path to checkpoint for resuming training.
        
    Returns:
        Dictionary containing final metrics for this fold.
    """
    # Set random seed
    set_random_seed(config.seed)
    
    # Update config for current fold
    if fold_idx is not None:
        config.current_fold = fold_idx
        config.__post_init__()
    
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
        canonical_weight=config.canonical_weight,
        perceived_weight=config.perceived_weight,
        error_weight=config.error_weight,
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
    best_canonical_accuracy = 0.0
    best_perceived_accuracy = 0.0
    best_error_accuracy = 0.0
    start_epoch = 1

    # Resume from checkpoint if specified
    if resume_checkpoint:
        start_epoch, resume_metrics = load_checkpoint(
            resume_checkpoint, model, wav2vec_optimizer, main_optimizer, config.device
        )
        best_canonical_accuracy = resume_metrics.get('canonical_accuracy', 0.0)
        best_perceived_accuracy = resume_metrics.get('perceived_accuracy', 0.0)
        best_error_accuracy = resume_metrics.get('error_accuracy', 0.0)
        checkpoint = torch.load(resume_checkpoint, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        logger.info("=" * 50)
        logger.info("Resuming Training")
        logger.info("=" * 50)
        logger.info(f"Training mode: {config.training_mode}")
        if fold_idx is not None:
            logger.info(f"Fold: {fold_idx}")
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best metrics: {resume_metrics}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting training with mode: {config.training_mode}")
        logger.info(f"Model type: {config.model_type}")
        if fold_idx is not None:
            logger.info(f"Fold: {fold_idx}/{config.num_folds}")
        logger.info(f"Experiment name: {config.experiment_name}")
        logger.info(f"Training for {config.num_epochs} epochs")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Training
        train_loss = trainer.train_epoch(train_dataloader, criterion, epoch)
        
        # Validation
        val_loss = trainer.validate_epoch(val_dataloader, criterion)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Evaluate all tasks
        current_canonical_accuracy = 0.0
        current_perceived_accuracy = 0.0
        current_error_accuracy = 0.0

        # Evaluate canonical phoneme recognition (multitask only)
        if config.has_canonical_component():
            logger.info(f"Epoch {epoch}: Evaluating canonical phoneme recognition...")
            canonical_results = evaluator.evaluate_canonical_recognition(
                model=model,
                dataloader=eval_dataloader,
                training_mode=config.training_mode,
                id_to_phoneme=id_to_phoneme
            )
            logger.info(f"Canonical PER: {canonical_results['per']:.4f}")
            current_canonical_accuracy = 1.0 - canonical_results['per']

        # Evaluate perceived phoneme recognition
        logger.info(f"Epoch {epoch}: Evaluating perceived phoneme recognition...")
        perceived_results = evaluator.evaluate_perceived_recognition(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"Perceived PER: {perceived_results['per']:.4f}")
        current_perceived_accuracy = 1.0 - perceived_results['per']

        # Evaluate error detection
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
        metrics = {
            'canonical_accuracy': best_canonical_accuracy,
            'perceived_accuracy': best_perceived_accuracy,
            'error_accuracy': best_error_accuracy
        }

        if config.save_best_canonical and current_canonical_accuracy > best_canonical_accuracy:
            best_canonical_accuracy = current_canonical_accuracy
            metrics['canonical_accuracy'] = best_canonical_accuracy
            model_path = os.path.join(config.output_dir, 'best_canonical.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best canonical accuracy: {best_canonical_accuracy:.4f}")

        if config.save_best_perceived and current_perceived_accuracy > best_perceived_accuracy:
            best_perceived_accuracy = current_perceived_accuracy
            metrics['perceived_accuracy'] = best_perceived_accuracy
            model_path = os.path.join(config.output_dir, 'best_perceived.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best perceived accuracy: {best_perceived_accuracy:.4f}")

        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            metrics['error_accuracy'] = best_error_accuracy
            model_path = os.path.join(config.output_dir, 'best_error.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best error accuracy: {best_error_accuracy:.4f}")

        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Save latest checkpoint
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                      epoch, val_loss, train_loss, metrics, latest_path)

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final metrics
    final_metrics = {
        'best_canonical_accuracy': best_canonical_accuracy,
        'best_perceived_accuracy': best_perceived_accuracy,
        'best_error_accuracy': best_error_accuracy,
        'best_val_loss': best_val_loss,
        'completed_epochs': config.num_epochs,
        'training_mode': config.training_mode,
        'model_type': config.model_type,
        'fold': fold_idx
    }
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training completed!")
    logger.info(f"Best canonical accuracy: {best_canonical_accuracy:.4f}")
    logger.info(f"Best perceived accuracy: {best_perceived_accuracy:.4f}")
    logger.info(f"Best error accuracy: {best_error_accuracy:.4f}")

    return final_metrics


def train_model(config, resume_checkpoint=None):
    """Main training function with cross-validation support.
    
    Args:
        config: Configuration object.
        resume_checkpoint: Path to checkpoint for resuming training.
    """
    if config.use_cross_validation:
        logger.info(f"Starting {config.num_folds}-fold cross-validation...")
        
        all_fold_metrics = []
        
        for fold_idx in range(config.num_folds):
            logger.info("\n" + "="*80)
            logger.info(f"Training Fold {fold_idx + 1}/{config.num_folds}")
            logger.info("="*80)
            
            fold_metrics = train_single_fold(config, fold_idx=fold_idx)
            all_fold_metrics.append(fold_metrics)
        
        # Aggregate cross-validation results
        avg_metrics = {}
        metric_keys = ['best_canonical_accuracy', 'best_perceived_accuracy', 'best_error_accuracy', 'best_val_loss']
        
        for key in metric_keys:
            values = [m[key] for m in all_fold_metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
                avg_metrics[f'std_{key}'] = (sum((v - avg_metrics[f'avg_{key}'])**2 for v in values) / len(values))**0.5
        
        # Save cross-validation summary
        cv_summary_path = os.path.join(config.base_experiment_dir, f'{config.experiment_name}_cv_summary.json')
        cv_summary = {
            'num_folds': config.num_folds,
            'training_mode': config.training_mode,
            'model_type': config.model_type,
            'fold_metrics': all_fold_metrics,
            'average_metrics': avg_metrics
        }
        
        with open(cv_summary_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("Cross-Validation Summary")
        logger.info("="*80)
        for key, value in avg_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info(f"Results saved to: {cv_summary_path}")
        logger.info("="*80)
    else:
        # Standard single-split training
        train_single_fold(config, fold_idx=None, resume_checkpoint=resume_checkpoint)
