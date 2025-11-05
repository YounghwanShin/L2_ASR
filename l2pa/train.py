"""Training script with cross-validation support.

This module implements the main training logic including dataset loading,
model initialization, and training loop execution.
"""

import json
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import PronunciationDataset, collate_batch
from .evaluation.evaluator import ModelEvaluator
from .models.losses import UnifiedLoss
from .models.unified_model import UnifiedModel
from .training.trainer import ModelTrainer
from .training.utils import (
    load_checkpoint, 
    save_checkpoint,
    set_random_seed, 
    setup_experiment_directories
)

logger = logging.getLogger(__name__)


def train_model(config, resume_checkpoint=None):
  """Main training function with full training loop.
  
  Args:
    config: Configuration object containing hyperparameters.
    resume_checkpoint: Path to checkpoint file for resuming training.
  """
  set_random_seed(config.seed)
  setup_experiment_directories(config, resume=bool(resume_checkpoint))

  # Load phoneme mapping
  with open(config.phoneme_map, 'r') as f:
    phoneme_to_id = json.load(f)
  id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
  error_type_names = config.get_error_type_names()

  # Initialize model
  model = UnifiedModel(
      pretrained_model_name=config.pretrained_model,
      num_phonemes=config.num_phonemes,
      num_error_types=config.num_error_types,
      **config.get_model_config()
  )

  # Data parallel if multiple GPUs available
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

  model = model.to(config.device)

  # Initialize loss
  criterion = UnifiedLoss(
      training_mode=config.training_mode,
      canonical_weight=config.canonical_weight,
      perceived_weight=config.perceived_weight,
      error_weight=config.error_weight,
      focal_alpha=config.focal_alpha,
      focal_gamma=config.focal_gamma
  )

  # Initialize trainer and optimizers
  trainer = ModelTrainer(model, config, config.device, logger)
  wav2vec_optimizer, main_optimizer = trainer.get_optimizers()

  # Load datasets
  train_dataset = PronunciationDataset(
      config.train_data, 
      phoneme_to_id, 
      config.training_mode,
      config.max_length, 
      config.sampling_rate, 
      config.device
  )
  val_dataset = PronunciationDataset(
      config.val_data, 
      phoneme_to_id, 
      config.training_mode,
      config.max_length, 
      config.sampling_rate, 
      config.device
  )
  test_dataset = PronunciationDataset(
      config.test_data, 
      phoneme_to_id, 
      config.training_mode,
      config.max_length, 
      config.sampling_rate, 
      config.device
  )
  
  # Create dataloaders
  train_dataloader = DataLoader(
      train_dataset, 
      batch_size=config.batch_size, 
      shuffle=True,
      collate_fn=lambda batch: collate_batch(batch, config.training_mode)
  )
  val_dataloader = DataLoader(
      val_dataset, 
      batch_size=config.batch_size, 
      shuffle=False,
      collate_fn=lambda batch: collate_batch(batch, config.training_mode)
  )
  test_dataloader = DataLoader(
      test_dataset, 
      batch_size=config.eval_batch_size, 
      shuffle=False,
      collate_fn=lambda batch: collate_batch(batch, config.training_mode)
  )

  evaluator = ModelEvaluator(config.device)

  # Initialize best metrics tracking
  best_metrics = {
      'canonical_per': float('inf'),
      'perceived_per': float('inf'),
      'error_accuracy': 0.0,
      'val_loss': float('inf')
  }
  start_epoch = 1

  # Resume from checkpoint if specified
  if resume_checkpoint:
    start_epoch, best_metrics = load_checkpoint(
        resume_checkpoint, 
        model, 
        wav2vec_optimizer,
        main_optimizer, 
        config.device
    )

  # Training loop
  for epoch in range(start_epoch, config.num_epochs + 1):
    train_loss = trainer.train_epoch(train_dataloader, criterion, epoch)
    val_loss = trainer.validate_epoch(val_dataloader, criterion)
    
    logger.info(
        f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}"
    )

    # Evaluate on test set
    metrics = {}

    if config.has_canonical_task():
      canonical_results = evaluator.evaluate_phoneme_recognition(
          model, 
          test_dataloader, 
          config.training_mode,
          id_to_phoneme, 
          'canonical'
      )
      metrics['canonical_per'] = canonical_results['per']
      logger.info(f"Canonical PER: {canonical_results['per']:.4f}")

    if config.has_perceived_task():
      perceived_results = evaluator.evaluate_phoneme_recognition(
          model, 
          test_dataloader, 
          config.training_mode,
          id_to_phoneme, 
          'perceived'
      )
      metrics['perceived_per'] = perceived_results['per']
      logger.info(f"Perceived PER: {perceived_results['per']:.4f}")

    if config.has_error_task():
      error_results = evaluator.evaluate_error_detection(
          model, 
          test_dataloader, 
          config.training_mode,
          error_type_names
      )
      metrics['error_accuracy'] = error_results['token_accuracy']
      logger.info(f"Error Accuracy: {error_results['token_accuracy']:.4f}")

    # Save best checkpoints based on configured metrics
    for metric_name in config.save_best_metrics:
      should_save = False
      
      if metric_name == 'canonical' and 'canonical_per' in metrics:
        if metrics['canonical_per'] < best_metrics['canonical_per']:
          best_metrics['canonical_per'] = metrics['canonical_per']
          should_save = True
      elif metric_name == 'perceived' and 'perceived_per' in metrics:
        if metrics['perceived_per'] < best_metrics['perceived_per']:
          best_metrics['perceived_per'] = metrics['perceived_per']
          should_save = True
      elif metric_name == 'error' and 'error_accuracy' in metrics:
        if metrics['error_accuracy'] > best_metrics['error_accuracy']:
          best_metrics['error_accuracy'] = metrics['error_accuracy']
          should_save = True
      elif metric_name == 'loss':
        if val_loss < best_metrics['val_loss']:
          best_metrics['val_loss'] = val_loss
          should_save = True
      
      if should_save:
        path = os.path.join(
            config.checkpoint_dir, 
            f'best_{metric_name}.pth'
        )
        save_checkpoint(
            model, 
            wav2vec_optimizer, 
            main_optimizer,
            epoch, 
            val_loss, 
            train_loss, 
            best_metrics, 
            path
        )

    # Save latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, 'latest.pth')
    save_checkpoint(
        model, 
        wav2vec_optimizer, 
        main_optimizer,
        epoch, 
        val_loss, 
        train_loss, 
        best_metrics, 
        latest_path
    )

    # Memory cleanup
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  # Save final metrics
  final_metrics = {
      **best_metrics, 
      'completed_epochs': config.num_epochs
  }
  if config.use_cross_validation:
    final_metrics['cv_fold'] = config.cv_fold
  
  metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
  with open(metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)

  logger.info(f"Training completed! Best metrics: {best_metrics}")
