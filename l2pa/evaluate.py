"""Evaluation script for trained models.

This module provides functionality to evaluate trained pronunciation
assessment models on test datasets with automatic configuration loading.
"""

import json
import logging
import os
from datetime import datetime

import pytz
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import PronunciationDataset, collate_batch
from .evaluation.evaluator import ModelEvaluator
from .models.unified_model import UnifiedModel
from .training.utils import get_checkpoint_config

logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_path, config):
  """Main evaluation function.
  
  Args:
    checkpoint_path: Path to model checkpoint.
    config: Configuration object.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Load checkpoint configuration if available
  ckpt_config = get_checkpoint_config(checkpoint_path)
  if ckpt_config:
    logger.info("Loading configuration from checkpoint")
    if 'pretrained_model' in ckpt_config:
      config.pretrained_model = ckpt_config['pretrained_model']
      logger.info(f"Using pretrained model: {config.pretrained_model}")
    if 'training_mode' in ckpt_config:
      config.training_mode = ckpt_config['training_mode']
      logger.info(f"Using training mode: {config.training_mode}")
  
  logger.info(f"Device: {device}, Mode: {config.training_mode}")
  logger.info(f"Pretrained model: {config.pretrained_model}")

  # Load phoneme mapping
  with open(config.phoneme_map, 'r') as f:
    phoneme_to_id = json.load(f)
  id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
  error_type_names = config.get_error_type_names()

  # Initialize model with configuration
  model_config = config.get_model_config()
  logger.info(f"Model hidden dimension: {model_config['hidden_dim']}")
  
  model = UnifiedModel(
      pretrained_model_name=config.pretrained_model,
      num_phonemes=config.num_phonemes,
      num_error_types=config.num_error_types,
      **model_config
  )

  # Load checkpoint
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  
  # Remove 'module.' prefix if present (from DataParallel)
  if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {
        k.replace('module.', ''): v 
        for k, v in state_dict.items()
    }
  
  model.load_state_dict(state_dict)
  model = model.to(device)
  model.eval()

  # Load test dataset
  test_dataset = PronunciationDataset(
      config.test_data, 
      phoneme_to_id, 
      config.training_mode,
      config.max_length, 
      config.sampling_rate, 
      device
  )
  test_dataloader = DataLoader(
      test_dataset, 
      batch_size=config.eval_batch_size, 
      num_workers=config.num_workers,
      shuffle=False,
      collate_fn=lambda batch: collate_batch(batch, config.training_mode)
  )

  evaluator = ModelEvaluator(device)

  # Prepare results dictionary
  eval_results = {
      'config': {
          'training_mode': config.training_mode,
          'model_type': config.model_type,
          'pretrained_model': config.pretrained_model,
          'checkpoint_path': checkpoint_path,
          'date': datetime.now(
              pytz.timezone('Asia/Seoul')
          ).strftime('%Y-%m-%d %H:%M:%S')
      }
  }

  # Evaluate each task
  if config.has_canonical_task():
    canonical_results = evaluator.evaluate_phoneme_recognition(
        model, 
        test_dataloader, 
        config.training_mode,
        id_to_phoneme, 
        'canonical'
    )
    logger.info(f"Canonical PER: {canonical_results['per']:.4f}")
    eval_results['canonical'] = canonical_results

  if config.has_perceived_task():
    perceived_results = evaluator.evaluate_phoneme_recognition(
        model, 
        test_dataloader, 
        config.training_mode,
        id_to_phoneme, 
        'perceived'
    )
    logger.info(f"Perceived PER: {perceived_results['per']:.4f}")
    eval_results['perceived'] = perceived_results

  if config.has_error_task():
    error_results = evaluator.evaluate_error_detection(
        model, 
        test_dataloader, 
        config.training_mode, 
        error_type_names
    )
    logger.info(f"Error Accuracy: {error_results['token_accuracy']:.4f}")
    eval_results['error'] = error_results

  # Save results
  results_dir = 'evaluation_results'
  os.makedirs(results_dir, exist_ok=True)
  
  exp_name = os.path.basename(
      os.path.dirname(os.path.dirname(checkpoint_path))
  )
  results_path = os.path.join(results_dir, f"{exp_name}_results.json")

  with open(results_path, 'w') as f:
    json.dump(eval_results, f, indent=2)

  logger.info(f"Results saved to: {results_path}")