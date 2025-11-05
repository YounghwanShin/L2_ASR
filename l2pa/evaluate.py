"""Evaluation script."""
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
from .models.unified_model import MultitaskPronunciationModel

logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_path, config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f'Device: {device}, Mode: {config.training_mode}')
  
  with open(config.phoneme_mapping_path, 'r') as f:
    phoneme_to_id = json.load(f)
  id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
  error_type_names = config.get_error_label_names()
  
  model = MultitaskPronunciationModel(
      pretrained_model_name=config.pretrained_model,
      num_phonemes=config.num_phonemes,
      num_error_types=config.num_error_types,
      **config.get_model_architecture_config()
  )
  
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
  
  model.load_state_dict(state_dict)
  model = model.to(device)
  model.eval()
  
  test_dataset = PronunciationDataset(config.test_labels_path, phoneme_to_id, config.training_mode, config.max_audio_length, config.sampling_rate)
  test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_batch)
  
  evaluator = ModelEvaluator(device)
  eval_results = {
      'config': {
          'training_mode': config.training_mode,
          'model_type': config.model_type,
          'checkpoint_path': checkpoint_path,
          'date': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
      }
  }
  
  if config.has_canonical_task():
    canonical_results = evaluator.evaluate_phoneme_recognition(model, test_dataloader, config.training_mode, id_to_phoneme, 'canonical')
    logger.info(f"Canonical PER: {canonical_results['per']:.4f}")
    eval_results['canonical'] = canonical_results
  
  if config.has_perceived_task():
    perceived_results = evaluator.evaluate_phoneme_recognition(model, test_dataloader, config.training_mode, id_to_phoneme, 'perceived')
    logger.info(f"Perceived PER: {perceived_results['per']:.4f}")
    eval_results['perceived'] = perceived_results
  
  if config.has_error_task():
    error_results = evaluator.evaluate_error_detection(model, test_dataloader, config.training_mode, error_type_names)
    logger.info(f"Error Accuracy: {error_results['token_accuracy']:.4f}")
    eval_results['error'] = error_results
  
  results_dir = 'evaluation_results'
  os.makedirs(results_dir, exist_ok=True)
  exp_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
  results_path = os.path.join(results_dir, f'{exp_name}_results.json')
  
  with open(results_path, 'w') as f:
    json.dump(eval_results, f, indent=2)
  
  logger.info(f'Results saved to: {results_path}')
