"""Evaluator module for pronunciation assessment model.

This module implements evaluation for phoneme recognition and error detection.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional

from .metrics import calculate_error_metrics, calculate_phoneme_metrics
from ..utils.audio import greedy_ctc_decode, compute_output_lengths, enable_specaugment, create_attention_mask


class ModelEvaluator:
  """Evaluator for pronunciation assessment model."""
  
  def __init__(self, device: str = 'cuda'):
    """Initialize evaluator.
    
    Args:
      device: Evaluation device.
    """
    self.device = device

  def clean_targets(self, labels: torch.Tensor, label_lengths: torch.Tensor) -> List[List[int]]:
    """Clean target labels by removing padding.
    
    Args:
      labels: Padded label tensor.
      label_lengths: Label sequence lengths.
      
    Returns:
      List of label sequences without padding.
    """
    return [labels[i][:length].cpu().numpy().tolist() 
            for i, length in enumerate(label_lengths)]

  def evaluate_phoneme_recognition(self,
                                  model,
                                  dataloader: DataLoader,
                                  training_mode: str,
                                  id_to_phoneme: Dict[str, str],
                                  target_type: str = 'perceived') -> Dict:
    """Evaluate phoneme recognition performance.
    
    Args:
      model: Model to evaluate.
      dataloader: Evaluation data loader.
      training_mode: Training mode.
      id_to_phoneme: ID to phoneme mapping.
      target_type: Type of target ('canonical' or 'perceived').
      
    Returns:
      Dictionary with evaluation metrics.
    """
    model.eval()
    enable_specaugment(model, False)
    
    all_predictions, all_targets = [], []
    all_canonical, all_perceived = [], []
    all_ids, all_speaker_ids = [], []

    logits_key = f'{target_type}_logits'
    labels_key = f'{target_type}_labels'
    lengths_key = f'{target_type}_lengths'

    with torch.no_grad():
      for batch_data in tqdm(dataloader, desc=f'{target_type.capitalize()} Phoneme Eval'):
        if labels_key not in batch_data:
          continue

        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)

        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)

        outputs = model(waveforms, attention_mask, training_mode)
        
        if logits_key not in outputs:
          continue

        logits = outputs[logits_key]
        input_lengths = compute_output_lengths(model, audio_lengths)
        input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))

        log_probs = torch.log_softmax(logits, dim=-1)
        predictions = greedy_ctc_decode(log_probs, input_lengths)
        targets = self.clean_targets(batch_data[labels_key], batch_data[lengths_key])

        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_ids.extend(batch_data['file_paths'])
        all_speaker_ids.extend(batch_data['speaker_ids'])

        if 'canonical_labels' in batch_data and 'perceived_labels' in batch_data:
          canonical = self.clean_targets(batch_data['canonical_labels'],
                                        batch_data['canonical_lengths'])
          perceived = self.clean_targets(batch_data['perceived_labels'],
                                        batch_data['perceived_lengths'])
          all_canonical.extend(canonical)
          all_perceived.extend(perceived)

    # Overall metrics
    results = calculate_phoneme_metrics(
        all_predictions, all_targets, all_canonical, all_perceived,
        all_ids, id_to_phoneme
    )

    # Per-speaker metrics
    by_speaker_results = {}
    speaker_data = defaultdict(lambda: {
        'predictions': [], 'targets': [],
        'canonical': [], 'perceived': [], 'ids': []
    })

    for i, (pred, target, file_id, speaker_id) in enumerate(
        zip(all_predictions, all_targets, all_ids, all_speaker_ids)):
      speaker_data[speaker_id]['predictions'].append(pred)
      speaker_data[speaker_id]['targets'].append(target)
      if i < len(all_canonical):
        speaker_data[speaker_id]['canonical'].append(all_canonical[i])
        speaker_data[speaker_id]['perceived'].append(all_perceived[i])
      speaker_data[speaker_id]['ids'].append(file_id)

    for speaker_id, data in speaker_data.items():
      by_speaker_results[speaker_id] = calculate_phoneme_metrics(
          data['predictions'], data['targets'], data['canonical'],
          data['perceived'], data['ids'], id_to_phoneme
      )

    results['by_speaker'] = by_speaker_results
    return results

  def evaluate_error_detection(self,
                               model,
                               dataloader: DataLoader,
                               training_mode: str,
                               error_type_names: Dict[int, str]) -> Dict:
    """Evaluate error detection performance.
    
    Args:
      model: Model to evaluate.
      dataloader: Evaluation data loader.
      training_mode: Training mode.
      error_type_names: Error ID to name mapping.
      
    Returns:
      Dictionary with evaluation metrics.
    """
    model.eval()
    all_predictions, all_targets = [], []
    all_ids, all_speaker_ids = [], []

    with torch.no_grad():
      for batch_data in tqdm(dataloader, desc='Error Detection Eval'):
        if 'error_labels' not in batch_data:
          continue

        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)

        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)

        outputs = model(waveforms, attention_mask, training_mode)

        if 'error_logits' not in outputs:
          continue

        error_logits = outputs['error_logits']
        input_lengths = compute_output_lengths(model, audio_lengths)
        input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))

        log_probs = torch.log_softmax(error_logits, dim=-1)
        predictions = greedy_ctc_decode(log_probs, input_lengths)
        targets = self.clean_targets(batch_data['error_labels'],
                                     batch_data['error_lengths'])

        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_ids.extend(batch_data['file_paths'])
        all_speaker_ids.extend(batch_data['speaker_ids'])

    # Overall metrics
    results = calculate_error_metrics(all_predictions, all_targets,
                                     all_ids, error_type_names)

    # Per-speaker metrics
    by_speaker_results = {}
    speaker_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'ids': []})

    for pred, target, file_id, speaker_id in zip(
        all_predictions, all_targets, all_ids, all_speaker_ids):
      speaker_data[speaker_id]['predictions'].append(pred)
      speaker_data[speaker_id]['targets'].append(target)
      speaker_data[speaker_id]['ids'].append(file_id)

    for speaker_id, data in speaker_data.items():
      by_speaker_results[speaker_id] = calculate_error_metrics(
          data['predictions'], data['targets'], data['ids'], error_type_names
      )

    results['by_speaker'] = by_speaker_results
    return results
