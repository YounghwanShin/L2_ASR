"""Model evaluator simplified version."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List

from .metrics import calculate_phoneme_metrics, calculate_error_metrics
from ..utils.audio import greedy_ctc_decode, compute_wav2vec_output_lengths, enable_specaugment, create_attention_mask


class ModelEvaluator:
  def __init__(self, device: str = 'cuda'):
    self.device = device
  
  def _clean_labels(self, labels, lengths):
    return [labels[i][:length].cpu().numpy().tolist() for i, length in enumerate(lengths)]
  
  def _decode_predictions(self, logits, audio_lengths, model):
    input_lengths = compute_wav2vec_output_lengths(model, audio_lengths)
    input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
    log_probs = torch.log_softmax(logits, dim=-1)
    return greedy_ctc_decode(log_probs, input_lengths)
  
  def evaluate_phoneme_recognition(self, model, dataloader, training_mode, id_to_phoneme, target_type='perceived'):
    model.eval()
    enable_specaugment(model, False)
    
    all_predictions = []
    all_targets = []
    all_canonical = []
    all_perceived = []
    all_ids = []
    all_speaker_ids = []
    
    logits_key = f'{target_type}_logits'
    labels_key = f'{target_type}_labels'
    lengths_key = f'{target_type}_lengths'
    
    with torch.no_grad():
      for batch_data in tqdm(dataloader, desc=f'{target_type.capitalize()} Eval'):
        if batch_data is None or labels_key not in batch_data:
          continue
        
        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)
        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)
        
        outputs = model(waveforms, attention_mask, training_mode)
        if logits_key not in outputs:
          continue
        
        predictions = self._decode_predictions(outputs[logits_key], audio_lengths, model)
        targets = self._clean_labels(batch_data[labels_key], batch_data[lengths_key])
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_ids.extend(batch_data['file_paths'])
        all_speaker_ids.extend(batch_data['speaker_ids'])
        
        if 'canonical_labels' in batch_data and 'perceived_labels' in batch_data:
          canonical = self._clean_labels(batch_data['canonical_labels'], batch_data['canonical_lengths'])
          perceived = self._clean_labels(batch_data['perceived_labels'], batch_data['perceived_lengths'])
          all_canonical.extend(canonical)
          all_perceived.extend(perceived)
    
    results = calculate_phoneme_metrics(all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme)
    
    speaker_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'canonical': [], 'perceived': [], 'ids': []})
    for i, (pred, target, file_id, speaker_id) in enumerate(zip(all_predictions, all_targets, all_ids, all_speaker_ids)):
      speaker_data[speaker_id]['predictions'].append(pred)
      speaker_data[speaker_id]['targets'].append(target)
      if i < len(all_canonical):
        speaker_data[speaker_id]['canonical'].append(all_canonical[i])
        speaker_data[speaker_id]['perceived'].append(all_perceived[i])
      speaker_data[speaker_id]['ids'].append(file_id)
    
    by_speaker_results = {}
    for speaker_id, data in speaker_data.items():
      by_speaker_results[speaker_id] = calculate_phoneme_metrics(
          data['predictions'], data['targets'], data['canonical'], data['perceived'], data['ids'], id_to_phoneme
      )
    
    results['by_speaker'] = by_speaker_results
    return results
  
  def evaluate_error_detection(self, model, dataloader, training_mode, error_type_names):
    model.eval()
    enable_specaugment(model, False)
    
    all_predictions = []
    all_targets = []
    all_ids = []
    all_speaker_ids = []
    
    with torch.no_grad():
      for batch_data in tqdm(dataloader, desc='Error Detection Eval'):
        if batch_data is None or 'error_labels' not in batch_data:
          continue
        
        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)
        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask = create_attention_mask(waveforms, normalized_lengths)
        
        outputs = model(waveforms, attention_mask, training_mode)
        if 'error_logits' not in outputs:
          continue
        
        predictions = self._decode_predictions(outputs['error_logits'], audio_lengths, model)
        targets = self._clean_labels(batch_data['error_labels'], batch_data['error_lengths'])
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_ids.extend(batch_data['file_paths'])
        all_speaker_ids.extend(batch_data['speaker_ids'])
    
    results = calculate_error_metrics(all_predictions, all_targets, all_ids, error_type_names)
    
    speaker_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'ids': []})
    for pred, target, file_id, speaker_id in zip(all_predictions, all_targets, all_ids, all_speaker_ids):
      speaker_data[speaker_id]['predictions'].append(pred)
      speaker_data[speaker_id]['targets'].append(target)
      speaker_data[speaker_id]['ids'].append(file_id)
    
    by_speaker_results = {}
    for speaker_id, data in speaker_data.items():
      by_speaker_results[speaker_id] = calculate_error_metrics(data['predictions'], data['targets'], data['ids'], error_type_names)
    
    results['by_speaker'] = by_speaker_results
    return results
