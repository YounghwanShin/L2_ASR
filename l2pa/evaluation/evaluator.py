"""Evaluator module for pronunciation assessment model.

This module implements comprehensive evaluation for phoneme recognition
and error detection, including per-speaker analysis and sample predictions.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional

from .metrics import calculate_error_metrics, calculate_phoneme_metrics
from ..utils.audio import greedy_ctc_decode, compute_output_lengths, enable_specaugment, create_attention_mask


class ModelEvaluator:
    """Evaluator class for pronunciation assessment model.
    
    Provides comprehensive evaluation metrics for both phoneme recognition
    and error detection tasks, including per-speaker breakdown.
    
    Attributes:
        device: Device for evaluation.
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initializes the evaluator.
        
        Args:
            device: Device for evaluation ('cuda' or 'cpu').
        """
        self.device = device

    def clean_targets(self, labels: torch.Tensor, label_lengths: torch.Tensor) -> List[List[int]]:
        """Cleans target labels by removing padding.
        
        Args:
            labels: Padded label tensor [batch_size, max_length].
            label_lengths: Length of each label sequence [batch_size].
            
        Returns:
            List of label sequences without padding.
        """
        return [labels[i][:length].cpu().numpy().tolist() for i, length in enumerate(label_lengths)]

    def evaluate_error_detection(self, 
                                model, 
                                dataloader: DataLoader, 
                                training_mode: str = 'phoneme_error', 
                                error_type_names: Optional[Dict[int, str]] = None) -> Dict:
        """Evaluates error detection performance.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            training_mode: Training mode.
            error_type_names: Mapping from error IDs to names.
            
        Returns:
            Dictionary containing error detection metrics.
        """
        if error_type_names is None:
            error_type_names = {0: 'blank', 1: 'deletion', 2: 'insertion', 3: 'substitution', 4: 'correct'}

        model.eval()
        all_predictions, all_targets, all_ids, all_speaker_ids = [], [], [], []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Error Detection Evaluation', dynamic_ncols=True)

            for batch_data in progress_bar:
                if 'error_labels' not in batch_data:
                    continue

                waveforms = batch_data['waveforms'].to(self.device)
                error_labels = batch_data['error_labels'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)

                normalized_lengths = audio_lengths.float() / waveforms.shape[1]
                attention_mask = create_attention_mask(waveforms, normalized_lengths)

                outputs = model(waveforms, attention_mask, training_mode=training_mode)

                if 'error_logits' not in outputs:
                    continue

                error_logits = outputs['error_logits']

                input_lengths = compute_output_lengths(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))

                log_probs = torch.log_softmax(error_logits, dim=-1)
                predictions = greedy_ctc_decode(log_probs, input_lengths)
                targets = self.clean_targets(error_labels, batch_data['error_lengths'])

                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_ids.extend(batch_data['file_paths'])
                all_speaker_ids.extend(batch_data['speaker_ids'])

        # Compute overall metrics
        results = calculate_error_metrics(all_predictions, all_targets, all_ids, error_type_names)

        # Compute per-speaker metrics
        by_speaker_results = {}
        speaker_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'ids': []})

        for pred, target, file_id, speaker_id in zip(all_predictions, all_targets, all_ids, all_speaker_ids):
            speaker_data[speaker_id]['predictions'].append(pred)
            speaker_data[speaker_id]['targets'].append(target)
            speaker_data[speaker_id]['ids'].append(file_id)

        for speaker_id, data in speaker_data.items():
            by_speaker_results[speaker_id] = calculate_error_metrics(
                data['predictions'], data['targets'], data['ids'], error_type_names
            )

        results['by_speaker'] = by_speaker_results
        return results

    def evaluate_phoneme_recognition(self, 
                                   model, 
                                   dataloader: DataLoader, 
                                   training_mode: str = 'phoneme_only', 
                                   id_to_phoneme: Optional[Dict[str, str]] = None) -> Dict:
        """Evaluates phoneme recognition performance.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            training_mode: Training mode.
            id_to_phoneme: Mapping from phoneme IDs to phoneme strings.
            
        Returns:
            Dictionary containing phoneme recognition metrics.
        """
        model.eval()
        enable_specaugment(model, False)
        all_predictions, all_targets, all_canonical, all_perceived, all_ids, all_speaker_ids = [], [], [], [], [], []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Phoneme Recognition Evaluation', dynamic_ncols=True)

            for batch_data in progress_bar:
                waveforms = batch_data['waveforms'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)

                normalized_lengths = audio_lengths.float() / waveforms.shape[1]
                attention_mask = create_attention_mask(waveforms, normalized_lengths)

                outputs = model(waveforms, attention_mask, training_mode=training_mode)
                phoneme_logits = outputs['phoneme_logits']

                input_lengths = compute_output_lengths(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))

                log_probs = torch.log_softmax(phoneme_logits, dim=-1)
                batch_phoneme_preds = greedy_ctc_decode(log_probs, input_lengths)

                batch_targets = self.clean_targets(batch_data['phoneme_labels'], batch_data['phoneme_lengths'])

                if 'canonical_labels' in batch_data:
                    batch_canonical = self.clean_targets(batch_data['canonical_labels'], batch_data['canonical_lengths'])
                    batch_perceived = self.clean_targets(batch_data['phoneme_labels'], batch_data['phoneme_lengths'])
                    all_canonical.extend(batch_canonical)
                    all_perceived.extend(batch_perceived)

                all_predictions.extend(batch_phoneme_preds)
                all_targets.extend(batch_targets)
                all_ids.extend(batch_data['file_paths'])
                all_speaker_ids.extend(batch_data['speaker_ids'])

        # Compute overall metrics
        results = calculate_phoneme_metrics(
            all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme
        )

        # Compute per-speaker metrics
        by_speaker_results = {}
        speaker_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'canonical': [], 'perceived': [], 'ids': []})

        for i, (pred, target, file_id, speaker_id) in enumerate(zip(all_predictions, all_targets, all_ids, all_speaker_ids)):
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

    def show_sample_predictions(self, 
                              model, 
                              eval_dataloader: DataLoader, 
                              id_to_phoneme: Dict[str, str], 
                              logger, 
                              training_mode: str = 'phoneme_only', 
                              error_type_names: Optional[Dict[int, str]] = None, 
                              num_samples: int = 3):
        """Shows sample predictions for debugging and analysis.
        
        Args:
            model: Model to evaluate.
            eval_dataloader: Evaluation data loader.
            id_to_phoneme: Mapping from phoneme IDs to strings.
            logger: Logger for output.
            training_mode: Training mode.
            error_type_names: Mapping from error IDs to names.
            num_samples: Number of samples to show.
        """
        model.eval()
        enable_specaugment(model, False)
        samples_shown = 0

        with torch.no_grad():
            for batch_data in eval_dataloader:
                if samples_shown >= num_samples:
                    break

                waveforms = batch_data['waveforms'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)

                input_lengths = compute_output_lengths(model, audio_lengths)
                normalized_lengths = audio_lengths.float() / waveforms.shape[1]
                attention_mask = create_attention_mask(waveforms, normalized_lengths)

                outputs = model(waveforms, attention_mask, training_mode=training_mode)

                phoneme_logits = outputs['phoneme_logits']
                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
                phoneme_log_probs = torch.log_softmax(phoneme_logits, dim=-1)
                phoneme_predictions = greedy_ctc_decode(phoneme_log_probs, phoneme_input_lengths)

                if training_mode == 'phoneme_error' and 'error_logits' in outputs:
                    error_logits = outputs['error_logits']
                    error_input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
                    error_log_probs = torch.log_softmax(error_logits, dim=-1)
                    error_predictions = greedy_ctc_decode(error_log_probs, error_input_lengths)

                for i in range(min(waveforms.shape[0], num_samples - samples_shown)):
                    logger.info(f"\n--- Sample {samples_shown + 1} ({training_mode}) ---")
                    logger.info(f"File: {batch_data['file_paths'][i]}")

                    phoneme_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}")
                                    for pid in batch_data['phoneme_labels'][i][:batch_data['phoneme_lengths'][i]]]
                    phoneme_pred = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}")
                                  for pid in phoneme_predictions[i]]

                    logger.info(f"Phoneme Actual:    {' '.join(phoneme_actual)}")
                    logger.info(f"Phoneme Predicted: {' '.join(phoneme_pred)}")

                    if training_mode == 'phoneme_error' and 'error_labels' in batch_data:
                        error_actual = [error_type_names.get(int(label), str(label))
                                      for label in batch_data['error_labels'][i][:batch_data['error_lengths'][i]]]
                        error_pred = [error_type_names.get(int(pred), str(pred))
                                    for pred in error_predictions[i]]

                        logger.info(f"Error Actual:    {' '.join(error_actual)}")
                        logger.info(f"Error Predicted: {' '.join(error_pred)}")

                    samples_shown += 1
                    if samples_shown >= num_samples:
                        break

                if samples_shown >= num_samples:
                    break