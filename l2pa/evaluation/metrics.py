"""Evaluation metrics for pronunciation assessment.

This module implements metrics for phoneme recognition and error detection
including edit distance, PER, mispronunciation detection, and error classification.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, f1_score


def compute_edit_distance(
    reference: List,
    hypothesis: List
) -> Tuple[int, int, int, int]:
  """Computes Levenshtein edit distance with error breakdown.
  
  Args:
    reference: Reference sequence.
    hypothesis: Hypothesis sequence.
  
  Returns:
    Tuple of (substitutions, deletions, insertions, reference_length).
  """
  ref_len, hyp_len = len(reference), len(hypothesis)
  
  if ref_len == 0:
    return 0, 0, hyp_len, 0
  if hyp_len == 0:
    return 0, ref_len, 0, ref_len
  
  # Initialize DP table
  dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
  for i in range(ref_len + 1):
    dp[i][0] = i
  for j in range(hyp_len + 1):
    dp[0][j] = j
  
  # Fill DP table
  for i in range(1, ref_len + 1):
    for j in range(1, hyp_len + 1):
      if reference[i-1] == hypothesis[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(
            dp[i-1][j],      # Deletion
            dp[i][j-1],      # Insertion
            dp[i-1][j-1]     # Substitution
        )
  
  # Backtrack to count error types
  i, j = ref_len, hyp_len
  substitutions = deletions = insertions = 0
  
  while i > 0 or j > 0:
    if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
      i -= 1
      j -= 1
    elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
      substitutions += 1
      i -= 1
      j -= 1
    elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
      deletions += 1
      i -= 1
    elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
      insertions += 1
      j -= 1
    else:
      break
  
  return substitutions, deletions, insertions, ref_len


def convert_ids_to_phonemes(
    sequences: List[List[int]],
    id_to_phoneme: Dict[str, str]
) -> List[List[str]]:
  """Converts phoneme ID sequences to strings.
  
  Args:
    sequences: List of phoneme ID sequences.
    id_to_phoneme: ID to phoneme mapping.
  
  Returns:
    List of phoneme string sequences.
  """
  return [
      [id_to_phoneme.get(str(token_id), f'UNK_{token_id}') for token_id in seq]
      for seq in sequences
  ]


def remove_silence(sequences: List[List[str]]) -> List[List[str]]:
  """Removes silence tokens from phoneme sequences.
  
  Args:
    sequences: List of phoneme sequences.
  
  Returns:
    List of sequences without silence tokens.
  """
  return [[token for token in seq if token != 'sil'] for seq in sequences]


def calculate_phoneme_metrics(
    predictions: List[List[int]],
    targets: List[List[int]],
    canonical: List[List[int]],
    perceived: List[List[int]],
    ids: List[str],
    id_to_phoneme: Dict[str, str]
) -> Dict:
  """Calculates phoneme recognition metrics.
  
  Args:
    predictions: Predicted phoneme sequences.
    targets: Target phoneme sequences.
    canonical: Canonical phoneme sequences.
    perceived: Perceived phoneme sequences.
    ids: Sample IDs.
    id_to_phoneme: ID to phoneme mapping.
  
  Returns:
    Dictionary with phoneme recognition metrics.
  """
  # Convert to phonemes and remove silence
  pred_phonemes = remove_silence(
      convert_ids_to_phonemes(predictions, id_to_phoneme)
  )
  target_phonemes = remove_silence(
      convert_ids_to_phonemes(targets, id_to_phoneme)
  )
  
  # Calculate PER
  total_errors = 0
  total_phonemes = 0
  total_insertions = 0
  total_deletions = 0
  total_substitutions = 0
  
  for pred, target in zip(pred_phonemes, target_phonemes):
    subs, dels, ins, ref_len = compute_edit_distance(target, pred)
    total_errors += subs + dels + ins
    total_phonemes += ref_len
    total_insertions += ins
    total_deletions += dels
    total_substitutions += subs
  
  per = total_errors / total_phonemes if total_phonemes > 0 else 0.0
  
  # Calculate mispronunciation detection metrics
  misp_metrics = {
      'precision': 0.0,
      'recall': 0.0,
      'f1_score': 0.0,
      'true_acceptance': 0,
      'false_rejection': 0,
      'false_acceptance': 0,
      'true_rejection': 0
  }
  
  if canonical and perceived:
    misp_metrics = _calculate_mispronunciation_metrics(
        predictions, canonical, perceived, id_to_phoneme
    )
  
  return {
      'per': float(per),
      'mispronunciation_precision': float(misp_metrics['precision']),
      'mispronunciation_recall': float(misp_metrics['recall']),
      'mispronunciation_f1': float(misp_metrics['f1_score']),
      'total_phonemes': int(total_phonemes),
      'total_errors': int(total_errors),
      'insertions': int(total_insertions),
      'deletions': int(total_deletions),
      'substitutions': int(total_substitutions),
      'confusion_matrix': {
          'true_acceptance': int(misp_metrics['true_acceptance']),
          'false_rejection': int(misp_metrics['false_rejection']),
          'false_acceptance': int(misp_metrics['false_acceptance']),
          'true_rejection': int(misp_metrics['true_rejection'])
      }
  }


def _calculate_mispronunciation_metrics(
    predictions: List[List[int]],
    canonical: List[List[int]],
    perceived: List[List[int]],
    id_to_phoneme: Dict[str, str]
) -> Dict:
  """Calculates mispronunciation detection metrics.
  
  Args:
    predictions: Predicted phoneme sequences.
    canonical: Canonical phoneme sequences.
    perceived: Perceived phoneme sequences.
    id_to_phoneme: ID to phoneme mapping.
  
  Returns:
    Dictionary with mispronunciation detection metrics.
  """
  pred_phonemes = remove_silence(convert_ids_to_phonemes(predictions, id_to_phoneme))
  canonical_phonemes = remove_silence(convert_ids_to_phonemes(canonical, id_to_phoneme))
  perceived_phonemes = remove_silence(convert_ids_to_phonemes(perceived, id_to_phoneme))
  
  ta = fr = fa = tr = 0  # true_acceptance, false_rejection, false_acceptance, true_rejection
  
  for pred, canon, perc in zip(pred_phonemes, canonical_phonemes, perceived_phonemes):
    if not canon or not perc or not pred:
      continue
    
    max_len = max(len(canon), len(perc), len(pred))
    
    # Pad sequences
    canon_padded = canon + ['<pad>'] * (max_len - len(canon))
    perc_padded = perc + ['<pad>'] * (max_len - len(perc))
    pred_padded = pred + ['<pad>'] * (max_len - len(pred))
    
    for c, p, d in zip(canon_padded, perc_padded, pred_padded):
      if '<pad>' in [c, p, d]:
        continue
      
      is_correct = (c == p)
      predicted_correct = (c == d)
      
      if is_correct and predicted_correct:
        ta += 1
      elif is_correct and not predicted_correct:
        fr += 1
      elif not is_correct and predicted_correct:
        fa += 1
      else:
        tr += 1
  
  recall = tr / (tr + fa) if (tr + fa) > 0 else 0.0
  precision = tr / (tr + fr) if (tr + fr) > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
  
  return {
      'precision': precision,
      'recall': recall,
      'f1_score': f1,
      'true_acceptance': ta,
      'false_rejection': fr,
      'false_acceptance': fa,
      'true_rejection': tr
  }


def calculate_error_metrics(
    predictions: List[List[int]],
    targets: List[List[int]],
    ids: List[str],
    error_type_names: Dict[int, str]
) -> Dict:
  """Calculates error detection metrics.
  
  Args:
    predictions: Predicted error sequences.
    targets: Target error sequences.
    ids: Sample IDs.
    error_type_names: Error ID to name mapping.
  
  Returns:
    Dictionary with error detection metrics.
  """
  # Calculate token-level metrics
  total_sequences = len(predictions)
  total_tokens = 0
  total_errors = 0
  correct_sequences = 0
  
  for pred, target in zip(predictions, targets):
    subs, dels, ins, ref_len = compute_edit_distance(target, pred)
    errors = subs + dels + ins
    total_errors += errors
    total_tokens += ref_len
    if errors == 0:
      correct_sequences += 1
  
  sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
  token_accuracy = 1 - (total_errors / total_tokens) if total_tokens > 0 else 0.0
  avg_edit_distance = total_errors / total_sequences if total_sequences > 0 else 0.0
  
  # Calculate per-class metrics
  flat_predictions = [token for pred in predictions for token in pred]
  flat_targets = [token for target in targets for token in target]
  
  weighted_f1 = macro_f1 = 0.0
  class_metrics = {}
  
  if flat_predictions and flat_targets:
    try:
      min_len = min(len(flat_predictions), len(flat_targets))
      flat_predictions = flat_predictions[:min_len]
      flat_targets = flat_targets[:min_len]
      
      weighted_f1 = f1_score(flat_targets, flat_predictions, average='weighted', zero_division=0)
      macro_f1 = f1_score(flat_targets, flat_predictions, average='macro', zero_division=0)
      
      class_report = classification_report(
          flat_targets, flat_predictions, output_dict=True, zero_division=0
      )
      
      # Extract metrics for each error type (excluding blank)
      for class_id, class_name in error_type_names.items():
        if class_id != 0 and str(class_id) in class_report:
          class_metrics[class_name] = {
              'precision': float(class_report[str(class_id)]['precision']),
              'recall': float(class_report[str(class_id)]['recall']),
              'f1': float(class_report[str(class_id)]['f1-score']),
              'support': int(class_report[str(class_id)]['support'])
          }
    except Exception as e:
      print(f'Error calculating metrics: {e}')
  
  return {
      'sequence_accuracy': float(sequence_accuracy),
      'token_accuracy': float(token_accuracy),
      'avg_edit_distance': float(avg_edit_distance),
      'weighted_f1': float(weighted_f1),
      'macro_f1': float(macro_f1),
      'class_metrics': class_metrics,
      'total_sequences': int(total_sequences),
      'total_tokens': int(total_tokens)
  }
