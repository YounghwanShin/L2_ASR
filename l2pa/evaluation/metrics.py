"""Metrics module for pronunciation assessment evaluation.

This module implements various evaluation metrics including edit distance,
phoneme error rate (PER), mispronunciation detection metrics, and
error classification metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score


def compute_edit_distance_with_details(reference: List, hypothesis: List) -> Tuple[int, int, int, int, int]:
    """Computes edit distance with detailed error breakdown.
    
    Uses dynamic programming to compute Levenshtein distance and then
    backtracks to count specific error types.
    
    Args:
        reference: Reference sequence.
        hypothesis: Hypothesis sequence.
        
    Returns:
        Tuple of (total_errors, substitutions, deletions, insertions, reference_length).
    """
    ref_len, hyp_len = len(reference), len(hypothesis)

    if ref_len == 0:
        return hyp_len, 0, 0, hyp_len, 0
    if hyp_len == 0:
        return ref_len, 0, ref_len, 0, ref_len

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

    total_errors = dp[ref_len][hyp_len]

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

    return total_errors, substitutions, deletions, insertions, ref_len


def calculate_sequence_error_rate(ids: List, references: List[List], hypotheses: List[List]) -> List[Dict]:
    """Calculates sequence error rate details for a batch.
    
    Args:
        ids: List of sample IDs.
        references: List of reference sequences.
        hypotheses: List of hypothesis sequences.
        
    Returns:
        List of dictionaries containing error rate metrics for each sample.
    """
    details = []

    for sample_id, reference, hypothesis in zip(ids, references, hypotheses):
        ref_tokens = [str(token) for token in reference]
        hyp_tokens = [str(token) for token in hypothesis]

        total_errors, substitutions, deletions, insertions, num_ref = compute_edit_distance_with_details(
            ref_tokens, hyp_tokens
        )

        error_rate = total_errors / num_ref if num_ref > 0 else 0.0

        detail = {
            'id': sample_id,
            'error_rate': error_rate,
            'num_reference_tokens': num_ref,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'num_hypothesis_tokens': len(hyp_tokens)
        }

        details.append(detail)

    return details


def convert_ids_to_phonemes(sequences: List[List[int]], id_to_phoneme: Dict[str, str]) -> List[List[str]]:
    """Converts ID sequences to phoneme strings.
    
    Args:
        sequences: List of phoneme ID sequences.
        id_to_phoneme: Mapping from IDs to phoneme strings.
        
    Returns:
        List of phoneme string sequences.
    """
    return [[id_to_phoneme.get(str(token_id), f"UNK_{token_id}") for token_id in seq] for seq in sequences]


def remove_silence_tokens(sequences: List[List[str]]) -> List[List[str]]:
    """Removes silence tokens from sequences.
    
    Args:
        sequences: List of phoneme sequences.
        
    Returns:
        List of sequences with silence tokens removed.
    """
    return [[token for token in seq if token != "sil"] for seq in sequences]


def calculate_mispronunciation_metrics(all_predictions: List[List[int]], 
                                     all_canonical: List[List[int]], 
                                     all_perceived: List[List[int]], 
                                     id_to_phoneme: Dict[str, str]) -> Dict:
    """Calculates mispronunciation detection metrics.
    
    Computes precision, recall, and F1 score for detecting pronunciation
    errors by comparing predicted phonemes to canonical and perceived forms.
    
    Args:
        all_predictions: Predicted phoneme sequences.
        all_canonical: Canonical (correct) phoneme sequences.
        all_perceived: Perceived (actual) phoneme sequences.
        id_to_phoneme: Mapping from IDs to phoneme strings.
        
    Returns:
        Dictionary containing mispronunciation detection metrics.
    """
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    canonical_phonemes = convert_ids_to_phonemes(all_canonical, id_to_phoneme)
    perceived_phonemes = convert_ids_to_phonemes(all_perceived, id_to_phoneme)

    pred_phonemes = remove_silence_tokens(pred_phonemes)
    canonical_phonemes = remove_silence_tokens(canonical_phonemes)
    perceived_phonemes = remove_silence_tokens(perceived_phonemes)

    total_true_acceptance = 0
    total_false_rejection = 0
    total_false_acceptance = 0
    total_true_rejection = 0

    for pred, canonical, perceived in zip(pred_phonemes, canonical_phonemes, perceived_phonemes):
        if len(canonical) == 0 or len(perceived) == 0 or len(pred) == 0:
            continue

        max_len = max(len(canonical), len(perceived), len(pred))

        canonical_padded = canonical + ['<pad>'] * (max_len - len(canonical))
        perceived_padded = perceived + ['<pad>'] * (max_len - len(perceived))
        pred_padded = pred + ['<pad>'] * (max_len - len(pred))

        for canonical_phone, perceived_phone, predicted_phone in zip(canonical_padded, perceived_padded, pred_padded):
            if canonical_phone == '<pad>' or perceived_phone == '<pad>' or predicted_phone == '<pad>':
                continue

            is_correct = (canonical_phone == perceived_phone)
            predicted_correct = (canonical_phone == predicted_phone)

            if is_correct and predicted_correct:
                total_true_acceptance += 1
            elif is_correct and not predicted_correct:
                total_false_rejection += 1
            elif not is_correct and predicted_correct:
                total_false_acceptance += 1
            elif not is_correct and not predicted_correct:
                total_true_rejection += 1

    # Compute metrics
    if (total_true_rejection + total_false_acceptance) > 0:
        recall = total_true_rejection / (total_true_rejection + total_false_acceptance)
    else:
        recall = 0.0

    if (total_true_rejection + total_false_rejection) > 0:
        precision = total_true_rejection / (total_true_rejection + total_false_rejection)
    else:
        precision = 0.0

    if (precision + recall) > 0:
        f1_score_value = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score_value = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score_value,
        'true_acceptance': total_true_acceptance,
        'false_rejection': total_false_rejection,
        'false_acceptance': total_false_acceptance,
        'true_rejection': total_true_rejection
    }


def calculate_error_metrics(all_predictions: List[List[int]], 
                          all_targets: List[List[int]], 
                          all_ids: List[str], 
                          error_type_names: Dict[int, str]) -> Dict:
    """Calculates error detection metrics.
    
    Args:
        all_predictions: Predicted error sequences.
        all_targets: Target error sequences.
        all_ids: Sample IDs.
        error_type_names: Mapping from error IDs to names.
        
    Returns:
        Dictionary containing error detection metrics.
    """
    error_rate_details = calculate_sequence_error_rate(
        ids=all_ids,
        references=all_targets,
        hypotheses=all_predictions
    )

    total_sequences = len(error_rate_details)
    correct_sequences = sum(1 for detail in error_rate_details if detail['error_rate'] == 0.0)

    total_tokens = sum(detail['num_reference_tokens'] for detail in error_rate_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] 
                      for detail in error_rate_details)

    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    token_accuracy = 1 - (total_errors / total_tokens) if total_tokens > 0 else 0
    avg_edit_distance = total_errors / total_sequences if total_sequences > 0 else 0

    # Compute classification metrics
    flat_predictions = [token for pred in all_predictions for token in pred]
    flat_targets = [token for target in all_targets for token in target]

    weighted_f1 = macro_f1 = 0
    class_metrics = {}

    if len(flat_predictions) > 0 and len(flat_targets) > 0:
        try:
            min_len = min(len(flat_predictions), len(flat_targets))
            flat_predictions = flat_predictions[:min_len]
            flat_targets = flat_targets[:min_len]

            weighted_f1 = f1_score(flat_targets, flat_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(flat_targets, flat_predictions, average='macro', zero_division=0)

            class_report = classification_report(flat_targets, flat_predictions, output_dict=True, zero_division=0)

            # Exclude blank from evaluation
            evaluated_error_types = {k: v for k, v in error_type_names.items() if k != 0}
            for class_id, class_name in evaluated_error_types.items():
                if str(class_id) in class_report:
                    class_metrics[class_name] = {
                        'precision': float(class_report[str(class_id)]['precision']),
                        'recall': float(class_report[str(class_id)]['recall']),
                        'f1': float(class_report[str(class_id)]['f1-score']),
                        'support': int(class_report[str(class_id)]['support'])
                    }
        except Exception as e:
            print(f"Error calculating class metrics: {e}")

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


def calculate_phoneme_metrics(all_predictions: List[List[int]], 
                            all_targets: List[List[int]], 
                            all_canonical: List[List[int]], 
                            all_perceived: List[List[int]], 
                            all_ids: List[str], 
                            id_to_phoneme: Dict[str, str]) -> Dict:
    """Calculates phoneme recognition metrics.
    
    Args:
        all_predictions: Predicted phoneme sequences.
        all_targets: Target phoneme sequences.
        all_canonical: Canonical phoneme sequences.
        all_perceived: Perceived phoneme sequences.
        all_ids: Sample IDs.
        id_to_phoneme: Mapping from IDs to phoneme strings.
        
    Returns:
        Dictionary containing phoneme recognition metrics.
    """
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    target_phonemes = convert_ids_to_phonemes(all_targets, id_to_phoneme)

    pred_phonemes = remove_silence_tokens(pred_phonemes)
    target_phonemes = remove_silence_tokens(target_phonemes)

    per_details = calculate_sequence_error_rate(
        ids=all_ids,
        references=target_phonemes,
        hypotheses=pred_phonemes
    )

    total_phonemes = sum(detail['num_reference_tokens'] for detail in per_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] 
                      for detail in per_details)
    total_insertions = sum(detail['insertions'] for detail in per_details)
    total_deletions = sum(detail['deletions'] for detail in per_details)
    total_substitutions = sum(detail['substitutions'] for detail in per_details)

    per = total_errors / total_phonemes if total_phonemes > 0 else 0

    # Mispronunciation detection metrics
    mispronunciation_metrics = {
        'precision': 0.0, 
        'recall': 0.0, 
        'f1_score': 0.0, 
        'true_acceptance': 0, 
        'false_rejection': 0, 
        'false_acceptance': 0, 
        'true_rejection': 0
    }
    
    if all_canonical and all_perceived:
        mispronunciation_metrics = calculate_mispronunciation_metrics(
            all_predictions, all_canonical, all_perceived, id_to_phoneme
        )

    return {
        'per': float(per),
        'mispronunciation_precision': float(mispronunciation_metrics['precision']),
        'mispronunciation_recall': float(mispronunciation_metrics['recall']),
        'mispronunciation_f1': float(mispronunciation_metrics['f1_score']),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'confusion_matrix': {
            'true_acceptance': int(mispronunciation_metrics['true_acceptance']),
            'false_rejection': int(mispronunciation_metrics['false_rejection']),
            'false_acceptance': int(mispronunciation_metrics['false_acceptance']),
            'true_rejection': int(mispronunciation_metrics['true_rejection'])
        }
    }