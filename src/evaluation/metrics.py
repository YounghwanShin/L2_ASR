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


def edit_distance_with_details(ref: List, hyp: List) -> Tuple[int, int, int, int, int]:
    """Computes edit distance with detailed error breakdown.
    
    Uses dynamic programming to compute Levenshtein distance and then
    backtracks to count specific error types.
    
    Args:
        ref: Reference sequence.
        hyp: Hypothesis sequence.
        
    Returns:
        Tuple of (total_errors, substitutions, deletions, insertions, ref_length).
    """
    len_ref, len_hyp = len(ref), len(hyp)

    if len_ref == 0:
        return len_hyp, 0, 0, len_hyp, 0
    if len_hyp == 0:
        return len_ref, 0, len_ref, 0, len_ref

    # Initialize DP table
    dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]

    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )

    total_errors = dp[len_ref][len_hyp]

    # Backtrack to count error types
    i, j = len_ref, len_hyp
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
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

    return total_errors, substitutions, deletions, insertions, len_ref


def calculate_wer_details(ids: List, refs: List[List], hyps: List[List]) -> List[Dict]:
    """Calculates Word Error Rate (WER) details for a batch.
    
    Args:
        ids: List of sample IDs.
        refs: List of reference sequences.
        hyps: List of hypothesis sequences.
        
    Returns:
        List of dictionaries containing WER metrics for each sample.
    """
    details = []

    for i, (id_val, ref, hyp) in enumerate(zip(ids, refs, hyps)):
        ref_tokens = [str(token) for token in ref]
        hyp_tokens = [str(token) for token in hyp]

        total_errors, substitutions, deletions, insertions, num_ref = edit_distance_with_details(
            ref_tokens, hyp_tokens
        )

        wer = total_errors / num_ref if num_ref > 0 else 0.0

        detail = {
            'id': id_val,
            'WER': wer,
            'num_ref_tokens': num_ref,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'num_scored_tokens': num_ref,
            'num_hyp_tokens': len(hyp_tokens)
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


def remove_sil_tokens(sequences: List[List[str]]) -> List[List[str]]:
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

    pred_phonemes = remove_sil_tokens(pred_phonemes)
    canonical_phonemes = remove_sil_tokens(canonical_phonemes)
    perceived_phonemes = remove_sil_tokens(perceived_phonemes)

    total_ta, total_fr, total_fa, total_tr = 0, 0, 0, 0

    for pred, canonical, perceived in zip(pred_phonemes, canonical_phonemes, perceived_phonemes):
        if len(canonical) == 0 or len(perceived) == 0 or len(pred) == 0:
            continue

        max_len = max(len(canonical), len(perceived), len(pred))

        canonical_padded = canonical + ['<pad>'] * (max_len - len(canonical))
        perceived_padded = perceived + ['<pad>'] * (max_len - len(perceived))
        pred_padded = pred + ['<pad>'] * (max_len - len(pred))

        for c_phone, p_phone, pred_phone in zip(canonical_padded, perceived_padded, pred_padded):
            if c_phone == '<pad>' or p_phone == '<pad>' or pred_phone == '<pad>':
                continue

            is_correct = (c_phone == p_phone)
            pred_correct = (c_phone == pred_phone)

            if is_correct and pred_correct:
                total_ta += 1
            elif is_correct and not pred_correct:
                total_fr += 1
            elif not is_correct and pred_correct:
                total_fa += 1
            elif not is_correct and not pred_correct:
                total_tr += 1

    # Compute metrics
    if (total_tr + total_fa) > 0:
        recall = total_tr / (total_tr + total_fa)
    else:
        recall = 0.0

    if (total_tr + total_fr) > 0:
        precision = total_tr / (total_tr + total_fr)
    else:
        precision = 0.0

    if (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ta': total_ta,
        'fr': total_fr,
        'fa': total_fa,
        'tr': total_tr
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
    wer_details = calculate_wer_details(
        ids=all_ids,
        refs=all_targets,
        hyps=all_predictions
    )

    total_sequences = len(wer_details)
    correct_sequences = sum(1 for detail in wer_details if detail['WER'] == 0.0)

    total_tokens = sum(detail['num_ref_tokens'] for detail in wer_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] for detail in wer_details)

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
            eval_error_types = {k: v for k, v in error_type_names.items() if k != 0}
            for class_id, class_name in eval_error_types.items():
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

    pred_phonemes = remove_sil_tokens(pred_phonemes)
    target_phonemes = remove_sil_tokens(target_phonemes)

    per_details = calculate_wer_details(
        ids=all_ids,
        refs=target_phonemes,
        hyps=pred_phonemes
    )

    total_phonemes = sum(detail['num_ref_tokens'] for detail in per_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] for detail in per_details)
    total_insertions = sum(detail['insertions'] for detail in per_details)
    total_deletions = sum(detail['deletions'] for detail in per_details)
    total_substitutions = sum(detail['substitutions'] for detail in per_details)

    per = total_errors / total_phonemes if total_phonemes > 0 else 0

    # Mispronunciation detection metrics
    misp_metrics = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ta': 0, 'fr': 0, 'fa': 0, 'tr': 0}
    if all_canonical and all_perceived:
        misp_metrics = calculate_mispronunciation_metrics(all_predictions, all_canonical, all_perceived, id_to_phoneme)

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
            'true_acceptance': int(misp_metrics['ta']),
            'false_rejection': int(misp_metrics['fr']),
            'false_acceptance': int(misp_metrics['fa']),
            'true_rejection': int(misp_metrics['tr'])
        }
    }
