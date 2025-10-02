import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score


def edit_distance_with_details(ref: List, hyp: List) -> Tuple[int, int, int, int, int]:
    """편집 거리와 세부 에러 타입을 계산합니다.
    
    Args:
        ref: 참조 시퀀스
        hyp: 가설 시퀀스
        
    Returns:
        (총 에러수, 대체수, 삭제수, 삽입수, 참조 길이)
    """
    len_ref, len_hyp = len(ref), len(hyp)

    if len_ref == 0:
        return len_hyp, 0, 0, len_hyp, 0
    if len_hyp == 0:
        return len_ref, 0, len_ref, 0, len_ref

    # DP 테이블 초기화
    dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]

    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j

    # DP 테이블 채우기
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 삭제
                    dp[i][j-1],      # 삽입
                    dp[i-1][j-1]     # 대체
                )

    total_errors = dp[len_ref][len_hyp]

    # 백트래킹으로 에러 타입 계산
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
    """배치에 대한 WER 세부 정보를 계산합니다."""
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
    """ID 시퀀스를 음소 문자열로 변환합니다."""
    return [[id_to_phoneme.get(str(token_id), f"UNK_{token_id}") for token_id in seq] for seq in sequences]


def remove_sil_tokens(sequences: List[List[str]]) -> List[List[str]]:
    """sil 토큰을 제거합니다."""
    return [[token for token in seq if token != "sil"] for seq in sequences]


def calculate_mispronunciation_metrics(all_predictions: List[List[int]], 
                                     all_canonical: List[List[int]], 
                                     all_perceived: List[List[int]], 
                                     id_to_phoneme: Dict[str, str]) -> Dict:
    """잘못된 발음 탐지 메트릭을 계산합니다."""
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

    # 메트릭 계산
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
    """에러 탐지 메트릭을 계산합니다."""
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

    # 분류 메트릭 계산
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

            # 블랭크를 제외한 에러 타입들에 대한 메트릭
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
    """음소 인식 메트릭을 계산합니다."""
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

    # 잘못된 발음 탐지 메트릭
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