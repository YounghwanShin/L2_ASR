import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from model import ErrorDetectionModel, PhonemeRecognitionModel
from data import EvaluationDataset

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    
    # 삽입, 삭제, 대체 연산 추적을 위한 행렬
    ops = np.zeros((size_x, size_y, 3), dtype=np.int32)  # [DEL, INS, SUB]
    
    # 첫 행과 열의 연산 초기화
    for x in range(1, size_x):
        ops[x, 0, 0] = 1  # 삭제
    for y in range(1, size_y):
        ops[0, y, 1] = 1  # 삽입
    
    # 행렬 채우기
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                # 일치하는 경우
                matrix[x, y] = matrix[x-1, y-1]
                ops[x, y] = ops[x-1, y-1]
            else:
                # 삭제, 삽입, 대체 중 최소 비용 선택
                delete = matrix[x-1, y] + 1
                insert = matrix[x, y-1] + 1
                subst = matrix[x-1, y-1] + 1
                
                min_val = min(delete, insert, subst)
                matrix[x, y] = min_val
                
                if min_val == delete:
                    ops[x, y] = ops[x-1, y].copy()
                    ops[x, y, 0] += 1  # 삭제 +1
                elif min_val == insert:
                    ops[x, y] = ops[x, y-1].copy()
                    ops[x, y, 1] += 1  # 삽입 +1
                else:  # 대체
                    ops[x, y] = ops[x-1, y-1].copy()
                    ops[x, y, 2] += 1  # 대체 +1
    
    # 총 편집 거리와 각 편집 연산의 횟수 반환
    deletions, insertions, substitutions = ops[size_x-1, size_y-1]
    
    # NumPy 타입을 Python 네이티브 타입으로 변환
    distance = int(matrix[size_x-1, size_y-1])
    insertions = int(insertions)
    deletions = int(deletions)
    substitutions = int(substitutions)
    
    # 편집 거리, 삽입, 삭제, 대체 반환
    return distance, insertions, deletions, substitutions

def get_wav2vec2_output_lengths_official(model, input_lengths):
    """HuggingFace 공식 방법 사용 - 커스텀 모델 구조에 맞게 수정"""
    # DataParallel 처리
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 모델 구조에 따라 wav2vec2에 접근
    if hasattr(actual_model, 'encoder'):
        # ErrorDetectionModel인 경우: model.encoder.wav2vec2
        wav2vec_model = actual_model.encoder.wav2vec2
    elif hasattr(actual_model, 'error_model'):
        # PhonemeRecognitionModel인 경우: model.error_model.encoder.wav2vec2
        wav2vec_model = actual_model.error_model.encoder.wav2vec2
    else:
        # 직접 wav2vec2 모델인 경우
        wav2vec_model = actual_model
    
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)

def decode_ctc(log_probs, input_lengths, blank_idx=0):
    """CTC 디코딩 함수"""
    # 각 시간 단계에서 가장 확률이 높은 클래스 얻기
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()  # (batch_size, seq_len)
    
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []
    
    for b in range(batch_size):
        seq = []
        # 연속된 중복 제거 및 blank 토큰 제거
        prev = -1
        # 실제 길이까지만 디코딩
        actual_length = input_lengths[b].item()
        for t in range(min(greedy_preds.shape[1], actual_length)):  # 유효한 부분만 처리
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))  # NumPy int32를 Python int로 변환
            prev = pred
        decoded_seqs.append(seq)
    
    return decoded_seqs

def decode_and_remove_separators(log_probs, input_lengths, blank_idx=0, separator_idx=5):
    """CTC 디코딩 후 구분자 제거하여 원래 오류 시퀀스 복원"""
    # 기본 CTC 디코딩
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []
    
    for b in range(batch_size):
        seq = []
        prev = -1
        actual_length = input_lengths[b].item()
        
        # CTC 디코딩: 연속 중복 제거, blank 제거
        for t in range(min(greedy_preds.shape[1], actual_length)):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred
        
        # 구분자 제거하여 원래 오류 라벨 시퀀스 복원
        final_seq = [token for token in seq if token != separator_idx]
        decoded_seqs.append(final_seq)
    
    return decoded_seqs

def prepare_target_without_separators(error_labels, label_lengths, separator_idx=5):
    """타겟에서도 구분자 제거"""
    targets = []
    for labels, length in zip(error_labels, label_lengths):
        target_seq = labels[:length].cpu().numpy().tolist()
        # 구분자 제거
        clean_target = [token for token in target_seq if token != separator_idx]
        targets.append(clean_target)
    return targets

def collate_fn(batch):
    (
        waveforms, 
        error_labels, 
        perceived_phoneme_ids, 
        canonical_phoneme_ids,
        audio_lengths,
        error_label_lengths,
        perceived_lengths,
        canonical_lengths,
        wav_files
    ) = zip(*batch)
    
    # 가변 길이 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    # 오류 레이블 패딩
    max_error_len = max([labels.shape[0] for labels in error_labels])
    padded_error_labels = []
    
    for labels in error_labels:
        label_len = labels.shape[0]
        padding = max_error_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)
        padded_error_labels.append(padded_labels)
    
    # 인식된 음소 레이블 패딩
    max_perceived_len = max([ids.shape[0] for ids in perceived_phoneme_ids])
    padded_perceived_ids = []
    
    for ids in perceived_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_perceived_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_perceived_ids.append(padded_ids)
    
    # 정규 발음 음소 레이블 패딩
    max_canonical_len = max([ids.shape[0] for ids in canonical_phoneme_ids])
    padded_canonical_ids = []
    
    for ids in canonical_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_canonical_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_canonical_ids.append(padded_ids)
    
    # 텐서로 변환
    padded_waveforms = torch.stack(padded_waveforms)
    padded_error_labels = torch.stack(padded_error_labels)
    padded_perceived_ids = torch.stack(padded_perceived_ids)
    padded_canonical_ids = torch.stack(padded_canonical_ids)
    
    audio_lengths = torch.tensor(audio_lengths)
    error_label_lengths = torch.tensor(error_label_lengths)
    perceived_lengths = torch.tensor(perceived_lengths)
    canonical_lengths = torch.tensor(canonical_lengths)
    
    return (
        padded_waveforms, 
        padded_error_labels, 
        padded_perceived_ids, 
        padded_canonical_ids,
        audio_lengths,
        error_label_lengths,
        perceived_lengths,
        canonical_lengths,
        wav_files
    )

def evaluate_error_detection(model, dataloader, device, error_type_names=None):
    """오류 탐지 모델 평가 - 단순화된 버전"""
    if error_type_names is None:
        error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct', 5: 'separator'}
    
    model.eval()
    
    total_sequences = 0
    correct_sequences = 0
    total_edit_distance = 0
    total_tokens = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    # 토큰별 평가를 위한 리스트
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='오류 탐지 평가')
        
        for (waveforms, error_labels, _, _, audio_lengths, error_label_lengths, 
             _, _, wav_files) in progress_bar:
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            # 모델 순전파 (ErrorDetectionModel 또는 PhonemeRecognitionModel 모두 처리)
            if hasattr(model, 'error_model') or (hasattr(model, 'module') and hasattr(model.module, 'error_model')):
                # PhonemeRecognitionModel인 경우
                _, error_logits = model(waveforms, attention_mask)
            else:
                # ErrorDetectionModel인 경우
                error_logits = model(waveforms, attention_mask)
            
            # HuggingFace 공식 방법으로 정확한 길이 계산
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
            
            # CTC 디코딩 (구분자 제거)
            log_probs = torch.log_softmax(error_logits, dim=-1)
            predictions = decode_and_remove_separators(log_probs, input_lengths)
            
            # 타겟 준비 (구분자 제거)
            targets = prepare_target_without_separators(error_labels, error_label_lengths)
            
            # 시퀀스별 평가
            for pred, target in zip(predictions, targets):
                total_sequences += 1
                total_tokens += len(target)
                
                # 완전 일치
                if pred == target:
                    correct_sequences += 1
                
                # 편집 거리
                edit_dist, insertions, deletions, substitutions = levenshtein_distance(pred, target)
                total_edit_distance += edit_dist
                total_insertions += insertions
                total_deletions += deletions
                total_substitutions += substitutions
                
                # 토큰별 통계 (F1 계산용)
                all_predictions.extend(pred)
                all_targets.extend(target)
    
    # 주요 메트릭 계산
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    token_accuracy = 1 - (total_edit_distance / total_tokens) if total_tokens > 0 else 0
    avg_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0
    
    # sklearn으로 클래스별 메트릭 계산
    weighted_f1 = 0
    macro_f1 = 0
    class_metrics = {}
    
    if len(all_predictions) > 0 and len(all_targets) > 0:
        try:
            # 길이 맞추기
            min_len = min(len(all_predictions), len(all_targets))
            all_predictions = all_predictions[:min_len]
            all_targets = all_targets[:min_len]
            
            weighted_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
            
            # 클래스별 상세 정보
            class_report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
            
            # 오류 유형 이름으로 변환
            eval_error_types = {k: v for k, v in error_type_names.items() if k != 5}  # separator 제외
            for class_id, class_name in eval_error_types.items():
                if str(class_id) in class_report:
                    class_metrics[class_name] = {
                        'precision': float(class_report[str(class_id)]['precision']),
                        'recall': float(class_report[str(class_id)]['recall']),
                        'f1': float(class_report[str(class_id)]['f1-score']),
                        'support': int(class_report[str(class_id)]['support'])
                    }
        except Exception as e:
            print(f"클래스별 메트릭 계산 중 오류: {e}")
    
    return {
        'sequence_accuracy': float(sequence_accuracy),
        'token_accuracy': float(token_accuracy),
        'avg_edit_distance': float(avg_edit_distance),
        'weighted_f1': float(weighted_f1),
        'macro_f1': float(macro_f1),
        'class_metrics': class_metrics,
        'total_sequences': int(total_sequences),
        'total_tokens': int(total_tokens),
        'total_insertions': int(total_insertions),
        'total_deletions': int(total_deletions),
        'total_substitutions': int(total_substitutions)
    }

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    """음소 인식 모델 평가"""
    model.eval()
    
    total_phonemes = 0
    total_errors = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    per_sample_metrics = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='음소 인식 평가')
        
        for (waveforms, _, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, _, perceived_lengths, canonical_lengths, wav_files) in progress_bar:
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            # 모델 순전파
            phoneme_logits, _ = model(waveforms, attention_mask)
            
            # HuggingFace 공식 방법으로 정확한 길이 계산
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            
            # 음소 인식을 위한 CTC 디코딩
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            batch_phoneme_preds = decode_ctc(log_probs, input_lengths)
            
            # 배치의 각 샘플에 대해 PER 계산
            for i, (preds, true_phonemes, length, wav_file) in enumerate(
                zip(batch_phoneme_preds, perceived_phoneme_ids, perceived_lengths, wav_files)):
                
                # 패딩 제거한 참조 음소 시퀀스
                true_phonemes = true_phonemes[:length].cpu().numpy().tolist()
                
                # Python 기본 타입으로 변환 (NumPy int32 -> Python int)
                true_phonemes = [int(p) for p in true_phonemes]
                
                # PER 계산
                per, insertions, deletions, substitutions = levenshtein_distance(preds, true_phonemes)
                
                # 총 음소 수
                phoneme_count = len(true_phonemes)
                
                # 누적 통계 업데이트
                total_phonemes += phoneme_count
                total_errors += per
                total_insertions += insertions
                total_deletions += deletions
                total_substitutions += substitutions
                
                # 샘플별 결과 저장
                per_sample_metrics.append({
                    'wav_file': wav_file,
                    'per': float(per / phoneme_count) if phoneme_count > 0 else 0.0,
                    'insertions': insertions,
                    'deletions': deletions,
                    'substitutions': substitutions,
                    'true_phonemes': [id_to_phoneme.get(str(p), "UNK") for p in true_phonemes],
                    'pred_phonemes': [id_to_phoneme.get(str(p), "UNK") for p in preds]
                })
        
    # 전체 PER 계산
    per = total_errors / total_phonemes if total_phonemes > 0 else 0
    
    return {
        'per': float(per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'per_sample': per_sample_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='L2 음소 인식 및 오류 탐지 모델 평가')
    
    # 기본 설정
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='사용할 장치')
    parser.add_argument('--mode', type=str, choices=['error', 'phoneme', 'both'], default='both', help='평가 모드')
    
    # 데이터 설정
    parser.add_argument('--eval_data', type=str, required=True, help='평가 데이터 JSON 파일')
    parser.add_argument('--phoneme_map', type=str, required=True, help='음소-ID 매핑 JSON 파일')
    
    # 모델 설정
    parser.add_argument('--error_model_checkpoint', type=str, help='오류 탐지 모델 체크포인트 경로')
    parser.add_argument('--phoneme_model_checkpoint', type=str, help='음소 인식 모델 체크포인트 경로')
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53', help='wav2vec2 모델 이름')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--num_error_types', type=int, default=6, help='오류 유형 수 (blank + separator 포함)')
    
    # 평가 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='평가 결과 출력 디렉토리')
    parser.add_argument('--detailed', action='store_true', help='상세한 결과 출력')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 음소 매핑 로드
    logger.info(f"음소 매핑 파일 로드 중: {args.phoneme_map}")
    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    
    # ID를 음소로 변환하는 역매핑 생성
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    # 오류 유형 이름 매핑
    error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct', 5: 'separator'}
    
    # 평가 데이터셋 생성
    logger.info(f"평가 데이터셋 로드 중: {args.eval_data}")
    eval_dataset = EvaluationDataset(
        args.eval_data, phoneme_to_id, max_length=args.max_audio_length
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # 오류 탐지 모델 평가
    if args.mode in ['error', 'both']:
        if not args.error_model_checkpoint:
            logger.error("오류 탐지 평가를 위한 모델 체크포인트가 필요합니다.")
            if args.mode == 'error':
                sys.exit(1)
        else:
            logger.info("오류 탐지 모델 초기화")
            error_model = ErrorDetectionModel(
                pretrained_model_name=args.pretrained_model,
                hidden_dim=args.hidden_dim,
                num_error_types=args.num_error_types
            )
            
            # 체크포인트 로드
            logger.info(f"오류 탐지 모델 체크포인트 로드 중: {args.error_model_checkpoint}")
            state_dict = torch.load(args.error_model_checkpoint, map_location=args.device)
            
            # "module." 접두사 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 'module.' 접두사 제거
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            error_model.load_state_dict(new_state_dict)
            error_model = error_model.to(args.device)
            
            # 오류 탐지 평가
            logger.info("오류 탐지 평가 중...")
            error_detection_results = evaluate_error_detection(
                error_model, eval_dataloader, args.device, error_type_names
            )
            
            # 결과 출력
            logger.info("\n===== 오류 탐지 결과 =====")
            logger.info(f"시퀀스 정확도: {error_detection_results['sequence_accuracy']:.4f}")
            logger.info(f"토큰 정확도: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"평균 편집 거리: {error_detection_results['avg_edit_distance']:.4f}")
            logger.info(f"Weighted F1: {error_detection_results['weighted_f1']:.4f}")
            logger.info(f"Macro F1: {error_detection_results['macro_f1']:.4f}")
            
            logger.info("\n오류 유형별 메트릭:")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                logger.info(f"  {error_type}:")
                logger.info(f"    정밀도: {metrics['precision']:.4f}")
                logger.info(f"    재현율: {metrics['recall']:.4f}")
                logger.info(f"    F1 점수: {metrics['f1']:.4f}")
                logger.info(f"    Support: {metrics['support']}")
                
            # 결과 저장
            results['error_detection'] = error_detection_results
    
    # 음소 인식 모델 평가
    if args.mode in ['phoneme', 'both']:
        if not args.phoneme_model_checkpoint or (args.mode == 'both' and not args.error_model_checkpoint):
            logger.error("음소 인식 평가를 위한 모델 체크포인트가 필요합니다.")
            if args.mode == 'phoneme':
                sys.exit(1)
        else:
            logger.info("음소 인식 모델 초기화")
            phoneme_model = PhonemeRecognitionModel(
                pretrained_model_name=args.pretrained_model,
                error_model_checkpoint=args.error_model_checkpoint if args.mode == 'both' else None,
                hidden_dim=args.hidden_dim,
                num_phonemes=args.num_phonemes,
                num_error_types=args.num_error_types
            )
            
            # 체크포인트 로드
            logger.info(f"음소 인식 모델 체크포인트 로드 중: {args.phoneme_model_checkpoint}")
            state_dict = torch.load(args.phoneme_model_checkpoint, map_location=args.device)
            
            # "module." 접두사 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 'module.' 접두사 제거
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            phoneme_model.load_state_dict(new_state_dict)
            phoneme_model = phoneme_model.to(args.device)
            
            # 음소 인식 평가
            logger.info("음소 인식 평가 중...")
            phoneme_recognition_results = evaluate_phoneme_recognition(
                phoneme_model, eval_dataloader, args.device, id_to_phoneme
            )
            
            # 결과 출력
            logger.info("\n===== 음소 인식 결과 =====")
            logger.info(f"음소 오류율 (PER): {phoneme_recognition_results['per']:.4f}")
            logger.info(f"총 음소 수: {phoneme_recognition_results['total_phonemes']}")
            logger.info(f"총 오류 수: {phoneme_recognition_results['total_errors']}")
            logger.info(f"삽입: {phoneme_recognition_results['insertions']}")
            logger.info(f"삭제: {phoneme_recognition_results['deletions']}")
            logger.info(f"대체: {phoneme_recognition_results['substitutions']}")
            
            # 결과 저장
            results['phoneme_recognition'] = {
                'per': phoneme_recognition_results['per'],
                'total_phonemes': phoneme_recognition_results['total_phonemes'],
                'total_errors': phoneme_recognition_results['total_errors'],
                'insertions': phoneme_recognition_results['insertions'],
                'deletions': phoneme_recognition_results['deletions'],
                'substitutions': phoneme_recognition_results['substitutions']
            }
            
            # 상세 결과 저장
            if args.detailed:
                # 샘플별 PER 결과 저장
                per_sample_results_path = os.path.join(args.output_dir, 'per_sample_results.json')
                with open(per_sample_results_path, 'w') as f:
                    json.dump(phoneme_recognition_results['per_sample'], f, indent=2)
                logger.info(f"샘플별 PER 결과를 {per_sample_results_path}에 저장했습니다.")
    
    # 전체 결과 저장
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"평가 결과를 {results_path}에 저장했습니다.")

if __name__ == "__main__":
    main()