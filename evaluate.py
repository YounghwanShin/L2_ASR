import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from model import DualWav2VecWithErrorAwarePhonemeRecognition

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Levenshtein 거리 계산
def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    
    # 행렬 초기화
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    
    # 첫 행과 열 초기화
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

class EvaluationDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # 오류 유형 매핑: C (정확함), D (삭제), A/I (추가/삽입), S (대체)
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'I': 3, 'S': 2}
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        # 길이 제한
        if self.max_length is not None and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        # 오류 레이블 변환
        error_labels = item.get('error_labels', '')
        error_labels = [self.error_type_mapping.get(label, 0) for label in error_labels.split()]
        error_labels = torch.tensor(error_labels, dtype=torch.long)
        
        # 인식된 음소 레이블 변환
        perceived_phonemes = item.get('perceived_train_target', '').split()
        perceived_phoneme_ids = []
        for phoneme in perceived_phonemes:
            if phoneme in self.phoneme_to_id:
                perceived_phoneme_ids.append(self.phoneme_to_id[phoneme])
        
        perceived_phoneme_ids = torch.tensor(perceived_phoneme_ids, dtype=torch.long)
        
        # 정규 발음 음소 레이블 변환 (참고용)
        canonical_phonemes = item.get('canonical_aligned', '').split()
        canonical_phoneme_ids = []
        for phoneme in canonical_phonemes:
            if phoneme in self.phoneme_to_id:
                canonical_phoneme_ids.append(self.phoneme_to_id[phoneme])
        
        canonical_phoneme_ids = torch.tensor(canonical_phoneme_ids, dtype=torch.long)
        
        # 음성 길이와 레이블 길이
        audio_length = torch.tensor(waveform.shape[1], dtype=torch.long)
        error_label_length = torch.tensor(len(error_labels), dtype=torch.long)
        perceived_length = torch.tensor(len(perceived_phoneme_ids), dtype=torch.long)
        canonical_length = torch.tensor(len(canonical_phoneme_ids), dtype=torch.long)
        
        return (
            waveform.squeeze(0), 
            error_labels, 
            perceived_phoneme_ids, 
            canonical_phoneme_ids,
            audio_length,
            error_label_length,
            perceived_length,
            canonical_length,
            wav_file
        )

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

def decode_ctc(log_probs, input_lengths, blank_idx=0):
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

def evaluate_error_detection(model, dataloader, device, error_type_names=None):
    if error_type_names is None:
        error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct'}
    
    model.eval()
    
    # 오류 유형별 통계
    total_errors = 0
    correct_errors = 0
    
    # 오류 유형별 통계
    error_type_stats = {error_type: {'true': 0, 'pred': 0, 'correct': 0} for error_type in error_type_names.keys()}
    
    # 혼동 행렬
    confusion_matrix = np.zeros((len(error_type_names), len(error_type_names)), dtype=np.int32)
    
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
            
            # 모델 순전파
            _, _, error_logits = model(waveforms, attention_mask, return_error_probs=True)
            
            # 정확한 다운샘플링 비율 계산 (추가된 부분)
            input_seq_len = waveforms.size(1)
            output_seq_len = error_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산 (추가된 부분)
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
            
            # CTC 디코딩
            log_probs = torch.log_softmax(error_logits, dim=-1)
            
            # decode_ctc 함수 수정 (input_lengths 전달)
            batch_error_preds = decode_ctc(log_probs, input_lengths)
            
            # 배치의 각 샘플에 대해 오류 예측 정확도 계산
            for i, (preds, true_errors, length) in enumerate(zip(batch_error_preds, error_labels, error_label_lengths)):
                # 패딩 제거
                true_errors = true_errors[:length].cpu().numpy()
                
                # 최대한 맞추기 위해 더 짧은 시퀀스의 길이로 자르기 (정확한 정렬이 없는 경우)
                min_len = min(len(preds), len(true_errors))
                true_errors_trimmed = true_errors[:min_len]
                preds_trimmed = preds[:min_len]
                
                # 전체 정확도 계산
                correct_in_sample = (np.array(preds_trimmed) == true_errors_trimmed).sum()
                total_in_sample = min_len
                
                total_errors += total_in_sample
                correct_errors += correct_in_sample
                
                # 오류 유형별 통계 및 혼동 행렬 업데이트
                for t, p in zip(true_errors_trimmed, preds_trimmed):
                    error_type_stats[int(t)]['true'] += 1
                    error_type_stats[int(p)]['pred'] += 1
                    if t == p:
                        error_type_stats[int(t)]['correct'] += 1
                    
                    confusion_matrix[int(t), int(p)] += 1
    
    # 전체 정확도 계산
    accuracy = correct_errors / total_errors if total_errors > 0 else 0
    
    # 오류 유형별 정밀도, 재현율, F1 점수 계산
    error_type_metrics = {}
    for error_type, stats in error_type_stats.items():
        precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
        recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        error_type_metrics[error_type_names[error_type]] = {
            'precision': float(precision),  # NumPy 값을 Python 값으로 변환
            'recall': float(recall),
            'f1': float(f1),
            'support': int(stats['true'])
        }
    
    return {
        'accuracy': float(accuracy),
        'error_type_metrics': error_type_metrics,
        'confusion_matrix': confusion_matrix.tolist()  # NumPy 배열을 리스트로 변환
    }

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
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
            phoneme_logits, adjusted_probs = model(waveforms, attention_mask)
            
            # 정확한 다운샘플링 비율 계산 (추가된 부분)
            input_seq_len = waveforms.size(1)
            output_seq_len = phoneme_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산 (추가된 부분)
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
            
            # 음소 인식을 위한 CTC 디코딩
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # decode_ctc 함수 수정 (input_lengths 전달)
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
    
    # 데이터 설정
    parser.add_argument('--eval_data', type=str, required=True, help='평가 데이터 JSON 파일')
    parser.add_argument('--phoneme_map', type=str, required=True, help='음소-ID 매핑 JSON 파일')
    
    # 모델 설정
    parser.add_argument('--model_checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-960h', help='wav2vec2 모델 이름')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.25, help='어댑터 차원 비율')
    parser.add_argument('--error_influence_weight', type=float, default=0.2, help='오류 영향 가중치')
    
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
    error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct'}
    
    # 평가 데이터셋 생성
    logger.info(f"평가 데이터셋 로드 중: {args.eval_data}")
    eval_dataset = EvaluationDataset(
        args.eval_data, phoneme_to_id, max_length=args.max_audio_length
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # 모델 초기화
    logger.info("모델 초기화 중")
    model = DualWav2VecWithErrorAwarePhonemeRecognition(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        adapter_dim_ratio=args.adapter_dim_ratio,
        error_influence_weight=args.error_influence_weight,
        blank_index=0,
        sil_index=1
    )
    
    # 모델 체크포인트 로드
    logger.info(f"체크포인트 로드 중: {args.model_checkpoint}")
    state_dict = torch.load(args.model_checkpoint, map_location=args.device)

    # "module." 접두사 제거
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 'module.' 접두사 제거
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    
    # 오류 탐지 평가
    logger.info("오류 탐지 평가 중...")
    error_detection_results = evaluate_error_detection(model, eval_dataloader, args.device, error_type_names)
    
    # 음소 인식 평가
    logger.info("음소 인식 평가 중...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, args.device, id_to_phoneme)
    
    # 결과 출력
    logger.info("\n===== 오류 탐지 결과 =====")
    logger.info(f"전체 정확도: {error_detection_results['accuracy']:.4f}")
    
    logger.info("\n오류 유형별 메트릭:")
    for error_type, metrics in error_detection_results['error_type_metrics'].items():
        logger.info(f"  {error_type}:")
        logger.info(f"    정밀도: {metrics['precision']:.4f}")
        logger.info(f"    재현율: {metrics['recall']:.4f}")
        logger.info(f"    F1 점수: {metrics['f1']:.4f}")
        logger.info(f"    Support: {metrics['support']}")
    
    logger.info("\n===== 음소 인식 결과 =====")
    logger.info(f"음소 오류율 (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"총 음소 수: {phoneme_recognition_results['total_phonemes']}")
    logger.info(f"총 오류 수: {phoneme_recognition_results['total_errors']}")
    logger.info(f"삽입: {phoneme_recognition_results['insertions']}")
    logger.info(f"삭제: {phoneme_recognition_results['deletions']}")
    logger.info(f"대체: {phoneme_recognition_results['substitutions']}")
    
    # 상세 결과 저장
    if args.detailed:
        # 혼동 행렬 저장
        confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.json')
        with open(confusion_matrix_path, 'w') as f:
            json.dump(error_detection_results['confusion_matrix'], f, indent=2)
        logger.info(f"혼동 행렬을 {confusion_matrix_path}에 저장했습니다.")
        
        # 샘플별 PER 결과 저장
        per_sample_results_path = os.path.join(args.output_dir, 'per_sample_results.json')
        with open(per_sample_results_path, 'w') as f:
            json.dump(phoneme_recognition_results['per_sample'], f, indent=2)
        logger.info(f"샘플별 PER 결과를 {per_sample_results_path}에 저장했습니다.")
    
    # 전체 결과 저장
    results = {
        'error_detection': {
            'accuracy': error_detection_results['accuracy'],
            'error_type_metrics': error_detection_results['error_type_metrics']
        },
        'phoneme_recognition': {
            'per': phoneme_recognition_results['per'],
            'total_phonemes': phoneme_recognition_results['total_phonemes'],
            'total_errors': phoneme_recognition_results['total_errors'],
            'insertions': phoneme_recognition_results['insertions'],
            'deletions': phoneme_recognition_results['deletions'],
            'substitutions': phoneme_recognition_results['substitutions']
        }
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"평가 결과를 {results_path}에 저장했습니다.")

if __name__ == "__main__":
    main()