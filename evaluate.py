import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import PhonemeRecognitionModel
from data import PhonemeEvaluationDataset, phoneme_evaluation_collate_fn

def levenshtein_distance(seq1, seq2):
    """
    레벤슈타인 거리 계산 함수
    Returns:
        distance: 총 편집 거리
        insertions: 삽입 수
        deletions: 삭제 수
        substitutions: 대체 수
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    
    # 거리 행렬 초기화
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    
    # 편집 연산 추적
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
                # 일치
                matrix[x, y] = matrix[x-1, y-1]
                ops[x, y] = ops[x-1, y-1]
            else:
                # 최소 비용 연산 선택
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
    
    # 총 편집 거리와 각 편집 연산 횟수 반환
    deletions, insertions, substitutions = ops[size_x-1, size_y-1]
    
    # Python 기본 타입으로 변환
    distance = int(matrix[size_x-1, size_y-1])
    insertions = int(insertions)
    deletions = int(deletions)
    substitutions = int(substitutions)
    
    return distance, insertions, deletions, substitutions

def decode_ctc(log_probs, blank_idx=0):
    """
    CTC 그리디 디코딩
    """
    # 각 시간 단계에서 가장 확률이 높은 클래스 얻기
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []
    
    for b in range(batch_size):
        seq = []
        prev = -1
        for t in range(greedy_preds.shape[1]):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred
        decoded_seqs.append(seq)
    
    return decoded_seqs

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    """
    음소 인식 모델 평가
    """
    model.eval()
    
    total_phonemes = 0
    total_errors = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    per_sample_metrics = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='음소 인식 평가')
        
        for (waveforms, input_ids, attention_masks, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, perceived_lengths, canonical_lengths, wav_files) in progress_bar:
            
            waveforms = waveforms.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            audio_lengths = audio_lengths.to(device)
            
            # wav2vec용 어텐션 마스크 생성
            batch_size, audio_len = waveforms.shape
            audio_attention_mask = torch.ones((batch_size, audio_len), device=device)
            
            # 순전파
            phoneme_logits = model(
                waveforms, 
                input_ids, 
                audio_attention_mask=audio_attention_mask,
                text_attention_mask=attention_masks
            )
            
            # CTC 디코딩
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            batch_phoneme_preds = decode_ctc(log_probs)
            
            # 배치의 각 샘플 평가
            for i, (preds, true_phonemes, length, wav_file) in enumerate(
                zip(batch_phoneme_preds, perceived_phoneme_ids, perceived_lengths, wav_files)):
                
                # 패딩 제거한 참조 음소 시퀀스
                true_phonemes = true_phonemes[:length].cpu().numpy().tolist()
                
                # Python 기본 타입으로 변환
                true_phonemes = [int(p) for p in true_phonemes]
                
                # PER 계산
                per, insertions, deletions, substitutions = levenshtein_distance(preds, true_phonemes)
                
                # 총 음소 수
                phoneme_count = len(true_phonemes)
                
                # 통계 업데이트
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
    parser = argparse.ArgumentParser(description='L2 음소 인식 모델 평가')
    
    # 기본 설정
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='사용할 장치')
    
    # 데이터 설정
    parser.add_argument('--eval_data', type=str, required=True, help='평가 데이터 JSON 파일')
    parser.add_argument('--phoneme_map', type=str, required=True, help='음소-ID 매핑 JSON 파일')
    
    # 모델 설정
    parser.add_argument('--model_checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--pretrained_audio_model', type=str, default='facebook/wav2vec2-base-960h', 
                        help='사전학습된 wav2vec2 모델')
    parser.add_argument('--pretrained_text_model', type=str, default='bert-base-uncased', 
                        help='사전학습된 BERT 모델')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # 평가 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    parser.add_argument('--max_text_length', type=int, default=128, help='최대 텍스트 길이(토큰 단위)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='평가 결과 출력 디렉토리')
    parser.add_argument('--detailed', action='store_true', help='상세 결과 출력')
    
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
    
    # 평가 데이터셋 생성
    logger.info(f"평가 데이터셋 로드 중: {args.eval_data}")
    eval_dataset = PhonemeEvaluationDataset(
        args.eval_data, 
        phoneme_to_id, 
        text_model_name=args.pretrained_text_model,
        max_length=args.max_audio_length,
        max_text_length=args.max_text_length
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=phoneme_evaluation_collate_fn
    )
    
    # 모델 초기화
    logger.info("모델 초기화 중")
    model = PhonemeRecognitionModel(
        pretrained_audio_model=args.pretrained_audio_model,
        pretrained_text_model=args.pretrained_text_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
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
    
    # 음소 인식 평가
    logger.info("음소 인식 평가 중...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, args.device, id_to_phoneme)
    
    # 결과 출력
    logger.info("\n===== 음소 인식 결과 =====")
    logger.info(f"음소 오류율 (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"총 음소 수: {phoneme_recognition_results['total_phonemes']}")
    logger.info(f"총 오류 수: {phoneme_recognition_results['total_errors']}")
    logger.info(f"삽입: {phoneme_recognition_results['insertions']}")
    logger.info(f"삭제: {phoneme_recognition_results['deletions']}")
    logger.info(f"대체: {phoneme_recognition_results['substitutions']}")
    
    # 상세 결과 저장
    if args.detailed:
        # 샘플별 PER 결과 저장
        per_sample_results_path = os.path.join(args.output_dir, 'phoneme_recognition_per_sample.json')
        with open(per_sample_results_path, 'w') as f:
            json.dump(phoneme_recognition_results['per_sample'], f, indent=2)
        logger.info(f"샘플별 PER 결과를 {per_sample_results_path}에 저장했습니다.")
    
    # 전체 결과 저장
    results = {
        'phoneme_recognition': {
            'per': phoneme_recognition_results['per'],
            'total_phonemes': phoneme_recognition_results['total_phonemes'],
            'total_errors': phoneme_recognition_results['total_errors'],
            'insertions': phoneme_recognition_results['insertions'],
            'deletions': phoneme_recognition_results['deletions'],
            'substitutions': phoneme_recognition_results['substitutions']
        }
    }
    
    results_path = os.path.join(args.output_dir, 'phoneme_recognition_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"평가 결과를 {results_path}에 저장했습니다.")

if __name__ == "__main__":
    main()