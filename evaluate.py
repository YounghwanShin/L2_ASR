import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from model import DualWav2VecWithErrorAwarePhonemeRecognition
from sklearn.metrics import classification_report, accuracy_score
import editdistance

class EvalDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # 오류 유형 매핑 (train.py와 동일)
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'S': 2}
        self.error_id_to_type = {v: k for k, v in self.error_type_mapping.items()}
        
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
        
        # 지각된 음소 레이블 변환 (perceived_aligned)
        perceived_target = item.get('perceived_aligned', '')
        perceived_labels = []
        for phoneme in perceived_target.split():
            if phoneme in self.phoneme_to_id:
                perceived_labels.append(self.phoneme_to_id[phoneme])
        
        # 오류 레이블 변환
        error_labels = item.get('error_labels', '')
        error_ids = []
        for error_type in error_labels.split():
            if error_type in self.error_type_mapping:
                error_ids.append(self.error_type_mapping[error_type])
        
        return (
            waveform.squeeze(0),
            torch.tensor(perceived_labels, dtype=torch.long),
            torch.tensor(error_ids, dtype=torch.long),
            len(perceived_labels),
            len(error_ids),
            wav_file,
            item
        )

def collate_fn(batch):
    waveforms, perceived_labels, error_labels, perceived_lengths, error_lengths, wav_files, items = zip(*batch)
    
    # 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    audio_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    
    # 레이블 패딩
    max_perceived_len = max(perceived_lengths)
    max_error_len = max(error_lengths)
    
    padded_perceived = torch.zeros((len(batch), max_perceived_len), dtype=torch.long)
    padded_errors = torch.zeros((len(batch), max_error_len), dtype=torch.long)
    
    for i, (per, err, per_len, err_len) in enumerate(zip(
        perceived_labels, error_labels, perceived_lengths, error_lengths)):
        padded_perceived[i, :per_len] = per
        padded_errors[i, :err_len] = err
    
    return (
        torch.stack(padded_waveforms),
        padded_perceived,
        padded_errors,
        audio_lengths,
        torch.tensor(perceived_lengths),
        torch.tensor(error_lengths),
        wav_files,
        items
    )

def ctc_decode(logits, lengths, blank_index=0):
    """CTC 디코딩"""
    batch_size, max_len, vocab_size = logits.shape
    predictions = []
    
    for i in range(batch_size):
        # 로그 확률 계산
        log_probs = torch.log_softmax(logits[i], dim=-1).cpu().numpy()
        seq_len = lengths[i].item()
        
        # Greedy 디코딩
        best_path = np.argmax(log_probs[:seq_len], axis=-1)
        
        # CTC collapse
        decoded = []
        prev_label = None
        for label in best_path:
            if label != blank_index and label != prev_label:
                decoded.append(label)
            prev_label = label
        
        predictions.append(decoded)
    
    return predictions

def calculate_per(predictions, targets, id_to_phoneme, remove_sil=True):
    """PER (Phoneme Error Rate) 계산"""
    total_phonemes = 0
    total_errors = 0
    
    per_sample_errors = []
    
    for pred, target in zip(predictions, targets):
        # ID를 음소로 변환
        pred_phonemes = [id_to_phoneme.get(p, '<UNK>') for p in pred]
        target_phonemes = [id_to_phoneme.get(t, '<UNK>') for t in target]
        
        # 'sil' 제거 (선택적)
        if remove_sil:
            pred_phonemes = [p for p in pred_phonemes if p != 'sil']
            target_phonemes = [t for t in target_phonemes if t != 'sil']
        
        # Edit distance 계산
        errors = editdistance.eval(pred_phonemes, target_phonemes)
        total_errors += errors
        total_phonemes += len(target_phonemes)
        
        # 샘플별 PER
        sample_per = errors / max(len(target_phonemes), 1) * 100
        per_sample_errors.append(sample_per)
    
    # 전체 PER
    per = total_errors / max(total_phonemes, 1) * 100
    
    return per, per_sample_errors

def calculate_error_rate(predictions, targets, labels):
    """
    SpeechBrain 방식의 오류률 계산
    predictions: 디코딩된 시퀀스 리스트
    targets: 타겟 시퀀스 리스트
    labels: 라벨 이름 리스트
    """
    total_errors = 0
    total_tokens = 0
    
    for pred, target in zip(predictions, targets):
        # Edit distance 계산
        errors = editdistance.eval(pred, target)
        total_errors += errors
        total_tokens += len(target)
    
    error_rate = total_errors / max(total_tokens, 1) * 100 if total_tokens > 0 else 100
    
    return error_rate

def evaluate_stage1_error_detection(model, dataloader, device):
    """1단계: 오류 탐지만 평가 (SpeechBrain 방식)"""
    model.eval()
    
    all_predictions = []  # 디코딩된 시퀀스들
    all_targets = []      # 타겟 시퀀스들
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Stage 1: Error Detection Only"):
            waveforms, perceived_labels, error_labels, audio_lengths, perceived_lengths, error_lengths, _, _ = batch
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
            
            # 오류 탐지 예측
            _, _, error_logits = model(waveforms, attention_mask, return_error_probs=True)
            
            # wav2vec2 출력 길이 계산
            wav2vec_output_lengths = (audio_lengths / 20).long()
            
            # CTC 디코딩으로 예측값 계산
            error_preds = ctc_decode(error_logits, wav2vec_output_lengths)
            
            # 배치별로 처리
            for i in range(len(error_labels)):
                # 타겟 레이블 (CTC 형식에서 일반 시퀀스로 변환)
                target_len = error_lengths[i].item()
                target_labels = error_labels[i, :target_len].cpu().numpy()
                
                # blank 제거하고 중복 제거
                target_seq = []
                prev_label = None
                for label in target_labels:
                    if label != 0 and label != prev_label:  # blank=0 제거
                        target_seq.append(int(label))
                    prev_label = label
                
                # 예측값과 타겟값 저장
                all_predictions.append(error_preds[i])
                all_targets.append(target_seq)
    
    # 시퀀스별 정확도 계산
    correct = 0
    total = len(all_predictions)
    
    for pred, target in zip(all_predictions, all_targets):
        if len(pred) == len(target) and all(p == t for p, t in zip(pred, target)):
            correct += 1
    
    sequence_accuracy = correct / total if total > 0 else 0
    
    # 오류율 계산 (SpeechBrain 방식)
    error_types = ['D', 'S', 'A', 'C']  # blank 제외
    error_rate = calculate_error_rate(all_predictions, all_targets, error_types)
    
    # 클래스별 통계 계산
    class_stats = {}
    for class_id, class_name in enumerate(error_types, 1):
        class_pred_count = sum(1 for pred in all_predictions for p in pred if p == class_id)
        class_target_count = sum(1 for target in all_targets for t in target if t == class_id)
        
        # F1 스코어 계산을 위한 정보
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, target in zip(all_predictions, all_targets):
            # align sequences for comparison
            pred_set = set((i, p) for i, p in enumerate(pred))
            target_set = set((i, t) for i, t in enumerate(target))
            
            for i, p in enumerate(pred):
                if p == class_id:
                    if (i, p) in target_set:
                        true_positives += 1
                    else:
                        false_positives += 1
            
            for i, t in enumerate(target):
                if t == class_id and (i, t) not in pred_set:
                    false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_stats[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': class_target_count
        }
    
    return {
        'sequence_accuracy': sequence_accuracy,
        'error_rate': error_rate,
        'class_stats': class_stats,
        'error_types': error_types,
        'num_samples': total,
        'pred_samples': all_predictions[:5],  # 디버깅용
        'target_samples': all_targets[:5]    # 디버깅용
    }

def evaluate_stage2_full_model(model, dataloader, device, id_to_phoneme):
    """2단계: 오류 탐지 + 음소 인식 전체 평가"""
    model.eval()
    
    # 오류 탐지 결과
    all_error_predictions = []
    all_error_targets = []
    
    # 음소 인식 결과
    all_phoneme_predictions = []
    all_phoneme_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Stage 2: Full Model Evaluation"):
            waveforms, perceived_labels, error_labels, audio_lengths, perceived_lengths, error_lengths, wav_files, items = batch
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
            
            # 전체 모델 예측
            phoneme_logits, adjusted_probs, error_logits = model(waveforms, attention_mask, return_error_probs=True)
            
            # 오류 탐지 예측 (프레임별)
            frame_error_predictions = torch.argmax(error_logits, dim=-1)
            
            # 음소 인식 예측 (CTC 디코딩)
            phoneme_preds = ctc_decode(phoneme_logits, audio_lengths.to(device))
            
            # 결과 저장
            for i in range(len(waveforms)):
                # 오류 탐지
                seq_len = error_lengths[i].item()
                frame_pred = frame_error_predictions[i, :seq_len].cpu().numpy()
                target = error_labels[i, :seq_len].cpu().numpy()
                
                all_error_predictions.extend(frame_pred)
                all_error_targets.extend(target)
                
                # 음소 인식
                all_phoneme_predictions.append(phoneme_preds[i])
                per_len = perceived_lengths[i].item()
                all_phoneme_targets.append(perceived_labels[i, :per_len].cpu().numpy())
    
    # 오류 탐지 성능
    error_accuracy = accuracy_score(all_error_targets, all_error_predictions)
    error_types = ['blank', 'D', 'S', 'A', 'C']
    error_report = classification_report(all_error_targets, all_error_predictions, 
                                       target_names=error_types, output_dict=True)
    
    # 음소 인식 성능 (PER)
    per_perceived, per_perceived_samples = calculate_per(all_phoneme_predictions, all_phoneme_targets, id_to_phoneme)
    
    return {
        'error_detection': {
            'accuracy': error_accuracy,
            'report': error_report,
            'error_types': error_types
        },
        'phoneme_recognition': {
            'per_perceived': per_perceived,
            'per_perceived_samples': per_perceived_samples
        }
    }

def save_results(results, output_dir, stage):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    if stage == 1:
        # 1단계 결과 저장
        json_results = {
            'stage': 1,
            'error_detection': {
                'sequence_accuracy': results['sequence_accuracy'],
                'error_rate': results['error_rate'],  # SpeechBrain 방식
                'num_samples': results['num_samples'],
                'classification_report': results['class_stats']
            }
        }
        
        with open(os.path.join(output_dir, 'stage1_evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        with open(os.path.join(output_dir, 'stage1_detailed_results.txt'), 'w') as f:
            f.write("=== Stage 1: Error Detection Results ===\n")
            f.write(f"Total Samples: {results['num_samples']}\n")
            f.write(f"Sequence Accuracy: {results['sequence_accuracy']:.4f}\n")
            f.write(f"Error Rate: {results['error_rate']:.2f}%\n\n")
            f.write("Per-Class F1-scores:\n")
            for error_type, metrics in results['class_stats'].items():
                f.write(f"  {error_type}: {metrics['f1-score']:.4f} (support: {metrics['support']})\n")
            
            # 디버깅 정보 추가
            f.write(f"\nDebug Info (first 5 samples):\n")
            for i, (pred, target) in enumerate(zip(results['pred_samples'], results['target_samples'])):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Prediction: {pred}\n")
                f.write(f"  Target: {target}\n")
    
    else:  # stage == 2
        # 2단계 결과 저장
        json_results = {
            'stage': 2,
            'error_detection': {
                'sequence_accuracy': results['error_detection']['sequence_accuracy'],
                'error_rate': results['error_detection']['error_rate'],
                'classification_report': results['error_detection']['class_stats']
            },
            'phoneme_recognition': {
                'per_perceived': results['phoneme_recognition']['per_perceived'],
                'per_perceived_mean': np.mean(results['phoneme_recognition']['per_perceived_samples']),
                'per_perceived_std': np.std(results['phoneme_recognition']['per_perceived_samples']),
                'per_perceived_median': np.median(results['phoneme_recognition']['per_perceived_samples'])
            }
        }
        
        with open(os.path.join(output_dir, 'stage2_evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        with open(os.path.join(output_dir, 'stage2_detailed_results.txt'), 'w') as f:
            f.write("=== Stage 2: Full Model Results ===\n\n")
            f.write("Error Detection Results:\n")
            f.write(f"Sequence Accuracy: {results['error_detection']['sequence_accuracy']:.4f}\n")
            f.write(f"Error Rate: {results['error_detection']['error_rate']:.2f}%\n")
            f.write("Per-Class F1-scores:\n")
            for error_type, metrics in results['error_detection']['class_stats'].items():
                f.write(f"  {error_type}: {metrics['f1-score']:.4f} (support: {metrics['support']})\n")
            
            f.write(f"\nPhoneme Recognition Results:\n")
            f.write(f"PER (vs Perceived): {results['phoneme_recognition']['per_perceived']:.2f}%\n")
            f.write(f"PER Distribution:\n")
            f.write(f"  Mean: {np.mean(results['phoneme_recognition']['per_perceived_samples']):.2f}%\n")
            f.write(f"  Std:  {np.std(results['phoneme_recognition']['per_perceived_samples']):.2f}%\n")
            f.write(f"  Median: {np.median(results['phoneme_recognition']['per_perceived_samples']):.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description='L2 발음 오류 탐지 및 음소 인식 모델 평가')
    
    parser.add_argument('--eval_data', type=str, default='data/eval.json', help='평가 데이터 경로')
    parser.add_argument('--phoneme_map', type=str, default='data/phoneme_to_id.json', help='음소-ID 매핑 파일')
    
    # 1단계 또는 2단계 평가 선택
    parser.add_argument('--stage', type=int, choices=[1, 2], default=2, 
                       help='평가 단계 (1: 오류 탐지만, 2: 오류 탐지 + 음소 인식)')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='평가할 모델 체크포인트')
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='사용할 장치')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이')
    
    # 모델 관련 인자
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-960h', help='사전학습된 wav2vec2 모델')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.25, help='어댑터 차원 비율')
    parser.add_argument('--error_influence_weight', type=float, default=0.2, help='오류 영향 가중치')
    
    args = parser.parse_args()
    
    # 음소 매핑 로드
    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    
    # 모델 초기화
    model = DualWav2VecWithErrorAwarePhonemeRecognition(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        adapter_dim_ratio=args.adapter_dim_ratio,
        error_influence_weight=args.error_influence_weight,
        blank_index=0,
        sil_index=1
    )
    
    # 체크포인트 로드
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device))
    model = model.to(args.device)
    
    # 데이터셋 및 데이터로더 생성
    eval_dataset = EvalDataset(args.eval_data, phoneme_to_id, max_length=args.max_audio_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    if args.stage == 1:
        print("=== 1단계 평가: 오류 탐지만 ===")
        results = evaluate_stage1_error_detection(model, eval_dataloader, args.device)
        
        print(f"\n=== Stage 1 결과 ===")
        print(f"Total Samples: {results['num_samples']}")
        print(f"오류 탐지 시퀀스 정확도: {results['sequence_accuracy']:.4f}")
        print(f"오류 탐지 오류율: {results['error_rate']:.2f}%")
        print("Per-Class F1-scores:")
        for error_type, metrics in results['class_stats'].items():
            print(f"  {error_type}: {metrics['f1-score']:.4f} (support: {metrics['support']})")
        
        # 디버깅 정보 출력
        print(f"\nDebug Info (first 5 samples):")
        for i, (pred, target) in enumerate(zip(results['pred_samples'], results['target_samples'])):
            print(f"Sample {i+1}: pred={pred}, target={target}")
        
        save_results(results, args.output_dir, 1)
        
    else:  # args.stage == 2
        print("=== 2단계 평가: 오류 탐지 + 음소 인식 ===")
        results = evaluate_stage2_full_model(model, eval_dataloader, args.device, id_to_phoneme)
        
        print(f"\n=== Stage 2 결과 ===")
        print(f"오류 탐지 시퀀스 정확도: {results['error_detection']['sequence_accuracy']:.4f}")
        print(f"오류 탐지 오류율: {results['error_detection']['error_rate']:.2f}%")
        print("Per-Class F1-scores:")
        for error_type, metrics in results['error_detection']['class_stats'].items():
            print(f"  {error_type}: {metrics['f1-score']:.4f} (support: {metrics['support']})")
        print(f"\nL2 음소 인식 PER: {results['phoneme_recognition']['per_perceived']:.2f}%")
        
        save_results(results, args.output_dir, 2)
    
    print(f"\n결과가 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()