#!/usr/bin/env python3
"""
L2 발음 오류 탐지 및 음소 인식 모델 평가 스크립트
2단계 학습에 맞춘 평가: 1단계(오류 탐지), 2단계(오류 탐지 + 음소 인식)
"""

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

def evaluate_stage1_error_detection(model, dataloader, device):
    """1단계: 오류 탐지만 평가"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
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
            
            # 예측값 (CTC 디코딩)
            error_preds = ctc_decode(error_logits, error_lengths)
            
            # 배치별로 처리
            for i in range(len(error_labels)):
                pred_len = len(error_preds[i])
                target_len = error_lengths[i].item()
                
                all_predictions.extend(error_preds[i])
                all_targets.extend(error_labels[i, :target_len].cpu().numpy())
    
    # 정확도 계산
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # 상세 보고서
    error_types = ['blank', 'D', 'S', 'A', 'C']
    report = classification_report(all_targets, all_predictions, 
                                 target_names=error_types, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'error_types': error_types
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
            
            # 오류 탐지 예측
            error_preds = ctc_decode(error_logits, error_lengths)
            
            # 음소 인식 예측
            phoneme_preds = ctc_decode(phoneme_logits, audio_lengths.to(device))
            
            # 결과 저장
            for i in range(len(waveforms)):
                # 오류 탐지
                target_len = error_lengths[i].item()
                all_error_predictions.extend(error_preds[i])
                all_error_targets.extend(error_labels[i, :target_len].cpu().numpy())
                
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
                'accuracy': results['accuracy'],
                'classification_report': results['report']
            }
        }
        
        with open(os.path.join(output_dir, 'stage1_evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        with open(os.path.join(output_dir, 'stage1_detailed_results.txt'), 'w') as f:
            f.write("=== Stage 1: Error Detection Results ===\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")
            f.write("Per-Class F1-scores:\n")
            for error_type, metrics in results['report'].items():
                if error_type not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"  {error_type}: {metrics['f1-score']:.4f}\n")
    
    else:  # stage == 2
        # 2단계 결과 저장
        json_results = {
            'stage': 2,
            'error_detection': {
                'accuracy': results['error_detection']['accuracy'],
                'classification_report': results['error_detection']['report']
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
            f.write(f"Overall Accuracy: {results['error_detection']['accuracy']:.4f}\n")
            f.write("Per-Class F1-scores:\n")
            for error_type, metrics in results['error_detection']['report'].items():
                if error_type not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"  {error_type}: {metrics['f1-score']:.4f}\n")
            
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
        print(f"오류 탐지 정확도: {results['accuracy']:.4f}")
        print("Per-Class F1-scores:")
        for error_type, metrics in results['report'].items():
            if error_type not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"  {error_type}: {metrics['f1-score']:.4f}")
        
        save_results(results, args.output_dir, 1)
        
    else:  # args.stage == 2
        print("=== 2단계 평가: 오류 탐지 + 음소 인식 ===")
        results = evaluate_stage2_full_model(model, eval_dataloader, args.device, id_to_phoneme)
        
        print(f"\n=== Stage 2 결과 ===")
        print(f"오류 탐지 정확도: {results['error_detection']['accuracy']:.4f}")
        print("Per-Class F1-scores:")
        for error_type, metrics in results['error_detection']['report'].items():
            if error_type not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"  {error_type}: {metrics['f1-score']:.4f}")
        print(f"\nL2 음소 인식 PER: {results['phoneme_recognition']['per_perceived']:.2f}%")
        
        save_results(results, args.output_dir, 2)
    
    print(f"\n결과가 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()