import os
import sys
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn import CTCLoss

from model import DualWav2VecWithErrorAwarePhonemeRecognition

torch.autograd.set_detect_anomaly(True)

# 레벤슈타인 거리 계산을 위한 함수
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

class ErrorLabelDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # C: 정확함(4), D: 삭제(1), A: 추가/삽입(3), S: 대체(2)
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'S': 2}
        
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
        error_labels = [self.error_type_mapping[label] for label in error_labels.split()]
        error_labels = torch.tensor(error_labels, dtype=torch.long)

        label_length = torch.tensor(len(error_labels), dtype=torch.long)
        
        return waveform.squeeze(0), error_labels, label_length, wav_file

class PhonemeRecognitionDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
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
        
        # 음소 레이블 변환
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = []
        for phoneme in phoneme_target.split():
            if phoneme in self.phoneme_to_id:
                phoneme_labels.append(self.phoneme_to_id[phoneme])
        
        phoneme_labels = torch.tensor(phoneme_labels, dtype=torch.long)
        label_length = torch.tensor(len(phoneme_labels), dtype=torch.long)
        
        return waveform.squeeze(0), phoneme_labels, label_length, wav_file

# 평가용 데이터셋 - evaluate.py에서 가져옴
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

def evaluation_collate_fn(batch):
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

# 오류 탐지를 위한 배치 콜레이션 함수
def error_ctc_collate_fn(batch):
    waveforms, error_labels, label_lengths, wav_files = zip(*batch)
    
    # 가변 길이 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    audio_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    
    # CTC 손실을 위한 레이블 준비
    max_label_len = max([labels.shape[0] for labels in error_labels])
    padded_error_labels = []
    
    for labels in error_labels:
        label_len = labels.shape[0]
        padding = max_label_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)
        padded_error_labels.append(padded_labels)
    
    padded_waveforms = torch.stack(padded_waveforms)
    padded_error_labels = torch.stack(padded_error_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return padded_waveforms, padded_error_labels, audio_lengths, label_lengths, wav_files

# 음소 인식을 위한 배치 콜레이션 함수
def phoneme_collate_fn(batch):
    waveforms, phoneme_labels, label_lengths, wav_files = zip(*batch)
    
    # 가변 길이 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    audio_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    
    # 음소 레이블 패딩
    max_phoneme_len = max([labels.shape[0] for labels in phoneme_labels])
    padded_phoneme_labels = []
    
    for labels in phoneme_labels:
        label_len = labels.shape[0]
        padding = max_phoneme_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)
        padded_phoneme_labels.append(padded_labels)
    
    padded_waveforms = torch.stack(padded_waveforms)
    padded_phoneme_labels = torch.stack(padded_phoneme_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return padded_waveforms, padded_phoneme_labels, audio_lengths, label_lengths, wav_files

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
            
            # 정확한 다운샘플링 비율 계산
            input_seq_len = waveforms.size(1)
            output_seq_len = error_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
            
            # CTC 디코딩
            log_probs = torch.log_softmax(error_logits, dim=-1)
            
            # decode_ctc 함수 호출 (input_lengths 전달)
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
            
            # 정확한 다운샘플링 비율 계산
            input_seq_len = waveforms.size(1)
            output_seq_len = phoneme_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산
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

def train_error_detection_ctc(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=0.5):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'에폭 {epoch} [오류 탐지]')
    
    for batch_idx, (waveforms, error_labels, audio_lengths, label_lengths, _) in enumerate(progress_bar):
        waveforms = waveforms.to(device)
        error_labels = error_labels.to(device)
        audio_lengths = audio_lengths.to(device)
        label_lengths = label_lengths.to(device)
        
        # wav2vec용 어텐션 마스크 생성
        attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
        attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
        
        # 순전파
        phoneme_logits, adjusted_probs, error_logits = model(waveforms, attention_mask, return_error_probs=True)
        
        # CTC 손실 계산
        log_probs = torch.log_softmax(error_logits, dim=-1)
        
        # 정확한 다운샘플링 비율 계산 (입력 길이와 특성 길이 비교)
        input_seq_len = waveforms.size(1)
        output_seq_len = error_logits.size(1)
        
        # 입력 길이를 기반으로 출력 길이 계산
        input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
        
        # 유효한 길이 보장
        input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
        
        loss = criterion(log_probs.transpose(0, 1), error_labels, input_lengths, label_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix({
            '손실': running_loss / (batch_idx + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate_error_detection_ctc(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='검증 [오류 탐지]')
        
        for batch_idx, (waveforms, error_labels, audio_lengths, label_lengths, _) in enumerate(progress_bar):
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
            
            # 순전파
            phoneme_logits, adjusted_probs, error_logits = model(waveforms, attention_mask, return_error_probs=True)
            
            # CTC 손실 계산
            log_probs = torch.log_softmax(error_logits, dim=-1)
            
            # 정확한 다운샘플링 비율 계산 (입력 길이와 특성 길이 비교)
            input_seq_len = waveforms.size(1)
            output_seq_len = error_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            
            # 유효한 길이 보장
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
            
            loss = criterion(log_probs.transpose(0, 1), error_labels, input_lengths, label_lengths)
            
            running_loss += loss.item()
            
            progress_bar.set_postfix({
                '검증_손실': running_loss / (batch_idx + 1)
            })
    
    val_loss = running_loss / len(dataloader)
    return val_loss

def train_phoneme_recognition(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'에폭 {epoch} [음소 인식]')
    
    for batch_idx, (waveforms, phoneme_labels, audio_lengths, label_lengths, _) in enumerate(progress_bar):
        waveforms = waveforms.to(device)
        phoneme_labels = phoneme_labels.to(device)
        audio_lengths = audio_lengths.to(device)
        label_lengths = label_lengths.to(device)
        
        # 어텐션 마스크 생성
        attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
        attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
        
        # 순전파
        phoneme_logits, adjusted_probs = model(waveforms, attention_mask)
        
        # CTC 손실 계산
        log_probs = torch.log_softmax(phoneme_logits, dim=-1)
        
        # 정확한 다운샘플링 비율 계산 (입력 길이와 특성 길이 비교)
        input_seq_len = waveforms.size(1)
        output_seq_len = phoneme_logits.size(1)
        
        # 입력 길이를 기반으로 출력 길이 계산
        input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
        
        # 유효한 길이 보장
        input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
        
        loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix({
            '손실': running_loss / (batch_idx + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate_phoneme_recognition(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='검증 [음소 인식]')
        
        for batch_idx, (waveforms, phoneme_labels, audio_lengths, label_lengths, _) in enumerate(progress_bar):
            waveforms = waveforms.to(device)
            phoneme_labels = phoneme_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            # 순전파
            phoneme_logits, adjusted_probs = model(waveforms, attention_mask)
            
            # CTC 손실 계산
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # 정확한 다운샘플링 비율 계산 (입력 길이와 특성 길이 비교)
            input_seq_len = waveforms.size(1)
            output_seq_len = phoneme_logits.size(1)
            
            # 입력 길이를 기반으로 출력 길이 계산
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            
            # 유효한 길이 보장
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)
            
            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
            
            running_loss += loss.item()
            
            progress_bar.set_postfix({
                '검증_손실': running_loss / (batch_idx + 1)
            })
    
    val_loss = running_loss / len(dataloader)
    return val_loss

def seed_everything(seed):
    """재현성을 위한 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='L2 발음 오류 탐지 및 음소 인식을 위한 이중 wav2vec2 모델 학습')
    
    # 기본 설정
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1, help='학습 단계 (1: 오류 탐지, 2: 음소 인식)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='사용할 장치')
    
    # 데이터 설정
    parser.add_argument('--error_train_data', type=str, default='data/errors_train.json', help='오류 탐지 학습 데이터')
    parser.add_argument('--error_val_data', type=str, default='data/errors_val.json', help='오류 탐지 검증 데이터')
    parser.add_argument('--phoneme_train_data', type=str, default='data/perceived_train.json', help='음소 인식 학습 데이터')
    parser.add_argument('--phoneme_val_data', type=str, default='data/perceived_val.json', help='음소 인식 검증 데이터')
    parser.add_argument('--phoneme_map', type=str, default='data/phoneme_to_id.json', help='음소-ID 매핑')
    
    # 평가 데이터셋
    parser.add_argument('--eval_data', type=str, default='data/eval.json', help='평가 데이터셋 경로')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='평가 배치 크기')
    
    # 모델 설정
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-960h', help='사전학습된 wav2vec2 모델')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.25, help='어댑터 차원 비율')
    parser.add_argument('--error_influence_weight', type=float, default=0.2, help='오류 영향 가중치')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=10, help='에폭 수')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='그라디언트 클리핑을 위한 최대 노름값')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='models', help='모델 체크포인트 출력 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results', help='결과 출력 디렉토리')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='로드할 모델 체크포인트 경로')
    
    # 평가 관련 설정 (추가됨)
    parser.add_argument('--evaluate_every_epoch', action='store_true', help='각 에폭마다 평가 진행')
    
    args = parser.parse_args()
    
    # 재현성을 위한 시드 설정
    seed_everything(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, f'train_stage{args.stage}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(args.result_dir, f'hyperparams_stage{args.stage}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 음소 매핑 로드
    if os.path.exists(args.phoneme_map):
        with open(args.phoneme_map, 'r') as f:
            phoneme_to_id = json.load(f)
    else:
        logger.error(f"음소-ID 매핑 파일({args.phoneme_map})이 필요합니다. 이 파일을 생성한 후 다시 시도하세요.")
        sys.exit(1)
    
    # ID를 음소로 변환하는 역매핑 생성 (평가용)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    # 오류 유형 이름 매핑 (평가용)
    error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct'}
    
    # 모델 초기화
    model = DualWav2VecWithErrorAwarePhonemeRecognition(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        adapter_dim_ratio=args.adapter_dim_ratio,
        error_influence_weight=args.error_influence_weight,
        blank_index=0,  # CTC 빈칸 인덱스
        sil_index=1     # sil 인덱스
    )
        
    if args.model_checkpoint:
        logger.info(f"{args.model_checkpoint}에서 체크포인트 로드 중")
        
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
    
    if torch.cuda.device_count() > 1:
        logger.info(f"{torch.cuda.device_count()}개의 GPU가 감지되었습니다. DataParallel 사용")
        model = nn.DataParallel(model)

    model = model.to(args.device)
    
    eval_dataloader = None
    if args.evaluate_every_epoch and args.eval_data is not None:
        logger.info(f"평가 데이터셋 로드 중: {args.eval_data}")
        eval_dataset = EvaluationDataset(
            args.eval_data, phoneme_to_id, max_length=args.max_audio_length
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn
        )
    
    # 학습 단계 설정
    if args.stage == 1:
        # 1단계: 오류 탐지 학습
        logger.info("1단계: 오류 탐지 학습")
        
        train_dataset = ErrorLabelDataset(args.error_train_data, max_length=args.max_audio_length)
        val_dataset = ErrorLabelDataset(args.error_val_data, max_length=args.max_audio_length)
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=error_ctc_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=error_ctc_collate_fn
        )
        
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        best_val_loss = float('inf') 
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            train_loss = train_error_detection_ctc(
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=0.5
            )
            
            val_loss = validate_error_detection_ctc(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
            
            # 오류 탐지 평가 (추가됨)
            if args.evaluate_every_epoch and eval_dataloader is not None:
                logger.info(f"에폭 {epoch}: 오류 탐지 평가 중...")
                error_detection_results = evaluate_error_detection(
                    model, eval_dataloader, args.device, error_type_names
                )
                
                logger.info(f"오류 탐지 정확도: {error_detection_results['accuracy']:.4f}")
                
                # 오류 유형별 메트릭 로깅
                for error_type, metrics in error_detection_results['error_type_metrics'].items():
                    logger.info(f"  {error_type}:")
                    logger.info(f"    정밀도: {metrics['precision']:.4f}")
                    logger.info(f"    재현율: {metrics['recall']:.4f}")
                    logger.info(f"    F1 점수: {metrics['f1']:.4f}")
            
            # 결과 저장
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            # 평가 결과 추가
            if args.evaluate_every_epoch and eval_dataloader is not None:
                epoch_results['error_detection'] = error_detection_results
            
            with open(os.path.join(args.result_dir, f'error_detection_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_results, f, indent=4)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_error_detection.pth'))
                logger.info(f"검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_error_detection.pth'))
    
    elif args.stage == 2:
        # 2단계: 음소 인식 학습
        logger.info("2단계: 음소 인식 학습")
        
        # 오류 탐지 헤드 고정
        if isinstance(model, nn.DataParallel):
            for param in model.module.error_detection_head.parameters():
                param.requires_grad = False
        else:
            for param in model.error_detection_head.parameters():
                param.requires_grad = False
        
        train_dataset = PhonemeRecognitionDataset(
            args.phoneme_train_data, phoneme_to_id, max_length=args.max_audio_length
        )
        val_dataset = PhonemeRecognitionDataset(
            args.phoneme_val_data, phoneme_to_id, max_length=args.max_audio_length
        )
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=phoneme_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=phoneme_collate_fn
        )
        
        criterion = CTCLoss(blank=0, reduction='mean')
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=args.learning_rate
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            train_loss = train_phoneme_recognition(
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=1.0
            )
            
            val_loss = validate_phoneme_recognition(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
            
            # 음소 인식 평가
            if args.evaluate_every_epoch and eval_dataloader is not None:
                logger.info(f"에폭 {epoch}: 음소 인식 평가 중...")
                phoneme_recognition_results = evaluate_phoneme_recognition(
                    model, eval_dataloader, args.device, id_to_phoneme
                )
                
                logger.info(f"음소 오류율 (PER): {phoneme_recognition_results['per']:.4f}")
                logger.info(f"총 음소 수: {phoneme_recognition_results['total_phonemes']}")
                logger.info(f"총 오류 수: {phoneme_recognition_results['total_errors']}")
                logger.info(f"삽입: {phoneme_recognition_results['insertions']}")
                logger.info(f"삭제: {phoneme_recognition_results['deletions']}")
                logger.info(f"대체: {phoneme_recognition_results['substitutions']}")
            
            # 결과 저장
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            # 평가 결과
            if args.evaluate_every_epoch and eval_dataloader is not None:
                epoch_results['phoneme_recognition'] = {
                    'per': phoneme_recognition_results['per'],
                    'total_phonemes': phoneme_recognition_results['total_phonemes'],
                    'total_errors': phoneme_recognition_results['total_errors'],
                    'insertions': phoneme_recognition_results['insertions'],
                    'deletions': phoneme_recognition_results['deletions'],
                    'substitutions': phoneme_recognition_results['substitutions']
                }
            
            with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_results, f, indent=4)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_phoneme_recognition.pth'))
                logger.info(f"검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_phoneme_recognition.pth'))
    
    logger.info("학습 완료!")

if __name__ == "__main__":
    main()