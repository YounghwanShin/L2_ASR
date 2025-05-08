import os
import sys
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn import CTCLoss
import torch.nn.functional as F

from model import DualWav2VecWithErrorAwarePhonemeRecognition, LearnableWav2Vec
import editdistance

torch.autograd.set_detect_anomaly(True)

class ErrorRateInfluencedLoss(nn.Module):
    def __init__(self, ctc_weight=1.0, error_rate_weight=0.5, blank=0):
        super(ErrorRateInfluencedLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
        self.ctc_weight = ctc_weight
        self.error_rate_weight = error_rate_weight
        self.blank = blank
    
    def ctc_decode(self, logits, lengths):
        """간단한 CTC 디코딩"""
        batch_size, max_len, vocab_size = logits.shape
        predictions = []
        
        for i in range(batch_size):
            log_probs = torch.log_softmax(logits[i], dim=-1).cpu().numpy()
            seq_len = lengths[i].item()
            
            best_path = np.argmax(log_probs[:seq_len], axis=-1)
            
            decoded = []
            prev_label = None
            for label in best_path:
                if label != self.blank and label != prev_label:
                    decoded.append(label)
                prev_label = label
            
            predictions.append(decoded)
        
        return predictions
    
    def calculate_batch_error_rate(self, predictions, targets, target_lengths):
        """배치의 오류율 계산"""
        total_errors = 0
        total_tokens = 0
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            target_len = target_lengths[i].item()
            target_seq = target[:target_len].cpu().numpy()
            
            # Target sequence에서 blank 제거
            target_cleaned = []
            prev_label = None
            for label in target_seq:
                if label != self.blank and label != prev_label:
                    target_cleaned.append(label)
                prev_label = label
            
            errors = editdistance.eval(pred, target_cleaned)
            total_errors += errors
            total_tokens += len(target_cleaned)
        
        if total_tokens == 0:
            return 0.0
        
        return total_errors / total_tokens
    
    def forward(self, logits, targets, input_lengths, target_lengths):
        # 1. CTC Loss 계산
        log_probs = F.log_softmax(logits, dim=-1)
        ctc_loss = self.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        
        # 2. 예측 수행 (그래디언트 전파되지 않도록)
        with torch.no_grad():
            predictions = self.ctc_decode(logits, input_lengths)
        
        # 3. Error Rate 계산
        error_rate = self.calculate_batch_error_rate(predictions, targets, target_lengths)
        
        # 4. Error Rate를 보조 손실로 변환
        # Error Rate를 미분 가능한 형태로 근사
        # Soft error rate: 예측 분포와 타겟 분포 간의 거리 측정
        soft_error_loss = 0.0
        for i in range(len(predictions)):
            pred_probs = F.softmax(logits[i, :input_lengths[i]], dim=-1)
            target_one_hot = F.one_hot(targets[i, :target_lengths[i]], num_classes=logits.size(-1)).float()
            
            # KL divergence를 사용하여 분포 간 차이 측정
            if target_one_hot.size(0) > 0:
                # 시퀀스 길이 맞추기
                min_len = min(pred_probs.size(0), target_one_hot.size(0))
                pred_probs_aligned = pred_probs[:min_len]
                target_one_hot_aligned = target_one_hot[:min_len]
                
                kl_loss = F.kl_div(pred_probs_aligned.log(), target_one_hot_aligned, reduction='batchmean')
                soft_error_loss += kl_loss
        
        soft_error_loss /= len(predictions)
        
        # 5. 총 손실
        total_loss = self.ctc_weight * ctc_loss + self.error_rate_weight * soft_error_loss
        
        return total_loss, ctc_loss, error_rate

class WeightedCTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True, class_weights=None):
        super(WeightedCTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        self.reduction = reduction
        self.class_weights = class_weights  # [weight_for_blank, weight_for_D, weight_for_S, weight_for_A, weight_for_C]
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        losses = self.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        
        # 클래스 가중치 적용
        if self.class_weights is not None:
            weighted_losses = []
            for i, target in enumerate(targets):
                target_labels = target[:target_lengths[i]]
                # 타겟의 클래스별 빈도 계산
                weights = []
                for label in target_labels:
                    if label.item() < len(self.class_weights):
                        weights.append(self.class_weights[label.item()])
                    else:
                        weights.append(1.0)
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                else:
                    avg_weight = 1.0
                
                weighted_losses.append(losses[i] * avg_weight)
            
            losses = torch.stack(weighted_losses)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ErrorLabelDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # 오류 유형 매핑: C (정확함), D (삭제), A (추가/삽입), S (대체)
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'S': 2}
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        # 필요시 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 필요시 리샘플링
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        # 필요시 길이 제한
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
        self.id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        # 필요시 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 필요시 리샘플링
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        # 필요시 길이 제한
        if self.max_length is not None and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        # 음소 레이블 변환
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = []
        for phoneme in phoneme_target.split():
            if phoneme in self.phoneme_to_id:
                phoneme_labels.append(self.phoneme_to_id[phoneme])
        
        phoneme_labels = torch.tensor(phoneme_labels, dtype=torch.long)
        
        # 음소 레이블 길이
        label_length = torch.tensor(len(phoneme_labels), dtype=torch.long)
        
        return waveform.squeeze(0), phoneme_labels, label_length, wav_file

# 오류 탐지를 위한 배치 콜레이션 함수
def error_ctc_collate_fn(batch):
    waveforms, error_labels, label_lengths, wav_files = zip(*batch)
    
    # 가변 길이 오디오를 위한 패딩
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
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)  # CTC blank 인덱스 (0)
        padded_error_labels.append(padded_labels)
    
    padded_waveforms = torch.stack(padded_waveforms)
    padded_error_labels = torch.stack(padded_error_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return padded_waveforms, padded_error_labels, audio_lengths, label_lengths, wav_files

# 음소 인식을 위한 배치 콜레이션 함수
def phoneme_collate_fn(batch):
    waveforms, phoneme_labels, label_lengths, wav_files = zip(*batch)
    
    # 가변 길이 오디오를 위한 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    # 오디오 마스크 생성
    audio_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    
    # 음소 레이블 패딩
    max_phoneme_len = max([labels.shape[0] for labels in phoneme_labels])
    padded_phoneme_labels = []
    
    for labels in phoneme_labels:
        label_len = labels.shape[0]
        padding = max_phoneme_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)  # 0은 blank 인덱스
        padded_phoneme_labels.append(padded_labels)
    
    padded_waveforms = torch.stack(padded_waveforms)
    padded_phoneme_labels = torch.stack(padded_phoneme_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return padded_waveforms, padded_phoneme_labels, audio_lengths, label_lengths, wav_files

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

def calculate_error_rate(predictions, targets):
    """오류율 계산"""
    total_errors = 0
    total_tokens = 0
    
    for pred, target in zip(predictions, targets):
        # Edit distance 계산
        errors = editdistance.eval(pred, target)
        total_errors += errors
        total_tokens += len(target)
    
    error_rate = total_errors / max(total_tokens, 1) * 100
    return error_rate

def calculate_per(predictions, targets, id_to_phoneme, remove_sil=True):
    """PER (Phoneme Error Rate) 계산"""
    total_phonemes = 0
    total_errors = 0
    
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
    
    # 전체 PER
    per = total_errors / max(total_phonemes, 1) * 100
    
    return per

def train_error_detection_with_error_rate(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=0.5):
    model.train()
    running_total_loss = 0.0
    running_ctc_loss = 0.0
    running_error_rate = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'에폭 {epoch} [오류 탐지 + ER]')
    
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
        
        # 모델 출력 시퀀스 길이 계산
        input_lengths = torch.full(size=(error_logits.size(0),), fill_value=error_logits.size(1), 
                                 dtype=torch.long).to(device)
        
        # Error Rate 영향 손실 계산
        total_loss, ctc_loss, error_rate = criterion(error_logits, error_labels, input_lengths, label_lengths)
        
        optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        running_total_loss += total_loss.item()
        running_ctc_loss += ctc_loss.item()
        running_error_rate += error_rate
        
        progress_bar.set_postfix({
            '총손실': running_total_loss / (batch_idx + 1),
            'CTC손실': running_ctc_loss / (batch_idx + 1),
            '오류율': running_error_rate / (batch_idx + 1)
        })
    
    epoch_loss = running_total_loss / len(dataloader)
    epoch_ctc_loss = running_ctc_loss / len(dataloader)
    epoch_error_rate = running_error_rate / len(dataloader)
    
    return epoch_loss, epoch_ctc_loss, epoch_error_rate

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
        
        # CTC 손실 계산을 위해 log_softmax 적용
        log_probs = torch.log_softmax(error_logits, dim=-1)
        
        # 모델 출력 시퀀스 길이 계산
        input_lengths = torch.full(size=(log_probs.size(0),), fill_value=log_probs.size(1), 
                                 dtype=torch.long).to(device)
        
        # CTC 손실 계산
        loss = criterion(log_probs, error_labels, input_lengths, label_lengths)
        
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
    running_total_loss = 0.0
    running_ctc_loss = 0.0
    
    # 오류율 계산을 위한 변수
    all_predictions = []
    all_targets = []
    
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
            
            # 모델 출력 시퀀스 길이 계산
            input_lengths = torch.full(size=(error_logits.size(0),), fill_value=error_logits.size(1), 
                                      dtype=torch.long).to(device)
            
            # ErrorRateInfluencedLoss 처리
            total_loss, ctc_loss, error_rate = criterion(error_logits, error_labels, input_lengths, label_lengths)
            running_total_loss += total_loss.item()
            running_ctc_loss += ctc_loss.item()
            
            # 오류율 계산을 위한 디코딩
            wav2vec_output_lengths = (audio_lengths / 20).long()
            error_preds = ctc_decode(error_logits, wav2vec_output_lengths)
            
            # 타겟 시퀀스 준비
            for i in range(len(error_labels)):
                target_len = label_lengths[i].item()
                target_labels = error_labels[i, :target_len].cpu().numpy()
                
                # blank 제거하고 중복 제거
                target_seq = []
                prev_label = None
                for label in target_labels:
                    if label != 0 and label != prev_label:  # blank=0 제거
                        target_seq.append(int(label))
                    prev_label = label
                
                all_predictions.append(error_preds[i])
                all_targets.append(target_seq)
            
            # 진행 상황 표시줄 업데이트
            progress_bar.set_postfix({
                '총손실': running_total_loss / (batch_idx + 1),
                'CTC손실': running_ctc_loss / (batch_idx + 1)
            })
    
    val_total_loss = running_total_loss / len(dataloader)
    val_ctc_loss = running_ctc_loss / len(dataloader)
    error_rate = calculate_error_rate(all_predictions, all_targets)
    
    return val_total_loss, error_rate

# 음소 인식 학습 함수
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
        
        # CTC 손실 준비
        log_probs = torch.log_softmax(phoneme_logits, dim=-1)
        
        # 입력 시퀀스 길이 계산
        input_lengths = torch.full(size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long).to(device)
        
        # CTC 손실 계산
        loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # 그라디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        
        # 진행 상황 표시줄 업데이트
        progress_bar.set_postfix({
            '손실': running_loss / (batch_idx + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss

# 음소 인식 검증 함수
def validate_phoneme_recognition(model, dataloader, criterion, device, id_to_phoneme):
    model.eval()
    running_loss = 0.0
    
    # PER 계산을 위한 변수
    all_predictions = []
    all_targets = []
    
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
            
            # CTC 손실 준비
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # 입력 시퀀스 길이 계산
            input_lengths = torch.full(size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long).to(device)
            
            # CTC 손실 계산
            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
            
            # 통계 업데이트
            running_loss += loss.item()
            
            # PER 계산을 위한 디코딩
            phoneme_preds = ctc_decode(phoneme_logits, audio_lengths.to(device))
            
            # 타겟 시퀀스 준비
            for i in range(len(waveforms)):
                # 음소 인식
                all_predictions.append(phoneme_preds[i])
                per_len = label_lengths[i].item()
                all_targets.append(phoneme_labels[i, :per_len].cpu().numpy())
            
            # 진행 상황 표시줄 업데이트
            progress_bar.set_postfix({
                '검증_손실': running_loss / (batch_idx + 1)
            })
    
    val_loss = running_loss / len(dataloader)
    per = calculate_per(all_predictions, all_targets, id_to_phoneme)
    
    return val_loss, per

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
    
    # 체크포인트 저장 기준 설정
    parser.add_argument('--save_strategy', type=str, default='best', choices=['best', 'last', 'both'], 
                       help='체크포인트 저장 전략 (best: 최적 모델만, last: 마지막 모델만, both: 둘 다)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='models', help='모델 체크포인트 출력 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results', help='결과 출력 디렉토리')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='로드할 모델 체크포인트 경로')
    
    # Error Rate Loss 관련 파라미터 추가
    parser.add_argument('--ctc_weight', type=float, default=1.0, help='CTC loss 가중치')
    parser.add_argument('--error_rate_weight', type=float, default=0.3, help='Error rate loss 가중치')
    
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
    
    # 음소 매핑 로드 (필수)
    if os.path.exists(args.phoneme_map):
        with open(args.phoneme_map, 'r') as f:
            phoneme_to_id = json.load(f)
        id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    else:
        logger.error(f"음소-ID 매핑 파일({args.phoneme_map})이 필요합니다. 이 파일을 생성한 후 다시 시도하세요.")
        sys.exit(1)
    
    # 모델 초기화 - 새로운 모델 클래스 사용
    model = DualWav2VecWithErrorAwarePhonemeRecognition(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        adapter_dim_ratio=args.adapter_dim_ratio,
        error_influence_weight=args.error_influence_weight,
        blank_index=0,  # CTC 빈칸 인덱스
        sil_index=1     # sil 인덱스
    )
    
    # 체크포인트 로드(가능한 경우)
    if args.model_checkpoint:
        logger.info(f"{args.model_checkpoint}에서 체크포인트 로드 중")
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device))
    
    model = model.to(args.device)
    
    # 학습 단계 설정
    if args.stage == 1:
        # 1단계: 오류 탐지 학습
        logger.info("1단계: 오류 탐지 학습")
        
        # 데이터셋 로드
        train_dataset = ErrorLabelDataset(args.error_train_data, max_length=args.max_audio_length)
        val_dataset = ErrorLabelDataset(args.error_val_data, max_length=args.max_audio_length)
        
        # 기존 collate_fn 대신 CTC용 collate_fn 사용
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=error_ctc_collate_fn  # CTC용 콜레이트 함수
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=error_ctc_collate_fn  # CTC용 콜레이트 함수
        )
        
        # Error Rate Influenced Loss 사용
        criterion = ErrorRateInfluencedLoss(
            ctc_weight=args.ctc_weight, 
            error_rate_weight=args.error_rate_weight,
            blank=0
        )
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 학습 루프
        best_val_loss = float('inf') 
        best_error_rate = float('inf')
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            # Error Rate 영향 학습 함수 사용
            train_loss, train_ctc_loss, train_error_rate = train_error_detection_with_error_rate(
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=0.5
            )
            
            # CTC 검증 함수 사용 (손실과 오류율 모두 반환)
            val_loss, error_rate = validate_error_detection_ctc(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 총손실: {train_loss:.4f}, CTC손실: {train_ctc_loss:.4f}, "
                       f"학습오류율: {train_error_rate:.2f}%, 검증총손실: {val_loss:.4f}, 검증오류율: {error_rate:.2f}%")
            
            # 결과 저장 - 손실과 오류율 모두 저장
            with open(os.path.join(args.result_dir, f'error_detection_epoch{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_total_loss': train_loss,
                    'train_ctc_loss': train_ctc_loss,
                    'train_error_rate': train_error_rate,
                    'val_loss': val_loss,
                    'val_error_rate': error_rate
                }, f, indent=4)
            
            # 최고 모델 저장 - 오류율과 손실 모두 고려
            if args.save_strategy in ['best', 'both']:
                # 오류율이 더 중요하지만, 손실도 고려
                is_best = (error_rate < best_error_rate) or (error_rate == best_error_rate and val_loss < best_val_loss)
                
                if is_best:
                    best_val_loss = val_loss
                    best_error_rate = error_rate
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_error_detection.pth'))
                    logger.info(f"오류율 {error_rate:.2f}%, 검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            if args.save_strategy in ['last', 'both']:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_error_detection.pth'))
    
    elif args.stage == 2:
        # 2단계: 음소 인식 학습
        logger.info("2단계: 음소 인식 학습")
        
        # 오류 탐지 헤드 고정
        for param in model.error_detection_head.parameters():
            param.requires_grad = False
        
        # 데이터셋 로드
        train_dataset = PhonemeRecognitionDataset(
            args.phoneme_train_data, phoneme_to_id, max_length=args.max_audio_length
        )
        val_dataset = PhonemeRecognitionDataset(
            args.phoneme_val_data, phoneme_to_id, max_length=args.max_audio_length
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=phoneme_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=phoneme_collate_fn
        )
        
        # 손실 함수와 옵티마이저 설정
        criterion = CTCLoss(blank=0, reduction='mean')
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=args.learning_rate
        )
        
        # 학습 루프
        best_val_loss = float('inf')
        best_per = float('inf')
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            # 학습
            train_loss = train_phoneme_recognition(
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=1.0
            )
            
            # 검증 (손실과 PER 모두 반환)
            val_loss, per = validate_phoneme_recognition(
                model, val_dataloader, criterion, args.device, id_to_phoneme
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}, PER: {per:.2f}%")
            
            # 결과 저장
            with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'per': per
                }, f, indent=4)
            
            # 최고 모델 저장 - PER이 더 중요하지만, 손실도 고려
            if args.save_strategy in ['best', 'both']:
                is_best = (per < best_per) or (per == best_per and val_loss < best_val_loss)
                
                if is_best:
                    best_val_loss = val_loss
                    best_per = per
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_phoneme_recognition.pth'))
                    logger.info(f"PER {per:.2f}%, 검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            if args.save_strategy in ['last', 'both']:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_phoneme_recognition.pth'))
    
    logger.info("학습 완료!")

if __name__ == "__main__":
    main()