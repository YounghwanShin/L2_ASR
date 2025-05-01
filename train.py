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

from model import DualWav2VecWithErrorAwarePhonemeRecognition

# 오류 탐지를 위한 Focal Loss 구현
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

# 1단계 학습을 위한 오류 레이블 데이터셋
class ErrorLabelDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # 오류 유형 매핑: C (정확함), D (삭제), A (추가/삽입), S (대체)
        self.error_type_mapping = {'C': 3, 'D': 0, 'A': 2, 'S': 1}
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        # 오디오 로드
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
        
        return waveform.squeeze(0), error_labels, wav_file

# 2단계 학습을 위한 음소 인식 데이터셋
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
        
        # 오디오 로드
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
def error_collate_fn(batch):
    waveforms, error_labels, wav_files = zip(*batch)
    
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
    
    # 오류 레이블 패딩
    max_error_len = max([labels.shape[0] for labels in error_labels])
    padded_error_labels = []
    
    for labels in error_labels:
        label_len = labels.shape[0]
        padding = max_error_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=-100)  # -100은 손실 계산에서 무시됨
        padded_error_labels.append(padded_labels)
    
    # 텐서로 변환
    padded_waveforms = torch.stack(padded_waveforms)
    padded_error_labels = torch.stack(padded_error_labels)
    
    return padded_waveforms, padded_error_labels, audio_lengths, wav_files

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
    
    # 텐서로 변환
    padded_waveforms = torch.stack(padded_waveforms)
    padded_phoneme_labels = torch.stack(padded_phoneme_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return padded_waveforms, padded_phoneme_labels, audio_lengths, label_lengths, wav_files

# 오류 탐지 학습 함수
def train_error_detection(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f'에폭 {epoch} [오류 탐지]')
    
    for batch_idx, (waveforms, error_labels, audio_lengths, _) in enumerate(progress_bar):
        waveforms = waveforms.to(device)
        error_labels = error_labels.to(device)
        
        # wav2vec용 어텐션 마스크 생성
        attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
        attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
        
        # 순전파
        phoneme_logits, adjusted_probs, error_probs = model(waveforms, attention_mask, return_error_probs=True)
        
        # 손실 계산
        error_probs_reshaped = error_probs.view(-1, 4)  # (batch_size * seq_len, 4)
        error_labels_reshaped = error_labels.view(-1)   # (batch_size * seq_len)
        
        # -100 값 무시 (패딩)
        mask = error_labels_reshaped != -100
        error_probs_masked = error_probs_reshaped[mask]
        error_labels_masked = error_labels_reshaped[mask]
        
        loss = criterion(error_probs_masked, error_labels_masked)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        
        # 정확도 계산
        _, predicted = torch.max(error_probs_masked, 1)
        total_correct += (predicted == error_labels_masked).sum().item()
        total_samples += error_labels_masked.size(0)
        
        # 진행 상황 표시줄 업데이트
        progress_bar.set_postfix({
            '손실': running_loss / (batch_idx + 1),
            '정확도': 100. * total_correct / total_samples if total_samples > 0 else 0
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    
    return epoch_loss, epoch_acc

# 오류 탐지 검증 함수
def validate_error_detection(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='검증 [오류 탐지]')
        
        for batch_idx, (waveforms, error_labels, audio_lengths, _) in enumerate(progress_bar):
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            
            # 어텐션 마스크 생성
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
            
            # 순전파
            phoneme_logits, adjusted_probs, error_probs = model(waveforms, attention_mask, return_error_probs=True)
            
            # 손실 계산
            error_probs_reshaped = error_probs.view(-1, 4)
            error_labels_reshaped = error_labels.view(-1)
            
            # -100 값 무시
            mask = error_labels_reshaped != -100
            error_probs_masked = error_probs_reshaped[mask]
            error_labels_masked = error_labels_reshaped[mask]
            
            loss = criterion(error_probs_masked, error_labels_masked)
            
            # 통계 업데이트
            running_loss += loss.item()
            
            # 정확도 계산
            _, predicted = torch.max(error_probs_masked, 1)
            total_correct += (predicted == error_labels_masked).sum().item()
            total_samples += error_labels_masked.size(0)
            
            # 진행 상황 표시줄 업데이트
            progress_bar.set_postfix({
                '검증_손실': running_loss / (batch_idx + 1),
                '검증_정확도': 100. * total_correct / total_samples if total_samples > 0 else 0
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    
    return val_loss, val_acc

# 음소 인식 학습 함수
def train_phoneme_recognition(model, dataloader, criterion, optimizer, device, epoch):
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
            
            # CTC 손실 준비
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # 입력 시퀀스 길이 계산
            input_lengths = torch.full(size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long).to(device)
            
            # CTC 손실 계산
            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
            
            # 통계 업데이트
            running_loss += loss.item()
            
            # 진행 상황 표시줄 업데이트
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
    
    # 모델 설정
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-960h', help='사전학습된 wav2vec2 모델')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.25, help='어댑터 차원 비율')
    parser.add_argument('--unfreeze_top_percent', type=float, default=0.5, help='상위 레이어 언프리징 비율')
    parser.add_argument('--error_influence_weight', type=float, default=0.2, help='오류 영향 가중치')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=10, help='에폭 수')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='models', help='모델 체크포인트 출력 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results', help='결과 출력 디렉토리')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='로드할 모델 체크포인트 경로')
    
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
    else:
        logger.error(f"음소-ID 매핑 파일({args.phoneme_map})이 필요합니다. 이 파일을 생성한 후 다시 시도하세요.")
        sys.exit(1)
    
    # 모델 초기화
    model = DualWav2VecWithErrorAwarePhonemeRecognition(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        adapter_dim_ratio=args.adapter_dim_ratio,
        unfreeze_top_percent=args.unfreeze_top_percent,
        error_influence_weight=args.error_influence_weight,
        training_stage=args.stage,
        blank_index=0  # CTC 빈칸 인덱스
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
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=error_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=error_collate_fn
        )
        
        # 손실 함수와 옵티마이저 설정
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 학습 루프
        best_val_acc = 0.0
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            # 학습
            train_loss, train_acc = train_error_detection(
                model, train_dataloader, criterion, optimizer, args.device, epoch
            )
            
            # 검증
            val_loss, val_acc = validate_error_detection(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 학습 정확도: {train_acc:.2f}%, "
                       f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.2f}%")
            
            # 결과 저장
            with open(os.path.join(args.result_dir, f'error_detection_epoch{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, f, indent=4)
            
            # 최고 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_error_detection.pth'))
                logger.info(f"검증 정확도 {val_acc:.2f}%로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_error_detection.pth'))
    
    elif args.stage == 2:
        # 2단계: 음소 인식 학습
        logger.info("2단계: 음소 인식 학습")
        
        # 모델의 학습 단계 변경
        model.set_training_stage(2)
        
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
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            # 학습
            train_loss = train_phoneme_recognition(
                model, train_dataloader, criterion, optimizer, args.device, epoch
            )
            
            # 검증
            val_loss = validate_phoneme_recognition(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
            
            # 결과 저장
            with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, f, indent=4)
            
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