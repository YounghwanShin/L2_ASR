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
from torch.nn import CrossEntropyLoss

from model import DualWav2VecWithErrorAwarePhonemeRecognition, LearnableWav2Vec

torch.autograd.set_detect_anomaly(True)

class ErrorLabelFrameDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000, wav2vec_stride=320):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.wav2vec_stride = wav2vec_stride  # wav2vec2의 다운샘플링 비율 (320은 20ms 단위)
        
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
        
        # 프레임별 오류 레이블 변환 및 wav2vec 출력 길이에 맞게 다운샘플링
        frame_errors = item.get('frame_errors', [])
        
        # wav2vec2 출력 프레임 수 계산
        wav_length = waveform.shape[-1]
        # wav2vec2는 일반적으로 ~20ms 마다 하나의 출력을 생성 (320 샘플)
        # 이는 대략 1/20 다운샘플링 비율
        num_wav2vec_frames = wav_length // self.wav2vec_stride + (1 if wav_length % self.wav2vec_stride > 0 else 0)
        
        # 원본 프레임 수
        num_orig_frames = len(frame_errors)
        
        # 레이블 다운샘플링 (wav2vec2 출력 프레임 수에 맞게)
        if num_orig_frames > 0:
            # 리샘플링 비율 계산
            resample_ratio = num_orig_frames / num_wav2vec_frames
            downsampled_labels = []
            
            for i in range(num_wav2vec_frames):
                # 원본 프레임 인덱스 계산
                orig_frame_idx = min(int(i * resample_ratio), num_orig_frames - 1)
                error = frame_errors[orig_frame_idx]
                
                if error in self.error_type_mapping:
                    downsampled_labels.append(self.error_type_mapping[error])
                else:
                    # 알 수 없는 오류 유형은 'C'(Correct)로 간주
                    downsampled_labels.append(self.error_type_mapping['C'])
        else:
            # 프레임 정보가 없으면 모든 프레임을 'C'로 설정
            downsampled_labels = [self.error_type_mapping['C']] * num_wav2vec_frames
        
        error_labels = torch.tensor(downsampled_labels, dtype=torch.long)
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

# 프레임 기반 오류 탐지를 위한 배치 콜레이션 함수
def error_frame_collate_fn(batch):
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
    
    # 프레임별 레이블 패딩
    max_label_len = max([labels.shape[0] for labels in error_labels])
    padded_error_labels = []
    
    for labels in error_labels:
        label_len = labels.shape[0]
        padding = max_label_len - label_len
        # 패딩 위치에는 무시할 값(-100)을 넣음 (CrossEntropyLoss에서 무시됨)
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=-100)
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

def train_error_detection_frame(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=0.5):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_valid_predictions = 0  # 패딩을 제외한 예측 수
    
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
        
        # error_logits는 [batch_size, seq_len, num_classes] 형태
        # 각 프레임에 대한 예측 계산
        error_preds = torch.argmax(error_logits, dim=-1)
        
        # 손실 계산 (CrossEntropyLoss는 [B, C, L] 형태를 기대하므로 차원 변경)
        # error_logits: [B, L, C] -> [B, C, L]로 변경
        error_logits_transposed = error_logits.transpose(1, 2)
        
        # 손실 계산 (ignore_index=-100으로 패딩 무시)
        loss = criterion(error_logits_transposed, error_labels)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 통계 계산
        running_loss += loss.item()
        
        # 정확도 계산 (패딩 제외)
        mask = (error_labels != -100)  # 패딩 마스크
        correct_predictions += ((error_preds == error_labels) & mask).sum().item()
        total_valid_predictions += mask.sum().item()
        
        # 진행 상황 표시줄 업데이트
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100.0 * correct_predictions / max(total_valid_predictions, 1)
        
        progress_bar.set_postfix({
            '손실': f'{avg_loss:.4f}',
            '정확도': f'{accuracy:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct_predictions / max(total_valid_predictions, 1)
    
    return epoch_loss, epoch_accuracy

def validate_error_detection_frame(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_valid_predictions = 0
    
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
            
            # 각 프레임에 대한 예측 계산
            error_preds = torch.argmax(error_logits, dim=-1)
            
            # 손실 계산 (CrossEntropyLoss는 [B, C, L] 형태를 기대하므로 차원 변경)
            error_logits_transposed = error_logits.transpose(1, 2)
            loss = criterion(error_logits_transposed, error_labels)
            
            # 통계 계산
            running_loss += loss.item()
            
            # 정확도 계산 (패딩 제외)
            mask = (error_labels != -100)  # 패딩 마스크
            correct_predictions += ((error_preds == error_labels) & mask).sum().item()
            total_valid_predictions += mask.sum().item()
            
            # 진행 상황 표시줄 업데이트
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct_predictions / max(total_valid_predictions, 1)
            
            progress_bar.set_postfix({
                '검증_손실': f'{avg_loss:.4f}',
                '정확도': f'{accuracy:.2f}%'
            })
    
    val_loss = running_loss / len(dataloader)
    val_accuracy = 100.0 * correct_predictions / max(total_valid_predictions, 1)
    
    return val_loss, val_accuracy

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
    parser.add_argument('--error_train_data', type=str, default='data/errors_train_frames.json', help='오류 탐지 학습 데이터')
    parser.add_argument('--error_val_data', type=str, default='data/errors_val_frames.json', help='오류 탐지 검증 데이터')
    parser.add_argument('--wav2vec_stride', type=int, default=320, help='wav2vec2 다운샘플링 스트라이드')
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
        logger.info("1단계: 프레임별 오류 탐지 학습")
        
        # 데이터셋 로드
        train_dataset = ErrorLabelFrameDataset(args.error_train_data, max_length=args.max_audio_length, wav2vec_stride=args.wav2vec_stride)
        val_dataset = ErrorLabelFrameDataset(args.error_val_data, max_length=args.max_audio_length, wav2vec_stride=args.wav2vec_stride)
        
        # 프레임 기반 collate_fn 사용
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=error_frame_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=error_frame_collate_fn
        )
        
        # CrossEntropyLoss 사용 (ignore_index=-100으로 패딩 무시)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 학습 루프
        best_val_accuracy = 0.0
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            # 프레임 기반 학습 함수 사용
            train_loss, train_accuracy = train_error_detection_frame(
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=args.max_grad_norm
            )
            
            # 프레임 기반 검증 함수 사용
            val_loss, val_accuracy = validate_error_detection_frame(
                model, val_dataloader, criterion, args.device
            )
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 학습 정확도: {train_accuracy:.2f}%, "
                       f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_accuracy:.2f}%")
            
            # 결과 저장
            with open(os.path.join(args.result_dir, f'error_detection_frame_epoch{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, f, indent=4)
            
            # 최고 모델 저장 - 정확도 기준
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_error_detection_frame.pth'))
                logger.info(f"검증 정확도 {val_accuracy:.2f}%로 새로운 최고 모델 저장")
            
            # 마지막 모델 저장
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_error_detection_frame.pth'))
    
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
        criterion = nn.CTCLoss(blank=0, reduction='mean')
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
                model, train_dataloader, criterion, optimizer, args.device, epoch, max_grad_norm=1.0
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