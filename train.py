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
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ErrorDetectionModel, PhonemeRecognitionModel
from data import ErrorLabelDataset, PhonemeRecognitionDataset, EvaluationDataset
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition, decode_ctc, collate_fn

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

def train_error_detection(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, max_grad_norm=0.5):
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
        error_logits = model(waveforms, attention_mask)
        
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

def validate_error_detection(model, dataloader, criterion, device):
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
            error_logits = model(waveforms, attention_mask)
            
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

def train_phoneme_recognition(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, max_grad_norm=1.0):
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
        phoneme_logits, _ = model(waveforms, attention_mask)
        
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
            phoneme_logits, _ = model(waveforms, attention_mask)
            
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
    parser = argparse.ArgumentParser(description='L2 발음 오류 탐지 및 음소 인식을 위한 분리 모델 학습')
    
    # 기본 설정
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--mode', type=str, choices=['error', 'phoneme'], required=True, 
                        help='학습 모드 (error: 오류 탐지, phoneme: 음소 인식)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='사용할 장치')
    
    # 데이터 설정
    parser.add_argument('--error_train_data', type=str, help='오류 탐지 학습 데이터')
    parser.add_argument('--error_val_data', type=str, help='오류 탐지 검증 데이터')
    parser.add_argument('--phoneme_train_data', type=str, help='음소 인식 학습 데이터')
    parser.add_argument('--phoneme_val_data', type=str, help='음소 인식 검증 데이터')
    parser.add_argument('--phoneme_map', type=str, help='음소-ID 매핑')
    
    # 평가 데이터셋
    parser.add_argument('--eval_data', type=str, help='평가 데이터셋 경로')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='평가 배치 크기')
    parser.add_argument('--evaluate_every_epoch', action='store_true', help='각 에폭마다 평가 진행')
    
    # 모델 설정
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53', 
                        help='사전학습된 wav2vec2 모델')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--num_error_types', type=int, default=6, help='오류 유형 수 (blank + separator 포함)')
    parser.add_argument('--error_model_checkpoint', type=str, default=None, 
                        help='음소 인식 모드에서 사용할 오류 탐지 모델 체크포인트')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=10, help='에폭 수')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='그라디언트 클리핑을 위한 최대 노름값')
    
    # 학습률 스케줄러 설정
    parser.add_argument('--use_scheduler', action='store_true', help='학습률 스케줄러 사용 여부')
    parser.add_argument('--scheduler_patience', type=int, default=2, help='학습률 감소 전 기다릴 에폭 수')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='학습률 감소 비율')
    parser.add_argument('--scheduler_threshold', type=float, default=0.001, help='개선으로 간주할 최소 변화량')
    parser.add_argument('--scheduler_cooldown', type=int, default=1, help='감소 후 감시 재개 전 대기 에폭 수')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help='최소 학습률')
    
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
            logging.FileHandler(os.path.join(args.result_dir, f'train_{args.mode}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(args.result_dir, f'hyperparams_{args.mode}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 음소 매핑 로드 (음소 인식 모드에서만 필요)
    phoneme_to_id = None
    id_to_phoneme = {}
    if args.phoneme_map:
        with open(args.phoneme_map, 'r') as f:
            phoneme_to_id = json.load(f)
        id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    # 오류 유형 이름 매핑
    error_type_names = {0: 'blank', 1: 'deletion', 2: 'substitution', 3: 'insertion', 4: 'correct', 5: 'separator'}
    
    # 모델 초기화
    if args.mode == 'error':
        logger.info("오류 탐지 모델 초기화")
        model = ErrorDetectionModel(
            pretrained_model_name=args.pretrained_model,
            hidden_dim=args.hidden_dim,
            num_error_types=args.num_error_types
        )
        
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    elif args.mode == 'phoneme':
        logger.info("음소 인식 모델 초기화")
        if not args.error_model_checkpoint:
            logger.error("음소 인식 모드에서는 오류 탐지 모델 체크포인트가 필요합니다.")
            sys.exit(1)
            
        model = PhonemeRecognitionModel(
            pretrained_model_name=args.pretrained_model,
            error_model_checkpoint=args.error_model_checkpoint,
            hidden_dim=args.hidden_dim,
            num_phonemes=args.num_phonemes,
            num_error_types=args.num_error_types
        )
        
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 이전 체크포인트가 있으면 로드
    if args.model_checkpoint:
        logger.info(f"{args.model_checkpoint}에서 체크포인트 로드 중")
        state_dict = torch.load(args.model_checkpoint, map_location=args.device)
        
        # "module." 접두사 제거
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
    
    # 여러 GPU 사용 시 DataParallel 적용
    if torch.cuda.device_count() > 1:
        logger.info(f"{torch.cuda.device_count()}개의 GPU가 감지되었습니다. DataParallel 사용")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 학습률 스케줄러 설정
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=args.scheduler_threshold,
            threshold_mode='rel',
            cooldown=args.scheduler_cooldown,
            min_lr=args.scheduler_min_lr
        )
        logger.info("학습률 스케줄러(ReduceLROnPlateau) 초기화됨")
    
    # 평가 데이터셋 로드 (각 에폭마다 평가 수행)
    eval_dataloader = None
    if args.evaluate_every_epoch and args.eval_data:
        logger.info(f"평가 데이터셋 로드 중: {args.eval_data}")
        
        if args.mode == 'phoneme' and not phoneme_to_id:
            logger.error("음소 인식 모드에서는 음소 매핑이 필요합니다.")
            sys.exit(1)
        
        eval_dataset = EvaluationDataset(
            args.eval_data, phoneme_to_id, max_length=args.max_audio_length
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn
        )
    
    # 데이터셋 및 데이터로더 설정
    if args.mode == 'error':
        if not args.error_train_data or not args.error_val_data:
            logger.error("오류 탐지 모드에서는 오류 학습 및 검증 데이터 경로가 필요합니다.")
            sys.exit(1)
            
        train_dataset = ErrorLabelDataset(args.error_train_data, max_length=args.max_audio_length)
        val_dataset = ErrorLabelDataset(args.error_val_data, max_length=args.max_audio_length)
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=error_ctc_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=error_ctc_collate_fn
        )
        
        # 학습 시작
        best_val_loss = float('inf')
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            train_loss = train_error_detection(
                model, train_dataloader, criterion, optimizer, args.device, epoch, 
                scheduler=scheduler, max_grad_norm=args.max_grad_norm
            )
            
            val_loss = validate_error_detection(
                model, val_dataloader, criterion, args.device
            )
            
            # 스케줄러 단계 업데이트
            if scheduler is not None:
                scheduler.step(val_loss)
                logger.info(f"현재 학습률: {optimizer.param_groups[0]['lr']:.2e}")
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
            
            # 각 에폭마다 평가 수행
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if args.evaluate_every_epoch and eval_dataloader:
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
                
                # 평가 결과 저장
                epoch_metrics['error_detection'] = error_detection_results
            
            # 에폭 결과 저장
            with open(os.path.join(args.result_dir, f'error_detection_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_metrics, f, indent=4)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(args.output_dir, f'best_error_detection.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장: {model_path}")
            
            # 마지막 모델 저장
            model_path = os.path.join(args.output_dir, f'last_error_detection.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"마지막 모델 저장: {model_path}")
            
    elif args.mode == 'phoneme':
        if not args.phoneme_train_data or not args.phoneme_val_data:
            logger.error("음소 인식 모드에서는 음소 학습 및 검증 데이터 경로가 필요합니다.")
            sys.exit(1)
            
        if not phoneme_to_id:
            logger.error("음소 인식 모드에서는 음소-ID 매핑이 필요합니다.")
            sys.exit(1)
            
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
        
        # 학습 시작
        best_val_loss = float('inf')
        
        for epoch in range(1, args.num_epochs + 1):
            logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
            
            train_loss = train_phoneme_recognition(
                model, train_dataloader, criterion, optimizer, args.device, epoch, 
                scheduler=scheduler, max_grad_norm=args.max_grad_norm
            )
            
            val_loss = validate_phoneme_recognition(
                model, val_dataloader, criterion, args.device
            )
            
            # 스케줄러 단계 업데이트
            if scheduler is not None:
                scheduler.step(val_loss)
                logger.info(f"현재 학습률: {optimizer.param_groups[0]['lr']:.2e}")
            
            logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
            
            # 각 에폭마다 평가 수행
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if args.evaluate_every_epoch and eval_dataloader:
                # 음소 인식 평가
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
                
                # 평가 결과 저장
                epoch_metrics['phoneme_recognition'] = {
                    'per': phoneme_recognition_results['per'],
                    'total_phonemes': phoneme_recognition_results['total_phonemes'],
                    'total_errors': phoneme_recognition_results['total_errors'],
                    'insertions': phoneme_recognition_results['insertions'],
                    'deletions': phoneme_recognition_results['deletions'],
                    'substitutions': phoneme_recognition_results['substitutions']
                }
            
            # 에폭 결과 저장
            with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_metrics, f, indent=4)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(args.output_dir, f'best_phoneme_recognition.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장: {model_path}")
            
            # 마지막 모델 저장
            model_path = os.path.join(args.output_dir, f'last_phoneme_recognition.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"마지막 모델 저장: {model_path}")
    
    logger.info("학습 완료!")

if __name__ == "__main__":
    main()