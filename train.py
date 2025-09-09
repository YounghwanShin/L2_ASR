import os
import json
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from src.utils import (
    make_attn_mask,
    get_model_class,
    detect_model_type_from_checkpoint,
    setup_experiment_dirs,
    enable_wav2vec2_specaug,
    get_wav2vec2_output_lengths_official,
    show_sample_predictions,
    evaluate_error_detection,
    evaluate_phoneme_recognition,
)
from src.data_prepare import UnifiedDataset, collate_fn

logger = logging.getLogger(__name__)

def calculate_target_length(phoneme_lengths, error_lengths=None, training_mode='phoneme_only'):
    """
    훈련 모드에 따라 적절한 목표 길이를 계산
    
    Args:
        phoneme_lengths: Phoneme 시퀀스 길이
        error_lengths: Error 시퀀스 길이 (옵션)
        training_mode: 훈련 모드
    
    Returns:
        목표 길이 텐서
    """
    if training_mode == 'phoneme_only':
        return phoneme_lengths.float()
    elif training_mode == 'phoneme_error':
        # Error 정보가 있으면 더 긴 길이를 목표로 설정
        if error_lengths is not None:
            return torch.max(phoneme_lengths, error_lengths).float()
        else:
            return phoneme_lengths.float()
    elif training_mode == 'phoneme_error_length':
        # Length prediction이 주요 목적이므로 phoneme 길이를 기준으로 함
        # 하지만 error 정보가 있으면 약간의 가중치를 부여
        if error_lengths is not None:
            # Phoneme 길이를 기본으로 하되, error 길이가 더 길면 약간 반영
            target_lengths = phoneme_lengths.float()
            longer_error_mask = error_lengths > phoneme_lengths
            target_lengths[longer_error_mask] = (
                phoneme_lengths[longer_error_mask].float() * 0.8 + 
                error_lengths[longer_error_mask].float() * 0.2
            )
            return target_lengths
        else:
            return phoneme_lengths.float()
    else:
        return phoneme_lengths.float()

def train_epoch(model, dataloader, criterion, wav2vec_optimizer, main_optimizer,
                device, epoch, scaler, gradient_accumulation=1, config=None):
    """훈련 에폭 실행"""
    model.train()
    if config and config.wav2vec2_specaug:
        enable_wav2vec2_specaug(model, True)

    total_loss = 0.0
    error_loss_sum = 0.0
    phoneme_loss_sum = 0.0
    length_loss_sum = 0.0
    error_count = 0
    phoneme_count = 0
    length_count = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue

        # 배치 데이터 준비
        waveforms = batch_data['waveforms'].to(device)
        audio_lengths = batch_data['audio_lengths'].to(device)
        phoneme_labels = batch_data['phoneme_labels'].to(device)
        phoneme_lengths = batch_data['phoneme_lengths'].to(device)

        # Attention mask 및 입력 길이 계산
        input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
        wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
        attention_mask = make_attn_mask(waveforms, wav_lens_norm)

        with torch.amp.autocast('cuda'):
            # 모델 순전파
            outputs = model(waveforms, attention_mask=attention_mask, training_mode=config.training_mode)

            # Phoneme CTC를 위한 입력 길이 조정
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            # Error detection 관련 데이터 준비
            error_targets = None
            error_input_lengths = None
            error_target_lengths = None
            batch_error_lengths = None

            if config.has_error_component() and 'error_labels' in batch_data:
                error_labels = batch_data['error_labels'].to(device)
                batch_error_lengths = batch_data['error_lengths'].to(device)
                error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                # 유효한 error 샘플들만 필터링
                valid_error_mask = batch_error_lengths > 0
                if valid_error_mask.any():
                    error_targets = error_labels[valid_error_mask]
                    error_input_lengths = error_input_lengths[valid_error_mask]
                    error_target_lengths = batch_error_lengths[valid_error_mask]

            # 길이 예측을 위한 목표 길이 계산
            target_lengths = None
            if config.has_length_component():
                target_lengths = calculate_target_length(
                    phoneme_lengths, 
                    batch_error_lengths, 
                    config.training_mode
                )

            # 통합 손실 계산
            loss, loss_dict = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=phoneme_input_lengths,
                phoneme_target_lengths=phoneme_lengths,
                error_targets=error_targets,
                error_input_lengths=error_input_lengths,
                error_target_lengths=error_target_lengths,
                target_lengths=target_lengths
            )

            # Gradient accumulation을 위한 손실 스케일링
            accumulated_loss = loss / gradient_accumulation

            # 손실 통계 업데이트
            if 'error_loss' in loss_dict:
                error_loss_sum += loss_dict['error_loss']
                error_count += 1
            if 'phoneme_loss' in loss_dict:
                phoneme_loss_sum += loss_dict['phoneme_loss']
                phoneme_count += 1
            if 'length_loss' in loss_dict:
                length_loss_sum += loss_dict['length_loss']
                length_count += 1

        # 역전파
        if accumulated_loss > 0:
            if scaler:
                scaler.scale(accumulated_loss).backward()
            else:
                accumulated_loss.backward()

        # Gradient accumulation 처리
        if (batch_idx + 1) % gradient_accumulation == 0:
            if scaler:
                scaler.step(wav2vec_optimizer)
                scaler.step(main_optimizer)
                scaler.update()
            else:
                wav2vec_optimizer.step()
                main_optimizer.step()
            wav2vec_optimizer.zero_grad()
            main_optimizer.zero_grad()

            total_loss += accumulated_loss.item() * gradient_accumulation if accumulated_loss > 0 else 0

        # 메모리 정리
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

        # Progress bar 업데이트
        avg_total = total_loss / max(((batch_idx + 1) // gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        avg_length = length_loss_sum / max(length_count, 1) if length_count > 0 else 0

        progress_dict = {
            'Total': f'{avg_total:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        }
        if config.has_error_component():
            progress_dict['Error'] = f'{avg_error:.4f}'
        if config.has_length_component():
            progress_dict['Length'] = f'{avg_length:.4f}'

        progress_bar.set_postfix(progress_dict)

    torch.cuda.empty_cache()
    return total_loss / (len(dataloader) // gradient_accumulation)

def validate_epoch(model, dataloader, criterion, device, config):
    """검증 에폭 실행"""
    model.eval()
    enable_wav2vec2_specaug(model, False)
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            # 배치 데이터 준비
            waveforms = batch_data['waveforms'].to(device)
            audio_lengths = batch_data['audio_lengths'].to(device)
            phoneme_labels = batch_data['phoneme_labels'].to(device)
            phoneme_lengths = batch_data['phoneme_lengths'].to(device)

            # Attention mask 및 입력 길이 계산
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)

            # 모델 순전파
            outputs = model(waveforms, attention_mask=attention_mask, training_mode=config.training_mode)

            # Phoneme CTC를 위한 입력 길이 조정
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            # Error detection 관련 데이터 준비
            error_targets = None
            error_input_lengths = None
            error_target_lengths = None
            batch_error_lengths = None

            if config.has_error_component() and 'error_labels' in batch_data:
                error_labels = batch_data['error_labels'].to(device)
                batch_error_lengths = batch_data['error_lengths'].to(device)
                error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                # 유효한 error 샘플들만 필터링
                valid_error_mask = batch_error_lengths > 0
                if valid_error_mask.any():
                    error_targets = error_labels[valid_error_mask]
                    error_input_lengths = error_input_lengths[valid_error_mask]
                    error_target_lengths = batch_error_lengths[valid_error_mask]

            # 길이 예측을 위한 목표 길이 계산
            target_lengths = None
            if config.has_length_component():
                target_lengths = calculate_target_length(
                    phoneme_lengths, 
                    batch_error_lengths, 
                    config.training_mode
                )

            # 통합 손실 계산
            loss, _ = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=phoneme_input_lengths,
                phoneme_target_lengths=phoneme_lengths,
                error_targets=error_targets,
                error_input_lengths=error_input_lengths,
                error_target_lengths=error_target_lengths,
                target_lengths=target_lengths
            )

            total_loss += loss.item() if loss > 0 else 0
            progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

def seed_everything(seed):
    """시드 고정으로 재현 가능한 학습 환경 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
        'main_optimizer_state_dict': main_opt.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'metrics': metrics,
        'saved_time': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(checkpoint_path, model, wav2vec_optimizer, main_optimizer, device):
    """체크포인트 로드"""
    logger.info(f"체크포인트 로딩: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    wav2vec_optimizer.load_state_dict(checkpoint['wav2vec_optimizer_state_dict'])
    main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint.get('metrics', {})
    logger.info(f"에폭 {checkpoint['epoch']}에서 재개")
    if 'saved_time' in checkpoint:
        logger.info(f"체크포인트 저장 시점: {checkpoint['saved_time']}")
    logger.info(f"이전 메트릭: {best_metrics}")
    return start_epoch, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', type=str, choices=['phoneme_only', 'phoneme_error', 'phoneme_error_length'], help='훈련 모드')
    parser.add_argument('--model_type', type=str, choices=['simple', 'transformer'], help='모델 아키텍처')
    parser.add_argument('--config', type=str, help='config 값 오버라이드 (key=value 형식)')
    parser.add_argument('--train_data', type=str, help='훈련 데이터 경로 오버라이드')
    parser.add_argument('--val_data', type=str, help='검증 데이터 경로 오버라이드')
    parser.add_argument('--eval_data', type=str, help='평가 데이터 경로 오버라이드')
    parser.add_argument('--phoneme_map', type=str, help='음소 맵 경로 오버라이드')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 오버라이드')
    parser.add_argument('--resume', type=str, help='체크포인트에서 훈련 재개')
    parser.add_argument('--experiment_name', type=str, help='실험 이름 오버라이드')
    args = parser.parse_args()

    # 설정 로드 및 오버라이드
    config = Config()

    if args.training_mode:
        config.training_mode = args.training_mode
    if args.model_type:
        config.model_type = args.model_type
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.output_dir:
        config.output_dir = args.output_dir

    # 체크포인트에서 재개하는 경우 모델 타입 자동 감지
    if args.resume:
        detected_model_type = detect_model_type_from_checkpoint(args.resume)
        config.model_type = detected_model_type
        logger.info(f"체크포인트에서 모델 타입 자동 감지: {detected_model_type}")

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)

    # 추가 설정 오버라이드
    if args.config:
        for override in args.config.split(','):
            key, value = override.split('=')
            if hasattr(config, key):
                attr_type = type(getattr(config, key))
                if attr_type == bool:
                    setattr(config, key, value.lower() == 'true')
                else:
                    setattr(config, key, attr_type(value))

    config.__post_init__()

    # 실험 환경 설정
    seed_everything(config.seed)
    setup_experiment_dirs(config, resume=bool(args.resume))

    # 음소 맵 로드
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}

    # 모델 및 손실 함수 초기화
    model_class, loss_class = get_model_class(config.model_type)
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **config.get_model_config()
    )

    # 멀티 GPU 지원
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(config.device)

    # 손실 함수 설정 (길이 손실 포함)
    criterion = loss_class(
        training_mode=config.training_mode,
        error_weight=config.error_weight,
        phoneme_weight=config.phoneme_weight,
        length_weight=config.length_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        length_loss_type='smooth_l1',  # 길이 손실 타입
        length_beta=1.0  # SmoothL1Loss beta 파라미터
    )

    # 옵티마이저 설정 (차별적 학습률)
    wav2vec_params = []
    main_params = []
    for name, param in model.named_parameters():
        if 'encoder.wav2vec2' in name:
            wav2vec_params.append(param)
        else:
            main_params.append(param)

    wav2vec_optimizer = optim.AdamW(wav2vec_params, lr=config.wav2vec_lr)
    main_optimizer = optim.AdamW(main_params, lr=config.main_lr)

    # Mixed precision 학습을 위한 스케일러
    scaler = torch.amp.GradScaler('cuda')

    # 데이터셋 및 데이터로더 설정
    train_dataset = UnifiedDataset(
        config.train_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    val_dataset = UnifiedDataset(
        config.val_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    eval_dataset = UnifiedDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )

    # 최적 성능 추적 변수들
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

    # 체크포인트에서 재개
    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(
            args.resume, model, wav2vec_optimizer, main_optimizer, config.device
        )
        if 'error_accuracy' in resume_metrics:
            best_error_accuracy = resume_metrics['error_accuracy']
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        checkpoint = torch.load(args.resume, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        logger.info("=" * 50)
        logger.info("훈련 재개")
        logger.info("=" * 50)
        logger.info(f"훈련 모드: {config.training_mode}")
        logger.info(f"에폭 {start_epoch}에서 재개")
        logger.info(f"현재까지 최고 에러 정확도: {best_error_accuracy:.4f}")
        logger.info(f"현재까지 최고 음소 정확도: {best_phoneme_accuracy:.4f}")
        logger.info(f"현재까지 최고 검증 손실: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"훈련 모드로 학습 시작: {config.training_mode}")
        logger.info(f"모델 타입: {config.model_type}")
        logger.info(f"실험명: {config.experiment_name}")
        logger.info(f"{config.num_epochs} 에폭 동안 훈련 시작")
        logger.info(f"SpecAugment 활성화: {config.wav2vec2_specaug}")
        logger.info(f"길이 손실 활성화: {config.has_length_component()}")
        logger.info(f"에러 감지 활성화: {config.has_error_component()}")
        logger.info(f"Focal Loss 사용 (기본 파라미터)")
        if config.has_length_component():
            logger.info(f"길이 예측 헤드를 통한 직접적인 길이 학습")

    # 메인 훈련 루프
    for epoch in range(start_epoch, config.num_epochs + 1):
        # 훈련
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            config.device, epoch, scaler, config.gradient_accumulation, config
        )
        
        # 검증
        val_loss = validate_epoch(model, val_dataloader, criterion, config.device, config)
        logger.info(f"에폭 {epoch}: 훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")

        # 샘플 예측 출력
        if True:  # 매 에폭마다 샘플 예측 확인
            logger.info(f"에폭 {epoch} - 샘플 예측")
            logger.info("=" * 50)
            show_sample_predictions(
                model=model,
                eval_dataloader=eval_dataloader,
                device=config.device,
                id_to_phoneme=id_to_phoneme,
                logger=logger,
                training_mode=config.training_mode,
                error_type_names=error_type_names
            )

        # 음소 인식 성능 평가
        logger.info(f"에폭 {epoch}: 음소 인식 평가 중...")
        phoneme_recognition_results = evaluate_phoneme_recognition(
            model=model,
            dataloader=eval_dataloader,
            device=config.device,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"음소 에러율 (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"음소 정확도: {1.0 - phoneme_recognition_results['per']:.4f}")

        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        current_error_accuracy = 0.0

        # 에러 감지 성능 평가 (필요한 경우)
        if config.has_error_component():
            logger.info(f"에폭 {epoch}: 에러 감지 평가 중...")
            error_detection_results = evaluate_error_detection(
                model=model,
                dataloader=eval_dataloader,
                device=config.device,
                training_mode=config.training_mode,
                error_type_names=error_type_names
            )
            logger.info(f"에러 토큰 정확도: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"에러 가중 F1: {error_detection_results['weighted_f1']:.4f}")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                if error_type != 'blank':
                    logger.info(f"  {error_type}: 정밀도={metrics['precision']:.4f}, 재현율={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            current_error_accuracy = error_detection_results['token_accuracy']

        # 최적 모델 저장
        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_error.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 에러 정확도: {best_error_accuracy:.4f}")

        if config.save_best_phoneme and current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 음소 정확도: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")

        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 검증 손실: {best_val_loss:.4f}")

        # 최신 체크포인트 저장
        latest_metrics = {
            'error_accuracy': best_error_accuracy,
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                      epoch, val_loss, train_loss, latest_metrics, latest_path)

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 훈련 완료 후 최종 메트릭 저장
    final_metrics = {
        'best_error_accuracy': best_error_accuracy,
        'best_phoneme_accuracy': best_phoneme_accuracy,
        'best_val_loss': best_val_loss,
        'completed_epochs': config.num_epochs,
        'training_mode': config.training_mode,
        'model_type': config.model_type,
        'experiment_name': config.experiment_name
    }
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("훈련 완료!")
    logger.info(f"훈련 모드: {config.training_mode}")
    logger.info(f"최고 에러 정확도: {best_error_accuracy:.4f}")
    logger.info(f"최고 음소 정확도: {best_phoneme_accuracy:.4f}")
    logger.info(f"최종 메트릭 저장 위치: {metrics_path}")

if __name__ == "__main__":
    main()