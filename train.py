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
    calculate_soft_length,
    show_sample_predictions,
    evaluate_error_detection,
    evaluate_phoneme_recognition,
)
from models.loss_functions import LogCoshLengthLoss
from src.data_prepare import UnifiedDataset, collate_fn

logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, wav2vec_optimizer, main_optimizer,
                device, epoch, scaler, gradient_accumulation=1, config=None):
    model.train()
    if config and config.wav2vec2_specaug:
        enable_wav2vec2_specaug(model, True)
    
    length_loss_fn = LogCoshLengthLoss()

    total_loss = 0.0
    error_loss_sum = 0.0
    phoneme_loss_sum = 0.0
    length_loss_sum = 0.0
    error_count = 0
    phoneme_count = 0
    length_count = 0.0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue

        accumulated_loss = 0.0

        waveforms = batch_data['waveforms'].to(device)
        audio_lengths = batch_data['audio_lengths'].to(device)
        phoneme_labels = batch_data['phoneme_labels'].to(device)
        phoneme_lengths = batch_data['phoneme_lengths'].to(device)

        input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
        wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
        attention_mask = make_attn_mask(waveforms, wav_lens_norm)

        with torch.amp.autocast('cuda'):
            outputs = model(waveforms, attention_mask=attention_mask, training_mode=config.training_mode)

            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            error_targets = None
            error_input_lengths = None
            error_target_lengths = None

            if config.has_error_component() and 'error_labels' in batch_data:
                error_labels = batch_data['error_labels'].to(device)
                error_lengths = batch_data['error_lengths'].to(device)
                error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
                
                valid_error_mask = error_lengths > 0
                if valid_error_mask.any():
                    error_targets = error_labels[valid_error_mask]
                    error_input_lengths = error_input_lengths[valid_error_mask]
                    error_target_lengths = error_lengths[valid_error_mask]

            loss, loss_dict = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=phoneme_input_lengths,
                phoneme_target_lengths=phoneme_lengths,
                error_targets=error_targets,
                error_input_lengths=error_input_lengths,
                error_target_lengths=error_target_lengths
            )
            
            if config.has_length_component():
                os.makedirs(config.length_logs_dir, exist_ok=True)
                length_logs_path = os.path.join(config.length_logs_dir, f'length_logs_epoch_{epoch}.json')
                
                phoneme_logits = outputs['phoneme_logits']
                soft_length = calculate_soft_length(phoneme_logits, config)
                soft_length = torch.clamp(soft_length, max=80)

                length_loss = length_loss_fn(
                    soft_length,
                    phoneme_lengths.float()
                )

                length_loss_sum += length_loss
                length_count += 1

                soft_lengths = [int(s) for s in soft_length.tolist()]
                target_lengths = phoneme_lengths.tolist()
                length_diffs = [s - t for s, t in zip(soft_lengths, target_lengths)]
                length_dict = {
                    'epoch_num': epoch,
                    'batch_idx': batch_idx,
                    'soft_lengths': soft_lengths,
                    'target_lengths': target_lengths,
                    'length_diffs': length_diffs
                }
                    
                with open(length_logs_path, 'a') as f:
                    json.dump(length_dict, f)
                    f.write("\n")

                loss = loss + (config.length_weight * length_loss)

            accumulated_loss = loss / gradient_accumulation
            if 'error_loss' in loss_dict:
                error_loss_sum += loss_dict['error_loss']
                error_count += 1
            if 'phoneme_loss' in loss_dict:
                phoneme_loss_sum += loss_dict['phoneme_loss']
                phoneme_count += 1

        if accumulated_loss > 0:
            if scaler:
                scaler.scale(accumulated_loss).backward()
            else:
                accumulated_loss.backward()

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

        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

        avg_total = total_loss / max(((batch_idx + 1) // gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        avg_length = length_loss_sum / max(length_count, 1) if config.has_length_component() else 0

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
    model.eval()
    enable_wav2vec2_specaug(model, False)
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            waveforms = batch_data['waveforms'].to(device)
            audio_lengths = batch_data['audio_lengths'].to(device)
            phoneme_labels = batch_data['phoneme_labels'].to(device)
            phoneme_lengths = batch_data['phoneme_lengths'].to(device)

            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)

            outputs = model(waveforms, attention_mask=attention_mask, training_mode=config.training_mode)

            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            error_targets = None
            error_input_lengths = None
            error_target_lengths = None

            if config.has_error_component() and 'error_labels' in batch_data:
                error_labels = batch_data['error_labels'].to(device)
                error_lengths = batch_data['error_lengths'].to(device)
                error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
                
                valid_error_mask = error_lengths > 0
                if valid_error_mask.any():
                    error_targets = error_labels[valid_error_mask]
                    error_input_lengths = error_input_lengths[valid_error_mask]
                    error_target_lengths = error_lengths[valid_error_mask]

            loss, _ = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=phoneme_input_lengths,
                phoneme_target_lengths=phoneme_lengths,
                error_targets=error_targets,
                error_input_lengths=error_input_lengths,
                error_target_lengths=error_target_lengths
            )

            if config.has_length_component():
                phoneme_logits = outputs['phoneme_logits']
                soft_length = calculate_soft_length(phoneme_logits, config)

                length_loss = LogCoshLengthLoss()(
                    soft_length,
                    phoneme_lengths.float()
                )
                loss = loss + config.length_weight * length_loss

            total_loss += loss.item() if loss > 0 else 0
            progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path):
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
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    wav2vec_optimizer.load_state_dict(checkpoint['wav2vec_optimizer_state_dict'])
    main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint.get('metrics', {})
    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    if 'saved_time' in checkpoint:
        logger.info(f"Checkpoint saved at: {checkpoint['saved_time']}")
    logger.info(f"Previous metrics: {best_metrics}")
    return start_epoch, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', type=str, choices=['phoneme_only', 'phoneme_error', 'phoneme_error_length'], help='Training mode')
    parser.add_argument('--model_type', type=str, choices=['simple', 'transformer'], help='Model architecture')
    parser.add_argument('--config', type=str, help='Override config values in format key=value')
    parser.add_argument('--train_data', type=str, help='Override train data path')
    parser.add_argument('--val_data', type=str, help='Override validation data path')
    parser.add_argument('--eval_data', type=str, help='Override evaluation data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint path')
    parser.add_argument('--experiment_name', type=str, help='Override experiment name')
    args = parser.parse_args()

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

    if args.resume:
        detected_model_type = detect_model_type_from_checkpoint(args.resume)
        config.model_type = detected_model_type
        logger.info(f"Auto-detected model type from checkpoint: {detected_model_type}")

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)

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

    seed_everything(config.seed)
    setup_experiment_dirs(config, resume=bool(args.resume))

    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}

    model_class, loss_class = get_model_class(config.model_type)
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **config.get_model_config()
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(config.device)

    criterion = loss_class(
        training_mode=config.training_mode,
        error_weight=config.error_weight,
        phoneme_weight=config.phoneme_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma
    )

    wav2vec_params = []
    main_params = []
    for name, param in model.named_parameters():
        if 'encoder.wav2vec2' in name:
            wav2vec_params.append(param)
        else:
            main_params.append(param)

    wav2vec_optimizer = optim.AdamW(wav2vec_params, lr=config.wav2vec_lr)
    main_optimizer = optim.AdamW(main_params, lr=config.main_lr)

    scaler = torch.amp.GradScaler('cuda')

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

    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

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
        logger.info("RESUMING TRAINING")
        logger.info("=" * 50)
        logger.info(f"Training mode: {config.training_mode}")
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best error accuracy so far: {best_error_accuracy:.4f}")
        logger.info(f"Best phoneme accuracy so far: {best_phoneme_accuracy:.4f}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting training with training mode: {config.training_mode}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"SpecAugment enabled: {config.wav2vec2_specaug}")
        logger.info(f"Length Loss enabled: {config.has_length_component()}")
        logger.info(f"Error Detection enabled: {config.has_error_component()}")
        logger.info(f"Using Focal Loss with default parameters")

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            config.device, epoch, scaler, config.gradient_accumulation, config
        )
        val_loss = validate_epoch(model, val_dataloader, criterion, config.device, config)
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if True:
            logger.info(f"Epoch {epoch} - Sample Predictions")
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

        logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
        phoneme_recognition_results = evaluate_phoneme_recognition(
            model=model, 
            dataloader=eval_dataloader, 
            device=config.device, 
            training_mode=config.training_mode, 
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")

        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        current_error_accuracy = 0.0

        if config.has_error_component():
            logger.info(f"Epoch {epoch}: Evaluating error detection...")
            error_detection_results = evaluate_error_detection(
                model=model, 
                dataloader=eval_dataloader, 
                device=config.device, 
                training_mode=config.training_mode, 
                error_type_names=error_type_names
            )
            logger.info(f"Error Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"Error Weighted F1: {error_detection_results['weighted_f1']:.4f}")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                if error_type != 'blank':
                    logger.info(f"  {error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            current_error_accuracy = error_detection_results['token_accuracy']

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
            logger.info(f"New best error accuracy: {best_error_accuracy:.4f}")

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
            logger.info(f"New best phoneme accuracy: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")

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
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        latest_metrics = {
            'error_accuracy': best_error_accuracy,
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                      epoch, val_loss, train_loss, latest_metrics, latest_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    logger.info("Training completed!")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Best Error Accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f}")
    logger.info(f"Final metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()