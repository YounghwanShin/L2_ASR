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
from utils import (
    make_attn_mask,
    get_model_class,
    detect_model_type_from_checkpoint,
    setup_experiment_dirs,
    enable_wav2vec2_specaug,
    get_wav2vec2_output_lengths_official,
    calculate_soft_length,
    show_sample_predictions,
    decode_ctc,
)
from models.loss_functions import LogCoshLengthLoss
from data_prepare import BaseDataset, collate_fn
from multitask_eval import evaluate_error_detection, evaluate_phoneme_recognition

logger = logging.getLogger(__name__)

# ======================
# Device 설정
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_dtype = "cuda"
    scaler = torch.amp.GradScaler(device_type="cuda")
else:
    device = torch.device("cpu")
    amp_dtype = "cpu"
    scaler = None

print("Using device:", device)


def train_epoch(model, dataloader, criterion, wav2vec_optimizer, main_optimizer,
                device, epoch, scaler, gradient_accumulation=1, config=None):
    model.train()
    if config and config.wav2vec2_specaug:
        enable_wav2vec2_specaug(model, True)

    total_loss = 0.0
    error_loss_sum = 0.0
    phoneme_loss_sum = 0.0
    error_count = 0
    phoneme_count = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue

        accumulated_loss = 0.0

        waveforms = batch_data['waveforms'].to(device)
        audio_lengths = batch_data['audio_lengths'].to(device)

        input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
        wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
        attention_mask = make_attn_mask(waveforms, wav_lens_norm)

        # AMP 사용 (cuda일 때만)
        autocast_ctx = torch.amp.autocast(device_type=amp_dtype) if amp_dtype == "cuda" else torch.no_grad()
        with autocast_ctx:
            outputs = model(waveforms, attention_mask=attention_mask, task_mode=config.task_mode['multi_train'])

            error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            has_error = any(el is not None for el in batch_data['error_labels'])
            has_phoneme = any(pl is not None for pl in batch_data['phoneme_labels'])

            batch_error_labels = None
            batch_error_lengths = None
            batch_phoneme_labels = None
            batch_phoneme_lengths = None

            if has_error:
                valid_error_indices = [i for i, el in enumerate(batch_data['error_labels']) if el is not None]
                if valid_error_indices:
                    error_labels = [batch_data['error_labels'][i] for i in valid_error_indices]
                    error_lengths = [batch_data['error_lengths'][i] for i in valid_error_indices]

                    max_error_len = max(l.shape[0] for l in error_labels)
                    batch_error_labels = torch.stack([
                        torch.nn.functional.pad(l, (0, max_error_len - l.shape[0]), value=0)
                        for l in error_labels
                    ]).to(device)
                    batch_error_lengths = torch.stack(error_lengths).to(device)

                    error_outputs = {
                        'error_logits': outputs['error_logits'][valid_error_indices]
                    }
                    error_input_lengths_filtered = error_input_lengths[valid_error_indices]

            if has_phoneme:
                valid_phoneme_indices = [i for i, pl in enumerate(batch_data['phoneme_labels']) if pl is not None]
                if valid_phoneme_indices:
                    phoneme_labels = [batch_data['phoneme_labels'][i] for i in valid_phoneme_indices]
                    phoneme_lengths = [batch_data['phoneme_lengths'][i] for i in valid_phoneme_indices]

                    max_phoneme_len = max(l.shape[0] for l in phoneme_labels)
                    batch_phoneme_labels = torch.stack([
                        torch.nn.functional.pad(l, (0, max_phoneme_len - l.shape[0]), value=0)
                        for l in phoneme_labels
                    ]).to(device)
                    batch_phoneme_lengths = torch.stack(phoneme_lengths).to(device)

                    phoneme_outputs = {
                        'phoneme_logits': outputs['phoneme_logits'][valid_phoneme_indices]
                    }
                    phoneme_input_lengths_filtered = phoneme_input_lengths[valid_phoneme_indices]

            combined_outputs = {}
            if has_error and batch_error_labels is not None:
                combined_outputs.update(error_outputs)
            if has_phoneme and batch_phoneme_labels is not None:
                combined_outputs.update(phoneme_outputs)

            if combined_outputs:
                loss, loss_dict = criterion(
                    combined_outputs,
                    error_targets=batch_error_labels,
                    phoneme_targets=batch_phoneme_labels,
                    error_input_lengths=error_input_lengths_filtered if has_error and batch_error_labels is not None else None,
                    phoneme_input_lengths=phoneme_input_lengths_filtered if has_phoneme and batch_phoneme_labels is not None else None,
                    error_target_lengths=batch_error_lengths,
                    phoneme_target_lengths=batch_phoneme_lengths
                )

            length_loss = 0.0
            if has_phoneme:
                phoneme_logits = outputs['phoneme_logits']
                soft_length = calculate_soft_length(phoneme_logits)

                length_loss = LogCoshLengthLoss()(
                    soft_length,
                    batch_phoneme_lengths.float()
                )
                loss = loss + config.length_weight * length_loss

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

        if (batch_idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_total = total_loss / max(((batch_idx + 1) // gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)

        progress_bar.set_postfix({
            'Total': f'{avg_total:.4f}',
            'Error': f'{avg_error:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        })

    if torch.cuda.is_available():
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

            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)

            outputs = model(waveforms, attention_mask=attention_mask, task_mode=config.task_mode['multi_train'])

            error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

            has_error = any(el is not None for el in batch_data['error_labels'])
            has_phoneme = any(pl is not None for pl in batch_data['phoneme_labels'])

            batch_error_labels = None
            batch_error_lengths = None
            batch_phoneme_labels = None
            batch_phoneme_lengths = None

            if has_error:
                valid_error_indices = [i for i, el in enumerate(batch_data['error_labels']) if el is not None]
                if valid_error_indices:
                    error_labels = [batch_data['error_labels'][i] for i in valid_error_indices]
                    error_lengths = [batch_data['error_lengths'][i] for i in valid_error_indices]

                    max_error_len = max(l.shape[0] for l in error_labels)
                    batch_error_labels = torch.stack([
                        torch.nn.functional.pad(l, (0, max_error_len - l.shape[0]), value=0)
                        for l in error_labels
                    ]).to(device)
                    batch_error_lengths = torch.stack(error_lengths).to(device)

                    error_outputs = {
                        'error_logits': outputs['error_logits'][valid_error_indices]
                    }
                    error_input_lengths_filtered = error_input_lengths[valid_error_indices]

            if has_phoneme:
                valid_phoneme_indices = [i for i, pl in enumerate(batch_data['phoneme_labels']) if pl is not None]
                if valid_phoneme_indices:
                    phoneme_labels = [batch_data['phoneme_labels'][i] for i in valid_phoneme_indices]
                    phoneme_lengths = [batch_data['phoneme_lengths'][i] for i in valid_phoneme_indices]

                    max_phoneme_len = max(l.shape[0] for l in phoneme_labels)
                    batch_phoneme_labels = torch.stack([
                        torch.nn.functional.pad(l, (0, max_phoneme_len - l.shape[0]), value=0)
                        for l in phoneme_labels
                    ]).to(device)
                    batch_phoneme_lengths = torch.stack(phoneme_lengths).to(device)

                    phoneme_outputs = {
                        'phoneme_logits': outputs['phoneme_logits'][valid_phoneme_indices]
                    }
                    phoneme_input_lengths_filtered = phoneme_input_lengths[valid_phoneme_indices]

            combined_outputs = {}
            if has_error and batch_error_labels is not None:
                combined_outputs.update(error_outputs)
            if has_phoneme and batch_phoneme_labels is not None:
                combined_outputs.update(phoneme_outputs)

            if combined_outputs:
                loss, _ = criterion(
                    combined_outputs,
                    error_targets=batch_error_labels,
                    phoneme_targets=batch_phoneme_labels,
                    error_input_lengths=error_input_lengths_filtered if has_error and batch_error_labels is not None else None,
                    phoneme_input_lengths=phoneme_input_lengths_filtered if has_phoneme and batch_phoneme_labels is not None else None,
                    error_target_lengths=batch_error_lengths,
                    phoneme_target_lengths=batch_phoneme_lengths
                )

            if has_phoneme:
                phoneme_logits = outputs['phoneme_logits']
                soft_length = calculate_soft_length(phoneme_logits)

                length_loss = LogCoshLengthLoss()(
                    soft_length,
                    batch_phoneme_lengths.float()
                )
                loss = loss + config.length_weight * length_loss

            total_loss += loss.item() if loss > 0 else 0
            progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

    if torch.cuda.is_available():
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

    model = model.to(device)

    criterion = loss_class(
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

    train_dataset = BaseDataset(
        config.train_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        task_mode=config.task_mode['multi_train'],
        error_task_ratio=config.error_task_ratio
    )
    val_dataset = BaseDataset(
        config.val_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        task_mode=config.task_mode['multi_train'],
        error_task_ratio=config.error_task_ratio
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, task_mode=config.task_mode['multi_train'])
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, task_mode=config.task_mode['multi_train'])
    )
    eval_dataset = BaseDataset(
        config.eval_data, phoneme_to_id,
        task_mode=config.task_mode['multi_eval'],
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, task_mode=config.task_mode['multi_eval'])
    )

    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(
            args.resume, model, wav2vec_optimizer, main_optimizer, device
        )
        if 'error_accuracy' in resume_metrics:
            best_error_accuracy = resume_metrics['error_accuracy']
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        checkpoint = torch.load(args.resume, map_location=device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        logger.info("=" * 50)
        logger.info("RESUMING TRAINING")
        logger.info("=" * 50)
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best error accuracy so far: {best_error_accuracy:.4f}")
        logger.info(f"Best phoneme accuracy so far: {best_phoneme_accuracy:.4f}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting training with model type: {config.model_type}")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"SpecAugment enabled: {config.wav2vec2_specaug}")
        logger.info(f"Using Focal Loss with default parameters")

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            device, epoch, scaler, config.gradient_accumulation, config
        )
        val_loss = validate_epoch(model, val_dataloader, criterion, device, config)
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch} - Sample Predictions")
            logger.info("=" * 50)
            show_sample_predictions(model, eval_dataloader, device, id_to_phoneme, logger=logger, error_type_names=error_type_names)

        logger.info(f"Epoch {epoch}: Evaluating error detection...")
        error_detection_results = evaluate_error_detection(model, eval_dataloader, device, error_type_names)
        logger.info(f"Error Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
        logger.info(f"Error Weighted F1: {error_detection_results['weighted_f1']:.4f}")
        for error_type, metrics in error_detection_results['class_metrics'].items():
            if error_type != 'blank':
                logger.info(f"  {error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
        phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, device, id_to_phoneme)
        logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")

        current_error_accuracy = error_detection_results['token_accuracy']
        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']

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
        'model_type': config.model_type,
        'experiment_name': config.experiment_name
    }
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training completed!")
    logger.info(f"Best Error Accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f}")
    logger.info(f"Final metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
