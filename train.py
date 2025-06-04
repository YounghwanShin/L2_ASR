import os
import json
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from model import SimpleMultiTaskModel, MultiTaskLoss
from data_prepare import MultiTaskDataset, EvaluationDataset, multitask_collate_fn, evaluation_collate_fn
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition, get_wav2vec2_output_lengths_official

def get_wav2vec2_output_lengths_official(model, input_lengths):
    actual_model = model.module if hasattr(model, 'module') else model
    wav2vec_model = actual_model.encoder.wav2vec2
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)

def train_epoch(model, dataloader, criterion, wav2vec_optimizer, main_optimizer, 
                device, epoch, scaler, gradient_accumulation=1):
    model.train()
    
    total_loss = 0.0
    error_loss_sum = 0.0
    phoneme_loss_sum = 0.0
    error_count = 0
    phoneme_count = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        accumulated_loss = 0.0
        
        if 'error' in batch_data:
            data = batch_data['error']
            waveforms = data['waveforms'].to(device)
            audio_lengths = data['audio_lengths'].to(device)
            error_labels = data['labels'].to(device)
            error_label_lengths = data['label_lengths'].to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms, attention_mask=attention_mask, task='error')
                input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
                
                error_loss, error_loss_dict = criterion(
                    outputs, 
                    error_targets=error_labels,
                    error_input_lengths=input_lengths,
                    error_target_lengths=error_label_lengths
                )
                
                accumulated_loss += error_loss / gradient_accumulation
                error_loss_sum += error_loss_dict.get('error_loss', 0.0)
                error_count += 1
        
        if 'phoneme' in batch_data:
            data = batch_data['phoneme']
            waveforms = data['waveforms'].to(device)
            audio_lengths = data['audio_lengths'].to(device)
            phoneme_labels = data['labels'].to(device)
            phoneme_label_lengths = data['label_lengths'].to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms, attention_mask=attention_mask, task='phoneme')
                input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
                
                phoneme_loss, phoneme_loss_dict = criterion(
                    outputs,
                    phoneme_targets=phoneme_labels,
                    phoneme_input_lengths=input_lengths,
                    phoneme_target_lengths=phoneme_label_lengths
                )
                
                accumulated_loss += phoneme_loss / gradient_accumulation
                phoneme_loss_sum += phoneme_loss_dict.get('phoneme_loss', 0.0)
                phoneme_count += 1
        
        if accumulated_loss > 0:
            scaler.scale(accumulated_loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation == 0:
            scaler.step(wav2vec_optimizer)
            scaler.step(main_optimizer)
            scaler.update()
            wav2vec_optimizer.zero_grad()
            main_optimizer.zero_grad()
            
            total_loss += accumulated_loss.item() * gradient_accumulation
        
        avg_total = total_loss / max(((batch_idx + 1) // gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        
        progress_bar.set_postfix({
            'Total': f'{avg_total:.4f}',
            'Error': f'{avg_error:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        })
    
    return total_loss / (len(dataloader) // gradient_accumulation)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            accumulated_loss = 0.0
            
            if 'error' in batch_data:
                data = batch_data['error']
                waveforms = data['waveforms'].to(device)
                audio_lengths = data['audio_lengths'].to(device)
                error_labels = data['labels'].to(device)
                error_label_lengths = data['label_lengths'].to(device)
                
                attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
                attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
                
                outputs = model(waveforms, attention_mask=attention_mask, task='error')
                input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
                
                error_loss, _ = criterion(
                    outputs,
                    error_targets=error_labels,
                    error_input_lengths=input_lengths,
                    error_target_lengths=error_label_lengths
                )
                accumulated_loss += error_loss
            
            if 'phoneme' in batch_data:
                data = batch_data['phoneme']
                waveforms = data['waveforms'].to(device)
                audio_lengths = data['audio_lengths'].to(device)
                phoneme_labels = data['labels'].to(device)
                phoneme_label_lengths = data['label_lengths'].to(device)
                
                attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
                attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
                
                outputs = model(waveforms, attention_mask=attention_mask, task='phoneme')
                input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
                input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
                
                phoneme_loss, _ = criterion(
                    outputs,
                    phoneme_targets=phoneme_labels,
                    phoneme_input_lengths=input_lengths,
                    phoneme_target_lengths=phoneme_label_lengths
                )
                accumulated_loss += phoneme_loss
            
            total_loss += accumulated_loss.item()
            progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(dataloader)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, wav2vec_opt, main_opt, scheduler, epoch, val_loss, train_loss, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
        'main_optimizer_state_dict': main_opt.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Override config values in format key=value')
    parser.add_argument('--train_data', type=str, help='Override train data path')
    parser.add_argument('--val_data', type=str, help='Override validation data path')
    parser.add_argument('--eval_data', type=str, help='Override evaluation data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
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
    
    if args.config:
        for override in args.config.split(','):
            key, value = override.split('=')
            if hasattr(config, key):
                attr_type = type(getattr(config, key))
                setattr(config, key, attr_type(value))
    
    seed_everything(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    model = SimpleMultiTaskModel(
        pretrained_model_name=config.pretrained_model,
        hidden_dim=config.hidden_dim,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        dropout=config.dropout
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(config.device)
    
    criterion = MultiTaskLoss(
        error_weight=config.error_weight,
        phoneme_weight=config.phoneme_weight
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
    
    scheduler = ReduceLROnPlateau(main_optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)
    
    train_dataset = MultiTaskDataset(
        config.train_data, phoneme_to_id, 
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        task_mode=config.task_mode,
        error_task_ratio=config.error_task_ratio
    )
    
    val_dataset = MultiTaskDataset(
        config.val_data, phoneme_to_id, 
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        task_mode=config.task_mode,
        error_task_ratio=config.error_task_ratio
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        collate_fn=multitask_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=multitask_collate_fn
    )
    
    eval_dataset = EvaluationDataset(
        config.eval_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn
    )
    
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    
    logger.info(f"Starting training for {config.num_epochs} epochs")
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            config.device, epoch, scaler, config.gradient_accumulation
        )
        
        val_loss = validate_epoch(model, val_dataloader, criterion, config.device)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        error_detection_results = evaluate_error_detection(model, eval_dataloader, config.device, error_type_names)
        phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, config.device, id_to_phoneme)
        
        current_error_accuracy = error_detection_results['token_accuracy']
        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        
        logger.info(f"Error Accuracy: {current_error_accuracy:.4f}")
        logger.info(f"Phoneme Accuracy: {current_phoneme_accuracy:.4f}")
        
        metrics = {
            'error_accuracy': current_error_accuracy,
            'phoneme_accuracy': current_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        
        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            model_path = os.path.join(config.output_dir, 'best_error.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, scheduler, 
                          epoch, val_loss, train_loss, metrics, model_path)
            
        if config.save_best_phoneme and current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, scheduler, 
                          epoch, val_loss, train_loss, metrics, model_path)
        
        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, scheduler, 
                          epoch, val_loss, train_loss, metrics, model_path)
    
    logger.info("Training completed!")
    logger.info(f"Best Error Accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f}")

if __name__ == "__main__":
    main()