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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MultiTaskModel, MultiTaskLoss
from data_prepare import MultiTaskDataset, EvaluationDataset, multitask_collate_fn, evaluation_collate_fn
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition, show_multitask_samples, get_wav2vec2_output_lengths_official

def train_multitask_epoch(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=0.5):
    model.train()
    
    total_loss = 0.0
    error_loss_sum = 0.0
    phoneme_loss_sum = 0.0
    error_count = 0
    phoneme_count = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Multi-task]')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        optimizer.zero_grad()
        
        batch_total_loss = 0.0
        
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
            
            error_loss, error_loss_dict = criterion(
                outputs, 
                error_targets=error_labels,
                error_input_lengths=input_lengths,
                error_target_lengths=error_label_lengths
            )
            
            batch_total_loss += error_loss
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
            
            outputs = model(waveforms, attention_mask=attention_mask, task='phoneme')
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
            
            phoneme_loss, phoneme_loss_dict = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=input_lengths,
                phoneme_target_lengths=phoneme_label_lengths
            )
            
            batch_total_loss += phoneme_loss
            phoneme_loss_sum += phoneme_loss_dict.get('phoneme_loss', 0.0)
            phoneme_count += 1
        
        if batch_total_loss > 0:
            batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += batch_total_loss.item()
        
        avg_total = total_loss / (batch_idx + 1) if batch_idx > 0 else 0
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        
        progress_bar.set_postfix({
            'Total': f'{avg_total:.4f}',
            'Error': f'{avg_error:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        })
    
    return total_loss / len(dataloader)

def validate_multitask_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation [Multi-task]')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_total_loss = 0.0
            
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
                batch_total_loss += error_loss
            
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
                batch_total_loss += phoneme_loss
            
            total_loss += batch_total_loss.item()
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

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, train_loss, best_val_loss, best_accuracy, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_accuracy': best_accuracy,
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)

def main():
    parser = argparse.ArgumentParser(description='Multi-task L2 Pronunciation Training')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--phoneme_map', type=str, required=True)
    
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_phonemes', type=int, default=42)
    parser.add_argument('--num_error_types', type=int, default=3)
    parser.add_argument('--use_cross_attention', action='store_true', default=True)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--max_audio_length', type=int, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    
    parser.add_argument('--error_weight', type=float, default=1.0)
    parser.add_argument('--phoneme_weight', type=float, default=1.0)
    parser.add_argument('--adaptive_weights', action='store_true')
    parser.add_argument('--error_task_ratio', type=float, default=0.5)
    
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_patience', type=int, default=3)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--evaluate_every_epoch', action='store_true', default=True)
    parser.add_argument('--show_samples', action='store_true')
    parser.add_argument('--num_sample_show', type=int, default=3)
    
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'multitask_train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    with open(os.path.join(args.result_dir, 'multitask_hyperparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    logger.info("Initializing multi-task model")
    model = MultiTaskModel(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        num_error_types=args.num_error_types,
        use_cross_attention=args.use_cross_attention
    )
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    criterion = MultiTaskLoss(
        error_weight=args.error_weight,
        phoneme_weight=args.phoneme_weight,
        adaptive_weights=args.adaptive_weights
    )
    
    if args.adaptive_weights:
        criterion = criterion.to(args.device)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()), 
            lr=args.learning_rate
        )
        logger.info(f"Using adaptive loss weights - Error: {args.error_weight}, Phoneme: {args.phoneme_weight}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        logger.info(f"Using fixed loss weights - Error: {args.error_weight}, Phoneme: {args.phoneme_weight}")
    
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=args.scheduler_factor,
            patience=args.scheduler_patience, threshold=0.001,
            threshold_mode='rel', cooldown=1, min_lr=1e-6
        )
        logger.info("Learning rate scheduler initialized")
    
    logger.info("Creating multi-task datasets")
    train_dataset = MultiTaskDataset(
        args.train_data, phoneme_to_id, max_length=args.max_audio_length,
        task_mode='both', error_task_ratio=args.error_task_ratio
    )
    
    val_dataset = MultiTaskDataset(
        args.val_data, phoneme_to_id, max_length=args.max_audio_length,
        task_mode='both', error_task_ratio=args.error_task_ratio
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=multitask_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=multitask_collate_fn
    )
    
    eval_dataloader = None
    if args.evaluate_every_epoch:
        eval_dataset = EvaluationDataset(
            args.eval_data, phoneme_to_id, max_length=args.max_audio_length
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=evaluation_collate_fn
        )
    
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    
    logger.info(f"Starting multi-task training for {args.num_epochs} epochs")
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs} starting")
        
        train_loss = train_multitask_epoch(
            model, train_dataloader, criterion, optimizer, args.device, epoch, args.max_grad_norm
        )
        
        val_loss = validate_multitask_epoch(model, val_dataloader, criterion, args.device)
        
        if scheduler is not None:
            scheduler.step(val_loss)
            logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if args.show_samples and eval_dataloader:
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch} - Sample Predictions")
            logger.info(f"{'='*50}")
            show_multitask_samples(model, eval_dataloader, args.device, error_type_names, id_to_phoneme, args.num_sample_show)
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        current_error_accuracy = 0.0
        current_phoneme_accuracy = 0.0
        
        if args.evaluate_every_epoch and eval_dataloader:
            logger.info(f"Epoch {epoch}: Evaluating error detection...")
            error_detection_results = evaluate_error_detection(model, eval_dataloader, args.device, error_type_names)
            
            current_error_accuracy = error_detection_results['token_accuracy']
            logger.info(f"Error Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"Error Weighted F1: {error_detection_results['weighted_f1']:.4f}")
            
            for error_type, metrics in error_detection_results['class_metrics'].items():
                logger.info(f"  {error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            
            epoch_metrics['error_detection'] = error_detection_results
            
            logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
            phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, args.device, id_to_phoneme)
            
            current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
            logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
            logger.info(f"Phoneme Accuracy: {current_phoneme_accuracy:.4f}")
            
            epoch_metrics['phoneme_recognition'] = {
                'per': phoneme_recognition_results['per'],
                'accuracy': current_phoneme_accuracy,
                'total_phonemes': phoneme_recognition_results['total_phonemes'],
                'total_errors': phoneme_recognition_results['total_errors'],
                'insertions': phoneme_recognition_results['insertions'],
                'deletions': phoneme_recognition_results['deletions'],
                'substitutions': phoneme_recognition_results['substitutions']
            }
        
        with open(os.path.join(args.result_dir, f'multitask_epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_metrics, f, indent=4)
        
        if current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            model_path = os.path.join(args.output_dir, 'best_multitask_error.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, train_loss, best_val_loss, best_error_accuracy, model_path)
            logger.info(f"New best error accuracy: {current_error_accuracy:.4f}")
            
        if current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            model_path = os.path.join(args.output_dir, 'best_multitask_phoneme.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, train_loss, best_val_loss, best_phoneme_accuracy, model_path)
            logger.info(f"New best phoneme accuracy: {current_phoneme_accuracy:.4f} (PER: {1.0-current_phoneme_accuracy:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, 'best_multitask_loss.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, train_loss, best_val_loss, 0.0, model_path)
            logger.info(f"New best validation loss: {val_loss:.4f}")
        
        last_model_path = os.path.join(args.output_dir, 'last_multitask.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, train_loss, best_val_loss, 0.0, last_model_path)
    
    logger.info("Multi-task training completed!")
    logger.info(f"Best Error Accuracy: {best_error_accuracy:.4f}")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f} (PER: {1.0-best_phoneme_accuracy:.4f})")

if __name__ == "__main__":
    main()