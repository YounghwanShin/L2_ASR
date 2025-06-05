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
from phoneme_data_prepare import PhonemeDataset, PhonemeEvaluationDataset, phoneme_collate_fn, phoneme_evaluation_collate_fn
from phoneme_evaluate import evaluate_phoneme_recognition, get_wav2vec2_output_lengths_official, decode_ctc

logger = logging.getLogger(__name__)

def get_phoneme_model_class(model_type):
    if model_type == 'simple':
        from models.phoneme_model import SimplePhonemeModel, PhonemeLoss
        return SimplePhonemeModel, PhonemeLoss
    elif model_type == 'transformer':
        from models.phoneme_model_transformer import TransformerPhonemeModel, PhonemeLoss
        return TransformerPhonemeModel, PhonemeLoss
    else:
        raise ValueError(f"Unknown phoneme model type: {model_type}. Available: simple, transformer")

def setup_experiment_dirs(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    config_path = os.path.join(config.experiment_dir, 'config.json')
    config.save_config(config_path)
    
    log_file = os.path.join(config.log_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def show_sample_predictions(model, eval_dataloader, device, id_to_phoneme, num_samples=3):
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for batch_data in eval_dataloader:
            if samples_shown >= num_samples:
                break
                
            (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, perceived_lengths, canonical_lengths, wav_files) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            max_len = input_lengths.max().item()
            attention_mask = torch.arange(max_len).expand(waveforms.shape[0], max_len).to(device)
            attention_mask = (attention_mask < input_lengths.unsqueeze(1))
            
            outputs = model(waveforms, attention_mask)
            phoneme_logits = outputs['phoneme_logits']
            
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            phoneme_log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            phoneme_predictions = decode_ctc(phoneme_log_probs, phoneme_input_lengths)
            
            for i in range(min(waveforms.shape[0], num_samples - samples_shown)):
                logger.info(f"\n--- Phoneme Sample {samples_shown + 1} ---")
                logger.info(f"File: {wav_files[i]}")
                
                phoneme_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                for pid in perceived_phoneme_ids[i][:perceived_lengths[i]]]
                phoneme_pred = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                              for pid in phoneme_predictions[i]]
                
                logger.info(f"Phoneme Actual:    {' '.join(phoneme_actual)}")
                logger.info(f"Phoneme Predicted: {' '.join(phoneme_pred)}")
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
            
            if samples_shown >= num_samples:
                break

def train_epoch(model, dataloader, criterion, wav2vec_optimizer, main_optimizer, 
                device, epoch, scaler, gradient_accumulation=1):
    model.train()
    
    total_loss = 0.0
    phoneme_loss_sum = 0.0
    phoneme_count = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue
            
        waveforms = batch_data['waveforms'].to(device)
        audio_lengths = batch_data['audio_lengths'].to(device)
        phoneme_labels = batch_data['phoneme_labels'].to(device)
        phoneme_label_lengths = batch_data['phoneme_lengths'].to(device)
        
        input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
        max_len = input_lengths.max().item()
        attention_mask = torch.arange(max_len).expand(waveforms.shape[0], max_len).to(device)
        attention_mask = (attention_mask < input_lengths.unsqueeze(1))
        
        with torch.amp.autocast('cuda'):
            outputs = model(waveforms, attention_mask=attention_mask)
            input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
            
            phoneme_loss, phoneme_loss_dict = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=input_lengths,
                phoneme_target_lengths=phoneme_label_lengths
            )
            
            accumulated_loss = phoneme_loss / gradient_accumulation
            phoneme_loss_sum += phoneme_loss_dict.get('phoneme_loss', 0.0)
            phoneme_count += 1
        
        scaler.scale(accumulated_loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation == 0:
            scaler.step(wav2vec_optimizer)
            scaler.step(main_optimizer)
            scaler.update()
            wav2vec_optimizer.zero_grad()
            main_optimizer.zero_grad()
            
            total_loss += accumulated_loss.item() * gradient_accumulation
        
        del waveforms, audio_lengths, phoneme_labels, phoneme_label_lengths
        del attention_mask, outputs, phoneme_loss, accumulated_loss
        
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()
        
        avg_total = total_loss / max(((batch_idx + 1) // gradient_accumulation), 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        
        progress_bar.set_postfix({
            'Total': f'{avg_total:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        })
    
    torch.cuda.empty_cache()
    return total_loss / (len(dataloader) // gradient_accumulation)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
                
            waveforms = batch_data['waveforms'].to(device)
            audio_lengths = batch_data['audio_lengths'].to(device)
            phoneme_labels = batch_data['phoneme_labels'].to(device)
            phoneme_label_lengths = batch_data['phoneme_lengths'].to(device)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            max_len = input_lengths.max().item()
            attention_mask = torch.arange(max_len).expand(waveforms.shape[0], max_len).to(device)
            attention_mask = (attention_mask < input_lengths.unsqueeze(1))
            
            outputs = model(waveforms, attention_mask=attention_mask)
            input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
            
            phoneme_loss, _ = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=input_lengths,
                phoneme_target_lengths=phoneme_label_lengths
            )
            
            total_loss += phoneme_loss.item()
            
            del waveforms, audio_lengths, phoneme_labels, phoneme_label_lengths
            del attention_mask, outputs, phoneme_loss
            
            progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})
    
    torch.cuda.empty_cache()
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
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
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
    
    config.__post_init__()
    
    if config.experiment_name and '_' in config.experiment_name:
        timestamp_part = config.experiment_name.split('_', 1)[-1]
        config.experiment_name = f"phoneme_{config.model_type}_{timestamp_part}"
    else:
        config.experiment_name = f"phoneme_{config.model_type}_{config.experiment_name}"
    
    config.experiment_dir = os.path.join(config.base_experiment_dir, config.experiment_name)
    config.checkpoint_dir = os.path.join(config.experiment_dir, 'checkpoints')
    config.log_dir = os.path.join(config.experiment_dir, 'logs')
    config.result_dir = os.path.join(config.experiment_dir, 'results')
    config.output_dir = config.checkpoint_dir
    
    seed_everything(config.seed)
    setup_experiment_dirs(config)
    
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    model_class, loss_class = get_phoneme_model_class(config.model_type)
    
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        **config.get_model_config()
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(config.device)
    torch.cuda.empty_cache()
    
    criterion = loss_class()
    
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
    
    train_dataset = PhonemeDataset(
        config.train_data, phoneme_to_id, 
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    
    val_dataset = PhonemeDataset(
        config.val_data, phoneme_to_id, 
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        collate_fn=phoneme_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=phoneme_collate_fn
    )
    
    eval_dataset = PhonemeEvaluationDataset(
        config.eval_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=phoneme_evaluation_collate_fn
    )
    
    best_val_loss = float('inf')
    best_phoneme_accuracy = 0.0
    
    logger.info(f"Starting phoneme-only training with model type: {config.model_type}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Starting training for {config.num_epochs} epochs")
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            config.device, epoch, scaler, config.gradient_accumulation
        )
        
        val_loss = validate_epoch(model, val_dataloader, criterion, config.device)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch} - Sample Predictions")
            logger.info("=" * 50)
            show_sample_predictions(model, eval_dataloader, config.device, id_to_phoneme)
            torch.cuda.empty_cache()
        
        logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
        phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, config.device, id_to_phoneme)
        torch.cuda.empty_cache()
        
        logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
        
        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        
        metrics = {
            'phoneme_accuracy': current_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        
        if current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, scheduler, 
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best phoneme accuracy: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, scheduler, 
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        torch.cuda.empty_cache()
    
    final_metrics = {
        'best_phoneme_accuracy': best_phoneme_accuracy,
        'best_val_loss': best_val_loss
    }
    
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("Phoneme-only training completed!")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f}")

if __name__ == "__main__":
    main()