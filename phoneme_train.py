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

from config import Config
from phoneme_data_prepare import PhonemeDataset, PhonemeEvaluationDataset, phoneme_collate_fn, phoneme_evaluation_collate_fn
from phoneme_evaluate import evaluate_phoneme_recognition, get_wav2vec2_output_lengths_official, decode_ctc

logger = logging.getLogger(__name__)

def make_attn_mask(wavs, wav_lens):
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

def enable_wav2vec2_specaug(model, enable=True):
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model.encoder.wav2vec2, 'config'):
        actual_model.encoder.wav2vec2.config.apply_spec_augment = enable

def get_phoneme_model_class(model_type):
    if model_type == 'simple':
        from models.phoneme_model import SimplePhonemeModel, PhonemeLoss
        return SimplePhonemeModel, PhonemeLoss
    elif model_type == 'transformer':
        from models.phoneme_model_transformer import TransformerPhonemeModel, PhonemeLoss
        return TransformerPhonemeModel, PhonemeLoss
    else:
        raise ValueError(f"Unknown phoneme model type: {model_type}. Available: simple, transformer")

def detect_phoneme_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'

def setup_experiment_dirs(config, resume=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    config_path = os.path.join(config.experiment_dir, 'config.json')
    if not resume:
        config.save_config(config_path)
    
    log_file = os.path.join(config.log_dir, 'training.log')
    file_mode = 'a' if resume else 'w'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ]
    )

def show_sample_predictions(model, eval_dataloader, device, id_to_phoneme, num_samples=3):
    model.eval()
    enable_wav2vec2_specaug(model, False)
    samples_shown = 0
    
    with torch.no_grad():
        for batch_data in eval_dataloader:
            if samples_shown >= num_samples:
                break
                
            (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
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
                device, epoch, scaler, gradient_accumulation=1, config=None):
    model.train()
    if config and config.wav2vec2_specaug:
        enable_wav2vec2_specaug(model, True)
    
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
        wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
        attention_mask = make_attn_mask(waveforms, wav_lens_norm)
        
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
            phoneme_label_lengths = batch_data['phoneme_lengths'].to(device)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
            outputs = model(waveforms, attention_mask=attention_mask)
            input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
            
            phoneme_loss, _ = criterion(
                outputs,
                phoneme_targets=phoneme_labels,
                phoneme_input_lengths=input_lengths,
                phoneme_target_lengths=phoneme_label_lengths
            )
            
            total_loss += phoneme_loss.item()
            
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

def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
        'main_optimizer_state_dict': main_opt.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'metrics': metrics
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
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)
    
    if args.resume:
        detected_model_type = detect_phoneme_model_type_from_checkpoint(args.resume)
        config.model_type = detected_model_type
        logger.info(f"Auto-detected model type from checkpoint: {detected_model_type}")
    
    if args.config:
        for override in args.config.split(','):
            key, value = override.split('=')
            if hasattr(config, key):
                attr_type = type(getattr(config, key))
                setattr(config, key, attr_type(value))
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)
    elif config.experiment_name is None:
        model_prefix = 'phoneme_simple' if config.model_type == 'simple' else f'phoneme_{config.model_type}'
        config.experiment_name = model_prefix
    elif not config.experiment_name.startswith('phoneme_'):
        model_prefix = 'phoneme_simple' if config.model_type == 'simple' else f'phoneme_{config.model_type}'
        config.experiment_name = model_prefix
    
    config.experiment_dir = os.path.join(config.base_experiment_dir, config.experiment_name)
    config.checkpoint_dir = os.path.join(config.experiment_dir, 'checkpoints')
    config.log_dir = os.path.join(config.experiment_dir, 'logs')
    config.result_dir = os.path.join(config.experiment_dir, 'results')
    config.output_dir = config.checkpoint_dir
    
    seed_everything(config.seed)
    setup_experiment_dirs(config, resume=bool(args.resume))
    
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
    start_epoch = 1
    
    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(
            args.resume, model, wav2vec_optimizer, main_optimizer, config.device
        )
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        
        checkpoint = torch.load(args.resume, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        logger.info("=" * 50)
        logger.info("RESUMING PHONEME TRAINING")
        logger.info("=" * 50)
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best phoneme accuracy so far: {best_phoneme_accuracy:.4f}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"Starting phoneme-only training with model type: {config.model_type}")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"SpecAugment enabled: {config.wav2vec2_specaug}")
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_dataloader, criterion, wav2vec_optimizer, main_optimizer,
            config.device, epoch, scaler, config.gradient_accumulation, config
        )
        
        val_loss = validate_epoch(model, val_dataloader, criterion, config.device)
        
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
        if 'mpd_f1' in phoneme_recognition_results:
            logger.info(f"MPD F1 Score: {phoneme_recognition_results['mpd_f1']:.4f}")
        
        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        
        if current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            metrics = {
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            if 'mpd_f1' in phoneme_recognition_results:
                metrics['mpd_f1'] = phoneme_recognition_results['mpd_f1']
            
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, 
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best phoneme accuracy: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            if 'mpd_f1' in phoneme_recognition_results:
                metrics['mpd_f1'] = phoneme_recognition_results['mpd_f1']
            
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer, 
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        latest_metrics = {
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        if 'mpd_f1' in phoneme_recognition_results:
            latest_metrics['mpd_f1'] = phoneme_recognition_results['mpd_f1']
        
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer, 
                      epoch, val_loss, train_loss, latest_metrics, latest_path)
        
        torch.cuda.empty_cache()
    
    final_metrics = {
        'best_phoneme_accuracy': best_phoneme_accuracy,
        'best_val_loss': best_val_loss,
        'completed_epochs': config.num_epochs,
        'model_type': f"phoneme_{config.model_type}",
        'experiment_name': config.experiment_name
    }
    
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("Phoneme-only training completed!")
    logger.info(f"Best Phoneme Accuracy: {best_phoneme_accuracy:.4f}")
    logger.info(f"Final metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()