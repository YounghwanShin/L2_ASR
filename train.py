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
from torch.utils.data import DataLoader, Dataset
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio

from model import ErrorDetectionModel, PhonemeRecognitionModel
from data import ErrorLabelDataset, PhonemeRecognitionDataset, EvaluationDataset
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition, decode_ctc, collate_fn

class BalancedErrorLabelDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000, oversample_incorrect=True):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        self.oversample_incorrect = oversample_incorrect
        
        if oversample_incorrect:
            self._balance_dataset()
        
    def _balance_dataset(self):
        incorrect_files = []
        for wav_file in self.wav_files:
            error_labels = self.data[wav_file].get('error_labels', '').split()
            if 'I' in error_labels:
                incorrect_files.append(wav_file)
        
        # Oversample files with incorrect labels by 3x
        self.wav_files.extend(incorrect_files * 2)
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        if self.max_length and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        error_labels = item.get('error_labels', '').split()
        modified_labels = [self.error_mapping[label] for label in error_labels]
        
        return (
            waveform.squeeze(0),
            torch.tensor(modified_labels, dtype=torch.long),
            torch.tensor(len(modified_labels), dtype=torch.long),
            wav_file
        )

def get_wav2vec2_output_lengths_official(model, input_lengths):
    actual_model = model.module if hasattr(model, 'module') else model
    
    if hasattr(actual_model, 'encoder'):
        wav2vec_model = actual_model.encoder.wav2vec2
    elif hasattr(actual_model, 'error_model'):
        wav2vec_model = actual_model.error_model.encoder.wav2vec2
    else:
        wav2vec_model = actual_model
    
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)

def compute_entropy_regularization(log_probs, input_lengths):
    batch_size = log_probs.size(1)
    total_entropy = 0.0
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        probs = torch.exp(log_probs[:seq_len, b, :])
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        total_entropy += torch.mean(entropy)
    
    return total_entropy / batch_size

class AdaptiveEntropyRegularizer(nn.Module):
    def __init__(self, initial_beta=0.02, target_entropy_factor=0.6):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.target_entropy_factor = target_entropy_factor
        
    def get_target_entropy(self):
        return self.target_entropy_factor * torch.log(torch.tensor(4.0))
    
    def compute_loss(self, entropy):
        target_entropy = self.get_target_entropy()
        return self.beta * (target_entropy - entropy)

def generic_collate_fn(batch):
    if len(batch[0]) == 4:
        waveforms, labels, label_lengths, wav_files = zip(*batch)
    else:
        waveforms, labels, label_lengths, wav_files = zip(*batch)
    
    max_audio_len = max(waveform.shape[0] for waveform in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_audio_len - waveform.shape[0]))
        for waveform in waveforms
    ])
    
    max_label_len = max(labels_tensor.shape[0] for labels_tensor in labels)
    padded_labels = torch.stack([
        torch.nn.functional.pad(labels_tensor, (0, max_label_len - labels_tensor.shape[0]), value=0)
        for labels_tensor in labels
    ])
    
    return padded_waveforms, padded_labels, torch.tensor([waveform.shape[0] for waveform in waveforms]), torch.tensor(label_lengths), wav_files

def show_samples(model, dataloader, device, id_mapping, num_samples=3, mode='error'):
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (waveforms, labels, audio_lengths, label_lengths, wav_files) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            if mode == 'error':
                logits = model(waveforms, attention_mask)
            else:
                logits, _ = model(waveforms, attention_mask)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
            
            log_probs = torch.log_softmax(logits, dim=-1)
            greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
            predictions = []
            
            for b in range(greedy_preds.shape[0]):
                seq = []
                prev = 0
                actual_length = input_lengths[b].item()
                
                for t in range(min(greedy_preds.shape[1], actual_length)):
                    pred = greedy_preds[b, t]
                    if pred != 0 and pred != prev:
                        seq.append(int(pred))
                    prev = pred
                predictions.append(seq)
            
            if mode == 'error':
                targets = []
                for labels_tensor, length in zip(labels, label_lengths):
                    target_seq = labels_tensor[:length].cpu().numpy().tolist()
                    targets.append(target_seq)
                
                pred = predictions[0]
                target = targets[0]
                
                pred_symbols = [id_mapping.get(p, str(p)) for p in pred]
                target_symbols = [id_mapping.get(t, str(t)) for t in target]
            else:
                pred = predictions[0]
                target = labels[0][:label_lengths[0]].cpu().numpy().tolist()
                
                pred_symbols = [id_mapping.get(str(p), f"UNK({p})") for p in pred]
                target_symbols = [id_mapping.get(str(t), f"UNK({t})") for t in target]
            
            print(f"\n--- {mode.title()} Sample {batch_idx + 1} ---")
            print(f"File: {wav_files[0]}")
            print(f"Actual:  {' '.join(target_symbols)}")
            print(f"Predicted:  {' '.join(pred_symbols)}")
            print(f"Match:  {'✓' if pred == target else '✗'}")
            
            if len(target) > 0 and len(pred) > 0:
                correct = sum(1 for p, t in zip(pred, target) if p == t)
                accuracy = correct / max(len(target), len(pred))
                print(f"Token Accuracy: {accuracy:.3f}")
    
    model.train()

def train_model(model, dataloader, criterion, optimizer_step_fn, optimizer_zero_grad_fn, device, epoch, mode, max_grad_norm=0.5, entropy_regularizer=None):
    model.train()
    running_loss = running_ctc_loss = running_entropy_value = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [{mode}]')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        waveforms, labels, audio_lengths, label_lengths, _ = batch_data
            
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        audio_lengths = audio_lengths.to(device)
        label_lengths = label_lengths.to(device)
        
        attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
        attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
        
        if mode == 'error':
            logits = model(waveforms, attention_mask)
        else:
            logits, _ = model(waveforms, attention_mask)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
        input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
        
        ctc_loss = criterion(log_probs.transpose(0, 1), labels, input_lengths, label_lengths)
        total_loss = ctc_loss
        
        optimizer_zero_grad_fn()
        
        entropy_value = torch.tensor(0.0)
        if entropy_regularizer is not None and mode == 'error':
            entropy_value = compute_entropy_regularization(log_probs.transpose(0, 1), input_lengths)
            entropy_reg_loss = entropy_regularizer.compute_loss(entropy_value)
            total_loss = ctc_loss + entropy_reg_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if entropy_regularizer is not None:
            torch.nn.utils.clip_grad_norm_(entropy_regularizer.parameters(), 0.1)
        
        optimizer_step_fn()
        
        running_loss += total_loss.item()
        running_ctc_loss += ctc_loss.item()
        running_entropy_value += entropy_value.item()
        
        progress_bar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'CTC': f'{running_ctc_loss / (batch_idx + 1):.4f}',
            'Entropy': f'{running_entropy_value / (batch_idx + 1):.3f}'
        })
    
    return running_loss / len(dataloader)

def validate_model(model, dataloader, criterion, device, mode):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation [{mode}]')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            waveforms, labels, audio_lengths, label_lengths, _ = batch_data
                
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            audio_lengths = audio_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1).to(device)).float()
            
            if mode == 'error':
                logits = model(waveforms, attention_mask)
            else:
                logits, _ = model(waveforms, attention_mask)
            
            log_probs = torch.log_softmax(logits, dim=-1)
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
            
            loss = criterion(log_probs.transpose(0, 1), labels, input_lengths, label_lengths)
            running_loss += loss.item()
            
            progress_bar.set_postfix({'Val_Loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(dataloader)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, path, best_accuracy=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    
    if best_accuracy is not None:
        checkpoint['best_accuracy'] = best_accuracy
    
    if entropy_regularizer is not None:
        checkpoint['beta_optimizer_state_dict'] = beta_optimizer.state_dict()
        checkpoint['entropy_regularizer_state_dict'] = entropy_regularizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, entropy_regularizer, beta_optimizer, device, logger):
    checkpoint = torch.load(path, map_location=device)
    
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state restored")
    if 'beta_optimizer_state_dict' in checkpoint and entropy_regularizer is not None:
        beta_optimizer.load_state_dict(checkpoint['beta_optimizer_state_dict'])
        logger.info("Beta optimizer state restored")
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state restored")
    if 'entropy_regularizer_state_dict' in checkpoint and entropy_regularizer is not None:
        entropy_regularizer.load_state_dict(checkpoint['entropy_regularizer_state_dict'])
        logger.info("Entropy regularizer state restored")
    
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
    return checkpoint['epoch'] + 1, checkpoint['best_val_loss'], best_accuracy

def main():
    parser = argparse.ArgumentParser(description='L2 Pronunciation Error Detection and Phoneme Recognition Model Training')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['error', 'phoneme'], required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--error_train_data', type=str)
    parser.add_argument('--error_val_data', type=str)
    parser.add_argument('--phoneme_train_data', type=str)
    parser.add_argument('--phoneme_val_data', type=str)
    parser.add_argument('--phoneme_map', type=str)
    
    parser.add_argument('--eval_data', type=str)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--evaluate_every_epoch', action='store_true')
    
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_phonemes', type=int, default=42)
    parser.add_argument('--num_error_types', type=int, default=4)
    parser.add_argument('--error_model_checkpoint', type=str, default=None)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_audio_length', type=int, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--show_samples', action='store_true')
    parser.add_argument('--num_sample_show', type=int, default=3)
    
    parser.add_argument('--use_entropy_reg', action='store_true')
    parser.add_argument('--initial_beta', type=float, default=0.02)
    parser.add_argument('--target_entropy_factor', type=float, default=0.6)
    
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_patience', type=int, default=3)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--resume_from', type=str, default=None)
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
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
    
    with open(os.path.join(args.result_dir, f'hyperparams_{args.mode}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    phoneme_to_id = None
    id_to_phoneme = {}
    if args.phoneme_map:
        with open(args.phoneme_map, 'r') as f:
            phoneme_to_id = json.load(f)
        id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    if args.mode == 'error':
        logger.info("Initializing error detection model")
        model = ErrorDetectionModel(
            pretrained_model_name=args.pretrained_model,
            hidden_dim=args.hidden_dim,
            num_error_types=args.num_error_types
        )
    elif args.mode == 'phoneme':
        logger.info("Initializing phoneme recognition model")
        if not args.error_model_checkpoint:
            logger.error("Error model checkpoint required for phoneme recognition mode.")
            sys.exit(1)
            
        model = PhonemeRecognitionModel(
            pretrained_model_name=args.pretrained_model,
            error_model_checkpoint=args.error_model_checkpoint,
            hidden_dim=args.hidden_dim,
            num_phonemes=args.num_phonemes,
            num_error_types=args.num_error_types
        )
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    entropy_regularizer = None
    if args.use_entropy_reg and args.mode == 'error':
        entropy_regularizer = AdaptiveEntropyRegularizer(
            initial_beta=args.initial_beta,
            target_entropy_factor=args.target_entropy_factor
        )
        entropy_regularizer = entropy_regularizer.to(args.device)
        logger.info(f"Using adaptive entropy regularization - beta: {args.initial_beta}, target: {args.target_entropy_factor}")
    
    start_epoch = 1
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    if args.resume_from:
        logger.info(f"Resuming training from {args.resume_from}")
    elif args.model_checkpoint:
        logger.info(f"Loading checkpoint from {args.model_checkpoint}")
        state_dict = torch.load(args.model_checkpoint, map_location=args.device)
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    beta_optimizer = None
    if entropy_regularizer is not None:
        beta_optimizer = optim.AdamW(entropy_regularizer.parameters(), lr=args.learning_rate * 0.1)
        def optimizer_step():
            optimizer.step()
            beta_optimizer.step()
        def optimizer_zero_grad():
            optimizer.zero_grad()
            beta_optimizer.zero_grad()
    else:
        def optimizer_step():
            optimizer.step()
        def optimizer_zero_grad():
            optimizer.zero_grad()
    
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=0.001,
            threshold_mode='rel',
            cooldown=1,
            min_lr=1e-6
        )
        logger.info("Learning rate scheduler (ReduceLROnPlateau) initialized")
    
    if args.resume_from:
        start_epoch, best_val_loss, best_accuracy = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, entropy_regularizer, beta_optimizer, args.device, logger
        )
        logger.info(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}, best accuracy: {best_accuracy:.4f}")
    
    eval_dataloader = None
    if args.evaluate_every_epoch and args.eval_data:
        logger.info(f"Loading evaluation dataset: {args.eval_data}")
        
        if args.mode == 'phoneme' and not phoneme_to_id:
            logger.error("Phoneme mapping required for phoneme recognition mode.")
            sys.exit(1)
        
        eval_dataset = EvaluationDataset(args.eval_data, phoneme_to_id, max_length=args.max_audio_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    if args.mode == 'error':
        if not args.error_train_data or not args.error_val_data:
            logger.error("Error training and validation data paths required for error detection mode.")
            sys.exit(1)
            
        train_dataset = BalancedErrorLabelDataset(args.error_train_data, max_length=args.max_audio_length, oversample_incorrect=True)
        val_dataset = ErrorLabelDataset(args.error_val_data, max_length=args.max_audio_length)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=generic_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generic_collate_fn)
        
        for epoch in range(start_epoch, args.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{args.num_epochs} starting")
            
            train_loss = train_model(model, train_dataloader, criterion, optimizer_step, optimizer_zero_grad, args.device, epoch, 'error', args.max_grad_norm, entropy_regularizer)
            val_loss = validate_model(model, val_dataloader, criterion, args.device, 'error')
            
            if scheduler is not None:
                scheduler.step(val_loss)
                logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            if entropy_regularizer is not None:
                logger.info(f"Current beta: {entropy_regularizer.beta.item():.4f}")
                logger.info(f"Target entropy: {entropy_regularizer.get_target_entropy().item():.3f}")
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if args.show_samples:
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch} - Sample Predictions")
                logger.info(f"{'='*50}")
                show_samples(model, val_dataloader, args.device, error_type_names, args.num_sample_show, 'error')
            
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'beta': entropy_regularizer.beta.item() if entropy_regularizer is not None else None
            }
            
            current_accuracy = 0.0
            if args.evaluate_every_epoch and eval_dataloader:
                logger.info(f"Epoch {epoch}: Evaluating error detection...")
                error_detection_results = evaluate_error_detection(model, eval_dataloader, args.device, error_type_names)
                
                current_accuracy = error_detection_results['token_accuracy']
                logger.info(f"Sequence Accuracy: {error_detection_results['sequence_accuracy']:.4f}")
                logger.info(f"Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
                logger.info(f"Weighted F1: {error_detection_results['weighted_f1']:.4f}")
                
                for error_type, metrics in error_detection_results['class_metrics'].items():
                    logger.info(f"  {error_type}:")
                    logger.info(f"    Precision: {metrics['precision']:.4f}")
                    logger.info(f"    Recall: {metrics['recall']:.4f}")
                    logger.info(f"    F1 Score: {metrics['f1']:.4f}")
                
                epoch_metrics['error_detection'] = error_detection_results
            
            with open(os.path.join(args.result_dir, f'error_detection_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_metrics, f, indent=4)
            
            # Save best model based on accuracy
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                
                model_path = os.path.join(args.output_dir, f'best_error_detection.pth')
                save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, model_path, best_accuracy)
                logger.info(f"New best model saved with token accuracy {current_accuracy:.4f}: {model_path}")
                
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_error_detection_weights.pth'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                model_path = os.path.join(args.output_dir, f'best_error_detection_loss.pth')
                save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, model_path, best_accuracy)
                logger.info(f"Best validation loss model saved: {val_loss:.4f}")
            
            last_model_path = os.path.join(args.output_dir, f'last_error_detection.pth')
            save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, last_model_path, best_accuracy)
            
    elif args.mode == 'phoneme':
        if not args.phoneme_train_data or not args.phoneme_val_data:
            logger.error("Phoneme training and validation data paths required for phoneme recognition mode.")
            sys.exit(1)
            
        if not phoneme_to_id:
            logger.error("Phoneme-ID mapping required for phoneme recognition mode.")
            sys.exit(1)
            
        train_dataset = PhonemeRecognitionDataset(args.phoneme_train_data, phoneme_to_id, max_length=args.max_audio_length)
        val_dataset = PhonemeRecognitionDataset(args.phoneme_val_data, phoneme_to_id, max_length=args.max_audio_length)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=generic_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generic_collate_fn)
        
        for epoch in range(start_epoch, args.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{args.num_epochs} starting")
            
            train_loss = train_model(model, train_dataloader, criterion, optimizer_step, optimizer_zero_grad, args.device, epoch, 'phoneme', args.max_grad_norm)
            val_loss = validate_model(model, val_dataloader, criterion, args.device, 'phoneme')
            
            if scheduler is not None:
                scheduler.step(val_loss)
                logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if args.show_samples:
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch} - Sample Predictions")
                logger.info(f"{'='*50}")
                show_samples(model, val_dataloader, args.device, id_to_phoneme, args.num_sample_show, 'phoneme')
            
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            current_accuracy = 0.0
            if args.evaluate_every_epoch and eval_dataloader:
                logger.info(f"Epoch {epoch}: Evaluating phoneme recognition...")
                phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, args.device, id_to_phoneme)
                
                current_accuracy = 1.0 - phoneme_recognition_results['per']  # Convert PER to accuracy
                logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
                logger.info(f"Phoneme Accuracy: {current_accuracy:.4f}")
                logger.info(f"Total Phonemes: {phoneme_recognition_results['total_phonemes']}")
                logger.info(f"Total Errors: {phoneme_recognition_results['total_errors']}")
                
                epoch_metrics['phoneme_recognition'] = {
                    'per': phoneme_recognition_results['per'],
                    'accuracy': current_accuracy,
                    'total_phonemes': phoneme_recognition_results['total_phonemes'],
                    'total_errors': phoneme_recognition_results['total_errors'],
                    'insertions': phoneme_recognition_results['insertions'],
                    'deletions': phoneme_recognition_results['deletions'],
                    'substitutions': phoneme_recognition_results['substitutions']
                }
            
            with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
                json.dump(epoch_metrics, f, indent=4)
            
            # Save best model based on accuracy (1 - PER)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                
                model_path = os.path.join(args.output_dir, f'best_phoneme_recognition.pth')
                save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, model_path, best_accuracy)
                logger.info(f"New best model saved with phoneme accuracy {current_accuracy:.4f} (PER: {1.0-current_accuracy:.4f}): {model_path}")
                
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_phoneme_recognition_weights.pth'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                model_path = os.path.join(args.output_dir, f'best_phoneme_recognition_loss.pth')
                save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, model_path, best_accuracy)
                logger.info(f"Best validation loss model saved: {val_loss:.4f}")
            
            last_model_path = os.path.join(args.output_dir, f'last_phoneme_recognition.pth')
            save_checkpoint(model, optimizer, scheduler, entropy_regularizer, beta_optimizer, epoch, val_loss, train_loss, best_val_loss, last_model_path, best_accuracy)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()