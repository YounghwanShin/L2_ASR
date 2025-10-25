"""Trainer module for unified pronunciation assessment model.

This module implements the training loop, validation, and optimization
procedures for the multitask pronunciation assessment system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.audio import (
    make_attn_mask, enable_wav2vec2_specaug, get_wav2vec2_output_lengths
)


class UnifiedTrainer:
    """Trainer class for unified pronunciation assessment model.
    
    Handles training and validation loops with mixed precision training,
    gradient accumulation, and separate learning rates for Wav2Vec2 and
    other model components.
    
    Attributes:
        model: The model to train.
        config: Configuration object containing hyperparameters.
        device: Device for training ('cuda' or 'cpu').
        logger: Logger for training information.
        wav2vec_optimizer: Optimizer for Wav2Vec2 parameters.
        main_optimizer: Optimizer for other model parameters.
        scaler: Gradient scaler for mixed precision training.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: str = 'cuda',
                 logger = None):
        """Initializes the trainer.
        
        Args:
            model: Model to train.
            config: Configuration object.
            device: Device for training.
            logger: Logger object for training information.
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        # Setup optimizers with different learning rates
        self._setup_optimizers()
        
        # Setup gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')
    
    def _setup_optimizers(self):
        """Sets up optimizers with different learning rates.
        
        Wav2Vec2 parameters use a smaller learning rate to avoid disrupting
        the pretrained representations, while other parameters use a larger
        learning rate for faster adaptation.
        """
        wav2vec_params = []
        main_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder.wav2vec2' in name:
                wav2vec_params.append(param)
            else:
                main_params.append(param)
        
        self.wav2vec_optimizer = optim.AdamW(wav2vec_params, lr=self.config.wav2vec_lr)
        self.main_optimizer = optim.AdamW(main_params, lr=self.config.main_lr)
    
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   criterion, 
                   epoch: int) -> float:
        """Performs one training epoch.
        
        Args:
            dataloader: Training data loader.
            criterion: Loss function.
            epoch: Current epoch number.
            
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        if self.config.wav2vec2_specaug:
            enable_wav2vec2_specaug(self.model, True)

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

            # Prepare input data
            waveforms = batch_data['waveforms'].to(self.device)
            audio_lengths = batch_data['audio_lengths'].to(self.device)
            phoneme_labels = batch_data['phoneme_labels'].to(self.device)
            phoneme_lengths = batch_data['phoneme_lengths'].to(self.device)

            # Compute input lengths after Wav2Vec2 feature extraction
            input_lengths = get_wav2vec2_output_lengths(self.model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

                # Prepare error detection targets if applicable
                error_targets = None
                error_input_lengths = None
                error_target_lengths = None

                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
                    error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                    # Filter out samples with empty error labels
                    valid_error_mask = error_lengths > 0
                    if valid_error_mask.any():
                        error_targets = error_labels[valid_error_mask]
                        error_input_lengths = error_input_lengths[valid_error_mask]
                        error_target_lengths = error_lengths[valid_error_mask]

                # Compute loss
                loss, loss_dict = criterion(
                    outputs,
                    phoneme_targets=phoneme_labels,
                    phoneme_input_lengths=phoneme_input_lengths,
                    phoneme_target_lengths=phoneme_lengths,
                    error_targets=error_targets,
                    error_input_lengths=error_input_lengths,
                    error_target_lengths=error_target_lengths
                )

                # Scale loss for gradient accumulation
                accumulated_loss = loss / self.config.gradient_accumulation
                
                # Accumulate individual loss components
                if 'error_loss' in loss_dict:
                    error_loss_sum += loss_dict['error_loss']
                    error_count += 1
                if 'phoneme_loss' in loss_dict:
                    phoneme_loss_sum += loss_dict['phoneme_loss']
                    phoneme_count += 1

            # Backward pass
            if accumulated_loss > 0:
                self.scaler.scale(accumulated_loss).backward()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.main_optimizer)
                self.scaler.update()
                self.wav2vec_optimizer.zero_grad()
                self.main_optimizer.zero_grad()

                total_loss += accumulated_loss.item() * self.config.gradient_accumulation if accumulated_loss > 0 else 0

            # Periodic memory cleanup
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

            # Update progress bar
            self._update_progress_bar(progress_bar, total_loss, error_loss_sum, phoneme_loss_sum, 
                                    error_count, phoneme_count, batch_idx)

        torch.cuda.empty_cache()
        return total_loss / (len(dataloader) // self.config.gradient_accumulation)

    def validate_epoch(self, 
                      dataloader: DataLoader, 
                      criterion) -> float:
        """Performs validation.
        
        Args:
            dataloader: Validation data loader.
            criterion: Loss function.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        enable_wav2vec2_specaug(self.model, False)
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')

            for batch_idx, batch_data in enumerate(progress_bar):
                if batch_data is None:
                    continue

                # Prepare input data
                waveforms = batch_data['waveforms'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)
                phoneme_labels = batch_data['phoneme_labels'].to(self.device)
                phoneme_lengths = batch_data['phoneme_lengths'].to(self.device)

                # Compute input lengths after Wav2Vec2 feature extraction
                input_lengths = get_wav2vec2_output_lengths(self.model, audio_lengths)
                wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
                attention_mask = make_attn_mask(waveforms, wav_lens_norm)

                # Forward pass
                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

                # Prepare error detection targets if applicable
                error_targets = None
                error_input_lengths = None
                error_target_lengths = None

                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
                    error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                    # Filter out samples with empty error labels
                    valid_error_mask = error_lengths > 0
                    if valid_error_mask.any():
                        error_targets = error_labels[valid_error_mask]
                        error_input_lengths = error_input_lengths[valid_error_mask]
                        error_target_lengths = error_lengths[valid_error_mask]

                # Compute loss
                loss, _ = criterion(
                    outputs,
                    phoneme_targets=phoneme_labels,
                    phoneme_input_lengths=phoneme_input_lengths,
                    phoneme_target_lengths=phoneme_lengths,
                    error_targets=error_targets,
                    error_input_lengths=error_input_lengths,
                    error_target_lengths=error_target_lengths
                )

                total_loss += loss.item() if loss > 0 else 0
                progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

        torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def _update_progress_bar(self, progress_bar, total_loss, error_loss_sum, phoneme_loss_sum,
                           error_count, phoneme_count, batch_idx):
        """Updates progress bar with loss information.
        
        Args:
            progress_bar: tqdm progress bar object.
            total_loss: Accumulated total loss.
            error_loss_sum: Sum of error detection losses.
            phoneme_loss_sum: Sum of phoneme recognition losses.
            error_count: Number of error loss computations.
            phoneme_count: Number of phoneme loss computations.
            batch_idx: Current batch index.
        """
        avg_total = total_loss / max(((batch_idx + 1) // self.config.gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)

        progress_dict = {
            'Total': f'{avg_total:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        }
        if self.config.has_error_component():
            progress_dict['Error'] = f'{avg_error:.4f}'

        progress_bar.set_postfix(progress_dict)

    def get_optimizers(self):
        """Returns the optimizers.
        
        Returns:
            tuple: (wav2vec_optimizer, main_optimizer)
        """
        return self.wav2vec_optimizer, self.main_optimizer
