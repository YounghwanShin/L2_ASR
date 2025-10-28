"""Trainer module for pronunciation assessment model.

This module implements the training loop, validation, and optimization
procedures for the multi-task pronunciation assessment system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.audio import (
    create_attention_mask, enable_specaugment, compute_output_lengths
)


class ModelTrainer:
    """Trainer class for pronunciation assessment model.
    
    Handles training and validation loops with mixed precision training,
    gradient accumulation, and separate learning rates for different components.
    
    Attributes:
        model: The model to train.
        config: Configuration object containing hyperparameters.
        device: Device for training.
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
        
        Wav2Vec2 parameters use a smaller learning rate while other
        parameters use a larger rate for faster adaptation.
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
            enable_specaugment(self.model, True)

        total_loss = 0.0
        loss_components = {'canonical': 0.0, 'perceived': 0.0, 'error': 0.0}
        loss_counts = {'canonical': 0, 'perceived': 0, 'error': 0}

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            # Prepare input data
            waveforms = batch_data['waveforms'].to(self.device)
            audio_lengths = batch_data['audio_lengths'].to(self.device)

            # Compute input lengths after feature extraction
            input_lengths = compute_output_lengths(self.model, audio_lengths)
            normalized_lengths = audio_lengths.float() / waveforms.shape[1]
            attention_mask = create_attention_mask(waveforms, normalized_lengths)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                # Prepare loss arguments
                loss_kwargs = {}

                # Canonical phoneme targets
                if 'canonical_labels' in batch_data:
                    canonical_labels = batch_data['canonical_labels'].to(self.device)
                    canonical_lengths = batch_data['canonical_lengths'].to(self.device)
                    canonical_input_lengths = torch.clamp(input_lengths, min=1, max=outputs.get('canonical_logits', outputs['perceived_logits']).size(1))
                    loss_kwargs['canonical_targets'] = canonical_labels
                    loss_kwargs['canonical_input_lengths'] = canonical_input_lengths
                    loss_kwargs['canonical_target_lengths'] = canonical_lengths

                # Perceived phoneme targets
                perceived_labels = batch_data['perceived_labels'].to(self.device)
                perceived_lengths = batch_data['perceived_lengths'].to(self.device)
                perceived_input_lengths = torch.clamp(input_lengths, min=1, max=outputs.get('perceived_logits', outputs.get('phoneme_logits')).size(1))
                loss_kwargs['perceived_targets'] = perceived_labels
                loss_kwargs['perceived_input_lengths'] = perceived_input_lengths
                loss_kwargs['perceived_target_lengths'] = perceived_lengths

                # Error detection targets
                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
                    error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                    # Filter out samples with empty error labels
                    valid_error_mask = error_lengths > 0
                    if valid_error_mask.any():
                        loss_kwargs['error_targets'] = error_labels[valid_error_mask]
                        loss_kwargs['error_input_lengths'] = error_input_lengths[valid_error_mask]
                        loss_kwargs['error_target_lengths'] = error_lengths[valid_error_mask]

                # Compute loss
                loss, loss_dict = criterion(outputs, **loss_kwargs)

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation
                
                # Accumulate individual loss components
                if 'canonical_loss' in loss_dict:
                    loss_components['canonical'] += loss_dict['canonical_loss']
                    loss_counts['canonical'] += 1
                if 'perceived_loss' in loss_dict:
                    loss_components['perceived'] += loss_dict['perceived_loss']
                    loss_counts['perceived'] += 1
                if 'error_loss' in loss_dict:
                    loss_components['error'] += loss_dict['error_loss']
                    loss_counts['error'] += 1

            # Backward pass
            if scaled_loss > 0:
                self.scaler.scale(scaled_loss).backward()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.main_optimizer)
                self.scaler.update()
                self.wav2vec_optimizer.zero_grad()
                self.main_optimizer.zero_grad()

                total_loss += scaled_loss.item() * self.config.gradient_accumulation if scaled_loss > 0 else 0

            # Periodic memory cleanup
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

            # Update progress bar
            self._update_progress_bar(progress_bar, total_loss, loss_components, loss_counts, batch_idx)

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
        enable_specaugment(self.model, False)
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')

            for batch_idx, batch_data in enumerate(progress_bar):
                if batch_data is None:
                    continue

                # Prepare input data
                waveforms = batch_data['waveforms'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)

                # Compute input lengths
                input_lengths = compute_output_lengths(self.model, audio_lengths)
                normalized_lengths = audio_lengths.float() / waveforms.shape[1]
                attention_mask = create_attention_mask(waveforms, normalized_lengths)

                # Forward pass
                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                # Prepare loss arguments
                loss_kwargs = {}

                # Canonical phoneme targets
                if 'canonical_labels' in batch_data:
                    canonical_labels = batch_data['canonical_labels'].to(self.device)
                    canonical_lengths = batch_data['canonical_lengths'].to(self.device)
                    canonical_input_lengths = torch.clamp(input_lengths, min=1, max=outputs.get('canonical_logits', outputs['perceived_logits']).size(1))
                    loss_kwargs['canonical_targets'] = canonical_labels
                    loss_kwargs['canonical_input_lengths'] = canonical_input_lengths
                    loss_kwargs['canonical_target_lengths'] = canonical_lengths

                # Perceived phoneme targets
                perceived_labels = batch_data['perceived_labels'].to(self.device)
                perceived_lengths = batch_data['perceived_lengths'].to(self.device)
                perceived_input_lengths = torch.clamp(input_lengths, min=1, max=outputs.get('perceived_logits', outputs.get('phoneme_logits')).size(1))
                loss_kwargs['perceived_targets'] = perceived_labels
                loss_kwargs['perceived_input_lengths'] = perceived_input_lengths
                loss_kwargs['perceived_target_lengths'] = perceived_lengths

                # Error detection targets
                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
                    error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))

                    # Filter out samples with empty error labels
                    valid_error_mask = error_lengths > 0
                    if valid_error_mask.any():
                        loss_kwargs['error_targets'] = error_labels[valid_error_mask]
                        loss_kwargs['error_input_lengths'] = error_input_lengths[valid_error_mask]
                        loss_kwargs['error_target_lengths'] = error_lengths[valid_error_mask]

                # Compute loss
                loss, _ = criterion(outputs, **loss_kwargs)

                total_loss += loss.item() if loss > 0 else 0
                progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

        torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def _update_progress_bar(self, progress_bar, total_loss, loss_components, loss_counts, batch_idx):
        """Updates progress bar with loss information.
        
        Args:
            progress_bar: tqdm progress bar object.
            total_loss: Accumulated total loss.
            loss_components: Dictionary of individual loss sums.
            loss_counts: Dictionary of loss computation counts.
            batch_idx: Current batch index.
        """
        avg_total = total_loss / max(((batch_idx + 1) // self.config.gradient_accumulation), 1)
        
        progress_dict = {'Total': f'{avg_total:.4f}'}
        
        if self.config.training_mode == 'multitask':
            if loss_counts['canonical'] > 0:
                progress_dict['Canon'] = f"{loss_components['canonical'] / loss_counts['canonical']:.4f}"
            if loss_counts['perceived'] > 0:
                progress_dict['Perceiv'] = f"{loss_components['perceived'] / loss_counts['perceived']:.4f}"
            if loss_counts['error'] > 0:
                progress_dict['Error'] = f"{loss_components['error'] / loss_counts['error']:.4f}"
        else:
            if loss_counts['perceived'] > 0:
                progress_dict['Phoneme'] = f"{loss_components['perceived'] / loss_counts['perceived']:.4f}"
            if loss_counts['error'] > 0:
                progress_dict['Error'] = f"{loss_components['error'] / loss_counts['error']:.4f}"

        progress_bar.set_postfix(progress_dict)

    def get_optimizers(self):
        """Returns the optimizers.
        
        Returns:
            Tuple of (wav2vec_optimizer, main_optimizer).
        """
        return self.wav2vec_optimizer, self.main_optimizer
