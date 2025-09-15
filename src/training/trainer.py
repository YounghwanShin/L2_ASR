import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any

from ..models.losses import SmoothL1LengthLoss
from ..utils.audio import (
    make_attn_mask, enable_wav2vec2_specaug, get_wav2vec2_output_lengths,
    calculate_ctc_decoded_length
)


class UnifiedTrainer:
    """통합 모델 훈련 클래스"""
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: str = 'cuda',
                 logger = None):
        """
        Args:
            model: 훈련할 모델
            config: 설정 객체
            device: 훈련에 사용할 디바이스
            logger: 로깅 객체
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        # 길이 손실 함수
        self.length_loss_fn = SmoothL1LengthLoss()
        
        # 옵티마이저 설정
        self._setup_optimizers()
        
        # 스케일러 설정 (mixed precision training)
        self.scaler = torch.amp.GradScaler('cuda')
    
    def _setup_optimizers(self):
        """옵티마이저를 설정합니다."""
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
        """한 에포크 훈련을 수행합니다."""
        self.model.train()
        if self.config.wav2vec2_specaug:
            enable_wav2vec2_specaug(self.model, True)

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

            accumulated_loss = 0.0

            waveforms = batch_data['waveforms'].to(self.device)
            audio_lengths = batch_data['audio_lengths'].to(self.device)
            phoneme_labels = batch_data['phoneme_labels'].to(self.device)
            phoneme_lengths = batch_data['phoneme_lengths'].to(self.device)

            input_lengths = get_wav2vec2_output_lengths(self.model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)

            with torch.amp.autocast('cuda'):
                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

                error_targets = None
                error_input_lengths = None
                error_target_lengths = None

                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
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

                # 길이 손실 추가
                if self.config.has_length_component():
                    length_loss = self._calculate_length_loss(
                        outputs, phoneme_input_lengths, error_input_lengths, 
                        phoneme_lengths, epoch, batch_idx
                    )
                    length_loss_sum += length_loss
                    length_count += 1
                    loss = loss + (self.config.length_weight * length_loss)

                accumulated_loss = loss / self.config.gradient_accumulation
                
                if 'error_loss' in loss_dict:
                    error_loss_sum += loss_dict['error_loss']
                    error_count += 1
                if 'phoneme_loss' in loss_dict:
                    phoneme_loss_sum += loss_dict['phoneme_loss']
                    phoneme_count += 1

            # 역전파
            if accumulated_loss > 0:
                self.scaler.scale(accumulated_loss).backward()

            # 옵티마이저 스텝
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.main_optimizer)
                self.scaler.update()
                self.wav2vec_optimizer.zero_grad()
                self.main_optimizer.zero_grad()

                total_loss += accumulated_loss.item() * self.config.gradient_accumulation if accumulated_loss > 0 else 0

            # 메모리 정리
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

            # 프로그레스 바 업데이트
            self._update_progress_bar(progress_bar, total_loss, error_loss_sum, phoneme_loss_sum, 
                                    length_loss_sum, error_count, phoneme_count, length_count, batch_idx)

        torch.cuda.empty_cache()
        return total_loss / (len(dataloader) // self.config.gradient_accumulation)

    def validate_epoch(self, 
                      dataloader: DataLoader, 
                      criterion) -> float:
        """검증을 수행합니다."""
        self.model.eval()
        enable_wav2vec2_specaug(self.model, False)
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')

            for batch_idx, batch_data in enumerate(progress_bar):
                if batch_data is None:
                    continue

                waveforms = batch_data['waveforms'].to(self.device)
                audio_lengths = batch_data['audio_lengths'].to(self.device)
                phoneme_labels = batch_data['phoneme_labels'].to(self.device)
                phoneme_lengths = batch_data['phoneme_lengths'].to(self.device)

                input_lengths = get_wav2vec2_output_lengths(self.model, audio_lengths)
                wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
                attention_mask = make_attn_mask(waveforms, wav_lens_norm)

                outputs = self.model(waveforms, attention_mask=attention_mask, training_mode=self.config.training_mode)

                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))

                error_targets = None
                error_input_lengths = None
                error_target_lengths = None

                if self.config.has_error_component() and 'error_labels' in batch_data:
                    error_labels = batch_data['error_labels'].to(self.device)
                    error_lengths = batch_data['error_lengths'].to(self.device)
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

                # 길이 손실 추가
                if self.config.has_length_component():
                    length_loss = self._calculate_length_loss_validation(
                        outputs, phoneme_input_lengths, error_input_lengths, phoneme_lengths
                    )
                    loss = loss + self.config.length_weight * length_loss

                total_loss += loss.item() if loss > 0 else 0
                progress_bar.set_postfix({'Val_Loss': total_loss / (batch_idx + 1)})

        torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def _calculate_length_loss(self, outputs, phoneme_input_lengths, error_input_lengths, 
                             phoneme_lengths, epoch, batch_idx):
        """길이 손실을 계산하고 로깅합니다."""
        os.makedirs(self.config.length_logs_dir, exist_ok=True)
        length_logs_path = os.path.join(self.config.length_logs_dir, f'length_logs_epoch_{epoch}.json')

        phoneme_logits = outputs['phoneme_logits']
        error_logits = outputs['error_logits']
        
        phoneme_ctc_decoded_length = calculate_ctc_decoded_length(
            phoneme_logits, phoneme_input_lengths
        )
        phoneme_ctc_decoded_length = torch.clamp(phoneme_ctc_decoded_length, max=80)
        
        error_ctc_decoded_length = calculate_ctc_decoded_length(
            error_logits, 
            error_input_lengths if error_input_lengths is not None else phoneme_input_lengths
        )
        error_ctc_decoded_length = torch.clamp(error_ctc_decoded_length, max=80)
        
        combined_ctc_decoded_length = (phoneme_ctc_decoded_length + error_ctc_decoded_length) / 2.0

        length_loss = self.length_loss_fn(
            combined_ctc_decoded_length.detach(),
            phoneme_lengths.float()
        )

        # 길이 로깅
        self._log_length_details(epoch, batch_idx, phoneme_ctc_decoded_length, 
                               error_ctc_decoded_length, combined_ctc_decoded_length,
                               phoneme_lengths, length_logs_path)

        return length_loss

    def _calculate_length_loss_validation(self, outputs, phoneme_input_lengths, 
                                        error_input_lengths, phoneme_lengths):
        """검증 시 길이 손실을 계산합니다."""
        phoneme_logits = outputs['phoneme_logits']
        error_logits = outputs['error_logits']
        
        phoneme_ctc_decoded_length = calculate_ctc_decoded_length(
            phoneme_logits, phoneme_input_lengths
        )
        phoneme_ctc_decoded_length = torch.clamp(phoneme_ctc_decoded_length, max=80)
        
        error_ctc_decoded_length = calculate_ctc_decoded_length(
            error_logits, 
            error_input_lengths if error_input_lengths is not None else phoneme_input_lengths
        )
        error_ctc_decoded_length = torch.clamp(error_ctc_decoded_length, max=80)
        
        combined_ctc_decoded_length = (phoneme_ctc_decoded_length + error_ctc_decoded_length) / 2.0

        return self.length_loss_fn(
            combined_ctc_decoded_length.detach(),
            phoneme_lengths.float()
        )

    def _log_length_details(self, epoch, batch_idx, phoneme_decoded, error_decoded, 
                          combined_decoded, target_lengths, log_path):
        """길이 세부사항을 로깅합니다."""
        phoneme_decoded_lengths = [int(s) for s in phoneme_decoded.tolist()]
        error_decoded_lengths = [int(s) for s in error_decoded.tolist()]
        combined_decoded_lengths = [int(s) for s in combined_decoded.tolist()]
        target_lengths_list = target_lengths.tolist()
        length_diffs = [s - t for s, t in zip(combined_decoded_lengths, target_lengths_list)]
        
        length_dict = {
            'epoch_num': epoch,
            'batch_idx': batch_idx,
            'phoneme_decoded_lengths': phoneme_decoded_lengths,
            'error_decoded_lengths': error_decoded_lengths,
            'combined_decoded_lengths': combined_decoded_lengths,
            'target_lengths': target_lengths_list,
            'length_diffs': length_diffs
        }

        with open(log_path, 'a') as f:
            json.dump(length_dict, f)
            f.write("\n")

    def _update_progress_bar(self, progress_bar, total_loss, error_loss_sum, phoneme_loss_sum,
                           length_loss_sum, error_count, phoneme_count, length_count, batch_idx):
        """프로그레스 바를 업데이트합니다."""
        avg_total = total_loss / max(((batch_idx + 1) // self.config.gradient_accumulation), 1)
        avg_error = error_loss_sum / max(error_count, 1)
        avg_phoneme = phoneme_loss_sum / max(phoneme_count, 1)
        avg_length = length_loss_sum / max(length_count, 1) if self.config.has_length_component() else 0

        progress_dict = {
            'Total': f'{avg_total:.4f}',
            'Phoneme': f'{avg_phoneme:.4f}'
        }
        if self.config.has_error_component():
            progress_dict['Error'] = f'{avg_error:.4f}'
        if self.config.has_length_component():
            progress_dict['Length'] = f'{avg_length:.4f}'

        progress_bar.set_postfix(progress_dict)

    def get_optimizers(self):
        """옵티마이저들을 반환합니다."""
        return self.wav2vec_optimizer, self.main_optimizer