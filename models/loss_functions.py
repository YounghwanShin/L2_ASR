import torch
import torch.nn as nn

class FocalCTCLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, blank=0, zero_infinity=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        p_t = torch.exp(-ctc_losses)
        p_t = torch.clamp(p_t, min=1e-6, max=1.0)
        focal_losses = ctc_losses * (self.alpha * (1 - p_t) ** self.gamma)

        return focal_losses.mean()

class UnifiedLoss(nn.Module):
    def __init__(self, training_mode='phoneme_only', error_weight=1.0, phoneme_weight=1.0, focal_alpha=1.0, focal_gamma=2.0):
        super().__init__()
        self.training_mode = training_mode
        self.error_weight = error_weight
        self.phoneme_weight = phoneme_weight
        self.error_criterion = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True)
        self.phoneme_criterion = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True)
        
    def forward(self, outputs, phoneme_targets, phoneme_input_lengths, phoneme_target_lengths,
                error_targets=None, error_input_lengths=None, error_target_lengths=None):
        
        total_loss = 0.0
        loss_dict = {}
        
        phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
        phoneme_loss = self.phoneme_criterion(
            phoneme_log_probs.transpose(0, 1), 
            phoneme_targets, 
            phoneme_input_lengths, 
            phoneme_target_lengths
        )
        weighted_phoneme_loss = self.phoneme_weight * phoneme_loss
        total_loss += weighted_phoneme_loss
        loss_dict['phoneme_loss'] = phoneme_loss.item()
        
        if self.training_mode in ['phoneme_error', 'phoneme_error_length'] and 'error_logits' in outputs:
            if error_targets is not None and error_input_lengths is not None and error_target_lengths is not None:
                error_log_probs = torch.log_softmax(outputs['error_logits'], dim=-1)
                error_loss = self.error_criterion(
                    error_log_probs.transpose(0, 1), 
                    error_targets, 
                    error_input_lengths, 
                    error_target_lengths
                )
                weighted_error_loss = self.error_weight * error_loss
                total_loss += weighted_error_loss
                loss_dict['error_loss'] = error_loss.item()
            
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

class LogCoshLengthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_lengths, target_lengths):
        diff = input_lengths - target_lengths
        diff_ratio = diff / (target_lengths + 1e-8)
        length_losses = torch.mean(torch.log(torch.cosh(diff_ratio)))
        return length_losses