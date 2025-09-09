import torch
import torch.nn as nn

class FocalCTCLoss(nn.Module):
    """Focal Loss를 적용한 CTC Loss"""
    def __init__(self, alpha=1.0, gamma=2.0, blank=0, zero_infinity=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # CTC 손실 계산
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        
        # Focal Loss 적용
        p_t = torch.exp(-ctc_losses)
        p_t = torch.clamp(p_t, min=1e-6, max=1.0)
        focal_losses = ctc_losses * (self.alpha * (1 - p_t) ** self.gamma)

        return focal_losses.mean()

class LengthRegressionLoss(nn.Module):
    """길이 예측을 위한 회귀 손실"""
    def __init__(self, loss_type='smooth_l1', beta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(beta=beta)
        else:
            raise ValueError(f"지원하지 않는 손실 타입: {loss_type}")
    
    def forward(self, predicted_lengths, target_lengths):
        """
        Args:
            predicted_lengths: (batch_size,) 예측된 길이
            target_lengths: (batch_size,) 실제 목표 길이
        """
        # target_lengths를 float으로 변환
        if target_lengths.dtype != torch.float32:
            target_lengths = target_lengths.float()
            
        return self.loss_fn(predicted_lengths, target_lengths)

class UnifiedLoss(nn.Module):
    """통합 손실 함수 (Phoneme, Error, Length)"""
    def __init__(self, 
                 training_mode='phoneme_only', 
                 error_weight=1.0, 
                 phoneme_weight=1.0, 
                 length_weight=1.0,
                 focal_alpha=1.0, 
                 focal_gamma=2.0,
                 length_loss_type='smooth_l1',
                 length_beta=1.0):
        super().__init__()
        self.training_mode = training_mode
        self.error_weight = error_weight
        self.phoneme_weight = phoneme_weight
        self.length_weight = length_weight
        
        # CTC 손실들
        self.error_criterion = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True)
        self.phoneme_criterion = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True)
        
        # 길이 회귀 손실
        self.length_criterion = LengthRegressionLoss(loss_type=length_loss_type, beta=length_beta)

    def forward(self, outputs, 
                phoneme_targets, phoneme_input_lengths, phoneme_target_lengths,
                error_targets=None, error_input_lengths=None, error_target_lengths=None,
                target_lengths=None):
        """
        Args:
            outputs: 모델 출력 딕셔너리
            phoneme_targets: Phoneme 타겟 라벨
            phoneme_input_lengths: Phoneme 입력 길이
            phoneme_target_lengths: Phoneme 타겟 길이
            error_targets: Error 타겟 라벨 (옵션)
            error_input_lengths: Error 입력 길이 (옵션)
            error_target_lengths: Error 타겟 길이 (옵션)
            target_lengths: 길이 예측을 위한 타겟 길이 (옵션)
        """
        total_loss = 0.0
        loss_dict = {}

        # Phoneme recognition 손실 (모든 모드에서 필요)
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

        # Error detection 손실 (필요한 경우에만)
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

        # Length prediction 손실 (phoneme_error_length 모드에서만)
        if self.training_mode == 'phoneme_error_length' and 'length_prediction' in outputs:
            if target_lengths is not None:
                length_loss = self.length_criterion(
                    outputs['length_prediction'],
                    target_lengths
                )
                weighted_length_loss = self.length_weight * length_loss
                total_loss += weighted_length_loss
                loss_dict['length_loss'] = length_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict