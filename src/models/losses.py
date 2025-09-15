import torch
import torch.nn as nn


class FocalCTCLoss(nn.Module):
    """Focal Loss를 적용한 CTC Loss"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, blank: int = 0, zero_infinity: bool = True):
        """
        Args:
            alpha: 클래스 균형을 위한 가중치
            gamma: 어려운 샘플에 집중하기 위한 지수
            blank: CTC blank 토큰 인덱스
            zero_infinity: 무한대 손실을 0으로 처리할지 여부
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Focal CTC Loss 계산"""
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        
        p_t = torch.exp(-ctc_losses)
        p_t = torch.clamp(p_t, min=1e-6, max=1.0)
        focal_losses = ctc_losses * (self.alpha * (1 - p_t) ** self.gamma)

        return focal_losses.mean()


class UnifiedLoss(nn.Module):
    """통합 손실 함수 클래스"""
    
    def __init__(self, 
                 training_mode: str = 'phoneme_only', 
                 error_weight: float = 1.0, 
                 phoneme_weight: float = 1.0, 
                 focal_alpha: float = 1.0, 
                 focal_gamma: float = 2.0):
        """
        Args:
            training_mode: 훈련 모드
            error_weight: 에러 탐지 손실 가중치
            phoneme_weight: 음소 인식 손실 가중치
            focal_alpha: Focal Loss alpha 파라미터
            focal_gamma: Focal Loss gamma 파라미터
        """
        super().__init__()
        self.training_mode = training_mode
        self.error_weight = error_weight
        self.phoneme_weight = phoneme_weight
        
        self.error_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )
        self.phoneme_criterion = FocalCTCLoss(
            alpha=focal_alpha, gamma=focal_gamma, blank=0, zero_infinity=True
        )

    def forward(self, outputs, phoneme_targets, phoneme_input_lengths, phoneme_target_lengths,
                error_targets=None, error_input_lengths=None, error_target_lengths=None):
        """통합 손실 계산"""
        total_loss = 0.0
        loss_dict = {}

        # 음소 인식 손실
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

        # 에러 탐지 손실
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


class SmoothL1LengthLoss(nn.Module):
    """길이 예측을 위한 Smooth L1 Loss"""
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Smooth L1 Loss의 beta 파라미터
        """
        super().__init__()
        self.beta = beta

    def forward(self, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Smooth L1 길이 손실 계산"""
        diff = input_lengths - target_lengths
        abs_diff = torch.abs(diff)
        
        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )
        
        return torch.mean(smooth_l1)