import torch.nn as nn


class ErrorDetectionHead(nn.Module):
    """에러 탐지를 위한 출력 헤드"""
    
    def __init__(self, input_dim: int, num_error_types: int = 5, dropout: float = 0.1):
        """
        Args:
            input_dim: 입력 특성 차원
            num_error_types: 에러 타입 수 (blank, D, I, S, C)
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_error_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """에러 탐지 로짓 계산"""
        x = self.dropout(x)
        return self.linear(x)


class PhonemeRecognitionHead(nn.Module):
    """음소 인식을 위한 출력 헤드"""
    
    def __init__(self, input_dim: int, num_phonemes: int = 42, dropout: float = 0.1):
        """
        Args:
            input_dim: 입력 특성 차원
            num_phonemes: 음소 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """음소 인식 로짓 계산"""
        x = self.dropout(x)
        return self.linear(x)