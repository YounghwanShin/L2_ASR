import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoders import Wav2VecEncoder, SimpleEncoder, TransformerEncoder
from .heads import ErrorDetectionHead, PhonemeRecognitionHead


class UnifiedModel(nn.Module):
    """음소 인식과 에러 탐지를 위한 통합 모델"""
    
    def __init__(self,
                 pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
                 hidden_dim: int = 512,
                 num_phonemes: int = 42,
                 num_error_types: int = 5,
                 dropout: float = 0.1,
                 use_transformer: bool = False,
                 num_layers: int = 2,
                 num_heads: int = 8):
        """
        Args:
            pretrained_model_name: 사전훈련된 Wav2Vec2 모델명
            hidden_dim: 은닉층 차원
            num_phonemes: 음소 수
            num_error_types: 에러 타입 수 (blank, D, I, S, C)
            dropout: 드롭아웃 비율
            use_transformer: Transformer 인코더 사용 여부
            num_layers: Transformer 레이어 수
            num_heads: 어텐션 헤드 수
        """
        super().__init__()

        # Wav2Vec2 인코더
        self.encoder = Wav2VecEncoder(pretrained_model_name)

        # Wav2Vec2 출력 차원 가져오기
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size

        # 피처 인코더 선택
        if use_transformer:
            self.feature_encoder = TransformerEncoder(
                wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
            )
        else:
            self.feature_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)

        # 출력 헤드들
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)

    def forward(self, x, attention_mask=None, training_mode='phoneme_only'):
        """모델 순전파"""
        # Wav2Vec2 특성 추출
        features = self.encoder(x, attention_mask)

        # 피처 인코딩
        if hasattr(self.feature_encoder, 'transformer'):
            enhanced_features = self.feature_encoder(features, attention_mask)
        else:
            enhanced_features = self.feature_encoder(features)

        # 출력 계산
        outputs = {}
        outputs['phoneme_logits'] = self.phoneme_head(enhanced_features)

        if training_mode in ['phoneme_error', 'phoneme_error_length']:
            outputs['error_logits'] = self.error_head(enhanced_features)

        return outputs