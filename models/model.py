import torch.nn as nn
from transformers import Wav2Vec2Config

from models.utils_models import (
    ErrorDetectionHead, 
    PhonemeRecognitionHead, 
    LengthPredictionHead,
    Wav2VecEncoder, 
    SimpleEncoder, 
    TransformerEncoder
)

class UnifiedModel(nn.Module):
    def __init__(self,
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=512,
                 num_phonemes=42,
                 num_error_types=3,
                 dropout=0.1,
                 use_transformer=False,
                 num_layers=2,
                 num_heads=8):
        super().__init__()

        # Wav2Vec2 인코더
        self.encoder = Wav2VecEncoder(pretrained_model_name)

        # Wav2Vec2 설정에서 hidden dimension 가져오기
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size

        # Feature encoder (Simple 또는 Transformer)
        if use_transformer:
            self.feature_encoder = TransformerEncoder(
                wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
            )
        else:
            self.feature_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)

        # 태스크별 예측 헤드들
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        self.length_head = LengthPredictionHead(hidden_dim, dropout)

    def forward(self, x, attention_mask=None, training_mode='phoneme_only'):
        # Wav2Vec2 특성 추출
        features = self.encoder(x, attention_mask)

        # Feature encoding
        if hasattr(self.feature_encoder, 'transformer'):
            # TransformerEncoder인 경우
            enhanced_features = self.feature_encoder(features, attention_mask)
        else:
            # SimpleEncoder인 경우
            enhanced_features = self.feature_encoder(features)

        outputs = {}
        
        # Phoneme recognition (모든 모드에서 필요)
        outputs['phoneme_logits'] = self.phoneme_head(enhanced_features)
        
        # Error detection (필요한 경우에만)
        if training_mode in ['phoneme_error', 'phoneme_error_length']:
            outputs['error_logits'] = self.error_head(enhanced_features)
        
        # Length prediction (필요한 경우에만)
        if training_mode in ['phoneme_error_length']:
            outputs['length_prediction'] = self.length_head(enhanced_features, attention_mask)

        return outputs