import torch.nn as nn
from transformers import Wav2Vec2Config

from models.utils_models import ErrorDetectionHead, PhonemeRecognitionHead, Wav2VecEncoder, SimpleEncoder, TransformerEncoder

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
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        self.use_transformer = use_transformer
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        if use_transformer:
            self.feature_encoder = TransformerEncoder(
                wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
            )
        else:
            self.feature_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)
            
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None, training_mode='phoneme_only'):
        features = self.encoder(x, attention_mask)
        
        if self.use_transformer:
            enhanced_features = self.feature_encoder(features, attention_mask)
        else:
            enhanced_features = self.feature_encoder(features)
        
        outputs = {}
        outputs['phoneme_logits'] = self.phoneme_head(enhanced_features)
        
        if training_mode in ['phoneme_error', 'phoneme_error_length']:
            outputs['error_logits'] = self.error_head(enhanced_features)
            
        return outputs