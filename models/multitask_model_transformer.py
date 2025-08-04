import torch.nn as nn
from transformers import Wav2Vec2Config

from models.utils_models import ErrorDetectionHead, PhonemeRecognitionHead, Wav2VecEncoder, TransformerEncoder

class TransformerMultiTaskModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=1024,
                 num_phonemes=42,
                 num_error_types=3,
                 num_layers=2,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.transformer_encoder = TransformerEncoder(
            wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
        )
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None, task_mode=''):
        features = self.encoder(x, attention_mask)
        
        enhanced_features = self.transformer_encoder(features, attention_mask)
        
        outputs = {}
        
        if task_mode.startswith('error', 'multi'):
            outputs['error_logits'] = self.error_head(enhanced_features)
            
        if task_mode.startswith('phoneme', 'multi'):
            outputs['phoneme_logits'] = self.phoneme_head(enhanced_features)
            
        return outputs
