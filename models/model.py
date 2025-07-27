import torch.nn as nn
from transformers import Wav2Vec2Config

from utils import ErrorDetectionHead, PhonemeRecognitionHead, Wav2VecEncoder, SimpleEncoder

class SimpleMultiTaskModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=512,
                 num_phonemes=42,
                 num_error_types=3,
                 dropout=0.1):
        super().__init__()
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.shared_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None, task='both'):
        features = self.encoder(x, attention_mask)
        shared_features = self.shared_encoder(features)
        
        outputs = {}
        
        if task in ['error', 'both']:
            outputs['error_logits'] = self.error_head(shared_features)
            
        if task in ['phoneme', 'both']:
            outputs['phoneme_logits'] = self.phoneme_head(shared_features)
            
        return outputs
