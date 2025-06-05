import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        x = self.input_projection(x)
        x = self.dropout(x)
        
        x = self.transformer(x)
        return self.layer_norm(x)

class PhonemeRecognitionHead(nn.Module):
    def __init__(self, input_dim, num_phonemes=42, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class Wav2VecEncoder(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
                
    def forward(self, x, attention_mask=None):
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class TransformerPhonemeModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=1024,
                 num_phonemes=42,
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
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None):
        features = self.encoder(x, attention_mask)
        enhanced_features = self.transformer_encoder(features, attention_mask)
        phoneme_logits = self.phoneme_head(enhanced_features)
        return {'phoneme_logits': phoneme_logits}

class PhonemeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.phoneme_criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, phoneme_targets, phoneme_input_lengths, phoneme_target_lengths):
        phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
        phoneme_loss = self.phoneme_criterion(
            phoneme_log_probs.transpose(0, 1), 
            phoneme_targets, 
            phoneme_input_lengths, 
            phoneme_target_lengths
        )
        return phoneme_loss, {'phoneme_loss': phoneme_loss.item()}