import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class LowLevelEncoder(nn.Module):
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
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        return self.layer_norm(x)

class MidLevelEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_heads=8, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        x = self.projection(x)
        x = self.dropout(x)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        return self.layer_norm(x)

class AdaptiveGating(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, low_level, mid_level):
        gate = self.gate_projection(low_level)
        combined = gate * low_level + (1 - gate) * mid_level
        return self.layer_norm(combined)

class AttentionHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pre_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(input_dim // 2, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        attended, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.layer_norm(x + self.dropout(attended))
        
        x = self.pre_classifier(x)
        return self.classifier(x)

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

class HierarchicalMultiTaskModel(nn.Module):
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
        
        self.low_level_encoder = LowLevelEncoder(
            wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
        )
        
        self.error_mid_encoder = MidLevelEncoder(
            hidden_dim, hidden_dim, num_layers//2, num_heads, dropout
        )
        self.phoneme_mid_encoder = MidLevelEncoder(
            hidden_dim, hidden_dim, num_layers//2, num_heads, dropout
        )
        
        self.error_gating = AdaptiveGating(hidden_dim, hidden_dim//2, dropout)
        self.phoneme_gating = AdaptiveGating(hidden_dim, hidden_dim//2, dropout)
        
        self.error_head = AttentionHead(
            hidden_dim, num_error_types, num_heads//2, dropout
        )
        self.phoneme_head = AttentionHead(
            hidden_dim, num_phonemes, num_heads//2, dropout
        )
        
    def forward(self, x, attention_mask=None, task='both'):
        shared_features = self.encoder(x, attention_mask)
        
        low_level_features = self.low_level_encoder(shared_features, attention_mask)
        
        outputs = {}
        
        if task in ['error', 'both']:
            error_mid_features = self.error_mid_encoder(low_level_features, attention_mask)
            error_combined = self.error_gating(low_level_features, error_mid_features)
            outputs['error_logits'] = self.error_head(error_combined, attention_mask)
            
        if task in ['phoneme', 'both']:
            phoneme_mid_features = self.phoneme_mid_encoder(low_level_features, attention_mask)
            phoneme_combined = self.phoneme_gating(low_level_features, phoneme_mid_features)
            outputs['phoneme_logits'] = self.phoneme_head(phoneme_combined, attention_mask)
            
        return outputs

class MultiTaskLoss(nn.Module):
    def __init__(self, error_weight=1.0, phoneme_weight=1.0):
        super().__init__()
        self.error_weight = error_weight
        self.phoneme_weight = phoneme_weight
        self.error_criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.phoneme_criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, error_targets=None, phoneme_targets=None,
                error_input_lengths=None, phoneme_input_lengths=None,
                error_target_lengths=None, phoneme_target_lengths=None):
        
        total_loss = 0.0
        loss_dict = {}
        
        if 'error_logits' in outputs and error_targets is not None:
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
            
        if 'phoneme_logits' in outputs and phoneme_targets is not None:
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
            
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict