import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        batch_size, time_steps, features = x.size()
        x_reshaped = x.contiguous().view(-1, features)
        y = self.module(x_reshaped)
        output_shape = y.size(-1)
        return y.contiguous().view(batch_size, time_steps, output_shape)

class TemporalContextModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != hidden_dim:
            self.proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        residual = self.proj(x)
        lstm_out, _ = self.bilstm(x)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        return self.layer_norm(residual + self.dropout(attn_out))

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim // 4) for _ in range(4)
        ])
        
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4),
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        residual = self.residual_proj(x)
        x_conv = x.transpose(1, 2)
        
        features = []
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            feat = F.relu(bn(conv(x_conv)))
            features.append(feat)
        
        multi_scale_features = torch.cat(features, dim=1).transpose(1, 2)
        
        global_features = torch.mean(multi_scale_features, dim=1)
        scale_weights = self.scale_attention(global_features)
        
        weighted_features = []
        for i in range(4):
            start_idx = i * (multi_scale_features.size(-1) // 4)
            end_idx = (i + 1) * (multi_scale_features.size(-1) // 4)
            scale_feat = multi_scale_features[:, :, start_idx:end_idx]
            weight = scale_weights[:, i].unsqueeze(1).unsqueeze(2)
            weighted_features.append(scale_feat * weight)
        
        fused_features = torch.cat(weighted_features, dim=-1)
        output = self.output_proj(fused_features)
        return self.layer_norm(residual + self.dropout(output))

class CrossTaskAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, query, key_value):
        attn_out, _ = self.cross_attention(query, key_value, key_value)
        query = self.layer_norm1(query + self.dropout(attn_out))
        
        ff_out = self.feed_forward(query)
        return self.layer_norm2(query + self.dropout(ff_out))

class ErrorDetectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_error_types=4, dropout_rate=0.3):
        super().__init__()
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_norm = TimeDistributed(nn.LayerNorm(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_error_types))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_norm(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        return self.td_linear2(x)

class PhonemeRecognitionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_phonemes=42, dropout_rate=0.1):
        super().__init__()
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_norm1 = TimeDistributed(nn.LayerNorm(hidden_dim))
        self.td_dropout1 = TimeDistributed(nn.Dropout(dropout_rate))
        
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, hidden_dim // 2))
        self.td_norm2 = TimeDistributed(nn.LayerNorm(hidden_dim // 2))
        self.td_dropout2 = TimeDistributed(nn.Dropout(dropout_rate))
        
        self.td_linear3 = TimeDistributed(nn.Linear(hidden_dim // 2, num_phonemes))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_norm1(x)
        x = F.gelu(x)
        x = self.td_dropout1(x)
        
        x = self.td_linear2(x)
        x = self.td_norm2(x)
        x = F.gelu(x)
        x = self.td_dropout2(x)
        
        return self.td_linear3(x)

class LearnableWav2Vec(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
                
    def forward(self, x, attention_mask=None):
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class MultiTaskModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=1024,
                 num_phonemes=42,
                 num_error_types=3,
                 use_cross_attention=True):
        super().__init__()
        
        self.encoder = LearnableWav2Vec(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.temporal_context_module = TemporalContextModule(
            input_dim=wav2vec_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.2
        )
        
        self.multi_scale_fusion = MultiScaleFeatureFusion(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.error_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.phoneme_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.error_to_phoneme_attn = CrossTaskAttention(hidden_dim)
            self.phoneme_to_error_attn = CrossTaskAttention(hidden_dim)
        
        self.error_detection_head = ErrorDetectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_error_types=num_error_types,
            dropout_rate=0.3
        )
        
        self.phoneme_recognition_head = PhonemeRecognitionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_phonemes=num_phonemes,
            dropout_rate=0.1
        )
        
    def forward(self, x, attention_mask=None, task='both', **kwargs):
        features = self.encoder(x, attention_mask)
        temporal_features = self.temporal_context_module(features)
        shared_features = self.multi_scale_fusion(temporal_features)
        
        error_features = self.error_branch(shared_features)
        phoneme_features = self.phoneme_branch(shared_features)
        
        if self.use_cross_attention:
            error_features_enhanced = self.phoneme_to_error_attn(error_features, phoneme_features)
            phoneme_features_enhanced = self.error_to_phoneme_attn(phoneme_features, error_features)
        else:
            error_features_enhanced = error_features
            phoneme_features_enhanced = phoneme_features
        
        outputs = {}
        
        if task in ['error', 'both']:
            error_logits = self.error_detection_head(error_features_enhanced)
            outputs['error_logits'] = error_logits
            
        if task in ['phoneme', 'both']:
            phoneme_logits = self.phoneme_recognition_head(phoneme_features_enhanced)
            outputs['phoneme_logits'] = phoneme_logits
            
        return outputs

class MultiTaskLoss(nn.Module):
    def __init__(self, error_weight=1.0, phoneme_weight=1.0, adaptive_weights=False):
        super().__init__()
        self.error_weight = nn.Parameter(torch.tensor(error_weight)) if adaptive_weights else error_weight
        self.phoneme_weight = nn.Parameter(torch.tensor(phoneme_weight)) if adaptive_weights else phoneme_weight
        self.adaptive_weights = adaptive_weights
        
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
            
            if self.adaptive_weights:
                weighted_error_loss = torch.abs(self.error_weight) * error_loss
            else:
                weighted_error_loss = self.error_weight * error_loss
                
            total_loss += weighted_error_loss
            loss_dict['error_loss'] = error_loss.item()
            loss_dict['weighted_error_loss'] = weighted_error_loss.item()
            
        if 'phoneme_logits' in outputs and phoneme_targets is not None:
            phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
            phoneme_loss = self.phoneme_criterion(
                phoneme_log_probs.transpose(0, 1), 
                phoneme_targets, 
                phoneme_input_lengths, 
                phoneme_target_lengths
            )
            
            if self.adaptive_weights:
                weighted_phoneme_loss = torch.abs(self.phoneme_weight) * phoneme_loss
            else:
                weighted_phoneme_loss = self.phoneme_weight * phoneme_loss
                
            total_loss += weighted_phoneme_loss
            loss_dict['phoneme_loss'] = phoneme_loss.item()
            loss_dict['weighted_phoneme_loss'] = weighted_phoneme_loss.item()
            
        loss_dict['total_loss'] = total_loss.item()
        
        if self.adaptive_weights:
            loss_dict['error_weight'] = torch.abs(self.error_weight).item()
            loss_dict['phoneme_weight'] = torch.abs(self.phoneme_weight).item()
            
        return total_loss, loss_dict