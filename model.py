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
        
        self.conv_1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3)
        
        self.bn_1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_3 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_5 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_7 = nn.BatchNorm1d(hidden_dim // 4)
        
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
        
        feat_1 = F.relu(self.bn_1(self.conv_1(x_conv)))
        feat_3 = F.relu(self.bn_3(self.conv_3(x_conv)))
        feat_5 = F.relu(self.bn_5(self.conv_5(x_conv)))
        feat_7 = F.relu(self.bn_7(self.conv_7(x_conv)))
        
        multi_scale_features = torch.cat([feat_1, feat_3, feat_5, feat_7], dim=1)
        multi_scale_features = multi_scale_features.transpose(1, 2)
        
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
    def __init__(self, input_dim, hidden_dim=256, num_phonemes=42, dropout_rate=0.1):
        super().__init__()
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_norm = TimeDistributed(nn.LayerNorm(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_phonemes))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_norm(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        return self.td_linear2(x)

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

class ErrorDetectionModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=1024,
                 num_error_types=4):
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
        
        self.error_detection_head = ErrorDetectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_error_types=num_error_types,
            dropout_rate=0.3
        )
        
    def forward(self, x, attention_mask=None):
        features = self.encoder(x, attention_mask)
        temporal_features = self.temporal_context_module(features)
        enhanced_features = self.multi_scale_fusion(temporal_features)
        return self.error_detection_head(enhanced_features)

class PhonemeAdapter(nn.Module):
    def __init__(self, input_dim, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU()  
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x

class GatedErrorAttention(nn.Module):
    def __init__(self, error_dim, phoneme_dim, hidden_dim=512):
        super().__init__()
        self.error_proj = nn.Sequential(
            nn.Linear(error_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, phoneme_dim)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(phoneme_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, phoneme_dim),
            nn.Sigmoid()
        )
        
        self.cross_attention = nn.MultiheadAttention(
            phoneme_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(phoneme_dim)
        
    def forward(self, error_probs, phoneme_features):
        error_features = self.error_proj(error_probs)
        
        attn_out, _ = self.cross_attention(
            query=phoneme_features,
            key=error_features,
            value=error_features
        )
        
        concat_features = torch.cat([phoneme_features, attn_out], dim=-1)
        gate = self.gate(concat_features)
        
        output = gate * attn_out + (1 - gate) * phoneme_features
        return self.layer_norm(output)

class EnhancedPhonemeRecognitionHead(nn.Module):
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

class L2PhonemeTemporalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = self.residual_proj(x)
        lstm_out, _ = self.bilstm(x)
        return self.layer_norm(residual + self.dropout(lstm_out))

class PhonemeRecognitionModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 error_model_checkpoint=None,
                 hidden_dim=1024,
                 num_phonemes=42,
                 num_error_types=3):
        super().__init__()
        
        self.encoder = LearnableWav2Vec(pretrained_model_name)
        
        self.error_model = ErrorDetectionModel(
            pretrained_model_name=pretrained_model_name,
            hidden_dim=hidden_dim,
            num_error_types=num_error_types
        )
        
        if error_model_checkpoint:
            state_dict = torch.load(error_model_checkpoint, map_location='cpu')
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            self.error_model.load_state_dict(new_state_dict)
            
            for param in self.error_model.parameters():
                param.requires_grad = False
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.phoneme_adapter = PhonemeAdapter(wav2vec_dim, adapter_dim=256)
        
        self.l2_temporal_module = L2PhonemeTemporalModule(
            input_dim=wav2vec_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.1
        )
        
        self.error_attention = GatedErrorAttention(
            error_dim=num_error_types,
            phoneme_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.phoneme_recognition_head = EnhancedPhonemeRecognitionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_phonemes=num_phonemes,
            dropout_rate=0.1
        )
        
    def forward(self, x, attention_mask=None):
        phoneme_features = self.encoder(x, attention_mask)
        
        phoneme_features = self.phoneme_adapter(phoneme_features)
        
        with torch.no_grad():
            error_logits = self.error_model(x, attention_mask)
            error_probs = F.softmax(error_logits, dim=-1)
        
        temporal_features = self.l2_temporal_module(phoneme_features)
        
        enhanced_features = self.error_attention(error_probs, temporal_features)
        
        phoneme_logits = self.phoneme_recognition_head(enhanced_features)
        
        return phoneme_logits, error_logits