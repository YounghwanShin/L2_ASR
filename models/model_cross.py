import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class TaskSpecificEncoder(nn.Module):
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
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        
    def forward(self, x, attention_mask=None):
        x = self.input_projection(x)
        x = self.dropout(x)
        
        x = self.transformer(x)
        return self.layer_norm(x)

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim, cross_attention_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, query, key_value, attention_mask=None):
        residual = query
        query = self.layer_norm1(query)
        
        cross_out, _ = self.cross_attention(query, key_value, key_value, attn_mask=attention_mask)
        cross_out = self.dropout(cross_out)
        query = residual + cross_out
        
        residual = query
        query = self.layer_norm2(query)
        ff_out = self.feed_forward(query)
        ff_out = self.dropout(ff_out)
        query = residual + ff_out
        
        return query

class ErrorDetectionHead(nn.Module):
    def __init__(self, input_dim, num_error_types=3, dropout=0.1):
        super().__init__()
        self.pre_classifier = nn.Linear(input_dim, input_dim // 2)
        self.classifier = nn.Linear(input_dim // 2, num_error_types)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.pre_classifier.weight)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.pre_classifier.bias, 0)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.pre_classifier(x))
        x = self.dropout(x)
        return self.classifier(x)

class PhonemeRecognitionHead(nn.Module):
    def __init__(self, input_dim, num_phonemes=42, dropout=0.1):
        super().__init__()
        self.pre_classifier = nn.Linear(input_dim, input_dim // 2)
        self.classifier = nn.Linear(input_dim // 2, num_phonemes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.pre_classifier.weight)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.pre_classifier.bias, 0)
        nn.init.constant_(self.classifier.bias, 0)
        
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.pre_classifier(x))
        x = self.dropout(x)
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

class CrossAttentionMultiTaskModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=1024,
                 num_phonemes=42,
                 num_error_types=3,
                 num_layers=2,
                 num_heads=8,
                 cross_attention_dim=512,
                 dropout=0.1):
        super().__init__()
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.error_encoder = TaskSpecificEncoder(
            wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
        )
        self.phoneme_encoder = TaskSpecificEncoder(
            wav2vec_dim, hidden_dim, num_layers, num_heads, dropout
        )
        
        self.error_cross_attention = CrossAttentionModule(
            hidden_dim, cross_attention_dim, num_heads, dropout
        )
        self.phoneme_cross_attention = CrossAttentionModule(
            hidden_dim, cross_attention_dim, num_heads, dropout
        )
        
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None, task='both'):
        shared_features = self.encoder(x, attention_mask)
        
        error_features = self.error_encoder(shared_features, attention_mask)
        phoneme_features = self.phoneme_encoder(shared_features, attention_mask)
        
        error_enhanced = self.error_cross_attention(
            error_features, phoneme_features, attention_mask
        )
        phoneme_enhanced = self.phoneme_cross_attention(
            phoneme_features, error_features, attention_mask
        )
        
        outputs = {}
        
        if task in ['error', 'both']:
            outputs['error_logits'] = self.error_head(error_enhanced)
            
        if task in ['phoneme', 'both']:
            outputs['phoneme_logits'] = self.phoneme_head(phoneme_enhanced)
            
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