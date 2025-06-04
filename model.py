import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x)

class ErrorDetectionHead(nn.Module):
    def __init__(self, input_dim, num_error_types=3, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_error_types)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

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

class SimpleMultiTaskModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=512,
                 num_phonemes=42,
                 num_error_types=3):
        super().__init__()
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.shared_encoder = SimpleEncoder(wav2vec_dim, hidden_dim)
        self.error_head = ErrorDetectionHead(hidden_dim, num_error_types)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes)
        
    def forward(self, x, attention_mask=None, task='both'):
        features = self.encoder(x, attention_mask)
        shared_features = self.shared_encoder(features)
        
        outputs = {}
        
        if task in ['error', 'both']:
            outputs['error_logits'] = self.error_head(shared_features)
            
        if task in ['phoneme', 'both']:
            outputs['phoneme_logits'] = self.phoneme_head(shared_features)
            
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