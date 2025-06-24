import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class FocalCTCLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, blank=0, reduction='mean', zero_infinity=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        self.reduction = reduction
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_losses = torch.clamp(ctc_losses, min=1e-6)
        p_t = torch.exp(-ctc_losses)
        p_t = torch.clamp(p_t, min=1e-6, max=1.0)
        focal_weights = self.alpha * (1 - p_t) ** self.gamma
        focal_losses = focal_weights * ctc_losses
        
        if self.reduction == 'mean':
            return focal_losses.mean()
        elif self.reduction == 'sum':
            return focal_losses.sum()
        else:
            return focal_losses

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

class SimplePhonemeModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                 hidden_dim=512,
                 num_phonemes=42,
                 dropout=0.1):
        super().__init__()
        
        self.encoder = Wav2VecEncoder(pretrained_model_name)
        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.shared_encoder = SimpleEncoder(wav2vec_dim, hidden_dim, dropout)
        self.phoneme_head = PhonemeRecognitionHead(hidden_dim, num_phonemes, dropout)
        
    def forward(self, x, attention_mask=None):
        features = self.encoder(x, attention_mask)
        shared_features = self.shared_encoder(features)
        phoneme_logits = self.phoneme_head(shared_features)
        return {'phoneme_logits': phoneme_logits}

class PhonemeLoss(nn.Module):
    def __init__(self, focal_alpha=1.0, focal_gamma=2.0):
        super().__init__()
        self.phoneme_criterion = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, phoneme_targets, phoneme_input_lengths, phoneme_target_lengths):
        phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
        phoneme_loss = self.phoneme_criterion(
            phoneme_log_probs.transpose(0, 1), 
            phoneme_targets, 
            phoneme_input_lengths, 
            phoneme_target_lengths
        )
        return phoneme_loss, {'phoneme_loss': phoneme_loss.item()}