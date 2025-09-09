import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class ErrorDetectionHead(nn.Module):
    def __init__(self, input_dim, num_error_types=3, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_error_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        return self.linear(x)

class PhonemeRecognitionHead(nn.Module):
    def __init__(self, input_dim, num_phonemes=42, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_phonemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        return self.linear(x)

class LengthPredictionHead(nn.Module):
    """시퀀스 길이를 직접 예측하는 헤드"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.fc3 = nn.Linear(input_dim // 4, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len, hidden_dim)
        # Wav2Vec2 출력과 attention_mask의 길이가 다르므로 단순 평균 사용
        # Wav2Vec2가 이미 유효한 feature만 추출하므로 추가 masking 불필요
        
        # 전체 시퀀스에 대해 평균 pooling
        pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # MLP를 통해 길이 예측
        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # 양수 값만 출력하도록 ReLU 적용
        return F.relu(out).squeeze(-1)  # (batch_size,)

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

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x)

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
        # x: (batch_size, seq_len, input_dim)
        # attention_mask는 원본 오디오 길이 기준이므로 Wav2Vec2 출력과 길이가 다름
        # 따라서 TransformerEncoder에서는 attention_mask를 사용하지 않음
        
        x = self.input_projection(x)
        x = self.dropout(x)

        # attention_mask 사용하지 않음 (길이 불일치 때문)
        # Wav2Vec2에서 이미 masking이 적용되었으므로 추가 masking 불필요
        x = self.transformer(x)
        return self.layer_norm(x)