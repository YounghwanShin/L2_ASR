import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
    """Wav2Vec2 기반 오디오 인코더"""
    
    def __init__(self, pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"):
        """
        Args:
            pretrained_model_name: 사용할 Wav2Vec2 사전훈련 모델명
        """
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)

    def forward(self, x, attention_mask=None):
        """Wav2Vec2 인코딩 수행"""
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state


class SimpleEncoder(nn.Module):
    """간단한 피드포워드 인코더"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """간단한 인코딩 수행"""
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x)


class TransformerEncoder(nn.Module):
    """Transformer 기반 인코더"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 2, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            num_layers: Transformer 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
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
        """Transformer 인코딩 수행"""
        x = self.input_projection(x)
        x = self.dropout(x)

        x = self.transformer(x)
        return self.layer_norm(x)