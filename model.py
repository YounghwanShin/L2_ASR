import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class TimeDistributed(nn.Module):
    """시간 차원에 걸쳐 동일한 모듈을 적용하는 래퍼"""
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, x):
        # x: [batch_size, time_steps, features]
        batch_size, time_steps, features = x.size()
        x_reshaped = x.contiguous().view(-1, features)
        y = self.module(x_reshaped)
        output_shape = y.size(-1)
        y = y.contiguous().view(batch_size, time_steps, output_shape)
        return y

class BottleneckAdapter(nn.Module):
    """Bottleneck adapter for frozen models"""
    def __init__(self, dim=256, bottleneck_dim=64, dropout_rate=0.1):
        super(BottleneckAdapter, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # 초기화
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual

class FrozenWav2VecWithAdapter(nn.Module):
    """고정된 wav2vec 모델과 어댑터"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h", adapter_dim_ratio=1/4, 
                 dropout_rate=0.1):
        super(FrozenWav2VecWithAdapter, self).__init__()
        
        # wav2vec2 모델 로드        
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            pretrained_model_name,
            config=config
        )
        
        # 모든 파라미터 고정
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        # 출력 차원 파악
        config = self.wav2vec2.config
        hidden_size = config.hidden_size
        
        # Bottleneck Adapter 추가
        bottleneck_dim = int(hidden_size * adapter_dim_ratio)
        self.adapter = BottleneckAdapter(
            dim=hidden_size, 
            bottleneck_dim=bottleneck_dim,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x, attention_mask=None):
        # wav2vec2 모델 적용 (특성 추출)
        with torch.no_grad():
            outputs = self.wav2vec2(x, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        # Bottleneck Adapter 적용
        return self.adapter(hidden_states)

class LearnableWav2Vec(nn.Module):
    """학습 가능한 wav2vec 모델"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h"):
        super(LearnableWav2Vec, self).__init__()
       
        # wav2vec2 모델 로드
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
        
        # 학습 가능하게 설정
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
                
    def forward(self, x, attention_mask=None):
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class FeatureFusion(nn.Module):
    """특징 융합 모듈"""
    def __init__(self, input_dim1, input_dim2, output_dim=None):
        super(FeatureFusion, self).__init__()
        self.concat_dim = input_dim1 + input_dim2
        self.use_projection = output_dim is not None
        if self.use_projection:
            self.projection = nn.Linear(self.concat_dim, output_dim)
            
    def forward(self, x1, x2):
        fused_features = torch.cat([x1, x2], dim=-1)
        if self.use_projection:
            fused_features = self.projection(fused_features)
        return fused_features

# Stage 1: 오류 탐지 모델
class ErrorDetectionModel(nn.Module):
    """오류 탐지를 위한 독립 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-base-960h",
                hidden_dim=768,
                num_error_types=5,  # blank + deletion + substitution + insertion + correct
                adapter_dim_ratio=1/4):
        super(ErrorDetectionModel, self).__init__()
        
        # 첫 번째 wav2vec: 고정 파라미터 + 어댑터
        self.frozen_wav2vec = FrozenWav2VecWithAdapter(
            pretrained_model_name=pretrained_model_name,
            adapter_dim_ratio=adapter_dim_ratio,
            dropout_rate=0.1
        )
        
        # 두 번째 wav2vec: 학습 가능한 파라미터
        self.learnable_wav2vec = LearnableWav2Vec(
            pretrained_model_name=pretrained_model_name
        )
        
        # 특징 융합
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.feature_fusion = FeatureFusion(
            input_dim1=wav2vec_dim,
            input_dim2=wav2vec_dim,
            output_dim=hidden_dim
        )
        
        # 오류 탐지 헤드
        self.error_detection_head = nn.Sequential(
            TimeDistributed(nn.Linear(hidden_dim, hidden_dim // 2)),
            TimeDistributed(nn.BatchNorm1d(hidden_dim // 2)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.Dropout(0.3)),
            TimeDistributed(nn.Linear(hidden_dim // 2, num_error_types))
        )
            
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 입력 오디오 [batch_size, sequence_length]
            attention_mask: 어텐션 마스크
        Returns:
            error_logits: 오류 탐지 로짓 [batch_size, seq_len, num_error_types]
        """
        # 두 wav2vec 모델로부터 특징 추출
        features1 = self.frozen_wav2vec(x, attention_mask)
        features2 = self.learnable_wav2vec(x, attention_mask)
        
        # 특징 융합
        fused_features = self.feature_fusion(features1, features2)
        
        # 오류 탐지
        error_logits = self.error_detection_head(fused_features)
        
        return error_logits

# Stage 2: 음소 인식 모델
class PhonemeRecognitionModel(nn.Module):
    """음소 인식을 위한 독립 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-base-960h",
                hidden_dim=768,
                num_phonemes=42,  # 음소 + sil + blank
                adapter_dim_ratio=1/4):
        super(PhonemeRecognitionModel, self).__init__()
        
        # 첫 번째 wav2vec: 고정 파라미터 + 어댑터
        self.frozen_wav2vec = FrozenWav2VecWithAdapter(
            pretrained_model_name=pretrained_model_name,
            adapter_dim_ratio=adapter_dim_ratio,
            dropout_rate=0.1
        )
        
        # 두 번째 wav2vec: 학습 가능한 파라미터
        self.learnable_wav2vec = LearnableWav2Vec(
            pretrained_model_name=pretrained_model_name
        )
        
        # 특징 융합
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.feature_fusion = FeatureFusion(
            input_dim1=wav2vec_dim,
            input_dim2=wav2vec_dim,
            output_dim=hidden_dim
        )
        
        # 음소 인식 헤드
        self.phoneme_recognition_head = nn.Sequential(
            TimeDistributed(nn.Linear(hidden_dim, hidden_dim // 2)),
            TimeDistributed(nn.BatchNorm1d(hidden_dim // 2)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.Dropout(0.1)),
            TimeDistributed(nn.Linear(hidden_dim // 2, num_phonemes))
        )
            
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 입력 오디오 [batch_size, sequence_length]
            attention_mask: 어텐션 마스크
        Returns:
            phoneme_logits: 음소 인식 로짓 [batch_size, seq_len, num_phonemes]
        """
        # 두 wav2vec 모델로부터 특징 추출
        features1 = self.frozen_wav2vec(x, attention_mask)
        features2 = self.learnable_wav2vec(x, attention_mask)
        
        # 특징 융합
        fused_features = self.feature_fusion(features1, features2)
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(fused_features)
        
        return phoneme_logits