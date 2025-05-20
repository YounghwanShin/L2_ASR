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

class AttentionFeatureFusion(nn.Module):
    """어텐션 기반 특징 융합 모듈"""
    def __init__(self, feature_dim, num_heads=8, output_dim=None, dropout_rate=0.1):
        super(AttentionFeatureFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # 멀티헤드 어텐션 레이어
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 출력 투영 레이어 (선택적)
        self.use_projection = output_dim is not None
        if self.use_projection:
            self.projection = nn.Linear(feature_dim, output_dim)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(feature_dim)
        if self.use_projection:
            self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: Learnable Wav2Vec의 특징 [batch_size, seq_len, feature_dim]
            key_value_features: Frozen Wav2Vec의 특징 [batch_size, seq_len, feature_dim]
        Returns:
            fused_features: 융합된 특징 [batch_size, seq_len, feature_dim or output_dim]
        """
        # 멀티헤드 어텐션 적용 (Q: Learnable, K/V: Frozen)
        attn_output, _ = self.mha(
            query=self.norm1(query_features),
            key=key_value_features,
            value=key_value_features
        )
        
        # 잔차 연결 (residual connection)
        fusion_output = query_features + attn_output
        
        # 선택적 투영
        if self.use_projection:
            fusion_output = self.projection(fusion_output)
            fusion_output = self.norm2(fusion_output)
        
        return fusion_output

class PhonemeRecognitionModel(nn.Module):
    """음소 인식을 위한 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-base-960h",
                hidden_dim=768,
                num_phonemes=42,  # 음소 + sil + blank
                adapter_dim_ratio=1/4,
                num_heads=8):
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
        
        self.feature_fusion = AttentionFeatureFusion(
            feature_dim=wav2vec_dim,
            num_heads=num_heads,
            output_dim=hidden_dim,
            dropout_rate=0.1
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
        frozen_features = self.frozen_wav2vec(x, attention_mask)
        learnable_features = self.learnable_wav2vec(x, attention_mask)
        
        # 특징 융합 (learnable이 Query, frozen이 Key/Value)
        fused_features = self.feature_fusion(learnable_features, frozen_features)
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(fused_features)
        
        return phoneme_logits