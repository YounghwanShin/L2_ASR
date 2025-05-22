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
        
        # 시간 차원을 배치 차원으로 병합
        x_reshaped = x.contiguous().view(-1, features)
        
        # 모듈 적용
        y = self.module(x_reshaped)
        
        # 원래 차원으로 복원
        output_shape = y.size(-1)
        y = y.contiguous().view(batch_size, time_steps, output_shape)
        
        return y

class ErrorDetectionHead(nn.Module):
    """오류 탐지를 위한 헤드"""
    def __init__(self, input_dim, hidden_dim=256, num_error_types=6, dropout_rate=0.3):
        super(ErrorDetectionHead, self).__init__()
        
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_norm = TimeDistributed(nn.LayerNorm(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_error_types))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_norm(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        
        # 오류 유형 logits 출력
        return self.td_linear2(x)

class PhonemeRecognitionHead(nn.Module):
    """음소 인식을 위한 헤드"""
    def __init__(self, input_dim, hidden_dim=256, num_phonemes=42, dropout_rate=0.1):
        super(PhonemeRecognitionHead, self).__init__()
        
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_norm = TimeDistributed(nn.LayerNorm(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_phonemes))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_norm(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        
        # 음소 logits 출력
        return self.td_linear2(x)

class LearnableWav2Vec(nn.Module):
    """학습 가능한 wav2vec 모델"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-xlsr-53"):
        super(LearnableWav2Vec, self).__init__()
       
        # wav2vec2 모델 로드 (마스킹 비활성화)
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
                
    def forward(self, x, attention_mask=None):
        # wav2vec2 모델 적용 (특성 추출)
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class ErrorDetectionModel(nn.Module):
    """오류 탐지 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                hidden_dim=1024,
                num_error_types=6):
        super(ErrorDetectionModel, self).__init__()
        
        # 학습 가능한 wav2vec 인코더
        self.encoder = LearnableWav2Vec(pretrained_model_name)
        
        # 오류 탐지 헤드
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.error_detection_head = ErrorDetectionHead(
            input_dim=wav2vec_dim,
            hidden_dim=hidden_dim,
            num_error_types=num_error_types,
            dropout_rate=0.3
        )
        
    def forward(self, x, attention_mask=None):
        # 인코더로부터 특징 추출
        features = self.encoder(x, attention_mask)
        
        # 오류 탐지
        error_logits = self.error_detection_head(features)
        
        return error_logits

class ErrorAwareAttention(nn.Module):
    """오류 정보와 음소 정보 사이의 관계를 학습하는 Attention 모듈"""
    def __init__(self, error_dim, phoneme_dim, hidden_dim=256):
        super(ErrorAwareAttention, self).__init__()
        
        self.error_proj = nn.Linear(error_dim, hidden_dim)
        self.phoneme_proj = nn.Linear(phoneme_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.output_proj = nn.Linear(hidden_dim, phoneme_dim)
        
    def forward(self, error_features, phoneme_features):
        # 특징 투영
        error_proj = self.error_proj(error_features)
        phoneme_proj = self.phoneme_proj(phoneme_features)
        
        # Attention 적용
        attn_output, _ = self.attention(
            query=phoneme_proj,
            key=error_proj,
            value=error_proj
        )
        
        # 출력 투영
        enhanced_features = self.output_proj(attn_output)
        
        # 원래 특징에 residual 연결
        return enhanced_features + phoneme_features

class PhonemeRecognitionModel(nn.Module):
    """오류 인식 기반 음소 인식 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-large-xlsr-53",
                error_model_checkpoint=None,
                hidden_dim=1024,
                num_phonemes=42,
                num_error_types=6):
        super(PhonemeRecognitionModel, self).__init__()
        
        # 학습 가능한 wav2vec 인코더
        self.encoder = LearnableWav2Vec(pretrained_model_name)
        
        # 사전 학습된 오류 탐지 모델 로드 (오류 예측용)
        self.error_model = ErrorDetectionModel(
            pretrained_model_name=pretrained_model_name,
            hidden_dim=hidden_dim,
            num_error_types=num_error_types
        )
        
        # 체크포인트가 있으면 로드
        if error_model_checkpoint:
            state_dict = torch.load(error_model_checkpoint, map_location='cpu')
            
            # "module." 접두사 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 'module.' 접두사 제거
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            self.error_model.load_state_dict(new_state_dict)
            
            # 오류 모델 파라미터 고정
            for param in self.error_model.parameters():
                param.requires_grad = False
        
        # wav2vec 출력 차원 가져오기
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        # 에러 정보와 음소 정보를 결합하는 Attention 모듈
        self.error_aware_attention = ErrorAwareAttention(
            error_dim=num_error_types,
            phoneme_dim=wav2vec_dim,
            hidden_dim=hidden_dim
        )
        
        # 음소 인식 헤드
        self.phoneme_recognition_head = PhonemeRecognitionHead(
            input_dim=wav2vec_dim,
            hidden_dim=hidden_dim,
            num_phonemes=num_phonemes,
            dropout_rate=0.1
        )
        
    def forward(self, x, attention_mask=None):
        # 인코더로부터 특징 추출
        phoneme_features = self.encoder(x, attention_mask)
        
        # 오류 모델로부터 오류 예측
        with torch.no_grad():
            error_logits = self.error_model(x, attention_mask)
            error_probs = F.softmax(error_logits, dim=-1)
        
        # 오류 정보와 음소 정보 결합
        enhanced_features = self.error_aware_attention(error_probs, phoneme_features)
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(enhanced_features)
        
        return phoneme_logits, error_logits