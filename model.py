import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class BottleneckAdapter(nn.Module):
    def __init__(self, dim=256, bottleneck_dim=64, dropout_rate=0.1, layer_norm=True):
        super(BottleneckAdapter, self).__init__()
        self.layer_norm = layer_norm
        self.norm = nn.LayerNorm(dim)
        
        # 다운-업 프로젝션 레이어
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # identity mapping을 위한 초기화
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
    def __init__(self, input_dim, hidden_dim=256, num_error_types=5, dropout_rate=0.3):
        super(ErrorDetectionHead, self).__init__()
        
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_bn = TimeDistributed(nn.BatchNorm1d(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_error_types))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_bn(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        
        # 오류 유형 logits 출력
        return self.td_linear2(x)

class PhonemeRecognitionHead(nn.Module):
    """음소 인식을 위한 헤드"""
    def __init__(self, input_dim, hidden_dim=256, num_phonemes=42, dropout_rate=0.1):
        super(PhonemeRecognitionHead, self).__init__()
        
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_bn = TimeDistributed(nn.BatchNorm1d(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_phonemes))
        
    def forward(self, x):
        x = self.td_linear1(x)
        x = self.td_bn(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        
        # 음소 logits 출력
        return self.td_linear2(x)

class FrozenWav2VecWithAdapter(nn.Module):
    """고정된 wav2vec 모델과 어댑터"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h", adapter_dim_ratio=1/4, 
                 dropout_rate=0.1, layer_norm=True):
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
            
        # 모델의 출력 차원 파악
        config = self.wav2vec2.config
        hidden_size = config.hidden_size
        
        # Bottleneck Adapter 추가
        bottleneck_dim = int(hidden_size * adapter_dim_ratio)
        self.adapter = BottleneckAdapter(
            dim=hidden_size, 
            bottleneck_dim=bottleneck_dim,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm
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
       
        # wav2vec2 모델 로드 (마스킹 비활성화)
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
                
    def forward(self, x, attention_mask=None):
        # wav2vec2 모델 적용 (특성 추출)
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class FeatureFusion(nn.Module):
    """특징 융합 모듈"""
    def __init__(self, input_dim1, input_dim2, output_dim=None):
        super(FeatureFusion, self).__init__()
        
        self.concat_dim = input_dim1 + input_dim2
        
        # 선택적 선형 투영
        self.use_projection = output_dim is not None
        if self.use_projection:
            self.projection = nn.Linear(self.concat_dim, output_dim)
            
    def forward(self, x1, x2):
        # 두 특징을 연결
        fused_features = torch.cat([x1, x2], dim=-1)
        
        # 선형 투영
        if self.use_projection:
            fused_features = self.projection(fused_features)
        return fused_features

class ErrorAwarePhonemeDecoder(nn.Module):
    """오류 정보를 활용한 음소 디코더"""
    def __init__(self, error_influence_weight=0.2, blank_index=0, sil_index=1):
        super(ErrorAwarePhonemeDecoder, self).__init__()
        self.error_weight = error_influence_weight
        self.blank_index = blank_index  # CTC 디코딩용 blank 인덱스 
        self.sil_index = sil_index      # sil 토큰 인덱스
        
    def forward(self, phoneme_logits, error_probs):
        """
        Args:
            phoneme_logits: 음소 분류를 위한 logit 값들 [batch_size, time_steps, num_phonemes]
            error_probs: 오류 유형별 확률 [batch_size, time_steps, 4]
                        (순서: [deletion, substitution, add, correct])
        """
        # 기본 음소 확률 (소프트맥스 적용)
        phoneme_probs = F.softmax(phoneme_logits, dim=-1)
        batch_size, time_steps, num_phonemes = phoneme_probs.shape
        
        # 가장 높은 확률을 가진 음소
        max_probs, max_indices = torch.max(phoneme_probs, dim=-1, keepdim=True)
        
        # 오류 유형별 확률 추출
        deletion_probs = error_probs[:, :, 0:1]      # deletion
        substitution_probs = error_probs[:, :, 1:2]  # substitution
        add_probs = error_probs[:, :, 2:3]           # add (insertion)
        correct_probs = error_probs[:, :, 3:4]       # correct
        
        # 오류 유형별 효과 적용        
        # 1. Deletion 오류: sil 토큰의 확률 증가
        deletion_effect = phoneme_probs.clone()
        sil_mask = torch.zeros_like(phoneme_probs)
        sil_mask[:, :, self.sil_index] = 1.0
        deletion_effect = deletion_effect * (1.0 - 0.6 * (1.0 - sil_mask)) + 0.6 * sil_mask
        deletion_effect = deletion_effect / (deletion_effect.sum(dim=-1, keepdim=True) + 1e-8)

        # 2. Substitution 오류: 상위 3개 음소 확률을 균등하게 분배
        top3_values, top3_indices = torch.topk(phoneme_probs, k=min(3, num_phonemes), dim=-1)
        boost_mask = torch.zeros_like(phoneme_probs).scatter_(-1, top3_indices, 1.0)
        sub_effect = phoneme_probs.clone()
        sub_effect = sub_effect * (1 - 0.3 * boost_mask) + 0.3 * boost_mask * torch.mean(top3_values, dim=-1, keepdim=True)
        sub_effect = sub_effect / (sub_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. Add(insertion) 오류: 전체 분포 평탄화
        flat_dist = torch.ones_like(phoneme_probs) / (num_phonemes + 1e-8)
        add_effect = 0.7 * phoneme_probs + 0.3 * flat_dist
        add_effect = add_effect / (add_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Correct: 최대 확률 음소의 확률 증가
        correct_effect = phoneme_probs.clone()
        boost_mask = torch.zeros_like(phoneme_probs).scatter_(-1, max_indices, 1.0)
        correct_effect = correct_effect * (1.0 + 0.3 * boost_mask)
        correct_effect = correct_effect / (correct_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 오류 확률에 따라 가중 합산
        adjusted_probs = (
            deletion_probs * deletion_effect +
            substitution_probs * sub_effect +
            add_probs * add_effect +
            correct_probs * correct_effect
        )
        
        # 기존 확률과 조정된 확률 사이의 가중 평균
        final_probs = (1 - self.error_weight) * phoneme_probs + self.error_weight * adjusted_probs
        
        return final_probs

class DualWav2VecWithErrorAwarePhonemeRecognition(nn.Module):
    """오류 인식 기반 음소 인식을 위한 이중 wav2vec 모델"""
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-base-960h",
                hidden_dim=768,
                num_phonemes=42,  # 음소 + sil + blank
                num_error_types=5,  # blank + deletion + substitution + add + correct
                adapter_dim_ratio=1/4,
                error_influence_weight=0.2,
                blank_index=0,
                sil_index=1):
        super(DualWav2VecWithErrorAwarePhonemeRecognition, self).__init__()
        
        # 첫 번째 wav2vec: 고정 파라미터 + 어댑터
        self.frozen_wav2vec = FrozenWav2VecWithAdapter(
            pretrained_model_name=pretrained_model_name,
            adapter_dim_ratio=adapter_dim_ratio,
            dropout_rate=0.1,
            layer_norm=True
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
        self.error_detection_head = ErrorDetectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_error_types=num_error_types,
            dropout_rate=0.3
        )
        
        # 음소 인식 헤드
        self.phoneme_recognition_head = PhonemeRecognitionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_phonemes=num_phonemes,
            dropout_rate=0.1
        )
        
        # 오류 인식 결합 디코더
        self.error_aware_decoder = ErrorAwarePhonemeDecoder(
            error_influence_weight=error_influence_weight,
            blank_index=blank_index,
            sil_index=sil_index
        )
        
    def forward(self, x, attention_mask=None, return_error_probs=False):
        """
        Args:
            x: 입력 오디오 [batch_size, sequence_length]
            attention_mask: 어텐션 마스크
            return_error_probs: 오류 확률 반환 여부
        """
        # 두 wav2vec 모델로부터 특징 추출
        features1 = self.frozen_wav2vec(x, attention_mask)
        features2 = self.learnable_wav2vec(x, attention_mask)
        
        # 특징 융합
        fused_features = self.feature_fusion(features1, features2)
        
        # 오류 탐지
        error_logits = self.error_detection_head(fused_features)
        error_probs = F.softmax(error_logits, dim=-1)
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(fused_features)
        
        # 오류 인식 결합 디코딩
        adjusted_probs = self.error_aware_decoder(phoneme_logits, error_probs)
        
        if return_error_probs:
            return phoneme_logits, adjusted_probs, error_logits
        else:
            return phoneme_logits, adjusted_probs