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

        # 초기화 (identity mapping을 위한 초기화)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        residual = x
        
        x = self.norm(x)
            
        # Bottleneck 변환
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        # 잔차 연결
        output = x + residual
        return output
    
class SimpleCNN(nn.Module):
    def __init__(self, input_dim=768, num_classes=10, dropout_rate=0.5, activation='relu', use_residual=True):
        """        
        Args:
            input_dim (int): 입력 특징 벡터의 차원 (wav2vec 특징 차원, 기본값: 768)
            num_classes (int): 출력 클래스 수 (기본값: 10)
            dropout_rate (float): 드롭아웃 비율 (기본값: 0.5)
            activation (str): 활성화 함수 ('relu', 'leaky_relu', 'gelu') 
            use_residual (bool): 잔차 연결 사용 여부
        """
        super(SimpleCNN, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
            
        self.use_residual = use_residual
        
        # Conv 블록 1
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Conv 블록 2
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Conv 블록 3
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 잔차 연결을 위한 투영 레이어
        if use_residual:
            self.res_proj1 = nn.Conv1d(input_dim, 256, kernel_size=1)
            self.res_proj2 = nn.Conv1d(256, 128, kernel_size=1)
            self.res_proj3 = nn.Conv1d(128, 64, kernel_size=1)
        
        # 어댑티브 풀링
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 분류 레이어
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """        
        Args:
            x (torch.Tensor): wav2vec에서 추출된 특징 벡터 (batch_size, sequence_length, input_dim)
            
        Outputs:
            torch.Tensor: 분류 결과 (batch_size, num_classes)
        """
        # 입력 데이터 형태 변환: (batch_size, sequence_length, input_dim) -> (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # 잔차 연결을 위한 원본 저장
        if self.use_residual:
            residual1 = self.res_proj1(x)
        
        # Conv 블록 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # 잔차 연결 추가
        if self.use_residual:
            x = x + residual1
            
        x = self.pool(x)
        
        # 잔차 연결을 위한 원본 저장
        if self.use_residual:
            residual2 = self.res_proj2(x)
        
        # Conv 블록 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # 잔차 연결 추가
        if self.use_residual:
            x = x + residual2
            
        x = self.pool(x)
        
        # 잔차 연결을 위한 원본 저장
        if self.use_residual:
            residual3 = self.res_proj3(x)
        
        # Conv 블록 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        # 잔차 연결 추가
        if self.use_residual:
            x = x + residual3
            
        # 시퀀스 길이에 관계없이 작동하도록 어댑티브 풀링 적용
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)  # (batch_size, 64, 1) -> (batch_size, 64)
        
        x = self.dropout(x)        
        x = self.fc(x)
        
        return x

class TimeDistributed(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=256, num_phonemes=42, dropout_rate=0.1):
        super(PhonemeRecognitionHead, self).__init__()
        
        self.td_linear1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))
        self.td_bn = TimeDistributed(nn.BatchNorm1d(hidden_dim))
        self.td_dropout = TimeDistributed(nn.Dropout(dropout_rate))
        self.td_linear2 = TimeDistributed(nn.Linear(hidden_dim, num_phonemes))
        
    def forward(self, x):
        # x: [batch_size, time_steps, input_dim]
        x = self.td_linear1(x)
        x = self.td_bn(x)
        x = F.relu(x)
        x = self.td_dropout(x)
        
        # 음소 logits 출력
        phoneme_logits = self.td_linear2(x)
        
        return phoneme_logits

class FrozenWav2VecWithAdapter(nn.Module):
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
        
        # Bottleneck Adapter 추가 (차원을 1/4로 축소)
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
        adapted_features = self.adapter(hidden_states)

        return adapted_features

class LearnableWav2Vec(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h", unfreeze_top_percent=0.5, stage=1):
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
        hidden_states = outputs.last_hidden_state

        return hidden_states

class FeatureFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim=None):
        super(FeatureFusion, self).__init__()
        
        self.concat_dim = input_dim1 + input_dim2
        
        # 선택적 선형 투영 (출력 차원이 지정된 경우)
        self.use_projection = output_dim is not None
        if self.use_projection:
            self.projection = nn.Linear(self.concat_dim, output_dim)
            
    def forward(self, x1, x2):
        # 두 특징을 연결
        fused_features = torch.cat([x1, x2], dim=-1)
        
        # 선택적 선형 투영
        if self.use_projection:
            fused_features = self.projection(fused_features)
        return fused_features

class ErrorAwarePhonemeDecoder(nn.Module):
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
        
        # 오류 영향 적용
        
        # Deletion 오류: 원래 있어야 할 음소가 발음되지 않은 경우 - sil 토큰의 확률 증가
        deletion_effect = phoneme_probs.clone()
        sil_mask = torch.zeros_like(phoneme_probs)
        sil_mask[:, :, self.sil_index] = 1.0  # sil 토큰(ID=1)의 확률 증가
        deletion_effect = deletion_effect * (1.0 - 0.6 * (1.0 - sil_mask)) + 0.6 * sil_mask
        deletion_effect = deletion_effect / (deletion_effect.sum(dim=-1, keepdim=True) + 1e-8)

        # Substitution 오류: 다른 음소로 대체된 경우 - 상위 3개 음소 확률을 더 균등하게 분배
        top3_values, top3_indices = torch.topk(phoneme_probs, k=min(3, num_phonemes), dim=-1)
        boost_mask = torch.zeros_like(phoneme_probs).scatter_(-1, top3_indices, 1.0)
        sub_effect = phoneme_probs.clone()
        sub_effect = sub_effect * (1 - 0.3 * boost_mask) + 0.3 * boost_mask * torch.mean(top3_values, dim=-1, keepdim=True)
        # 합계가 1이 되도록 정규화
        sub_effect = sub_effect / (sub_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Add (insertion) 오류: 원래 없어야 할 음소가 추가된 경우 - 전체 분포 평탄화
        flat_dist = torch.ones_like(phoneme_probs) / (num_phonemes + 1e-8)
        add_effect = 0.7 * phoneme_probs + 0.3 * flat_dist
        # 합계가 1이 되도록 정규화
        add_effect = add_effect / (add_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Correct: 정확하게 발음된 경우 - 최대 확률 음소의 확률을 증가
        correct_effect = phoneme_probs.clone()
        boost_mask = torch.zeros_like(phoneme_probs).scatter_(-1, max_indices, 1.0)
        correct_effect = correct_effect * (1.0 + 0.3 * boost_mask)
        correct_effect = correct_effect / (correct_effect.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 모든 오류 효과를 오류 확률에 따라 가중 합산
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
    def __init__(self, 
                pretrained_model_name="facebook/wav2vec2-base-960h",
                hidden_dim=768,
                num_phonemes=42,  # ARPABET 음소 + sil + blank
                num_error_types=5,  # deletion, substitution, add, correct
                adapter_dim_ratio=1/4,
                unfreeze_top_percent=0.5,
                error_influence_weight=0.2,
                training_stage=1,
                blank_index=0,
                sil_index=1):
        super(DualWav2VecWithErrorAwarePhonemeRecognition, self).__init__()
        
        # 첫 번째 wav2vec2: 모든 파라미터 고정 + Bottleneck Adapter
        self.frozen_wav2vec = FrozenWav2VecWithAdapter(
            pretrained_model_name=pretrained_model_name,
            adapter_dim_ratio=adapter_dim_ratio,
            dropout_rate=0.1,
            layer_norm=True
        )
        
        # 두 번째 wav2vec2
        self.learnable_wav2vec = LearnableWav2Vec(
            pretrained_model_name=pretrained_model_name
            )
        
        # 특징 융합 (단순 연결 + 선형 투영)
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        wav2vec_dim = config.hidden_size
        
        self.feature_fusion = FeatureFusion(
            input_dim1=wav2vec_dim,
            input_dim2=wav2vec_dim,
            output_dim=hidden_dim
        )
        
        # 오류 탐지 헤드 (deletion, substitution, add, correct)
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
            sil_index=sil_index  # sil_index 추가
        )
        
    def forward(self, x, attention_mask=None, return_error_probs=False):
        """
        Args:
            x: 입력 오디오 특징 [batch_size, sequence_length]
            attention_mask: 어텐션 마스크 (선택 사항)
            return_error_probs: 오류 확률 반환 여부
        """
        # 첫 번째 wav2vec2 (고정 + Adapter)
        features1 = self.frozen_wav2vec(x, attention_mask)
        
        # 두 번째 wav2vec2
        features2 = self.learnable_wav2vec(x, attention_mask)
        
        # 특징 융합
        fused_features = self.feature_fusion(features1, features2)
        
        # 오류 탐지 (deletion, substitution, add, correct)
        error_probs = self.error_detection_head(fused_features)
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(fused_features)
        
        # 오류 인식 결합 디코딩
        adjusted_probs = self.error_aware_decoder(phoneme_logits, error_probs)
        
        if return_error_probs:
            return phoneme_logits, adjusted_probs, error_probs
        else:
            return phoneme_logits, adjusted_probs