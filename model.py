import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config, BertModel, BertConfig

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

class TextEncoder(nn.Module):
    """텍스트 인코딩을 위한 BERT 기반 모듈"""
    def __init__(self, pretrained_model_name="bert-base-uncased", output_dim=768):
        super(TextEncoder, self).__init__()
        
        # BERT 모델 로드
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.output_dim = output_dim
        
        # BERT 출력 차원이 목표 차원과 다를 경우 선형 투영
        if self.bert.config.hidden_size != output_dim:
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
            
    def forward(self, input_ids, attention_mask=None):
        # BERT 모델 적용
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 출력 차원 변환
        projected_states = self.projection(hidden_states)
        return projected_states

class LearnableWav2Vec(nn.Module):
    """학습 가능한 wav2vec 모델"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-xlsr-53"):
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

class ContextAwareAlignment(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # 양방향 LSTM
        self.text_context = nn.LSTM(text_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.audio_context = nn.LSTM(audio_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # 정규화
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        
        # 어텐션 컴포넌트
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 프로젝션 레이어
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, audio_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_features, audio_features, text_attention_mask=None):
        batch_size, audio_len, _ = audio_features.shape
        batch_size, text_len, _ = text_features.shape
        
        # 양방향 LSTM으로 문맥 추출
        text_ctx, _ = self.text_context(text_features)
        audio_ctx, _ = self.audio_context(audio_features)
        
        # 정규화
        text_ctx = self.text_norm(text_ctx)
        audio_ctx = self.audio_norm(audio_ctx)
        
        # 프로젝션
        q = self.q_proj(audio_ctx)  # [batch_size, audio_len, hidden_dim]
        k = self.k_proj(text_ctx)   # [batch_size, text_len, hidden_dim]
        v = self.v_proj(text_ctx)   # [batch_size, text_len, hidden_dim]
        
        # 다중 헤드로 분할
        q = q.view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, audio_len, head_dim]
        k = k.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)   # [batch_size, num_heads, text_len, head_dim]
        v = v.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)   # [batch_size, num_heads, text_len, head_dim]
        
        # 어텐션 스코어 계산
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, audio_len, text_len]
        
        # 마스킹 (필요한 경우)
        if text_attention_mask is not None:
            mask = text_attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, text_len]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스 적용
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중 평균 계산
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, audio_len, head_dim]
        
        # 헤드 결합
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, audio_len, -1)
        
        # 출력 투영
        output_features = self.out_proj(attn_output)
        
        return output_features

class CrossAttention(nn.Module):
    """크로스 어텐션 모듈 - 직접 구현"""
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 프로젝션 레이어
        self.q_proj = nn.Linear(query_dim, output_dim)
        self.k_proj = nn.Linear(key_dim, output_dim)
        self.v_proj = nn.Linear(value_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 레이어 정규화
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, query_len, query_dim]
            key: [batch_size, key_len, key_dim]
            value: [batch_size, value_len, value_dim]
            key_padding_mask: [batch_size, key_len], 선택적
        """
        batch_size, query_len, _ = query.shape
        batch_size, key_len, _ = key.shape
        
        residual = query
        
        # 정규화
        query = self.norm(query)
        
        # 프로젝션
        q = self.q_proj(query)  # [batch_size, query_len, output_dim]
        k = self.k_proj(key)    # [batch_size, key_len, output_dim]
        v = self.v_proj(value)  # [batch_size, value_len, output_dim]
        
        # 다중 헤드로 분할
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, query_len, head_dim]
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)    # [batch_size, num_heads, key_len, head_dim]
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)    # [batch_size, num_heads, key_len, head_dim]
        
        # 어텐션 스코어 계산
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, query_len, key_len]
        
        # 마스킹 (필요하고 크기가 일치하는 경우에만)
        if key_padding_mask is not None:
            # 마스크 크기 일치 확인
            expected_mask_shape = (batch_size, key_len)
            if key_padding_mask.shape == expected_mask_shape:
                # [batch_size, key_len] -> [batch_size, 1, 1, key_len]
                mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            else:
                print(f"마스크 크기 불일치: 기대={expected_mask_shape}, 실제={key_padding_mask.shape}, 마스크 무시됨")
        
        # 소프트맥스 적용
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중 평균 계산
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, query_len, head_dim]
        
        # 헤드 결합
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        
        # 출력 투영
        output = self.out_proj(attn_output)
        
        # 잔차 연결
        return output + residual

class PhonemeRecognitionModel(nn.Module):
    def __init__(self, 
                pretrained_audio_model="facebook/wav2vec2-large-xlsr-53",
                pretrained_text_model="bert-base-uncased",
                hidden_dim=1024,
                num_phonemes=42,
                num_attention_heads=8,
                dropout=0.1):
        super(PhonemeRecognitionModel, self).__init__()
        
        # wav2vec: 학습 가능한 파라미터
        self.learnable_wav2vec = LearnableWav2Vec(
            pretrained_model_name=pretrained_audio_model
        )
        
        # 텍스트 인코더
        # BERT 출력 차원 확인
        bert_config = BertConfig.from_pretrained(pretrained_text_model)
        text_dim = bert_config.hidden_size
        
        self.text_encoder = TextEncoder(
            pretrained_model_name=pretrained_text_model,
            output_dim=text_dim
        )
        
        # wav2vec 출력 차원 확인
        wav2vec_config = Wav2Vec2Config.from_pretrained(pretrained_audio_model)
        audio_dim = wav2vec_config.hidden_size
        
        # 문맥 인식 정렬 모듈 - 커스텀 구현
        self.alignment_module = ContextAwareAlignment(
            text_dim=text_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # 크로스 어텐션 - 커스텀 구현
        self.cross_attention = CrossAttention(
            query_dim=audio_dim,
            key_dim=audio_dim,  # 정렬된 특징의 차원
            value_dim=audio_dim, # 정렬된 특징의 차원
            output_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # 음소 인식 헤드
        self.phoneme_recognition_head = nn.Sequential(
            TimeDistributed(nn.Linear(hidden_dim, hidden_dim // 2)),
            TimeDistributed(nn.BatchNorm1d(hidden_dim // 2)),
            TimeDistributed(nn.GELU()),
            TimeDistributed(nn.Dropout(dropout)),
            TimeDistributed(nn.Linear(hidden_dim // 2, num_phonemes))
        )
            
    def forward(self, audio_input, text_input_ids, audio_attention_mask=None, text_attention_mask=None):
        """
        Args:
            audio_input: 입력 오디오 [batch_size, sequence_length]
            text_input_ids: 입력 텍스트 ID [batch_size, text_length]
            audio_attention_mask: 오디오 어텐션 마스크
            text_attention_mask: 텍스트 어텐션 마스크
        """
        # 오디오 특징 추출
        audio_features = self.learnable_wav2vec(audio_input, audio_attention_mask)
        
        # 텍스트 특징 추출
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 문맥 인식 정렬 수행 - 텍스트 특징을 오디오 시퀀스에 맞게 정렬
        aligned_text_features = self.alignment_module(
            text_features, 
            audio_features, 
            text_attention_mask
        )
        
        # 크로스 어텐션으로 오디오와 정렬된 텍스트 특징 융합
        fused_features = self.cross_attention(
            audio_features, 
            aligned_text_features, 
            aligned_text_features
        )
        
        # 음소 인식
        phoneme_logits = self.phoneme_recognition_head(fused_features)
        
        return phoneme_logits