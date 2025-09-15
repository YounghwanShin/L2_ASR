import torch
import torch.nn as nn
import numpy as np
from typing import List


def make_attn_mask(wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
    """오디오 입력에 대한 어텐션 마스크를 생성합니다.
    
    Args:
        wavs: 배치된 오디오 텐서 [batch_size, seq_len]
        wav_lens: 정규화된 오디오 길이 [batch_size]
        
    Returns:
        어텐션 마스크 [batch_size, seq_len]
    """
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask


def enable_wav2vec2_specaug(model: nn.Module, enable: bool = True):
    """Wav2Vec2 모델의 SpecAugment를 활성화/비활성화합니다.
    
    Args:
        model: Wav2Vec2를 포함한 모델
        enable: SpecAugment 활성화 여부
    """
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model.encoder.wav2vec2, 'config'):
        actual_model.encoder.wav2vec2.config.apply_spec_augment = enable


def get_wav2vec2_output_lengths(model: nn.Module, input_lengths: torch.Tensor) -> torch.Tensor:
    """Wav2Vec2 출력 길이를 계산합니다.
    
    Args:
        model: Wav2Vec2를 포함한 모델
        input_lengths: 입력 오디오 길이 [batch_size]
        
    Returns:
        출력 특성 길이 [batch_size]
    """
    actual_model = model.module if hasattr(model, 'module') else model
    wav2vec_model = actual_model.encoder.wav2vec2
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)


def decode_ctc(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """CTC 디코딩을 수행합니다.
    
    Args:
        log_probs: 로그 확률 [batch_size, seq_len, vocab_size]
        input_lengths: 입력 길이 [batch_size]
        blank_idx: 블랭크 토큰 인덱스
        
    Returns:
        디코딩된 시퀀스들
    """
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []

    for b in range(batch_size):
        seq = []
        prev = blank_idx
        actual_length = min(input_lengths[b].item(), greedy_preds.shape[1])

        for t in range(actual_length):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred

        decoded_seqs.append(seq)

    return decoded_seqs


def calculate_ctc_decoded_length(outputs: torch.Tensor, 
                               input_lengths: torch.Tensor, 
                               blank_idx: int = 0) -> torch.Tensor:
    """CTC 디코딩된 길이를 계산합니다.
    
    Args:
        outputs: 모델 출력 로짓 [batch_size, seq_len, vocab_size]
        input_lengths: 입력 길이 [batch_size]
        blank_idx: 블랭크 토큰 인덱스
        
    Returns:
        디코딩된 시퀀스 길이 [batch_size]
    """
    with torch.no_grad():
        log_probs = torch.log_softmax(outputs, dim=-1)
        decoded_seqs = decode_ctc(log_probs, input_lengths, blank_idx)
        
        lengths = torch.tensor([len(seq) for seq in decoded_seqs], 
                             device=outputs.device, dtype=torch.float32)
        
    return lengths