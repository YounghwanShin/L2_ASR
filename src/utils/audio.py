"""Audio utility functions for pronunciation assessment model.

This module provides utilities for audio processing, attention mask generation,
CTC decoding, and Wav2Vec2 output length computation.
"""

import torch
import torch.nn as nn
from typing import List


def make_attn_mask(wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
    """Creates attention mask for audio inputs.
    
    Args:
        wavs: Batched audio tensor [batch_size, seq_len].
        wav_lens: Normalized audio lengths [batch_size] with values in [0, 1].
        
    Returns:
        Attention mask [batch_size, seq_len] with 1 for valid positions and
        0 for padded positions.
    """
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask


def enable_wav2vec2_specaug(model: nn.Module, enable: bool = True):
    """Enables or disables SpecAugment in Wav2Vec2 model.
    
    SpecAugment is a data augmentation technique that masks parts of the
    input spectrogram during training to improve robustness.
    
    Args:
        model: Model containing Wav2Vec2.
        enable: Whether to enable SpecAugment.
    """
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model.encoder.wav2vec2, 'config'):
        actual_model.encoder.wav2vec2.config.apply_spec_augment = enable


def get_wav2vec2_output_lengths(model: nn.Module, input_lengths: torch.Tensor) -> torch.Tensor:
    """Computes Wav2Vec2 output lengths.
    
    Wav2Vec2 applies convolutions that downsample the input, so the output
    sequence length is shorter than the input. This function computes the
    exact output length for each sample in the batch.
    
    Args:
        model: Model containing Wav2Vec2.
        input_lengths: Input audio lengths [batch_size].
        
    Returns:
        Output feature lengths [batch_size] after Wav2Vec2 processing.
    """
    actual_model = model.module if hasattr(model, 'module') else model
    wav2vec_model = actual_model.encoder.wav2vec2
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)


def decode_ctc(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """Performs CTC greedy decoding.
    
    CTC (Connectionist Temporal Classification) decoding removes repeated tokens
    and blank tokens to produce the final sequence.
    
    Args:
        log_probs: Log probabilities [batch_size, seq_len, vocab_size].
        input_lengths: Input lengths [batch_size].
        blank_idx: Index of the blank token.
        
    Returns:
        List of decoded sequences, where each sequence is a list of token IDs.
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
            # Add token if it's not blank and not a repeat of previous token
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred

        decoded_seqs.append(seq)

    return decoded_seqs
