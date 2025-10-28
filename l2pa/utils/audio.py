"""Audio utility functions.

This module provides utilities for audio processing, attention mask generation,
CTC decoding, and Wav2Vec2 output length computation.
"""

import torch
import torch.nn as nn
from typing import List


def create_attention_mask(waveforms: torch.Tensor, normalized_lengths: torch.Tensor) -> torch.Tensor:
    """Creates attention mask for audio inputs.
    
    Args:
        waveforms: Batched audio tensor [batch_size, sequence_length].
        normalized_lengths: Normalized audio lengths [batch_size] with values in [0, 1].
        
    Returns:
        Attention mask [batch_size, sequence_length] with 1 for valid positions
        and 0 for padded positions.
    """
    absolute_lengths = (normalized_lengths * waveforms.shape[1]).long()
    attention_mask = waveforms.new(waveforms.shape).zero_().long()
    for i in range(len(absolute_lengths)):
        attention_mask[i, :absolute_lengths[i]] = 1
    return attention_mask


def enable_specaugment(model: nn.Module, enable: bool = True):
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


def compute_output_lengths(model: nn.Module, input_lengths: torch.Tensor) -> torch.Tensor:
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


def greedy_ctc_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """Performs CTC greedy decoding.
    
    CTC (Connectionist Temporal Classification) decoding removes repeated tokens
    and blank tokens to produce the final sequence.
    
    Args:
        log_probs: Log probabilities [batch_size, sequence_length, vocab_size].
        input_lengths: Input lengths [batch_size].
        blank_idx: Index of the blank token.
        
    Returns:
        List of decoded sequences, where each sequence is a list of token IDs.
    """
    predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
    batch_size = predictions.shape[0]
    decoded_sequences = []

    for batch_idx in range(batch_size):
        sequence = []
        previous_token = blank_idx
        actual_length = min(input_lengths[batch_idx].item(), predictions.shape[1])

        for time_step in range(actual_length):
            current_token = predictions[batch_idx, time_step]
            # Add token if it's not blank and not a repeat of previous token
            if current_token != blank_idx and current_token != previous_token:
                sequence.append(int(current_token))
            previous_token = current_token

        decoded_sequences.append(sequence)

    return decoded_sequences