"""Audio utility functions.

This module provides utilities for audio processing, attention masks,
CTC decoding, and Wav2Vec2 output length computation.
"""

import torch
import torch.nn as nn
from typing import List


def create_attention_mask(
    waveforms: torch.Tensor,
    normalized_lengths: torch.Tensor
) -> torch.Tensor:
  """Creates attention mask for audio inputs.
  
  Args:
    waveforms: Audio tensor of shape [batch_size, seq_len].
    normalized_lengths: Normalized lengths of shape [batch_size] in [0, 1].
    
  Returns:
    Attention mask of shape [batch_size, seq_len].
  """
  batch_size, seq_len = waveforms.shape
  absolute_lengths = (normalized_lengths * seq_len).long()
  attention_mask = torch.zeros_like(waveforms, dtype=torch.long)
  
  for i in range(batch_size):
    attention_mask[i, :absolute_lengths[i]] = 1
  
  return attention_mask


def enable_specaugment(model: nn.Module, enable: bool = True):
  """Enables or disables SpecAugment in Wav2Vec2 model.
  
  Args:
    model: Model containing Wav2Vec2 encoder.
    enable: Whether to enable SpecAugment.
  """
  actual_model = model.module if hasattr(model, 'module') else model
  if hasattr(actual_model.encoder.wav2vec2, 'config'):
    actual_model.encoder.wav2vec2.config.apply_spec_augment = enable


def compute_output_lengths(
    model: nn.Module,
    input_lengths: torch.Tensor
) -> torch.Tensor:
  """Computes Wav2Vec2 output sequence lengths.
  
  Args:
    model: Model containing Wav2Vec2 encoder.
    input_lengths: Input audio lengths of shape [batch_size].
    
  Returns:
    Output feature lengths of shape [batch_size].
  """
  actual_model = model.module if hasattr(model, 'module') else model
  wav2vec_model = actual_model.encoder.wav2vec2
  return wav2vec_model._get_feat_extract_output_lengths(input_lengths)


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_idx: int = 0
) -> List[List[int]]:
  """Performs greedy CTC decoding.
  
  Args:
    log_probs: Log probabilities of shape [batch_size, seq_len, vocab_size].
    input_lengths: Input sequence lengths of shape [batch_size].
    blank_idx: Index of CTC blank token.
    
  Returns:
    List of decoded sequences (list of token IDs).
  """
  predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
  batch_size = predictions.shape[0]
  decoded_sequences = []

  for batch_idx in range(batch_size):
    sequence = []
    previous_token = blank_idx
    actual_length = min(
        input_lengths[batch_idx].item(), 
        predictions.shape[1]
    )

    for time_step in range(actual_length):
      current_token = predictions[batch_idx, time_step]
      # Add token if it's not blank and different from previous
      if (current_token != blank_idx and 
          current_token != previous_token):
        sequence.append(int(current_token))
      previous_token = current_token

    decoded_sequences.append(sequence)

  return decoded_sequences
