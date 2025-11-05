"""Audio processing utilities.

Provides functions for:
  - Attention mask creation for variable-length audio
  - SpecAugment control
  - Wav2Vec2 output length computation
  - CTC greedy decoding
"""

import torch
import torch.nn as nn
from typing import List


def create_attention_mask(
    waveforms: torch.Tensor,
    normalized_lengths: torch.Tensor
) -> torch.Tensor:
  """Creates attention mask for variable-length audio.
  
  Args:
    waveforms: Audio tensor of shape [batch_size, audio_length].
    normalized_lengths: Normalized lengths in [0, 1] of shape [batch_size].
  
  Returns:
    Binary attention mask of shape [batch_size, audio_length] with 1s for
    valid positions and 0s for padding.
  """
  batch_size, audio_length = waveforms.shape
  absolute_lengths = (normalized_lengths * audio_length).long()
  
  # Create mask
  attention_mask = torch.zeros_like(waveforms).long()
  for i in range(batch_size):
    attention_mask[i, :absolute_lengths[i]] = 1
  
  return attention_mask


def enable_specaugment(model: nn.Module, enable: bool = True):
  """Enables or disables SpecAugment in Wav2Vec2 encoder.
  
  Args:
    model: Model containing Wav2Vec2 encoder.
    enable: Whether to enable SpecAugment augmentation.
  """
  # Handle DataParallel wrapper
  actual_model = model.module if hasattr(model, 'module') else model
  
  # Set SpecAugment flag
  if hasattr(actual_model.wav2vec_encoder.wav2vec2, 'config'):
    actual_model.wav2vec_encoder.wav2vec2.config.apply_spec_augment = enable


def compute_wav2vec_output_lengths(
    model: nn.Module,
    input_audio_lengths: torch.Tensor
) -> torch.Tensor:
  """Computes Wav2Vec2 output sequence lengths.
  
  Args:
    model: Model containing Wav2Vec2 encoder.
    input_audio_lengths: Input audio lengths of shape [batch_size].
  
  Returns:
    Output feature sequence lengths of shape [batch_size].
  """
  # Handle DataParallel wrapper
  actual_model = model.module if hasattr(model, 'module') else model
  wav2vec_model = actual_model.wav2vec_encoder.wav2vec2
  
  return wav2vec_model._get_feat_extract_output_lengths(input_audio_lengths)


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    sequence_lengths: torch.Tensor,
    blank_id: int = 0
) -> List[List[int]]:
  """Performs greedy CTC decoding.
  
  Removes blank tokens and consecutive duplicates to decode CTC outputs.
  
  Args:
    log_probs: Log probabilities of shape [batch_size, seq_len, vocab_size].
    sequence_lengths: Valid sequence lengths of shape [batch_size].
    blank_id: Index of the blank token.
  
  Returns:
    List of decoded sequences, where each sequence is a list of token IDs.
  """
  predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
  batch_size = predictions.shape[0]
  decoded_sequences = []
  
  for batch_idx in range(batch_size):
    sequence = []
    previous_token = blank_id
    valid_length = min(
        sequence_lengths[batch_idx].item(),
        predictions.shape[1]
    )
    
    for time_step in range(valid_length):
      current_token = predictions[batch_idx, time_step]
      
      # Add token if it's not blank and not a repeat
      if current_token != blank_id and current_token != previous_token:
        sequence.append(int(current_token))
      
      previous_token = current_token
    
    decoded_sequences.append(sequence)
  
  return decoded_sequences
