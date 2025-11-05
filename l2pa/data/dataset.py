"""Dataset module for pronunciation assessment.

This module provides efficient data loading for multitask pronunciation
assessment including phoneme recognition and error detection.
"""

import json
import os
from typing import Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class PronunciationDataset(Dataset):
  """Dataset for pronunciation assessment with multitask support.
  
  loads audio and labels for canonical/perceived phoneme recognition and error detection tasks.
  """
  
  def __init__(
      self,
      json_path: str,
      phoneme_to_id: Dict[str, int],
      training_mode: str,
      max_length: Optional[int] = None,
      sampling_rate: int = 16000,
      device: str = 'cuda'
  ):
    """Initializes the dataset.
    
    Args:
      json_path: Path to dataset JSON file.
      phoneme_to_id: Phoneme to ID mapping dictionary.
      training_mode: Training mode ('phoneme_only', 'phoneme_error', 'multitask').
      max_length: Maximum audio length in samples. If None, no filtering applied.
      sampling_rate: Target audio sampling rate.
      device: Device for tensor operations.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)
    
    self.phoneme_to_id = phoneme_to_id
    self.training_mode = training_mode
    self.sampling_rate = sampling_rate
    self.max_length = max_length
    self.device = device
    
    # Error label to ID mapping
    self.error_to_id = {'D': 1, 'I': 2, 'S': 3, 'C': 4}
    
    # Filter valid files based on requirements
    self.file_paths = self._get_valid_files()
    
    print(f'Loaded {len(self.file_paths)} valid samples from {json_path}')
  
  def _get_valid_files(self) -> List[str]:
    """Gets list of valid files based on training mode and file existence.
    
    Returns:
      List of valid file paths.
    """
    valid_files = []
    
    for file_path, item in self.data.items():
      # Check file existence
      if not os.path.exists(file_path):
        continue
      
      # Check required labels based on training mode
      has_perceived = bool(item.get('perceived_train_target', '').strip())
      has_canonical = bool(item.get('canonical_train_target', '').strip())
      
      # Validate based on training mode
      if self.training_mode == 'multitask' and not (has_canonical and has_perceived):
        continue
      elif not has_perceived:
        continue
      
      valid_files.append(file_path)
    
    return valid_files
  
  def __len__(self) -> int:
    """Returns the number of samples in the dataset."""
    return len(self.file_paths)
  
  def _load_audio(self, file_path: str) -> torch.Tensor:
    """Loads and preprocesses audio file.
    
    Args:
      file_path: Path to audio file.
    
    Returns:
      Preprocessed audio waveform tensor.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != self.sampling_rate:
      resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
      waveform = resampler(waveform)
    
    return waveform.squeeze(0)
  
  def _encode_phonemes(self, phoneme_str: str) -> torch.Tensor:
    """Encodes phoneme string to ID tensor.
    
    Args:
      phoneme_str: Space-separated phoneme string.
    
    Returns:
      Tensor of phoneme IDs.
    """
    if not phoneme_str or not phoneme_str.strip():
      return torch.tensor([], dtype=torch.long)
    
    phonemes = phoneme_str.split()
    ids = [self.phoneme_to_id.get(p, 0) for p in phonemes]
    return torch.tensor(ids, dtype=torch.long)
  
  def _encode_errors(self, error_str: str) -> torch.Tensor:
    """Encodes error label string to ID tensor.
    
    Args:
      error_str: Space-separated error label string.
    
    Returns:
      Tensor of error IDs.
    """
    if not error_str or not error_str.strip():
      return torch.tensor([], dtype=torch.long)
    
    errors = error_str.split()
    ids = [self.error_to_id.get(e, 0) for e in errors]
    return torch.tensor(ids, dtype=torch.long)
  
  def __getitem__(self, idx: int) -> Optional[Dict]:
    """Gets a single dataset item.
    
    Args:
      idx: Index of the item.
    
    Returns:
      Dictionary containing waveform, labels, and metadata.
      Returns None if item is invalid.
    """
    file_path = self.file_paths[idx]
    item = self.data[file_path]
    
    try:
      # Load audio
      waveform = self._load_audio(file_path)
      
      # Check length constraint
      if self.max_length and waveform.shape[0] > self.max_length:
        return None
      
      # Prepare result dictionary
      result = {
          'waveform': waveform,
          'audio_length': torch.tensor(waveform.shape[0], dtype=torch.long),
          'file_path': file_path,
          'speaker_id': item.get('spk_id', 'UNKNOWN')
      }
      
      # Encode canonical phonemes
      canonical_labels = self._encode_phonemes(
          item.get('canonical_train_target', '')
      )
      result['canonical_labels'] = canonical_labels
      result['canonical_length'] = torch.tensor(len(canonical_labels), dtype=torch.long)
      
      # Encode perceived phonemes
      perceived_labels = self._encode_phonemes(
          item.get('perceived_train_target', '')
      )
      result['perceived_labels'] = perceived_labels
      result['perceived_length'] = torch.tensor(len(perceived_labels), dtype=torch.long)
      
      # Encode error labels
      error_labels = self._encode_errors(
          item.get('error_labels', '')
      )
      result['error_labels'] = error_labels
      result['error_length'] = torch.tensor(len(error_labels), dtype=torch.long)
      
      return result
    
    except Exception as e:
      print(f'Error loading {file_path}: {str(e)}')
      return None


def collate_batch(batch: List[Dict]) -> Optional[Dict]:
  """Collates batch data with proper padding.
  
  Args:
    batch: List of dataset items.
  
  Returns:
    Dictionary with padded tensors, or None if batch is empty.
  """
  # Filter out None items
  batch = [item for item in batch if item is not None]
  
  if not batch:
    return None
  
  # Pad waveforms to maximum length in batch
  max_audio_len = max(item['waveform'].shape[0] for item in batch)
  padded_waveforms = torch.stack([
      torch.nn.functional.pad(
          item['waveform'],
          (0, max_audio_len - item['waveform'].shape[0])
      )
      for item in batch
  ])
  
  # Create result dictionary
  result = {
      'waveforms': padded_waveforms,
      'audio_lengths': torch.tensor([item['audio_length'] for item in batch]),
      'file_paths': [item['file_path'] for item in batch],
      'speaker_ids': [item['speaker_id'] for item in batch]
  }
  
  # Pad canonical labels if present
  canonical_labels = [item['canonical_labels'] for item in batch]
  if canonical_labels and all(len(l) > 0 for l in canonical_labels):
    max_len = max(l.shape[0] for l in canonical_labels)
    result['canonical_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in canonical_labels
    ])
    result['canonical_lengths'] = torch.tensor(
        [item['canonical_length'] for item in batch]
    )
  
  # Pad perceived labels if present
  perceived_labels = [item['perceived_labels'] for item in batch]
  if perceived_labels and all(len(l) > 0 for l in perceived_labels):
    max_len = max(l.shape[0] for l in perceived_labels)
    result['perceived_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in perceived_labels
    ])
    result['perceived_lengths'] = torch.tensor(
        [item['perceived_length'] for item in batch]
    )
  
  # Pad error labels if present
  error_labels = [item['error_labels'] for item in batch]
  if error_labels and all(len(l) > 0 for l in error_labels):
    max_len = max(l.shape[0] for l in error_labels)
    result['error_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in error_labels
    ])
    result['error_lengths'] = torch.tensor(
        [item['error_length'] for item in batch]
    )
  
  return result
