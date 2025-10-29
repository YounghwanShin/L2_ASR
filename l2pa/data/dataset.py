"""Dataset module for pronunciation assessment model.

This module implements the dataset class for loading audio data with
canonical phonemes, perceived phonemes, and error labels.
"""

import json
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List, Optional


class PronunciationDataset(Dataset):
  """Dataset for pronunciation assessment with multitask support."""
  
  def __init__(self,
               json_path: str,
               phoneme_to_id: Dict[str, int],
               training_mode: str,
               max_length: Optional[int] = None,
               sampling_rate: int = 16000,
               device: str = 'cuda'):
    """Initialize dataset.
    
    Args:
      json_path: Path to data JSON file.
      phoneme_to_id: Phoneme to ID mapping.
      training_mode: Training mode.
      max_length: Maximum audio length in samples.
      sampling_rate: Target sampling rate.
      device: Device for data loading.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)

    self.file_paths = list(self.data.keys())
    self.phoneme_to_id = phoneme_to_id
    self.training_mode = training_mode
    self.sampling_rate = sampling_rate
    self.max_length = max_length
    self.device = device
    
    # Error label mapping
    self.error_mapping = {'D': 1, 'I': 2, 'S': 3, 'C': 4}

    self.valid_files = []
    self._filter_valid_files()

    if self.max_length:
      self._filter_by_length()

  def _filter_valid_files(self):
    """Filter valid files based on training mode and label availability."""
    for file_path in self.file_paths:
      item = self.data[file_path]
      has_canonical = bool(item.get('canonical_train_target', '').strip())
      has_perceived = bool(item.get('perceived_train_target', '').strip())
      has_error = bool(item.get('error_labels', '').strip())

      if self.training_mode == 'phoneme_only' and has_perceived:
        self.valid_files.append(file_path)
      elif self.training_mode == 'phoneme_error' and has_perceived:
        self.valid_files.append(file_path)
      elif self.training_mode == 'multitask' and has_canonical and has_perceived:
        self.valid_files.append(file_path)
      else:
        self.valid_files.append(file_path)

  def _filter_by_length(self):
    """Filter files by audio length."""
    filtered_files = []
    excluded_count = 0

    for file_path in tqdm(self.valid_files, desc="Filtering by length"):
      try:
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
          waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.sampling_rate:
          resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
          waveform = resampler(waveform)

        if waveform.shape[1] <= self.max_length:
          filtered_files.append(file_path)
        else:
          excluded_count += 1

      except Exception as e:
        excluded_count += 1

    print(f"Length filtering: {len(self.valid_files)} → {len(filtered_files)} "
          f"({excluded_count} excluded)")
    self.valid_files = filtered_files

  def __len__(self):
    """Return number of samples."""
    return len(self.valid_files)

  def load_waveform(self, file_path: str) -> torch.Tensor:
    """Load and preprocess audio file.
    
    Args:
      file_path: Path to audio file.
      
    Returns:
      Preprocessed waveform tensor.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != self.sampling_rate:
      resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
      waveform = resampler(waveform)

    return waveform.squeeze(0)

  def __getitem__(self, idx: int) -> Dict:
    """Return a single dataset item.
    
    Args:
      idx: Index of the item.
      
    Returns:
      Dictionary with waveform, labels, and metadata.
    """
    file_path = self.valid_files[idx]
    item = self.data[file_path]

    waveform = self.load_waveform(file_path)
    result = {
        'waveform': waveform,
        'audio_length': torch.tensor(waveform.shape[0], dtype=torch.long),
        'file_path': file_path,
        'speaker_id': item.get('spk_id', 'UNKNOWN')
    }

    # Process canonical labels
    canonical = item.get('canonical_train_target', '')
    if canonical and canonical.strip():
      canonical_tokens = canonical.split()
      result['canonical_labels'] = torch.tensor(
          [self.phoneme_to_id.get(p, 0) for p in canonical_tokens],
          dtype=torch.long
      )
    else:
      result['canonical_labels'] = torch.tensor([], dtype=torch.long)
    result['canonical_length'] = torch.tensor(len(result['canonical_labels']))

    # Process perceived labels
    perceived = item.get('perceived_train_target', '')
    if perceived and perceived.strip():
      perceived_tokens = perceived.split()
      result['perceived_labels'] = torch.tensor(
          [self.phoneme_to_id.get(p, 0) for p in perceived_tokens],
          dtype=torch.long
      )
    else:
      result['perceived_labels'] = torch.tensor([], dtype=torch.long)
    result['perceived_length'] = torch.tensor(len(result['perceived_labels']))

    # Process error labels
    errors = item.get('error_labels', '')
    if errors and errors.strip():
      error_tokens = errors.split()
      error_ids = [self.error_mapping.get(e, 0) for e in error_tokens]
      result['error_labels'] = torch.tensor(error_ids, dtype=torch.long)
    else:
      result['error_labels'] = torch.tensor([], dtype=torch.long)
    result['error_length'] = torch.tensor(len(result['error_labels']))

    return result


def collate_batch(batch: List[Dict], training_mode: str = 'multitask') -> Optional[Dict]:
  """Collate batch data with proper padding.
  
  Args:
    batch: List of dataset items.
    training_mode: Current training mode.
    
  Returns:
    Collated batch dictionary with padded tensors.
  """
  valid_samples = [
      item for item in batch
      if item['perceived_labels'] is not None and len(item['perceived_labels']) > 0
  ]

  if not valid_samples:
    return None

  # Pad waveforms
  waveforms = [sample['waveform'] for sample in valid_samples]
  max_len = max(waveform.shape[0] for waveform in waveforms)
  padded_waveforms = torch.stack([
      torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
      for waveform in waveforms
  ])

  result = {
      'waveforms': padded_waveforms,
      'audio_lengths': torch.tensor([s['audio_length'] for s in valid_samples]),
      'file_paths': [s['file_path'] for s in valid_samples],
      'speaker_ids': [s['speaker_id'] for s in valid_samples]
  }

  # Pad canonical labels
  canonical_labels = [s['canonical_labels'] for s in valid_samples]
  if canonical_labels and all(len(l) > 0 for l in canonical_labels):
    max_canonical_len = max(l.shape[0] for l in canonical_labels)
    result['canonical_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_canonical_len - l.shape[0]), value=0)
        for l in canonical_labels
    ])
    result['canonical_lengths'] = torch.tensor([s['canonical_length'] for s in valid_samples])

  # Pad perceived labels
  perceived_labels = [s['perceived_labels'] for s in valid_samples]
  if perceived_labels and all(len(l) > 0 for l in perceived_labels):
    max_perceived_len = max(l.shape[0] for l in perceived_labels)
    result['perceived_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_perceived_len - l.shape[0]), value=0)
        for l in perceived_labels
    ])
    result['perceived_lengths'] = torch.tensor([s['perceived_length'] for s in valid_samples])

  # Pad error labels
  error_labels = [s['error_labels'] for s in valid_samples]
  if error_labels and all(len(l) > 0 for l in error_labels):
    max_error_len = max(l.shape[0] for l in error_labels)
    result['error_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_error_len - l.shape[0]), value=0)
        for l in error_labels
    ])
    result['error_lengths'] = torch.tensor([s['error_length'] for s in valid_samples])

  return result
