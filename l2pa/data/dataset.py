"""Dataset module for pronunciation assessment model.

This module implements efficient dataset loading with metadata caching
and lazy loading to minimize redundant file I/O operations.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class PronunciationDataset(Dataset):
  """Dataset for pronunciation assessment with multitask support.
  
  Features lazy loading and metadata caching to avoid redundant file reads.
  Validates data availability based on training mode requirements.
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
      json_path: Path to data JSON file containing file paths and labels.
      phoneme_to_id: Dictionary mapping phoneme strings to integer IDs.
      training_mode: Training mode ('phoneme_only', 'phoneme_error', 'multitask').
      max_length: Maximum audio length in samples (filters longer files).
      sampling_rate: Target sampling rate for audio.
      device: Device for data loading.
    """
    self.phoneme_to_id = phoneme_to_id
    self.training_mode = training_mode
    self.sampling_rate = sampling_rate
    self.max_length = max_length
    self.device = device
    self.error_label_to_id = {'D': 1, 'I': 2, 'S': 3, 'C': 4}

    # Load dataset metadata
    with open(json_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)

    # Build metadata index with validation
    self.samples = self._build_sample_index()
    print(f"Loaded {len(self.samples)} valid samples from {json_path}")

  def _build_sample_index(self) -> List[Dict]:
    """Builds index of valid samples with metadata.
    
    Validates data availability based on training mode and caches
    audio duration metadata for length filtering.
    
    Returns:
      List of dictionaries containing sample metadata.
    """
    samples = []
    
    for file_path, item in tqdm(
        self.data.items(), 
        desc="Building sample index"
    ):
      # Validate file existence
      if not os.path.exists(file_path):
        continue
      
      # Validate required labels based on training mode
      if not self._has_required_labels(item):
        continue
      
      # Get audio duration for length filtering
      duration_seconds = item.get('duration')
      if duration_seconds is None:
        # Fallback: load audio to get duration
        try:
          info = torchaudio.info(file_path)
          duration_seconds = info.num_frames / info.sample_rate
        except Exception:
          continue
      
      # Convert duration to samples
      audio_length = int(duration_seconds * self.sampling_rate)
      
      # Apply length filter if specified
      if self.max_length and audio_length > self.max_length:
        continue
      
      # Store sample metadata
      samples.append({
          'file_path': file_path,
          'audio_length': audio_length,
          'speaker_id': item.get('spk_id', 'UNKNOWN'),
          'labels': item
      })
    
    return samples

  def _has_required_labels(self, item: Dict) -> bool:
    """Checks if item has required labels for current training mode.
    
    Args:
      item: Dictionary containing label data.
      
    Returns:
      True if item has all required labels, False otherwise.
    """
    has_canonical = bool(item.get('canonical_train_target', '').strip())
    has_perceived = bool(item.get('perceived_train_target', '').strip())
    
    if self.training_mode == 'multitask':
      return has_canonical and has_perceived
    else:
      # phoneme_only and phoneme_error both require perceived
      return has_perceived

  def __len__(self) -> int:
    """Returns number of samples in dataset."""
    return len(self.samples)

  def _load_audio(self, file_path: str) -> torch.Tensor:
    """Loads and preprocesses audio file.
    
    Args:
      file_path: Path to audio file.
      
    Returns:
      Preprocessed audio waveform tensor of shape [num_samples].
      
    Raises:
      FileNotFoundError: If audio file does not exist.
      RuntimeError: If audio loading fails.
    """
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != self.sampling_rate:
      resampler = torchaudio.transforms.Resample(
          sample_rate, 
          self.sampling_rate
      )
      waveform = resampler(waveform)
    
    return waveform.squeeze(0)

  def _parse_phoneme_labels(
      self, 
      phoneme_string: str
  ) -> torch.Tensor:
    """Parses phoneme string into tensor of IDs.
    
    Args:
      phoneme_string: Space-separated phoneme string.
      
    Returns:
      Tensor of phoneme IDs.
    """
    if not phoneme_string or not phoneme_string.strip():
      return torch.tensor([], dtype=torch.long)
    
    phonemes = phoneme_string.split()
    ids = [self.phoneme_to_id.get(p, 0) for p in phonemes]
    return torch.tensor(ids, dtype=torch.long)

  def _parse_error_labels(self, error_string: str) -> torch.Tensor:
    """Parses error label string into tensor of IDs.
    
    Args:
      error_string: Space-separated error labels (D, I, S, C).
      
    Returns:
      Tensor of error label IDs.
    """
    if not error_string or not error_string.strip():
      return torch.tensor([], dtype=torch.long)
    
    errors = error_string.split()
    ids = [self.error_label_to_id.get(e, 0) for e in errors]
    return torch.tensor(ids, dtype=torch.long)

  def __getitem__(self, idx: int) -> Dict:
    """Returns a single sample.
    
    Args:
      idx: Sample index.
      
    Returns:
      Dictionary containing:
        - waveform: Audio tensor [num_samples]
        - audio_length: Length of audio
        - file_path: Path to audio file
        - speaker_id: Speaker identifier
        - canonical_labels: Canonical phoneme IDs (if available)
        - canonical_length: Number of canonical phonemes
        - perceived_labels: Perceived phoneme IDs
        - perceived_length: Number of perceived phonemes
        - error_labels: Error type IDs (if available)
        - error_length: Number of error labels
    """
    sample_metadata = self.samples[idx]
    labels = sample_metadata['labels']
    
    # Lazy load audio
    waveform = self._load_audio(sample_metadata['file_path'])
    
    # Build result dictionary
    result = {
        'waveform': waveform,
        'audio_length': torch.tensor(waveform.shape[0], dtype=torch.long),
        'file_path': sample_metadata['file_path'],
        'speaker_id': sample_metadata['speaker_id']
    }
    
    # Parse canonical labels
    canonical = labels.get('canonical_train_target', '')
    canonical_labels = self._parse_phoneme_labels(canonical)
    result['canonical_labels'] = canonical_labels
    result['canonical_length'] = torch.tensor(len(canonical_labels))
    
    # Parse perceived labels
    perceived = labels.get('perceived_train_target', '')
    perceived_labels = self._parse_phoneme_labels(perceived)
    result['perceived_labels'] = perceived_labels
    result['perceived_length'] = torch.tensor(len(perceived_labels))
    
    # Parse error labels
    errors = labels.get('error_labels', '')
    error_labels = self._parse_error_labels(errors)
    result['error_labels'] = error_labels
    result['error_length'] = torch.tensor(len(error_labels))
    
    return result


def collate_batch(
    batch: List[Dict], 
    training_mode: str = 'multitask'
) -> Optional[Dict]:
  """Collates batch samples with proper padding.
  
  Args:
    batch: List of sample dictionaries from dataset.
    training_mode: Current training mode.
    
  Returns:
    Dictionary containing batched and padded tensors, or None if no valid samples.
  """
  # Filter valid samples (must have perceived labels)
  valid_samples = [
      item for item in batch
      if item['perceived_labels'] is not None 
      and len(item['perceived_labels']) > 0
  ]
  
  if not valid_samples:
    return None
  
  # Pad waveforms to maximum length in batch
  waveforms = [sample['waveform'] for sample in valid_samples]
  max_audio_len = max(waveform.shape[0] for waveform in waveforms)
  padded_waveforms = torch.stack([
      torch.nn.functional.pad(
          waveform, 
          (0, max_audio_len - waveform.shape[0])
      )
      for waveform in waveforms
  ])
  
  # Build result dictionary
  result = {
      'waveforms': padded_waveforms,
      'audio_lengths': torch.tensor(
          [s['audio_length'] for s in valid_samples]
      ),
      'file_paths': [s['file_path'] for s in valid_samples],
      'speaker_ids': [s['speaker_id'] for s in valid_samples]
  }
  
  # Pad canonical labels
  canonical_labels = [s['canonical_labels'] for s in valid_samples]
  if canonical_labels and all(len(l) > 0 for l in canonical_labels):
    max_len = max(l.shape[0] for l in canonical_labels)
    result['canonical_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in canonical_labels
    ])
    result['canonical_lengths'] = torch.tensor(
        [s['canonical_length'] for s in valid_samples]
    )
  
  # Pad perceived labels
  perceived_labels = [s['perceived_labels'] for s in valid_samples]
  if perceived_labels and all(len(l) > 0 for l in perceived_labels):
    max_len = max(l.shape[0] for l in perceived_labels)
    result['perceived_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in perceived_labels
    ])
    result['perceived_lengths'] = torch.tensor(
        [s['perceived_length'] for s in valid_samples]
    )
  
  # Pad error labels
  error_labels = [s['error_labels'] for s in valid_samples]
  if error_labels and all(len(l) > 0 for l in error_labels):
    max_len = max(l.shape[0] for l in error_labels)
    result['error_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
        for l in error_labels
    ])
    result['error_lengths'] = torch.tensor(
        [s['error_length'] for s in valid_samples]
    )
  
  return result
