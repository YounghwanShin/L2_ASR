"""Dataset module for L2 pronunciation assessment.

Provides efficient data loading for multitask learning including:
  - Canonical phoneme recognition
  - Perceived phoneme recognition  
  - Error type classification (Deletion, Insertion, Substitution, Correct)
"""

import json
import os
from typing import Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class PronunciationDataset(Dataset):
  """Dataset for multitask pronunciation assessment.
  
  Loads audio files and corresponding labels for training multitask models
  that jointly learn phoneme recognition and pronunciation error detection.
  
  Attributes:
    data: Dictionary mapping file paths to label dictionaries.
    phoneme_to_id: Mapping from phoneme strings to integer IDs.
    error_to_id: Mapping from error labels (D/I/S/C) to integer IDs.
    training_mode: Mode determining which tasks are active.
    max_audio_length: Maximum allowed audio length in samples.
    target_sample_rate: Target sampling rate for audio resampling.
  """
  
  # Error label mapping
  ERROR_LABELS = {'D': 1, 'I': 2, 'S': 3, 'C': 4}
  BLANK_ID = 0
  
  def __init__(
      self,
      json_path: str,
      phoneme_to_id: Dict[str, int],
      training_mode: str,
      max_audio_length: Optional[int] = None,
      target_sample_rate: int = 16000
  ):
    """Initializes the pronunciation dataset.
    
    Args:
      json_path: Path to JSON file containing dataset annotations.
      phoneme_to_id: Dictionary mapping phoneme strings to IDs.
      training_mode: Training mode ('phoneme_only', 'phoneme_error', 'multitask').
      max_audio_length: Maximum audio length in samples. Longer files are skipped.
      target_sample_rate: Target sampling rate for audio processing.
    """
    self.phoneme_to_id = phoneme_to_id
    self.error_to_id = self.ERROR_LABELS
    self.training_mode = training_mode
    self.max_audio_length = max_audio_length
    self.target_sample_rate = target_sample_rate
    
    # Load dataset annotations
    with open(json_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)
    
    # Filter files based on existence and required labels
    self.valid_file_paths = self._filter_valid_files()
    
    print(f'Loaded {len(self.valid_file_paths)} samples from {json_path}')
  
  def _filter_valid_files(self) -> List[str]:
    """Filters files based on existence and label requirements.
    
    Returns:
      List of valid file paths that meet all requirements.
    """
    valid_paths = []
    
    for file_path, metadata in self.data.items():
      # Check file existence
      if not os.path.exists(file_path):
        continue
      
      # Check label requirements based on training mode
      has_perceived = bool(metadata.get('perceived_train_target', '').strip())
      has_canonical = bool(metadata.get('canonical_train_target', '').strip())
      
      # All modes require perceived phonemes
      if not has_perceived:
        continue
      
      # Multitask mode additionally requires canonical phonemes
      if self.training_mode == 'multitask' and not has_canonical:
        continue
      
      valid_paths.append(file_path)
    
    return valid_paths
  
  def _load_and_preprocess_audio(self, file_path: str) -> Optional[torch.Tensor]:
    """Loads and preprocesses audio file.
    
    Args:
      file_path: Path to audio file.
    
    Returns:
      Preprocessed mono waveform at target sample rate, or None if loading fails
      or audio exceeds maximum length.
    """
    try:
      waveform, sample_rate = torchaudio.load(file_path)
      
      # Convert stereo to mono
      if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
      
      # Resample if necessary
      if sample_rate != self.target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, self.target_sample_rate
        )
        waveform = resampler(waveform)
      
      waveform = waveform.squeeze(0)
      
      # Filter by length if specified
      if self.max_audio_length and waveform.shape[0] > self.max_audio_length:
        return None
      
      return waveform
      
    except Exception as e:
      print(f'Error loading {file_path}: {str(e)}')
      return None
  
  def _encode_phoneme_sequence(self, phoneme_string: str) -> torch.Tensor:
    """Encodes space-separated phoneme string to ID tensor.
    
    Args:
      phoneme_string: Space-separated phoneme sequence.
    
    Returns:
      Tensor of phoneme IDs. Returns empty tensor for empty input.
    """
    if not phoneme_string.strip():
      return torch.tensor([], dtype=torch.long)
    
    phonemes = phoneme_string.split()
    ids = [self.phoneme_to_id.get(p, self.BLANK_ID) for p in phonemes]
    return torch.tensor(ids, dtype=torch.long)
  
  def _encode_error_sequence(self, error_string: str) -> torch.Tensor:
    """Encodes space-separated error label string to ID tensor.
    
    Args:
      error_string: Space-separated error labels (D/I/S/C).
    
    Returns:
      Tensor of error IDs. Returns empty tensor for empty input.
    """
    if not error_string.strip():
      return torch.tensor([], dtype=torch.long)
    
    errors = error_string.split()
    ids = [self.error_to_id.get(e, self.BLANK_ID) for e in errors]
    return torch.tensor(ids, dtype=torch.long)
  
  def __len__(self) -> int:
    """Returns the number of valid samples."""
    return len(self.valid_file_paths)
  
  def __getitem__(self, idx: int) -> Optional[Dict]:
    """Retrieves a single training sample.
    
    Args:
      idx: Sample index.
    
    Returns:
      Dictionary containing waveform, labels, and metadata. Returns None if
      the sample cannot be loaded or processed.
    """
    file_path = self.valid_file_paths[idx]
    metadata = self.data[file_path]
    
    # Load and preprocess audio
    waveform = self._load_and_preprocess_audio(file_path)
    if waveform is None:
      return None
    
    # Prepare sample dictionary
    sample = {
        'waveform': waveform,
        'audio_length': torch.tensor(waveform.shape[0], dtype=torch.long),
        'file_path': file_path,
        'speaker_id': metadata.get('spk_id', 'UNKNOWN')
    }
    
    # Encode canonical phonemes
    canonical_phonemes = self._encode_phoneme_sequence(
        metadata.get('canonical_train_target', '')
    )
    sample['canonical_labels'] = canonical_phonemes
    sample['canonical_length'] = torch.tensor(
        len(canonical_phonemes), dtype=torch.long
    )
    
    # Encode perceived phonemes
    perceived_phonemes = self._encode_phoneme_sequence(
        metadata.get('perceived_train_target', '')
    )
    sample['perceived_labels'] = perceived_phonemes
    sample['perceived_length'] = torch.tensor(
        len(perceived_phonemes), dtype=torch.long
    )
    
    # Encode error labels
    error_labels = self._encode_error_sequence(
        metadata.get('error_labels', '')
    )
    sample['error_labels'] = error_labels
    sample['error_length'] = torch.tensor(len(error_labels), dtype=torch.long)
    
    return sample


def collate_batch(batch: List[Optional[Dict]]) -> Optional[Dict]:
  """Collates batch samples with padding for variable-length sequences.
  
  Args:
    batch: List of sample dictionaries from dataset.
  
  Returns:
    Dictionary with padded tensors and metadata lists. Returns None if batch
    is empty after filtering.
  """
  # Filter out None samples
  batch = [sample for sample in batch if sample is not None]
  
  if not batch:
    return None
  
  # Pad audio waveforms to maximum length in batch
  max_audio_length = max(sample['waveform'].shape[0] for sample in batch)
  padded_waveforms = torch.stack([
      torch.nn.functional.pad(
          sample['waveform'],
          (0, max_audio_length - sample['waveform'].shape[0])
      )
      for sample in batch
  ])
  
  # Create collated batch
  collated = {
      'waveforms': padded_waveforms,
      'audio_lengths': torch.tensor([s['audio_length'] for s in batch]),
      'file_paths': [s['file_path'] for s in batch],
      'speaker_ids': [s['speaker_id'] for s in batch]
  }
  
  # Pad and stack canonical labels if present and non-empty
  canonical_labels = [sample['canonical_labels'] for sample in batch]
  if canonical_labels and any(len(labels) > 0 for labels in canonical_labels):
    max_length = max(labels.shape[0] for labels in canonical_labels if len(labels) > 0)
    collated['canonical_labels'] = torch.stack([
        torch.nn.functional.pad(labels, (0, max_length - labels.shape[0]), value=0)
        for labels in canonical_labels
    ])
    collated['canonical_lengths'] = torch.tensor(
        [sample['canonical_length'] for sample in batch]
    )
  
  # Pad and stack perceived labels if present and non-empty
  perceived_labels = [sample['perceived_labels'] for sample in batch]
  if perceived_labels and any(len(labels) > 0 for labels in perceived_labels):
    max_length = max(labels.shape[0] for labels in perceived_labels if len(labels) > 0)
    collated['perceived_labels'] = torch.stack([
        torch.nn.functional.pad(labels, (0, max_length - labels.shape[0]), value=0)
        for labels in perceived_labels
    ])
    collated['perceived_lengths'] = torch.tensor(
        [sample['perceived_length'] for sample in batch]
    )
  
  # Pad and stack error labels if present and non-empty
  error_labels = [sample['error_labels'] for sample in batch]
  if error_labels and any(len(labels) > 0 for labels in error_labels):
    max_length = max(labels.shape[0] for labels in error_labels if len(labels) > 0)
    collated['error_labels'] = torch.stack([
        torch.nn.functional.pad(labels, (0, max_length - labels.shape[0]), value=0)
        for labels in error_labels
    ])
    collated['error_lengths'] = torch.tensor(
        [sample['error_length'] for sample in batch]
    )
  
  return collated
