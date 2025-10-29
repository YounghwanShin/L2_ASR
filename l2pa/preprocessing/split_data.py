"""Cross-validation data splitting utilities.

This module handles splitting the dataset into cross-validation folds
based on speaker IDs, with separate test speakers.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']


def create_phoneme_map(data_dict: Dict) -> Dict[str, int]:
  """Create phoneme to ID mapping from dataset.
  
  Args:
    data_dict: Dictionary of dataset samples.
    
  Returns:
    Dictionary mapping phoneme strings to integer IDs.
  """
  phoneme_set = set()
  
  for item in data_dict.values():
    for key in ['canonical_train_target', 'perceived_train_target']:
      phonemes = item.get(key, '')
      if phonemes:
        phoneme_set.update(phonemes.split())
  
  # Sort for consistency and add blank at index 0
  phonemes_sorted = ['<blank>'] + sorted(phoneme_set)
  return {phoneme: idx for idx, phoneme in enumerate(phonemes_sorted)}


def get_speaker_ids(data_dict: Dict) -> List[str]:
  """Extract unique speaker IDs from dataset.
  
  Args:
    data_dict: Dictionary of dataset samples.
    
  Returns:
    Sorted list of unique speaker IDs.
  """
  speakers = set(item.get('spk_id', '') for item in data_dict.values())
  return sorted(speakers - {''})


def split_test_train_speakers(
    all_speakers: List[str],
    test_speakers: List[str]
) -> tuple[List[str], List[str]]:
  """Split speakers into test and train sets.
  
  Args:
    all_speakers: List of all speaker IDs.
    test_speakers: List of speaker IDs for test set.
    
  Returns:
    Tuple of (test_speakers, train_speakers).
  """
  test_set = set(test_speakers)
  train_speakers = [s for s in all_speakers if s not in test_set]
  return list(test_set), train_speakers


def create_cv_folds(
    data_dict: Dict,
    train_speakers: List[str],
    output_dir: str
) -> int:
  """Create cross-validation folds for training speakers.
  
  Each fold uses one training speaker as validation set.
  
  Args:
    data_dict: Complete dataset dictionary.
    train_speakers: List of training speaker IDs.
    output_dir: Output directory for fold data.
    
  Returns:
    Number of folds created.
  """
  num_folds = len(train_speakers)
  
  for fold_idx, val_speaker in enumerate(train_speakers):
    fold_dir = Path(output_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_data = {}
    val_data = {}
    
    for file_path, item in data_dict.items():
      speaker_id = item.get('spk_id', '')
      if speaker_id == val_speaker:
        val_data[file_path] = item
      elif speaker_id in train_speakers:
        train_data[file_path] = item
    
    # Save fold data
    with open(fold_dir / 'train_labels.json', 'w', encoding='utf-8') as f:
      json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(fold_dir / 'val_labels.json', 'w', encoding='utf-8') as f:
      json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fold {fold_idx}: Val={val_speaker} "
          f"(Train={len(train_data)}, Val={len(val_data)})")
  
  return num_folds


def create_test_split(
    data_dict: Dict,
    test_speakers: List[str],
    output_dir: str
):
  """Create test split from test speakers.
  
  Args:
    data_dict: Complete dataset dictionary.
    test_speakers: List of test speaker IDs.
    output_dir: Output directory.
  """
  test_data = {
      path: item for path, item in data_dict.items()
      if item.get('spk_id', '') in test_speakers
  }
  
  test_path = Path(output_dir) / 'test_labels.json'
  with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
  
  print(f"Test set: {len(test_data)} samples from {len(test_speakers)} speakers")


def split_dataset_for_cv(
    input_path: str,
    output_dir: str,
    test_speakers: Optional[List[str]] = None
) -> Dict:
  """Split dataset into cross-validation folds.
  
  Creates:
    - test_labels.json: Fixed test set
    - fold_X/train_labels.json: Training set for fold X
    - fold_X/val_labels.json: Validation set for fold X
    - phoneme_map.json: Phoneme to ID mapping
  
  Args:
    input_path: Path to complete dataset JSON.
    output_dir: Output directory for splits.
    test_speakers: List of test speaker IDs.
    
  Returns:
    Dictionary with split statistics.
  """
  if test_speakers is None:
    test_speakers = DEFAULT_TEST_SPEAKERS
  
  print(f"Loading dataset from {input_path}...")
  with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  
  print(f"Total samples: {len(data)}")
  
  # Get speakers
  all_speakers = get_speaker_ids(data)
  test_speakers_actual, train_speakers = split_test_train_speakers(
      all_speakers, test_speakers
  )
  
  print(f"\nSpeaker distribution:")
  print(f"  Test speakers: {test_speakers_actual}")
  print(f"  Train speakers: {train_speakers}")
  print(f"  Total folds: {len(train_speakers)}")
  
  # Create output directory
  os.makedirs(output_dir, exist_ok=True)
  
  # Create test split
  create_test_split(data, test_speakers_actual, output_dir)
  
  # Create CV folds
  print(f"\nCreating {len(train_speakers)} cross-validation folds...")
  num_folds = create_cv_folds(data, train_speakers, output_dir)
  
  # Create phoneme map
  print("\nCreating phoneme map...")
  phoneme_map = create_phoneme_map(data)
  phoneme_map_path = Path(output_dir) / 'phoneme_map.json'
  with open(phoneme_map_path, 'w', encoding='utf-8') as f:
    json.dump(phoneme_map, f, indent=2)
  print(f"Total phonemes: {len(phoneme_map)}")
  
  # Statistics
  stats = {
      'total_samples': len(data),
      'num_folds': num_folds,
      'test_speakers': test_speakers_actual,
      'train_speakers': train_speakers,
      'num_phonemes': len(phoneme_map)
  }
  
  # Save statistics
  stats_path = Path(output_dir) / 'split_statistics.json'
  with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)
  
  print("\nDataset splitting complete!")
  return stats


if __name__ == '__main__':
  # Example usage
  split_dataset_for_cv(
      input_path='data/processed_with_error.json',
      output_dir='data'
  )
