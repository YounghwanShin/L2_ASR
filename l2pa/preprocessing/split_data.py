"""Data splitting for cross-validation.

Creates speaker-based cross-validation folds and test set.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


DEFAULT_TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']


def get_speakers(data_dict: Dict) -> List[str]:
  """Extracts unique speaker IDs.
  
  Args:
    data_dict: Dataset dictionary.
  
  Returns:
    Sorted list of speaker IDs.
  """
  speakers = set(item.get('spk_id', '') for item in data_dict.values())
  return sorted(speakers - {''})


def create_cv_folds(data_dict: Dict, train_speakers: List[str], output_dir: str) -> int:
  """Creates cross-validation folds.
  
  Args:
    data_dict: Complete dataset.
    train_speakers: Training speaker IDs.
    output_dir: Output directory.
  
  Returns:
    Number of folds created.
  """
  for fold_idx, val_speaker in enumerate(train_speakers):
    fold_dir = Path(output_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = {}
    val_data = {}
    
    for file_path, item in data_dict.items():
      speaker_id = item.get('spk_id', '')
      if speaker_id == val_speaker:
        val_data[file_path] = item
      elif speaker_id in train_speakers:
        train_data[file_path] = item
    
    with open(fold_dir / 'train_labels.json', 'w', encoding='utf-8') as f:
      json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(fold_dir / 'val_labels.json', 'w', encoding='utf-8') as f:
      json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f'Fold {fold_idx}: Val={val_speaker} (Train={len(train_data)}, Val={len(val_data)})')
  
  return len(train_speakers)


def split_dataset(input_path: str, output_dir: str, test_speakers: List[str] = None):
  """Splits dataset for cross-validation.
  
  Args:
    input_path: Input dataset JSON path.
    output_dir: Output directory.
    test_speakers: Test speaker IDs.
  """
  if test_speakers is None:
    test_speakers = DEFAULT_TEST_SPEAKERS
  
  print(f'Loading {input_path}...')
  with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  
  # Get speakers
  all_speakers = get_speakers(data)
  train_speakers = [s for s in all_speakers if s not in test_speakers]
  
  print(f'\nTotal samples: {len(data)}')
  print(f'Test speakers: {test_speakers}')
  print(f'Train speakers: {train_speakers}')
  
  os.makedirs(output_dir, exist_ok=True)
  
  # Create test split
  test_data = {
      path: item for path, item in data.items()
      if item.get('spk_id', '') in test_speakers
  }
  
  test_path = Path(output_dir) / 'test_labels.json'
  with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
  
  print(f'\nTest set: {len(test_data)} samples')
  
  # Create CV folds
  print(f'\nCreating {len(train_speakers)} CV folds...')
  num_folds = create_cv_folds(data, train_speakers, output_dir)
  
  # Save statistics
  stats = {
      'total_samples': len(data),
      'num_folds': num_folds,
      'test_speakers': test_speakers,
      'train_speakers': train_speakers
  }
  
  stats_path = Path(output_dir) / 'split_statistics.json'
  with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
  
  print('\nSplitting complete!')
