"""Data splitting utilities for cross-validation.

This module handles splitting the dataset into cross-validation folds
based on speaker IDs, with separate test speakers.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']


def verify_phoneme_mapping(data_dict: Dict, phoneme_map_path: str) -> bool:
    """Verifies that all phonemes in dataset exist in mapping."""
    with open(phoneme_map_path, 'r') as f:
        phoneme_to_id = json.load(f)
    
    dataset_phonemes = set()
    for item in data_dict.values():
        for key in ['canonical_train_target', 'perceived_train_target']:
            phonemes = item.get(key, '')
            if phonemes:
                dataset_phonemes.update(phonemes.split())
    
    missing_phonemes = dataset_phonemes - set(phoneme_to_id.keys())
    
    if missing_phonemes:
        print(f"Warning: Phonemes not in mapping: {missing_phonemes}")
        return False
    
    return True


def get_speaker_ids(data_dict: Dict) -> List[str]:
    """Extracts unique speaker IDs from dataset."""
    speakers = set(item.get('spk_id', '') for item in data_dict.values())
    return sorted(speakers - {''})


def split_test_train_speakers(
    all_speakers: List[str],
    test_speakers: List[str]
) -> tuple[List[str], List[str]]:
    """Splits speakers into test and train sets."""
    test_set = set(test_speakers)
    train_speakers = [s for s in all_speakers if s not in test_set]
    return list(test_set), train_speakers


def create_cv_folds(
    data_dict: Dict,
    train_speakers: List[str],
    output_dir: str
) -> int:
    """Creates cross-validation folds for training speakers."""
    num_folds = len(train_speakers)
    
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
        
        print(f"Fold {fold_idx}: Val={val_speaker} "
              f"(Train={len(train_data)}, Val={len(val_data)})")
    
    return num_folds


def create_test_split(
    data_dict: Dict,
    test_speakers: List[str],
    output_dir: str
):
    """Creates test split from test speakers."""
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
    phoneme_map_path: str,
    test_speakers: Optional[List[str]] = None
) -> Dict:
    """Splits dataset into cross-validation folds."""
    if test_speakers is None:
        test_speakers = DEFAULT_TEST_SPEAKERS
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    print(f"\nVerifying phoneme mapping...")
    if not verify_phoneme_mapping(data, phoneme_map_path):
        print("Warning: Some phonemes may not be in the mapping file")
    else:
        print("All phonemes verified in mapping")
    
    all_speakers = get_speaker_ids(data)
    test_speakers_actual, train_speakers = split_test_train_speakers(
        all_speakers, test_speakers
    )
    
    print(f"\nSpeaker distribution:")
    print(f"  Test speakers: {test_speakers_actual}")
    print(f"  Train speakers: {train_speakers}")
    print(f"  Total folds: {len(train_speakers)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    create_test_split(data, test_speakers_actual, output_dir)
    
    print(f"\nCreating {len(train_speakers)} cross-validation folds...")
    num_folds = create_cv_folds(data, train_speakers, output_dir)
    
    stats = {
        'total_samples': len(data),
        'num_folds': num_folds,
        'test_speakers': test_speakers_actual,
        'train_speakers': train_speakers
    }
    
    stats_path = Path(output_dir) / 'split_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset splitting complete")
    return stats
