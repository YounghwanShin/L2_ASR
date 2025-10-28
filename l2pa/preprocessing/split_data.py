"""Data splitting utilities for train/validation/test sets.

This module handles splitting the dataset into training, validation, and test sets
based on speaker IDs. Test speakers are predefined, and the remaining speakers
are used for training.
"""

import json
from pathlib import Path
from collections import defaultdict


# Test speakers as specified
TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']


def create_phoneme_map(data_dict):
    """Creates phoneme to ID mapping from dataset.
    
    Args:
        data_dict: Dictionary of dataset samples.
        
    Returns:
        Dictionary mapping phoneme strings to integer IDs.
    """
    phoneme_set = set()
    
    for item in data_dict.values():
        canonical = item.get('canonical_train_target', '')
        perceived = item.get('perceived_train_target', '')
        
        if canonical:
            phoneme_set.update(canonical.split())
        if perceived:
            phoneme_set.update(perceived.split())
    
    # Sort for consistency and add blank at index 0
    phonemes_sorted = ['<blank>'] + sorted(phoneme_set)
    phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(phonemes_sorted)}
    
    return phoneme_to_id


def split_dataset_by_speakers(input_path, output_dir, test_speakers=None):
    """Splits dataset into train and test sets based on speaker IDs.
    
    Args:
        input_path: Path to input JSON file with complete dataset.
        output_dir: Directory where split files will be saved.
        test_speakers: List of speaker IDs to use for test set.
    """
    if test_speakers is None:
        test_speakers = TEST_SPEAKERS
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    print(f"Test speakers: {test_speakers}")
    
    # Split data by speaker
    train_data = {}
    test_data = {}
    
    speaker_counts = defaultdict(int)
    
    for file_path, item in data.items():
        speaker_id = item.get('spk_id', '')
        speaker_counts[speaker_id] += 1
        
        if speaker_id in test_speakers:
            test_data[file_path] = item
        else:
            train_data[file_path] = item
    
    print("\nSpeaker distribution:")
    for speaker_id, count in sorted(speaker_counts.items()):
        split_type = "TEST" if speaker_id in test_speakers else "TRAIN"
        print(f"  {speaker_id}: {count} samples ({split_type})")
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data
    train_path = output_dir / 'train_labels.json'
    print(f"\nSaving train data to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Use train data as validation data (can be used with k-fold CV)
    val_path = output_dir / 'val_labels.json'
    print(f"Saving validation data to {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save test data
    test_path = output_dir / 'test_labels.json'
    print(f"Saving test data to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Create phoneme map
    print("\nCreating phoneme map...")
    phoneme_map = create_phoneme_map(data)
    phoneme_map_path = output_dir / 'phoneme_map.json'
    print(f"Saving phoneme map to {phoneme_map_path}...")
    print(f"Total phonemes: {len(phoneme_map)}")
    with open(phoneme_map_path, 'w', encoding='utf-8') as f:
        json.dump(phoneme_map, f, indent=2)
    
    print("\nDataset splitting complete!")