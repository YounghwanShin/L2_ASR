"""Data splitting utilities for train/validation/test sets.

This module handles dataset splitting with support for both standard
train/val/test splits and K-fold cross-validation based on speaker IDs.
"""

import json
import random
from pathlib import Path
from collections import defaultdict


# Default test speakers
TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']

# Default validation speakers
VALIDATION_SPEAKERS = []


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


def split_dataset_by_speakers(input_path, output_dir, test_speakers=None, val_speakers=None):
    """Splits dataset into train, validation, and test sets by speaker.
    
    Args:
        input_path: Path to input JSON file with complete dataset.
        output_dir: Directory where split files will be saved.
        test_speakers: List of speaker IDs for test set.
        val_speakers: List of speaker IDs for validation set.
    """
    if test_speakers is None:
        test_speakers = TEST_SPEAKERS
    if val_speakers is None:
        val_speakers = VALIDATION_SPEAKERS
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    print(f"Test speakers: {test_speakers}")
    print(f"Validation speakers: {val_speakers}")
    
    # Verify no overlap
    overlap = set(test_speakers) & set(val_speakers)
    if overlap:
        raise ValueError(f"Speakers cannot be in both test and validation: {overlap}")
    
    # Split data by speaker
    train_data = {}
    val_data = {}
    test_data = {}
    
    speaker_counts = defaultdict(int)
    
    for file_path, item in data.items():
        speaker_id = item.get('spk_id', '')
        speaker_counts[speaker_id] += 1
        
        if speaker_id in test_speakers:
            test_data[file_path] = item
        elif speaker_id in val_speakers:
            val_data[file_path] = item
        else:
            train_data[file_path] = item
    
    _print_speaker_distribution(speaker_counts, test_speakers, val_speakers)
    _save_splits(output_dir, train_data, val_data, test_data, data)


def create_cross_validation_splits(input_path, output_dir, num_folds=5, test_speakers=None, seed=42):
    """Creates K-fold cross-validation splits.
    
    Splits non-test speakers into K folds for cross-validation while
    keeping test speakers separate for final evaluation.
    
    Args:
        input_path: Path to input JSON file with complete dataset.
        output_dir: Directory where split files will be saved.
        num_folds: Number of cross-validation folds.
        test_speakers: List of speaker IDs for test set.
        seed: Random seed for reproducibility.
    """
    if test_speakers is None:
        test_speakers = TEST_SPEAKERS
    
    print(f"Creating {num_folds}-fold cross-validation splits...")
    print(f"Loading dataset from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    print(f"Test speakers (held out): {test_speakers}")
    
    # Separate test data
    test_data = {}
    train_val_data_by_speaker = defaultdict(dict)
    
    for file_path, item in data.items():
        speaker_id = item.get('spk_id', '')
        
        if speaker_id in test_speakers:
            test_data[file_path] = item
        else:
            train_val_data_by_speaker[speaker_id][file_path] = item
    
    # Get list of train/val speakers
    train_val_speakers = list(train_val_data_by_speaker.keys())
    
    print(f"\nTrain/Val speakers: {len(train_val_speakers)}")
    print(f"Test samples: {len(test_data)}")
    
    # Shuffle speakers for random fold assignment
    random.seed(seed)
    random.shuffle(train_val_speakers)
    
    # Create K folds
    fold_size = len(train_val_speakers) // num_folds
    folds = []
    
    for i in range(num_folds):
        start_idx = i * fold_size
        if i == num_folds - 1:
            # Last fold gets remaining speakers
            fold_speakers = train_val_speakers[start_idx:]
        else:
            fold_speakers = train_val_speakers[start_idx:start_idx + fold_size]
        folds.append(fold_speakers)
    
    # Create and save cross-validation splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_idx in range(num_folds):
        print(f"\n--- Fold {fold_idx} ---")
        
        # Validation speakers for this fold
        val_speakers = folds[fold_idx]
        
        # Training speakers: all other folds
        train_speakers = []
        for other_fold_idx in range(num_folds):
            if other_fold_idx != fold_idx:
                train_speakers.extend(folds[other_fold_idx])
        
        # Create train and validation data
        train_data = {}
        val_data = {}
        
        for speaker in train_speakers:
            train_data.update(train_val_data_by_speaker[speaker])
        
        for speaker in val_speakers:
            val_data.update(train_val_data_by_speaker[speaker])
        
        print(f"Train speakers: {len(train_speakers)} ({len(train_data)} samples)")
        print(f"Val speakers: {len(val_speakers)} ({len(val_data)} samples)")
        
        # Save fold splits
        train_path = output_dir / f'fold_{fold_idx}_train.json'
        val_path = output_dir / f'fold_{fold_idx}_val.json'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {train_path}")
        print(f"Saved: {val_path}")
    
    # Save test data (same for all folds)
    test_path = output_dir / 'test_labels.json'
    print(f"\nSaving test data to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Create phoneme map
    print("\nCreating phoneme map...")
    phoneme_map = create_phoneme_map(data)
    phoneme_map_path = output_dir / 'phoneme_map.json'
    with open(phoneme_map_path, 'w', encoding='utf-8') as f:
        json.dump(phoneme_map, f, indent=2)
    print(f"Total phonemes: {len(phoneme_map)}")
    print(f"Saved: {phoneme_map_path}")
    
    print("\n" + "="*80)
    print("Cross-validation splits created successfully!")
    print("="*80)


def _print_speaker_distribution(speaker_counts, test_speakers, val_speakers):
    """Prints speaker distribution across splits.
    
    Args:
        speaker_counts: Dictionary of speaker sample counts.
        test_speakers: List of test speaker IDs.
        val_speakers: List of validation speaker IDs.
    """
    print("\nSpeaker distribution:")
    for speaker_id, count in sorted(speaker_counts.items()):
        if speaker_id in test_speakers:
            split_type = "TEST"
        elif speaker_id in val_speakers:
            split_type = "VAL"
        else:
            split_type = "TRAIN"
        print(f"  {speaker_id}: {count} samples ({split_type})")


def _save_splits(output_dir, train_data, val_data, test_data, full_data):
    """Saves train/val/test splits and phoneme map.
    
    Args:
        output_dir: Output directory path.
        train_data: Training data dictionary.
        val_data: Validation data dictionary.
        test_data: Test data dictionary.
        full_data: Complete dataset dictionary.
    """
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = [
        ('train_labels.json', train_data),
        ('val_labels.json', val_data),
        ('test_labels.json', test_data)
    ]
    
    for filename, data in splits:
        path = output_dir / filename
        print(f"\nSaving {filename}...")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Create and save phoneme map
    print("\nCreating phoneme map...")
    phoneme_map = create_phoneme_map(full_data)
    phoneme_map_path = output_dir / 'phoneme_map.json'
    print(f"Total phonemes: {len(phoneme_map)}")
    with open(phoneme_map_path, 'w', encoding='utf-8') as f:
        json.dump(phoneme_map, f, indent=2)
    
    print("\nDataset splitting complete!")
