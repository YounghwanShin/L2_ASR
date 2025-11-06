"""Data splitting utilities for cross-validation.

This module handles splitting the dataset into cross-validation folds
based on speaker IDs, with separate test speakers. Also supports creating
disjoint text splits where test and train/val sets have no overlapping
transcripts.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


DEFAULT_TEST_SPEAKERS = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']
DEFAULT_VAL_SPEAKER = 'MBMPS'


def verify_phoneme_mapping(data_dict: Dict, phoneme_map_path: str) -> bool:
    """Verifies that all phonemes in dataset exist in mapping.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
        phoneme_map_path: Path to phoneme to ID mapping JSON file.
    
    Returns:
        True if all phonemes are in mapping, False otherwise.
    """
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
    """Extracts unique speaker IDs from dataset.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
    
    Returns:
        Sorted list of unique speaker IDs.
    """
    speakers = set(item.get('spk_id', '') for item in data_dict.values())
    return sorted(speakers - {''})


def split_test_train_speakers(
    all_speakers: List[str],
    test_speakers: List[str]
) -> tuple[List[str], List[str]]:
    """Splits speakers into test and train sets.
    
    Args:
        all_speakers: List of all speaker IDs.
        test_speakers: List of speaker IDs to use for testing.
    
    Returns:
        Tuple of (test_speaker_list, train_speaker_list).
    """
    test_set = set(test_speakers)
    train_speakers = [s for s in all_speakers if s not in test_set]
    return list(test_set), train_speakers


def create_cv_folds(
    data_dict: Dict,
    train_speakers: List[str],
    output_dir: str
) -> int:
    """Creates cross-validation folds for training speakers.
    
    Each fold uses one training speaker for validation and the rest for training.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
        train_speakers: List of speaker IDs to use for training/validation.
        output_dir: Output directory for fold data.
    
    Returns:
        Number of folds created.
    """
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
    """Creates test split from test speakers.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
        test_speakers: List of speaker IDs for test set.
        output_dir: Output directory for test data.
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
    phoneme_map_path: str,
    test_speakers: Optional[List[str]] = None
) -> Dict:
    """Splits dataset into cross-validation folds.
    
    Creates multiple training folds with speaker-based validation splits
    and a separate test set.
    
    Args:
        input_path: Path to input JSON file with preprocessed data.
        output_dir: Output directory for split data.
        phoneme_map_path: Path to phoneme mapping file.
        test_speakers: List of speaker IDs for test set.
    
    Returns:
        Dictionary containing split statistics.
    """
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


def get_unique_transcripts(data_dict: Dict) -> Dict[str, List[str]]:
    """Groups file paths by unique transcript text.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
    
    Returns:
        Dictionary mapping transcript text to list of file paths.
    """
    transcript_groups = defaultdict(list)
    
    for file_path, item in data_dict.items():
        transcript = item.get('wrd', '').strip()
        if transcript:
            transcript_groups[transcript].append(file_path)
    
    return dict(transcript_groups)


def select_test_transcripts(
    data_dict: Dict,
    test_speakers: List[str],
    num_transcripts: int = 100
) -> Set[str]:
    """Selects transcripts for test set from test speaker data.
    
    Selects transcripts that have samples from all test speakers to ensure
    balanced test set coverage.
    
    Args:
        data_dict: Dataset dictionary with file paths as keys.
        test_speakers: List of test speaker IDs.
        num_transcripts: Number of unique transcripts to select for test set.
    
    Returns:
        Set of selected transcript texts for test set.
    """
    transcript_speakers = defaultdict(set)
    
    for file_path, item in data_dict.items():
        speaker_id = item.get('spk_id', '')
        if speaker_id in test_speakers:
            transcript = item.get('wrd', '').strip()
            if transcript:
                transcript_speakers[transcript].add(speaker_id)
    
    available_transcripts = [
        transcript for transcript, speakers in transcript_speakers.items()
        if len(speakers) == len(test_speakers)
    ]
    
    if len(available_transcripts) < num_transcripts:
        print(f"Warning: Only {len(available_transcripts)} transcripts have all test speakers")
        print(f"Using all available transcripts instead of {num_transcripts}")
        return set(available_transcripts)
    
    available_transcripts.sort()
    selected = set(available_transcripts[:num_transcripts])
    
    return selected


def split_dataset_disjoint_text(
    input_path: str,
    output_dir: str,
    phoneme_map_path: str,
    test_speakers: Optional[List[str]] = None,
    val_speaker: str = DEFAULT_VAL_SPEAKER,
    num_test_transcripts: int = 100
) -> Dict:
    """Splits dataset with disjoint transcripts between test and train/val sets.
    
    Creates a split where test set uses specific transcripts that do not appear
    in training or validation sets. This prevents canonical phoneme information
    leakage between splits.
    
    Args:
        input_path: Path to input JSON file with preprocessed data.
        output_dir: Output directory for split data.
        phoneme_map_path: Path to phoneme mapping file.
        test_speakers: List of speaker IDs for test set.
        val_speaker: Speaker ID to use for validation set.
        num_test_transcripts: Number of unique transcripts for test set.
    
    Returns:
        Dictionary containing split statistics.
    """
    if test_speakers is None:
        test_speakers = DEFAULT_TEST_SPEAKERS
    
    print(f"\nCreating disjoint text split...")
    print(f"Loading dataset from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    print(f"\nVerifying phoneme mapping...")
    if not verify_phoneme_mapping(data, phoneme_map_path):
        print("Warning: Some phonemes may not be in the mapping file")
    else:
        print("All phonemes verified in mapping")
    
    print(f"\nSelecting {num_test_transcripts} test transcripts...")
    test_transcripts = select_test_transcripts(
        data, test_speakers, num_test_transcripts
    )
    print(f"Selected {len(test_transcripts)} unique transcripts for test set")
    
    split_dir = Path(output_dir) / 'disjoint_wrd_split'
    split_dir.mkdir(parents=True, exist_ok=True)
    
    test_data = {}
    val_data = {}
    train_data = {}
    
    for file_path, item in data.items():
        speaker_id = item.get('spk_id', '')
        transcript = item.get('wrd', '').strip()
        
        if transcript in test_transcripts and speaker_id in test_speakers:
            test_data[file_path] = item
        elif transcript not in test_transcripts:
            if speaker_id == val_speaker:
                val_data[file_path] = item
            elif speaker_id not in test_speakers:
                train_data[file_path] = item
    
    print(f"\nSplit statistics:")
    print(f"  Test: {len(test_data)} samples from {len(test_speakers)} speakers")
    print(f"  Val: {len(val_data)} samples from speaker {val_speaker}")
    print(f"  Train: {len(train_data)} samples")
    
    with open(split_dir / 'test_labels.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    with open(split_dir / 'val_labels.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(split_dir / 'train_labels.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    train_transcripts = set(
        item.get('wrd', '').strip() 
        for item in train_data.values()
    ) | set(
        item.get('wrd', '').strip() 
        for item in val_data.values()
    )
    
    overlap = test_transcripts & train_transcripts
    if overlap:
        print(f"\nWarning: {len(overlap)} transcripts overlap between test and train/val")
    else:
        print(f"\nVerified: No transcript overlap between test and train/val sets")
    
    stats = {
        'total_samples': len(data),
        'test_samples': len(test_data),
        'val_samples': len(val_data),
        'train_samples': len(train_data),
        'test_speakers': test_speakers,
        'val_speaker': val_speaker,
        'num_test_transcripts': len(test_transcripts),
        'num_train_val_transcripts': len(train_transcripts),
        'transcript_overlap': len(overlap)
    }
    
    stats_path = split_dir / 'split_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDisjoint text split saved to {split_dir}")
    return stats