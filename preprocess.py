"""Preprocessing entry point with cross-validation support.

This script provides preprocessing for L2-ARCTIC dataset including
extraction, error label generation, and cross-validation splits.
"""

import argparse
import logging
import os
from pathlib import Path

from l2pa.preprocessing.generate_labels import add_error_labels_to_dataset
from l2pa.preprocessing.preprocess_dataset import DatasetProcessor
from l2pa.preprocessing.split_data import (
    split_dataset_for_cv,
    split_dataset_disjoint_text
)


def verify_phoneme_mapping_exists(data_dir: str) -> bool:
    """Verifies that phoneme mapping file exists.
    
    Args:
        data_dir: Directory containing the phoneme mapping file.
    
    Returns:
        True if mapping file exists, False otherwise.
    """
    phoneme_map_path = Path(data_dir) / 'phoneme_to_id.json'
    if not phoneme_map_path.exists():
        print(f"Error: Phoneme mapping not found at {phoneme_map_path}")
        print("Please run download_dataset.sh to download the mapping file")
        return False
    return True


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description='L2-ARCTIC Dataset Preprocessing with Cross-Validation'
    )
    subparsers = parser.add_subparsers(dest='command', help='Preprocessing command')
    
    extract_parser = subparsers.add_parser(
        'extract', help='Extract phoneme data from L2-ARCTIC'
    )
    extract_parser.add_argument(
        '--data_root', type=str, default='data/l2arctic',
        help='L2-ARCTIC dataset root directory'
    )
    extract_parser.add_argument(
        '--output', type=str, default='data/preprocessed.json',
        help='Output JSON file path'
    )
    
    labels_parser = subparsers.add_parser(
        'labels', help='Generate error labels'
    )
    labels_parser.add_argument(
        '--input', type=str, default='data/preprocessed.json',
        help='Input JSON file'
    )
    labels_parser.add_argument(
        '--output', type=str, default='data/processed_with_error.json',
        help='Output JSON file'
    )
    
    split_parser = subparsers.add_parser(
        'split', help='Split dataset for cross-validation'
    )
    split_parser.add_argument(
        '--input', type=str, default='data/processed_with_error.json',
        help='Input JSON file'
    )
    split_parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory'
    )
    split_parser.add_argument(
        '--test_speakers', nargs='+',
        default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
        help='Speaker IDs for test set'
    )
    
    disjoint_parser = subparsers.add_parser(
        'split_disjoint', 
        help='Split dataset with disjoint transcripts between test and train/val'
    )
    disjoint_parser.add_argument(
        '--input', type=str, default='data/processed_with_error.json',
        help='Input JSON file'
    )
    disjoint_parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory'
    )
    disjoint_parser.add_argument(
        '--test_speakers', nargs='+',
        default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
        help='Speaker IDs for test set'
    )
    disjoint_parser.add_argument(
        '--val_speaker', type=str, default='MBMPS',
        help='Speaker ID for validation set'
    )
    disjoint_parser.add_argument(
        '--num_test_transcripts', type=int, default=100,
        help='Number of unique transcripts for test set'
    )
    
    all_parser = subparsers.add_parser(
        'all', help='Run all preprocessing steps'
    )
    all_parser.add_argument(
        '--data_root', type=str, default='data/l2arctic',
        help='L2-ARCTIC dataset root directory'
    )
    all_parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory'
    )
    all_parser.add_argument(
        '--test_speakers', nargs='+',
        default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
        help='Speaker IDs for test set'
    )
    all_parser.add_argument(
        '--val_speaker', type=str, default='MBMPS',
        help='Speaker ID for validation in disjoint split'
    )
    all_parser.add_argument(
        '--num_test_transcripts', type=int, default=100,
        help='Number of unique transcripts for test set in disjoint split'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'extract':
        print("Step 1: Extracting phoneme data from L2-ARCTIC...")
        processor = DatasetProcessor(args.data_root, args.output)
        processor.process_all_files()
    
    elif args.command == 'labels':
        print("Step 2: Generating error labels...")
        add_error_labels_to_dataset(args.input, args.output)
    
    elif args.command == 'split':
        output_dir = Path(args.output_dir)
        phoneme_map_path = output_dir / 'phoneme_to_id.json'
        
        if not verify_phoneme_mapping_exists(args.output_dir):
            return
        
        print("Step 3: Creating cross-validation splits...")
        split_dataset_for_cv(
            args.input,
            args.output_dir,
            str(phoneme_map_path),
            args.test_speakers
        )
    
    elif args.command == 'split_disjoint':
        output_dir = Path(args.output_dir)
        phoneme_map_path = output_dir / 'phoneme_to_id.json'
        
        if not verify_phoneme_mapping_exists(args.output_dir):
            return
        
        print("Creating disjoint text split...")
        split_dataset_disjoint_text(
            args.input,
            args.output_dir,
            str(phoneme_map_path),
            args.test_speakers,
            args.val_speaker,
            args.num_test_transcripts
        )
    
    elif args.command == 'all':
        print("Running all preprocessing steps...")
        output_dir = Path(args.output_dir)
        
        if not verify_phoneme_mapping_exists(args.output_dir):
            return
        
        print("\nStep 1: Extracting phoneme data...")
        preprocessed_path = output_dir / 'preprocessed.json'
        processor = DatasetProcessor(args.data_root, str(preprocessed_path))
        processor.process_all_files()
        
        print("\nStep 2: Generating error labels...")
        processed_path = output_dir / 'processed_with_error.json'
        add_error_labels_to_dataset(str(preprocessed_path), str(processed_path))
        
        print("\nStep 3: Creating cross-validation splits...")
        phoneme_map_path = output_dir / 'phoneme_to_id.json'
        split_dataset_for_cv(
            str(processed_path),
            args.output_dir,
            str(phoneme_map_path),
            args.test_speakers
        )
        
        print("\nStep 4: Creating disjoint text split...")
        split_dataset_disjoint_text(
            str(processed_path),
            args.output_dir,
            str(phoneme_map_path),
            args.test_speakers,
            args.val_speaker,
            args.num_test_transcripts
        )
        
        print("\n" + "="*80)
        print("Preprocessing complete")
        print("="*80)
        print("\nGenerated splits:")
        print("  1. Cross-validation: data/fold_0/, fold_1/, ..., test_labels.json")
        print("  2. Disjoint text split: data/disjoint_wrd_split/")
        print("\nNext steps:")
        print("  - Train with CV: python main.py train --training_mode multitask")
        print("  - Train with disjoint split: Modify config.py to use disjoint_wrd_split/")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()