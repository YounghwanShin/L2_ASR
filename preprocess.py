"""Preprocessing entry point.

This script provides a unified interface for dataset preprocessing,
including data extraction, error label generation, and train/validation/test splitting.
"""

import argparse
import logging
from pathlib import Path

from l2pa.preprocessing.preprocess_dataset import DatasetProcessor
from l2pa.preprocessing.generate_labels import add_error_labels_to_dataset
from l2pa.preprocessing.split_data import split_dataset_by_speakers


def main():
    """Main function for preprocessing."""
    parser = argparse.ArgumentParser(description='L2-ARCTIC Dataset Preprocessing')
    subparsers = parser.add_subparsers(dest='command', help='Preprocessing command')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract phoneme data from L2-ARCTIC')
    extract_parser.add_argument('--data_root', type=str, default='data/l2arctic',
                               help='L2-ARCTIC dataset root directory')
    extract_parser.add_argument('--output', type=str, default='data/preprocessed.json',
                               help='Output JSON file path')
    
    # Generate labels command
    labels_parser = subparsers.add_parser('labels', help='Generate error labels')
    labels_parser.add_argument('--input', type=str, default='data/preprocessed.json',
                              help='Input JSON file')
    labels_parser.add_argument('--output', type=str, default='data/processed_with_error.json',
                              help='Output JSON file')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--input', type=str, default='data/processed_with_error.json',
                             help='Input JSON file')
    split_parser.add_argument('--output_dir', type=str, default='data',
                             help='Output directory')
    split_parser.add_argument('--test_speakers', nargs='+',
                             default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
                             help='Speaker IDs for test set')
    split_parser.add_argument('--val_speakers', nargs='+',
                             default=[],
                             help='Speaker IDs for validation set')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all preprocessing steps')
    all_parser.add_argument('--data_root', type=str, default='data/l2arctic',
                           help='L2-ARCTIC dataset root directory')
    all_parser.add_argument('--output_dir', type=str, default='data',
                           help='Output directory')
    all_parser.add_argument('--test_speakers', nargs='+',
                           default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
                           help='Speaker IDs for test set')
    all_parser.add_argument('--val_speakers', nargs='+',
                           default=[],
                           help='Speaker IDs for validation set')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == 'extract':
        print("Step 1: Extracting phoneme data from L2-ARCTIC...")
        processor = DatasetProcessor(args.data_root, args.output)
        processor.process_all_files()
        
    elif args.command == 'labels':
        print("Step 2: Generating error labels...")
        add_error_labels_to_dataset(args.input, args.output)
        
    elif args.command == 'split':
        print("Step 3: Splitting dataset...")
        split_dataset_by_speakers(args.input, args.output_dir, args.test_speakers, args.val_speakers)
        
    elif args.command == 'all':
        print("Running all preprocessing steps...")
        
        # Step 1: Extract
        print("\nStep 1: Extracting phoneme data...")
        preprocessed_path = Path(args.output_dir) / 'preprocessed.json'
        processor = DatasetProcessor(args.data_root, str(preprocessed_path))
        processor.process_all_files()
        
        # Step 2: Generate labels
        print("\nStep 2: Generating error labels...")
        processed_path = Path(args.output_dir) / 'processed_with_error.json'
        add_error_labels_to_dataset(str(preprocessed_path), str(processed_path))
        
        # Step 3: Split
        print("\nStep 3: Splitting dataset...")
        split_dataset_by_speakers(str(processed_path), args.output_dir, args.test_speakers, args.val_speakers)
        
        print("\n" + "="*80)
        print("Preprocessing complete!")
        print("="*80)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()