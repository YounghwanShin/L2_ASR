"""Preprocessing entry point with cross-validation support.

This script provides preprocessing for L2-ARCTIC dataset including
extraction, error label generation, and cross-validation splits.
"""

import argparse
import logging
from pathlib import Path

from l2pa.preprocessing.generate_labels import add_error_labels_to_dataset
from l2pa.preprocessing.preprocess_dataset import DatasetProcessor
from l2pa.preprocessing.split_data import split_dataset_for_cv


def main():
  """Main preprocessing function."""
  parser = argparse.ArgumentParser(
      description='L2-ARCTIC Dataset Preprocessing with Cross-Validation'
  )
  subparsers = parser.add_subparsers(dest='command', help='Preprocessing command')
  
  # Extract command
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
  
  # Generate labels command
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
  
  # Split command (cross-validation)
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
  
  # All command
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
  
  args = parser.parse_args()
  
  logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')
  
  if args.command == 'extract':
    print("Step 1: Extracting phoneme data from L2-ARCTIC...")
    processor = DatasetProcessor(args.data_root, args.output)
    processor.process_all_files()
  
  elif args.command == 'labels':
    print("Step 2: Generating error labels...")
    add_error_labels_to_dataset(args.input, args.output)
  
  elif args.command == 'split':
    print("Step 3: Creating cross-validation splits...")
    split_dataset_for_cv(args.input, args.output_dir, args.test_speakers)
  
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
    
    # Step 3: Create CV splits
    print("\nStep 3: Creating cross-validation splits...")
    split_dataset_for_cv(str(processed_path), args.output_dir, args.test_speakers)
    
    print("\n" + "="*80)
    print("Preprocessing complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Train all CV folds: python main.py train --training_mode multitask")
    print("  2. Train specific fold: python main.py train --training_mode multitask --cv_fold 0")
    print("  3. Evaluate: python main.py eval --checkpoint path/to/checkpoint.pth")
  
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
