"""Preprocessing entry point."""
import argparse
import logging

from l2pa.preprocessing.preprocess_dataset import preprocess_dataset
from l2pa.preprocessing.generate_labels import add_error_labels
from l2pa.preprocessing.split_data import split_dataset


def run_all_preprocessing(data_root: str, output_dir: str, test_speakers: list):
  """Runs all preprocessing steps in sequence.
  
  Args:
      data_root: L2-ARCTIC dataset root directory.
      output_dir: Output directory for processed files.
      test_speakers: List of test speaker IDs.
  """
  import os
  
  print("\n" + "="*80)
  print("Step 1/3: Extracting phonemes from dataset")
  print("="*80)
  preprocessed_path = os.path.join(output_dir, 'preprocessed.json')
  preprocess_dataset(data_root, preprocessed_path)
  
  print("\n" + "="*80)
  print("Step 2/3: Generating error labels")
  print("="*80)
  processed_with_error_path = os.path.join(output_dir, 'processed_with_error.json')
  add_error_labels(preprocessed_path, processed_with_error_path)
  
  print("\n" + "="*80)
  print("Step 3/3: Splitting dataset for cross-validation")
  print("="*80)
  split_dataset(processed_with_error_path, output_dir, test_speakers)
  
  print("\n" + "="*80)
  print("Preprocessing completed successfully!")
  print("="*80)


def main():
  parser = argparse.ArgumentParser(description='L2-ARCTIC Preprocessing')
  subparsers = parser.add_subparsers(dest='command', help='Command')
  
  # All preprocessing
  all_parser = subparsers.add_parser('all', help='Run all preprocessing steps')
  all_parser.add_argument('--data_root', type=str, default='data/l2arctic',
                          help='L2-ARCTIC dataset root directory')
  all_parser.add_argument('--output_dir', type=str, default='data',
                          help='Output directory for processed files')
  all_parser.add_argument('--test_speakers', nargs='+',
                          default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
                          help='Test speaker IDs')
  
  # Dataset preprocessing
  dataset_parser = subparsers.add_parser('dataset', help='Extract phonemes from dataset')
  dataset_parser.add_argument('--data_root', type=str, default='data/l2arctic',
                              help='L2-ARCTIC dataset root directory')
  dataset_parser.add_argument('--output', type=str, default='data/preprocessed.json',
                              help='Output JSON file path')
  
  # Error label generation
  labels_parser = subparsers.add_parser('labels', help='Generate error labels')
  labels_parser.add_argument('--input', type=str, default='data/preprocessed.json',
                             help='Input preprocessed JSON file')
  labels_parser.add_argument('--output', type=str, default='data/processed_with_error.json',
                             help='Output JSON file with error labels')
  
  # Cross-validation split
  split_parser = subparsers.add_parser('split', help='Split for cross-validation')
  split_parser.add_argument('--input', type=str, default='data/processed_with_error.json',
                            help='Input JSON file with error labels')
  split_parser.add_argument('--output_dir', type=str, default='data',
                            help='Output directory for splits')
  split_parser.add_argument('--test_speakers', nargs='+',
                            default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'],
                            help='Test speaker IDs')
  
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
  
  if args.command == 'all':
    run_all_preprocessing(args.data_root, args.output_dir, args.test_speakers)
  elif args.command == 'dataset':
    preprocess_dataset(args.data_root, args.output)
  elif args.command == 'labels':
    add_error_labels(args.input, args.output)
  elif args.command == 'split':
    split_dataset(args.input, args.output_dir, args.test_speakers)
  else:
    parser.print_help()


if __name__ == '__main__':
  main()
