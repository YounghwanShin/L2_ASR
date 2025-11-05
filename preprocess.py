"""Preprocessing entry point."""

import argparse
import logging

from l2pa.preprocessing.generate_labels import add_error_labels
from l2pa.preprocessing.split_data import split_dataset


def main():
  """Main preprocessing function."""
  parser = argparse.ArgumentParser(description='L2-ARCTIC Preprocessing')
  subparsers = parser.add_subparsers(dest='command', help='Command')
  
  # Generate labels
  labels_parser = subparsers.add_parser('labels', help='Generate error labels')
  labels_parser.add_argument('--input', type=str, default='data/preprocessed.json')
  labels_parser.add_argument('--output', type=str, default='data/processed_with_error.json')
  
  # Split dataset
  split_parser = subparsers.add_parser('split', help='Split for cross-validation')
  split_parser.add_argument('--input', type=str, default='data/processed_with_error.json')
  split_parser.add_argument('--output_dir', type=str, default='data')
  split_parser.add_argument('--test_speakers', nargs='+',
                           default=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK'])
  
  args = parser.parse_args()
  
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
  
  if args.command == 'labels':
    add_error_labels(args.input, args.output)
  elif args.command == 'split':
    split_dataset(args.input, args.output_dir, args.test_speakers)
  else:
    parser.print_help()


if __name__ == '__main__':
  main()
