"""L2-ARCTIC dataset preprocessing.

Extracts phoneme alignments from TextGrid files.
Note: This is a simplified version. For full preprocessing,
please use the original preprocess_dataset.py with NLTK/g2p support.
"""

import os
import json
from pathlib import Path


def preprocess_dataset(data_root, output_path):
  """Preprocesses L2-ARCTIC dataset.
  
  Args:
    data_root: L2-ARCTIC root directory.
    output_path: Output JSON path.
  
  Note:
    This is a simplified version. For full implementation with
    TextGrid parsing, CMUdict, and G2P support, refer to the
    original preprocessing code.
  """
  print(f'Preprocessing L2-ARCTIC from {data_root}')
  print('Note: This is a simplified preprocessing script.')
  print('For full TextGrid parsing, use the original implementation.')
  
  data_root = Path(data_root)
  result = {}
  
  # Example: Iterate through speaker directories
  for speaker_dir in data_root.iterdir():
    if not speaker_dir.is_dir():
      continue
    
    speaker_id = speaker_dir.name
    print(f'Processing speaker: {speaker_id}')
    
    # Process files (simplified - actual implementation would parse TextGrids)
    wav_dir = speaker_dir / 'wav'
    if wav_dir.exists():
      for wav_file in wav_dir.glob('*.wav'):
        # In full implementation, would extract phonemes from TextGrid
        result[str(wav_file)] = {
            'wav': str(wav_file),
            'spk_id': speaker_id,
            'canonical_train_target': '',  # Would be extracted from TextGrid
            'perceived_train_target': '',  # Would be extracted from TextGrid
            'wrd': ''  # Would be read from transcript
        }
  
  # Save result
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
  
  print(f'Saved to {output_path}')
  print(f'Total files: {len(result)}')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_root', type=str, default='data/l2arctic')
  parser.add_argument('--output', type=str, default='data/preprocessed.json')
  args = parser.parse_args()
  
  preprocess_dataset(args.data_root, args.output)
