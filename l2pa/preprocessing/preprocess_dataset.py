"""L2-ARCTIC dataset preprocessing.

Extracts phoneme alignments from TextGrid files and creates dataset JSON.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    TEXTGRID_AVAILABLE = False
    print("Warning: textgrid package not available. Install with: pip install praat-textgrids")

try:
    import nltk
    from nltk.corpus import cmudict
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("Warning: g2p_en not available. Install with: pip install g2p-en")


class L2ArcticPreprocessor:
    """Preprocessor for L2-ARCTIC dataset."""
    
    def __init__(self):
        """Initializes preprocessor with phoneme dictionary."""
        self.cmu_dict = None
        self.g2p = None
        
        if NLTK_AVAILABLE:
            try:
                self.cmu_dict = cmudict.dict()
            except LookupError:
                print("CMUDict not found. Downloading...")
                nltk.download('cmudict')
                self.cmu_dict = cmudict.dict()
        
        if G2P_AVAILABLE:
            self.g2p = G2p()
    
    def get_phonemes(self, word: str) -> Optional[str]:
        """Gets phonemes for a word.
        
        Args:
            word: Word to convert to phonemes.
        
        Returns:
            Space-separated phoneme string or None if not found.
        """
        word_lower = word.lower()
        
        # Try CMU dictionary first
        if self.cmu_dict and word_lower in self.cmu_dict:
            phonemes = self.cmu_dict[word_lower][0]
            # Remove stress markers
            phonemes = [p.rstrip('012') for p in phonemes]
            return ' '.join(phonemes)
        
        # Fall back to G2P
        if self.g2p:
            phonemes = self.g2p(word)
            # Clean up G2P output
            phonemes = [p for p in phonemes if p.isalpha()]
            return ' '.join(phonemes)
        
        return None
    
    def extract_from_textgrid(self, textgrid_path: str) -> Dict:
        """Extracts canonical and perceived phonemes from TextGrid.
        
        Args:
            textgrid_path: Path to TextGrid file.
        
        Returns:
            Dictionary with canonical and perceived phoneme sequences.
        """
        if not TEXTGRID_AVAILABLE:
            return {'canonical': '', 'perceived': ''}
        
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            canonical_phonemes = []
            perceived_phonemes = []
            
            # Extract from appropriate tiers
            for tier in tg:
                if 'canonical' in tier.name.lower():
                    for interval in tier:
                        if interval.mark.strip():
                            canonical_phonemes.append(interval.mark.strip())
                elif 'perceived' in tier.name.lower() or 'phones' in tier.name.lower():
                    for interval in tier:
                        if interval.mark.strip():
                            perceived_phonemes.append(interval.mark.strip())
            
            return {
                'canonical': ' '.join(canonical_phonemes),
                'perceived': ' '.join(perceived_phonemes)
            }
        except Exception as e:
            print(f"Error processing {textgrid_path}: {e}")
            return {'canonical': '', 'perceived': ''}
    
    def preprocess_dataset(self, data_root: str, output_path: str):
        """Preprocesses L2-ARCTIC dataset.
        
        Args:
            data_root: Root directory of L2-ARCTIC dataset.
            output_path: Output JSON file path.
        """
        print(f'Preprocessing L2-ARCTIC from {data_root}')
        data_root = Path(data_root)
        result = {}
        
        # Iterate through speaker directories
        for speaker_dir in sorted(data_root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            print(f'Processing speaker: {speaker_id}')
            
            wav_dir = speaker_dir / 'wav'
            textgrid_dir = speaker_dir / 'TextGrid'
            transcript_dir = speaker_dir / 'transcript'
            
            if not wav_dir.exists():
                continue
            
            # Process each audio file
            for wav_file in sorted(wav_dir.glob('*.wav')):
                file_id = wav_file.stem
                
                # Get TextGrid if available
                textgrid_file = textgrid_dir / f'{file_id}.TextGrid' if textgrid_dir.exists() else None
                transcript_file = transcript_dir / f'{file_id}.txt' if transcript_dir.exists() else None
                
                # Extract phonemes
                canonical = ''
                perceived = ''
                transcript = ''
                
                if textgrid_file and textgrid_file.exists():
                    phoneme_data = self.extract_from_textgrid(str(textgrid_file))
                    canonical = phoneme_data['canonical']
                    perceived = phoneme_data['perceived']
                
                if transcript_file and transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                
                # Create entry
                result[str(wav_file)] = {
                    'wav': str(wav_file),
                    'spk_id': speaker_id,
                    'canonical_train_target': canonical,
                    'perceived_train_target': perceived,
                    'wrd': transcript
                }
        
        # Save result
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f'\nSaved to {output_path}')
        print(f'Total files: {len(result)}')


def preprocess_dataset(data_root: str, output_path: str):
    """Main preprocessing function.
    
    Args:
        data_root: L2-ARCTIC root directory.
        output_path: Output JSON path.
    """
    preprocessor = L2ArcticPreprocessor()
    preprocessor.preprocess_dataset(data_root, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/l2arctic')
    parser.add_argument('--output', type=str, default='data/preprocessed.json')
    args = parser.parse_args()
    
    preprocess_dataset(args.data_root, args.output)
