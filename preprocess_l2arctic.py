"""L2-ARCTIC dataset preprocessing for pronunciation assessment."""

import os
import json
import re
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import torchaudio
from textgrid import TextGrid, IntervalTier
from nltk.corpus import cmudict
import nltk
from g2p_en import G2p

# ARPA phonemes
ARPA_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
    's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh',
    'sil', 'sp', 'spn'
]


def is_sil(s):
    """Check if phoneme is silence."""
    return s.lower() in {"sil", "sp", "spn", "pau", ""}


def normalize_phone(s, is_rm_annotation=True, is_phoneme_canonical=True, keep_artificial_sil=False):
    """Normalize phoneme labels from L2-ARCTIC annotations.
    
    Args:
        s: Phoneme annotation string (e.g., "m, n, s")
        is_rm_annotation: Remove annotation and return only phoneme
        is_phoneme_canonical: Return canonical (True) or perceived (False)
        keep_artificial_sil: Keep artificial silence from annotations
    
    Returns:
        Normalized phoneme or None
    """
    t = s.lower()
    pattern = re.compile(r"[^a-z,]")
    parse_tag = pattern.sub("", t)
    
    if is_sil(parse_tag):
        return "sil"
    
    if len(parse_tag) == 0:
        return None
    
    parts = parse_tag.split(",")
    
    if len(parts) == 1:
        phoneme = parts[0]
        return 'ah' if phoneme == 'ax' else phoneme
    
    if is_rm_annotation:
        if keep_artificial_sil:
            phoneme = parts[0] if is_phoneme_canonical else (parts[1] if len(parts) > 1 else parts[0])
        else:
            error_type = parts[2] if len(parts) > 2 else None
            
            if is_phoneme_canonical:
                if error_type == 'a':
                    return None
                phoneme = parts[0]
            else:
                if error_type == 'd':
                    return None
                phoneme = parts[1] if len(parts) > 1 else parts[0]
    else:
        return parse_tag
    
    return 'ah' if phoneme == 'ax' else phoneme


def remove_repetitive_sil(phone_list):
    """Remove consecutive silence tokens."""
    if not phone_list:
        return phone_list
    
    remove_sil_mask = [phone == "sil" for phone in phone_list]
    
    for i, is_sil in enumerate(remove_sil_mask):
        if is_sil:
            if i == len(remove_sil_mask) - 1:
                remove_sil_mask[i] = False
            elif not remove_sil_mask[i + 1]:
                remove_sil_mask[i] = False
    
    return [phone for i, phone in enumerate(phone_list) if not remove_sil_mask[i]]


def normalize_tier_mark(tier, mode="NormalizePhoneCanonical", keep_artificial_sil=False):
    """Normalize marks in an IntervalTier."""
    tier = copy.deepcopy(tier)
    tier_out = IntervalTier()
    
    for each_interval in tier.intervals:
        if mode == "NormalizePhoneCanonical":
            p = normalize_phone(each_interval.mark, True, True, keep_artificial_sil)
        elif mode == "NormalizePhonePerceived":
            p = normalize_phone(each_interval.mark, True, False, keep_artificial_sil)
        else:
            continue
        
        if p is None:
            continue
        
        each_interval.mark = p
        tier_out.addInterval(each_interval)
    
    return tier_out


def tier_to_list(tier):
    """Convert tier intervals to list of phoneme strings."""
    return [interval.mark for interval in tier]


class L2ArcticProcessor:
    """Processor for L2-ARCTIC dataset."""
    
    def __init__(self, data_root, output_path):
        self.data_root = Path(data_root)
        self.output_path = Path(output_path)
        
        # Initialize CMUdict and G2P
        self.cmu_dict = cmudict.dict()
        self.g2p = G2p()
        
        # Contractions mapping
        self.contractions = {
            "i'm": ['i', 'am'], "i'll": ['i', 'will'], "we'll": ['we', 'will'],
            "you'll": ['you', 'will'], "he'll": ['he', 'will'], "she'll": ['she', 'will'],
            "they'll": ['they', 'will'], "it'll": ['it', 'will'], "we're": ['we', 'are'],
            "you're": ['you', 'are'], "they're": ['they', 'are'], "i've": ['i', 'have'],
            "we've": ['we', 'have'], "you've": ['you', 'have'], "they've": ['they', 'have'],
            "isn't": ['is', 'not'], "aren't": ['are', 'not'], "wasn't": ['was', 'not'],
            "weren't": ['were', 'not'], "hasn't": ['has', 'not'], "haven't": ['have', 'not'],
            "hadn't": ['had', 'not'], "won't": ['will', 'not'], "wouldn't": ['would', 'not'],
            "don't": ['do', 'not'], "doesn't": ['does', 'not'], "didn't": ['did', 'not'],
            "can't": ['can', 'not'], "couldn't": ['could', 'not'], "shouldn't": ['should', 'not'],
            "mightn't": ['might', 'not'], "mustn't": ['must', 'not']
        }
        
        # Statistics
        self.total = 0
        self.success = 0
        self.annotation_used = 0
        self.textgrid_used = 0
    
    def get_cmu_phonemes_for_text(self, text):
        """Generate canonical phonemes from text using CMUdict."""
        phonemes = []
        words = text.lower().split()
        
        for word in words:
            # Check contractions
            if word in self.contractions:
                for part in self.contractions[word]:
                    if part in self.cmu_dict:
                        word_phonemes = self.cmu_dict[part][0]
                        word_phonemes = [re.sub(r'[0-9]', '', p).lower() for p in word_phonemes]
                        phonemes.extend(word_phonemes)
                    else:
                        g2p_phonemes = self.g2p(part)
                        phonemes.extend([p.lower() for p in g2p_phonemes if p not in [' ', '']])
            elif word in self.cmu_dict:
                word_phonemes = self.cmu_dict[word][0]
                word_phonemes = [re.sub(r'[0-9]', '', p).lower() for p in word_phonemes]
                phonemes.extend(word_phonemes)
            else:
                # Fallback to G2P
                g2p_phonemes = self.g2p(word)
                phonemes.extend([p.lower() for p in g2p_phonemes if p not in [' ', '']])
        
        # Convert ax to ah
        phonemes = ['ah' if p == 'ax' else p for p in phonemes]
        return phonemes
    
    def get_phonemes(self, tg, keep_artificial_sil=False, rm_repetitive_sil=True):
        """Extract phonemes from TextGrid."""
        phone_tier = tg.getFirst("phones")
        
        perceived_tier = normalize_tier_mark(phone_tier, "NormalizePhonePerceived", keep_artificial_sil)
        canonical_tier = normalize_tier_mark(phone_tier, "NormalizePhoneCanonical", keep_artificial_sil)
        
        canonical_phones = tier_to_list(canonical_tier)
        perceived_phones = tier_to_list(perceived_tier)
        
        if rm_repetitive_sil:
            canonical_phones = remove_repetitive_sil(canonical_phones)
            perceived_phones = remove_repetitive_sil(perceived_phones)
        
        return " ".join(canonical_phones), " ".join(perceived_phones)
    
    def remove_all_sil(self, phonemes):
        """Remove all silence tokens from phoneme list."""
        return [p for p in phonemes if p not in ['sil', 'sp', 'spn']]
    
    def process_file(self, speaker_id, filename):
        """Process a single file."""
        # Check for annotation file first
        annotation_path = self.data_root / speaker_id / 'annotation' / f'{filename}.TextGrid'
        textgrid_path = self.data_root / speaker_id / 'textgrid' / f'{filename}.TextGrid'
        wav_path = self.data_root / speaker_id / 'wav' / f'{filename}.wav'
        transcript_path = self.data_root / speaker_id / 'transcript' / f'{filename}.txt'
        
        # Determine which TextGrid to use
        if annotation_path.exists():
            tg_path = annotation_path
            is_annotation = True
            self.annotation_used += 1
        elif textgrid_path.exists():
            tg_path = textgrid_path
            is_annotation = False
            self.textgrid_used += 1
        else:
            return None
        
        if not wav_path.exists() or not transcript_path.exists():
            return None
        
        try:
            # Parse TextGrid
            tg = TextGrid()
            tg.read(str(tg_path))
            
            # Get duration
            duration = tg.maxTime
            
            # Read transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                wrd = f.read().strip()
            
            # Extract phonemes with alignment
            if is_annotation:
                canonical_aligned, perceived_aligned = self.get_phonemes(
                    tg, keep_artificial_sil=True, rm_repetitive_sil=False
                )
            else:
                _, perceived_aligned = self.get_phonemes(
                    tg, keep_artificial_sil=True, rm_repetitive_sil=False
                )
                canonical_phonemes = self.get_cmu_phonemes_for_text(wrd)
                canonical_aligned = " ".join(canonical_phonemes)
            
            # Extract training targets
            if is_annotation:
                canonical_train_target, perceived_train_target = self.get_phonemes(
                    tg, keep_artificial_sil=False, rm_repetitive_sil=True
                )
                # Remove all silence from canonical train target
                canonical_phones = canonical_train_target.split()
                canonical_phones = self.remove_all_sil(canonical_phones)
                canonical_train_target = " ".join(canonical_phones)
            else:
                _, perceived_train_target = self.get_phonemes(
                    tg, keep_artificial_sil=False, rm_repetitive_sil=True
                )
                canonical_phonemes = self.get_cmu_phonemes_for_text(wrd)
                canonical_train_target = " ".join(canonical_phonemes)
            
            # Construct relative wav path
            wav_relative = f"data/l2arctic/{speaker_id}/wav/{filename}.wav"
            
            return {
                'wav': wav_relative,
                'duration': float(duration),
                'spk_id': speaker_id,
                'canonical_aligned': canonical_aligned,
                'perceived_aligned': perceived_aligned,
                'canonical_train_target': canonical_train_target,
                'perceived_train_target': perceived_train_target,
                'wrd': wrd
            }
        except Exception as e:
            print(f"Error processing {speaker_id}/{filename}: {str(e)}")
            return None
    
    def process_all(self):
        """Process all speakers and files."""
        print(f"Processing L2-ARCTIC dataset from {self.data_root}")
        
        result = {}
        speaker_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            print(f"Processing speaker: {speaker_id}")
            
            wav_dir = speaker_dir / 'wav'
            if not wav_dir.exists():
                continue
            
            wav_files = sorted(list(wav_dir.glob('*.wav')))
            
            for wav_file in wav_files:
                filename = wav_file.stem
                self.total += 1
                
                data = self.process_file(speaker_id, filename)
                
                if data:
                    result[data['wav']] = data
                    self.success += 1
                
                if self.total % 100 == 0:
                    print(f"  Processed {self.total} files, {self.success} successful")
        
        # Save result
        print(f"Saving results to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print("\n" + "="*80)
        print(f"Total files processed: {self.total}")
        print(f"Successful: {self.success}")
        print(f"Annotation files used: {self.annotation_used}")
        print(f"TextGrid files used: {self.textgrid_used}")
        print("="*80)


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    data_root = script_dir / 'data' / 'l2arctic'
    output_path = script_dir / 'data' / 'preprocessed.json'
    
    if not data_root.exists():
        print(f"Error: Data directory not found: {data_root}")
        print("Please ensure the L2-ARCTIC dataset is in data/l2arctic/")
        return
    
    processor = L2ArcticProcessor(str(data_root), str(output_path))
    processor.process_all()
    
    print(f"\nPreprocessing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()