"""Dataset preprocessing for L2-ARCTIC pronunciation assessment.

This module extracts phoneme alignments and transcriptions from L2-ARCTIC
TextGrid files and audio data.
"""

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

# ARPA phonemes used in the dataset
ARPA_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
    's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh',
    'sil', 'sp', 'spn'
]


def is_silence(phoneme):
    """Checks if phoneme represents silence.
    
    Args:
        phoneme: Phoneme string to check.
        
    Returns:
        True if phoneme is a silence marker, False otherwise.
    """
    return phoneme.lower() in {"sil", "sp", "spn", "pau", ""}


def normalize_phoneme(annotation, remove_annotation=True, use_canonical=True, keep_artificial_sil=False):
    """Normalizes phoneme labels from L2-ARCTIC annotations.
    
    Args:
        annotation: Phoneme annotation string (e.g., "m, n, s").
        remove_annotation: Whether to remove annotation and return only phoneme.
        use_canonical: Whether to return canonical (True) or perceived (False) phoneme.
        keep_artificial_sil: Whether to keep artificial silence from annotations.
        
    Returns:
        Normalized phoneme string or None if invalid.
    """
    text = annotation.lower()
    pattern = re.compile(r"[^a-z,]")
    parse_tag = pattern.sub("", text)
    
    if is_silence(parse_tag):
        return "sil"
    
    if len(parse_tag) == 0:
        return None
    
    parts = parse_tag.split(",")
    
    if len(parts) == 1:
        phoneme = parts[0]
        return 'ah' if phoneme == 'ax' else phoneme
    
    if remove_annotation:
        if keep_artificial_sil:
            phoneme = parts[0] if use_canonical else (parts[1] if len(parts) > 1 else parts[0])
        else:
            error_type = parts[2] if len(parts) > 2 else None
            
            if use_canonical:
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


def remove_consecutive_silence(phoneme_list):
    """Removes consecutive silence tokens from phoneme list.
    
    Args:
        phoneme_list: List of phoneme strings.
        
    Returns:
        List with consecutive silence tokens removed.
    """
    if not phoneme_list:
        return phoneme_list
    
    remove_mask = [phone == "sil" for phone in phoneme_list]
    
    for i, is_sil in enumerate(remove_mask):
        if is_sil:
            if i == len(remove_mask) - 1:
                remove_mask[i] = False
            elif not remove_mask[i + 1]:
                remove_mask[i] = False
    
    return [phone for i, phone in enumerate(phoneme_list) if not remove_mask[i]]


def normalize_tier(tier, mode="canonical", keep_artificial_sil=False):
    """Normalizes phoneme marks in an IntervalTier.
    
    Args:
        tier: IntervalTier object to normalize.
        mode: Normalization mode ('canonical' or 'perceived').
        keep_artificial_sil: Whether to keep artificial silence markers.
        
    Returns:
        Normalized IntervalTier object.
    """
    tier = copy.deepcopy(tier)
    normalized_tier = IntervalTier()
    
    use_canonical = (mode == "canonical")
    
    for interval in tier.intervals:
        phoneme = normalize_phoneme(interval.mark, True, use_canonical, keep_artificial_sil)
        
        if phoneme is None:
            continue
        
        interval.mark = phoneme
        normalized_tier.addInterval(interval)
    
    return normalized_tier


def tier_to_phoneme_list(tier):
    """Converts IntervalTier to list of phoneme strings.
    
    Args:
        tier: IntervalTier object.
        
    Returns:
        List of phoneme strings from tier intervals.
    """
    return [interval.mark for interval in tier]


class DatasetProcessor:
    """Processor for L2-ARCTIC dataset preprocessing.
    
    Attributes:
        data_root: Root directory of L2-ARCTIC dataset.
        output_path: Output path for preprocessed JSON file.
        cmu_dict: CMU Pronouncing Dictionary.
        g2p: Grapheme-to-phoneme converter.
        contractions: Dictionary of English contractions and their expansions.
        total: Total number of files processed.
        success: Number of successfully processed files.
        annotation_used: Number of annotation files used.
        textgrid_used: Number of textgrid files used.
    """
    
    def __init__(self, data_root, output_path):
        """Initializes the dataset processor.
        
        Args:
            data_root: Root directory of L2-ARCTIC dataset.
            output_path: Output path for preprocessed JSON file.
        """
        self.data_root = Path(data_root)
        self.output_path = Path(output_path)
        
        # Initialize CMUdict and G2P
        self.cmu_dict = cmudict.dict()
        self.g2p = G2p()
        
        # English contractions mapping
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
        
        # Processing statistics
        self.total = 0
        self.success = 0
        self.annotation_used = 0
        self.textgrid_used = 0
    
    def get_canonical_phonemes_from_text(self, text):
        """Generates canonical phonemes from text using CMU dictionary.
        
        Args:
            text: Input text string.
            
        Returns:
            List of canonical phoneme strings.
        """
        phonemes = []
        words = text.lower().split()
        
        for word in words:
            # Handle contractions
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
        
        # Normalize ax to ah
        phonemes = ['ah' if p == 'ax' else p for p in phonemes]
        return phonemes
    
    def extract_phonemes_from_textgrid(self, textgrid, keep_artificial_sil=False, remove_consecutive_sil=True):
        """Extracts phoneme sequences from TextGrid file.
        
        Args:
            textgrid: TextGrid object.
            keep_artificial_sil: Whether to keep artificial silence markers.
            remove_consecutive_sil: Whether to remove consecutive silence tokens.
            
        Returns:
            Tuple of (canonical_phonemes, perceived_phonemes) as space-separated strings.
        """
        phone_tier = textgrid.getFirst("phones")
        
        perceived_tier = normalize_tier(phone_tier, "perceived", keep_artificial_sil)
        canonical_tier = normalize_tier(phone_tier, "canonical", keep_artificial_sil)
        
        canonical_phones = tier_to_phoneme_list(canonical_tier)
        perceived_phones = tier_to_phoneme_list(perceived_tier)
        
        if remove_consecutive_sil:
            canonical_phones = remove_consecutive_silence(canonical_phones)
            perceived_phones = remove_consecutive_silence(perceived_phones)
        
        return " ".join(canonical_phones), " ".join(perceived_phones)
    
    def remove_all_silence(self, phonemes):
        """Removes all silence tokens from phoneme list.
        
        Args:
            phonemes: List of phoneme strings.
            
        Returns:
            List with all silence tokens removed.
        """
        return [p for p in phonemes if p not in ['sil', 'sp', 'spn']]
    
    def process_single_file(self, speaker_id, filename):
        """Processes a single audio file and its annotations.
        
        Args:
            speaker_id: Speaker identifier.
            filename: Base filename without extension.
            
        Returns:
            Dictionary containing processed data or None if processing fails.
        """
        annotation_path = self.data_root / speaker_id / 'annotation' / f'{filename}.TextGrid'
        textgrid_path = self.data_root / speaker_id / 'textgrid' / f'{filename}.TextGrid'
        wav_path = self.data_root / speaker_id / 'wav' / f'{filename}.wav'
        transcript_path = self.data_root / speaker_id / 'transcript' / f'{filename}.txt'
        
        # Determine which TextGrid to use
        if annotation_path.exists():
            tg_path = annotation_path
            has_annotation = True
            self.annotation_used += 1
        elif textgrid_path.exists():
            tg_path = textgrid_path
            has_annotation = False
            self.textgrid_used += 1
        else:
            return None
        
        if not wav_path.exists() or not transcript_path.exists():
            return None
        
        try:
            tg = TextGrid()
            tg.read(str(tg_path))
            
            duration = tg.maxTime
            
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            # Extract phonemes with alignment
            if has_annotation:
                canonical_aligned, perceived_aligned = self.extract_phonemes_from_textgrid(
                    tg, keep_artificial_sil=True, remove_consecutive_sil=False
                )
            else:
                _, perceived_aligned = self.extract_phonemes_from_textgrid(
                    tg, keep_artificial_sil=True, remove_consecutive_sil=False
                )
                canonical_phonemes = self.get_canonical_phonemes_from_text(transcript)
                canonical_aligned = " ".join(canonical_phonemes)
            
            # Extract training targets
            if has_annotation:
                canonical_train_target, perceived_train_target = self.extract_phonemes_from_textgrid(
                    tg, keep_artificial_sil=False, remove_consecutive_sil=True
                )
                canonical_phones = canonical_train_target.split()
                canonical_phones = self.remove_all_silence(canonical_phones)
                canonical_train_target = " ".join(canonical_phones)
            else:
                _, perceived_train_target = self.extract_phonemes_from_textgrid(
                    tg, keep_artificial_sil=False, remove_consecutive_sil=True
                )
                canonical_phonemes = self.get_canonical_phonemes_from_text(transcript)
                canonical_train_target = " ".join(canonical_phonemes)
            
            # Store absolute path for reliability
            wav_absolute = str(wav_path.resolve())
            
            return {
                'wav': wav_absolute,
                'duration': float(duration),
                'spk_id': speaker_id,
                'canonical_aligned': canonical_aligned,
                'perceived_aligned': perceived_aligned,
                'canonical_train_target': canonical_train_target,
                'perceived_train_target': perceived_train_target,
                'wrd': transcript
            }
        except Exception as e:
            print(f"Error processing {speaker_id}/{filename}: {str(e)}")
            return None
    
    def process_all_files(self):
        """Processes all speakers and files in the dataset.
        
        Returns:
            Dictionary mapping file paths to processed data.
        """
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
                
                data = self.process_single_file(speaker_id, filename)
                
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
        
        return result