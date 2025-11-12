"""Dataset preprocessing for L2-ARCTIC pronunciation assessment.

This module extracts phoneme alignments and transcriptions from L2-ARCTIC
TextGrid files and audio data for pronunciation error detection.
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

ARPA_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
    's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh',
    'sil', 'sp', 'spn'
]


def normalize_phoneme_symbol(phoneme):
  """Normalizes a single phoneme symbol.
  
  Removes stress markers (0, 1, 2) and handles special cases.
  
  Args:
    phoneme: Raw phoneme string from CMU Dict or G2P.
    
  Returns:
    Normalized phoneme string.
  """
  # Remove stress markers
  phoneme = re.sub(r'[0-9]', '', phoneme).lower()
  
  # Handle special cases
  phoneme_map = {
      'ax': 'ah',
      'axr': 'er',
      'hh': 'hh',
      'er': 'er',
      'err': 'er',
  }
  
  return phoneme_map.get(phoneme, phoneme)


def is_silence(phoneme):
  """Checks if phoneme is a silence token.
  
  Args:
    phoneme: Phoneme string.
    
  Returns:
    True if phoneme is silence, False otherwise.
  """
  return phoneme.lower() in {"sil", "sp", "spn", "pau", ""}


def normalize_phoneme(annotation, remove_annotation=True, use_canonical=True, keep_artificial_sil=False):
  """Normalizes phoneme annotation from TextGrid.
  
  Processes annotated phonemes with error markers (deletion, insertion, substitution)
  and extracts either canonical (correct) or perceived (actual) pronunciation.
  
  Args:
    annotation: Raw annotation string from TextGrid.
    remove_annotation: Whether to remove error annotation markers.
    use_canonical: Whether to extract canonical (correct) phoneme.
    keep_artificial_sil: Whether to keep artificially inserted silence markers.
    
  Returns:
    Normalized phoneme string or None if should be skipped.
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
    return normalize_phoneme_symbol(phoneme)
  
  if remove_annotation:
    if keep_artificial_sil:
      phoneme = parts[0] if use_canonical else (parts[1] if len(parts) > 1 else parts[0])
    else:
      error_type = parts[2] if len(parts) > 2 else None
      
      if use_canonical:
        if error_type == 'a':  # Artificially added
          return None
        phoneme = parts[0]
      else:
        if error_type == 'd':  # Deleted
          return None
        phoneme = parts[1] if len(parts) > 1 else parts[0]
  else:
    return parse_tag
  
  return normalize_phoneme_symbol(phoneme)


def remove_consecutive_silence(phoneme_list):
  """Removes consecutive silence tokens, keeping only the last one.
  
  Args:
    phoneme_list: List of phoneme strings.
    
  Returns:
    List with consecutive silences removed.
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
  """Normalizes all intervals in a TextGrid tier.
  
  Args:
    tier: TextGrid tier object.
    mode: Normalization mode - 'canonical' or 'perceived'.
    keep_artificial_sil: Whether to keep artificially inserted silence.
    
  Returns:
    Normalized IntervalTier.
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
  """Converts TextGrid tier to list of phoneme strings.
  
  Args:
    tier: IntervalTier object.
    
  Returns:
    List of phoneme strings.
  """
  return [interval.mark for interval in tier]


class DatasetProcessor:
  """Processor for L2-ARCTIC dataset preprocessing.
  
  Extracts phoneme-level pronunciation data for error detection training.
  """
  
  def __init__(self, data_root, output_path):
    """Initializes dataset processor.
    
    Args:
      data_root: Root directory of L2-ARCTIC dataset.
      output_path: Path to output JSON file.
    """
    self.data_root = Path(data_root)
    self.output_path = Path(output_path)
    
    # Load CMU Dictionary
    try:
      self.cmu_dict = cmudict.dict()
    except LookupError:
      print("CMU Dictionary not found. Downloading...")
      nltk.download('cmudict')
      self.cmu_dict = cmudict.dict()
    
    # Initialize G2P converter
    self.g2p = G2p()
    
    # Contraction expansions for proper phoneme generation
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
    
    self.total = 0
    self.success = 0
    self.annotation_used = 0
    self.textgrid_used = 0
  
  def get_canonical_phonemes_from_text(self, text):
    """Generates canonical phoneme sequence from text.
    
    Uses CMU Dictionary with G2P fallback for out-of-vocabulary words.
    Properly removes stress markers and normalizes phoneme symbols.
    
    Args:
      text: Input text string.
      
    Returns:
      List of canonical phonemes.
    """
    phonemes = []
    words = text.lower().split()
    
    for word in words:
      # Handle contractions
      if word in self.contractions:
        for part in self.contractions[word]:
          word_phonemes = self._get_word_phonemes(part)
          phonemes.extend(word_phonemes)
      else:
        word_phonemes = self._get_word_phonemes(word)
        phonemes.extend(word_phonemes)
    
    return phonemes
  
  def _get_word_phonemes(self, word):
    """Gets phoneme sequence for a single word.
    
    Args:
      word: Single word string (lowercase).
      
    Returns:
      List of normalized phonemes.
    """
    phonemes = []
    
    # Try CMU Dictionary first
    if word in self.cmu_dict:
      word_phonemes = self.cmu_dict[word][0]
      phonemes = [normalize_phoneme_symbol(p) for p in word_phonemes]
    else:
      # Fallback to G2P
      g2p_phonemes = self.g2p(word)
      phonemes = [
          normalize_phoneme_symbol(p) 
          for p in g2p_phonemes 
          if p not in [' ', '']
      ]
    
    return phonemes
  
  def extract_phonemes_from_textgrid(self, textgrid, keep_artificial_sil=False, remove_consecutive_sil=True):
    """Extracts phoneme sequences from TextGrid file.
    
    Args:
      textgrid: Loaded TextGrid object.
      keep_artificial_sil: Whether to keep artificially inserted silence.
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
    """Processes a single audio file and extracts phoneme information.
    
    Args:
      speaker_id: Speaker identifier.
      filename: Base filename without extension.
      
    Returns:
      Dictionary containing extracted information or None if processing fails.
    """
    annotation_path = self.data_root / speaker_id / 'annotation' / f'{filename}.TextGrid'
    textgrid_path = self.data_root / speaker_id / 'textgrid' / f'{filename}.TextGrid'
    wav_path = self.data_root / speaker_id / 'wav' / f'{filename}.wav'
    transcript_path = self.data_root / speaker_id / 'transcript' / f'{filename}.txt'
    
    # Determine which TextGrid file to use
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
    
    # Check required files exist
    if not wav_path.exists() or not transcript_path.exists():
      return None
    
    try:
      # Load TextGrid
      tg = TextGrid()
      tg.read(str(tg_path))
      
      duration = tg.maxTime
      
      # Load transcript
      with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
      
      # Extract phonemes with alignment information
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
      
      # Extract training targets (clean phonemes for model training)
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
    """Processes all files in the L2-ARCTIC dataset.
    
    Returns:
      Dictionary mapping file paths to extracted information.
    """
    print(f"Processing L2-ARCTIC dataset from {self.data_root}")
    
    result = {}
    speaker_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
    
    if not speaker_dirs:
      raise ValueError(f"No speaker directories found in {self.data_root}")
    
    for speaker_dir in speaker_dirs:
      speaker_id = speaker_dir.name
      print(f"Processing speaker: {speaker_id}")
      
      wav_dir = speaker_dir / 'wav'
      if not wav_dir.exists():
        print(f"  Warning: No wav directory for {speaker_id}")
        continue
      
      wav_files = sorted(list(wav_dir.glob('*.wav')))
      
      if not wav_files:
        print(f"  Warning: No wav files for {speaker_id}")
        continue
      
      for wav_file in wav_files:
        filename = wav_file.stem
        self.total += 1
        
        data = self.process_single_file(speaker_id, filename)
        
        if data:
          result[data['wav']] = data
          self.success += 1
        
        if self.total % 100 == 0:
          print(f"  Processed {self.total} files, {self.success} successful")
    
    if self.success == 0:
      raise ValueError("No files were successfully processed. Please check your data directory.")
    
    print(f"Saving results to {self.output_path}")
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(self.output_path, 'w', encoding='utf-8') as f:
      json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"Total files processed: {self.total}")
    print(f"Successful: {self.success}")
    print(f"Annotation files used: {self.annotation_used}")
    print(f"TextGrid files used: {self.textgrid_used}")
    print("="*80)
    
    return result