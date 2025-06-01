import json
import argparse
from difflib import SequenceMatcher
import re
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from g2p_en import G2p

def get_canonical_phonemes(text):
    """
    Convert text to canonical phonemes using g2p-en
    """
    g2p = G2p()
    phonemes = g2p(text.strip().lower())
    
    # Convert to format similar to your data
    canonical = ' '.join(phonemes)
    # Add silence tokens at start/end like in your data
    canonical = f"sil {canonical} sil"
    
    return canonical

def normalize_phonemes(phoneme_str):
    """
    Normalize phoneme representation to match your data format
    """
    if not phoneme_str:
        return ""
    
    # Common phoneme mappings to match your data format
    phoneme_mapping = {
        'AA0': 'aa', 'AA1': 'aa', 'AA2': 'aa',
        'AE0': 'ae', 'AE1': 'ae', 'AE2': 'ae',
        'AH0': 'ah', 'AH1': 'ah', 'AH2': 'ah',
        'AO0': 'ao', 'AO1': 'ao', 'AO2': 'ao',
        'AW0': 'aw', 'AW1': 'aw', 'AW2': 'aw',
        'AY0': 'ay', 'AY1': 'ay', 'AY2': 'ay',
        'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh',
        'EH0': 'eh', 'EH1': 'eh', 'EH2': 'eh',
        'ER0': 'er', 'ER1': 'er', 'ER2': 'er',
        'EY0': 'ey', 'EY1': 'ey', 'EY2': 'ey',
        'F': 'f', 'G': 'g', 'HH': 'hh',
        'IH0': 'ih', 'IH1': 'ih', 'IH2': 'ih',
        'IY0': 'iy', 'IY1': 'iy', 'IY2': 'iy',
        'JH': 'jh', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng',
        'OW0': 'ow', 'OW1': 'ow', 'OW2': 'ow',
        'OY0': 'oy', 'OY1': 'oy', 'OY2': 'oy',
        'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
        'T': 't', 'TH': 'th',
        'UH0': 'uh', 'UH1': 'uh', 'UH2': 'uh',
        'UW0': 'uw', 'UW1': 'uw', 'UW2': 'uw',
        'V': 'v', 'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
    }
    
    phonemes = phoneme_str.split()
    normalized = []
    
    for phoneme in phonemes:
        if phoneme in phoneme_mapping:
            normalized.append(phoneme_mapping[phoneme])
        else:
            # Handle unknown phonemes
            clean_phoneme = re.sub(r'[0-9]', '', phoneme)
            mapped_phoneme = phoneme_mapping.get(clean_phoneme, phoneme.lower())
            normalized.append(mapped_phoneme)
    
    return ' '.join(normalized)

def align_sequences(seq1, seq2):
    """
    Align two phoneme sequences and return error labels
    """
    matcher = SequenceMatcher(None, seq1, seq2)
    error_labels = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Phonemes match
            for k in range(i1, i2):
                error_labels.append('C')  # Correct
                
        elif tag == 'replace':
            # Substitution - phonemes don't match
            len1, len2 = i2 - i1, j2 - j1
            max_len = max(len1, len2)
            
            for k in range(max_len):
                error_labels.append('I')  # Incorrect
                    
        elif tag == 'delete':
            # Deletion - phoneme in canonical but not in perceived
            for k in range(i1, i2):
                error_labels.append('I')  # Incorrect
                
        elif tag == 'insert':
            # Insertion - phoneme in perceived but not in canonical
            for k in range(j1, j2):
                error_labels.append('I')  # Incorrect
    
    return error_labels

def generate_error_labels_from_alignment(canonical_aligned, perceived_train_target):
    """
    Generate error labels by comparing canonical and perceived phonemes
    """
    canonical_phonemes = canonical_aligned.split()
    perceived_phonemes = perceived_train_target.split()
    
    error_labels = align_sequences(canonical_phonemes, perceived_phonemes)
    
    # Ensure error_labels length matches perceived_phonemes length
    if len(error_labels) != len(perceived_phonemes):
        # Simple fallback: if lengths don't match, mark all as correct
        error_labels = ['C'] * len(perceived_phonemes)
    
    return ' '.join(error_labels)

def process_single_item(item_data):
    """
    Process a single item for G2P conversion - for multiprocessing
    """
    wav_file, item = item_data
    
    try:
        augmented_item = item.copy()
        
        # Generate canonical from text using g2p-en
        if 'wrd' in item:
            text = item['wrd']
            canonical_phonemes = get_canonical_phonemes(text)
            canonical_normalized = normalize_phonemes(canonical_phonemes)
            augmented_item['canonical_aligned'] = canonical_normalized
            
            # Generate error labels
            if 'perceived_train_target' in item:
                perceived_train_target = item['perceived_train_target']
                error_labels = generate_error_labels_from_alignment(canonical_normalized, perceived_train_target)
                augmented_item['error_labels'] = error_labels
            
            return wav_file, augmented_item, True  # Success
        else:
            return wav_file, augmented_item, False  # No text field
            
    except Exception as e:
        print(f"G2P failed for '{item.get('wrd', 'UNKNOWN')}': {e}")
        return wav_file, item.copy(), False  # Failed

def process_batch(batch_data):
    """
    Process a batch of items - for better multiprocessing efficiency
    """
    results = []
    for item_data in batch_data:
        result = process_single_item(item_data)
        results.append(result)
    return results

def create_batches(items, batch_size):
    """
    Create batches from items for efficient processing
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def augment_phoneme_data_with_error_labels(phoneme_data_path, output_path, num_workers=None, batch_size=32):
    """
    Augment phoneme recognition data with error labels using g2p-en with multiprocessing
    """
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 cores by default
    
    print(f"Using {num_workers} workers for parallel G2P processing...")
    
    with open(phoneme_data_path, 'r', encoding='utf-8') as f:
        phoneme_data = json.load(f)
    
    print("Generating canonical phonemes using g2p-en with multiprocessing...")
    
    # Prepare data for processing
    items = list(phoneme_data.items())
    batches = list(create_batches(items, batch_size))
    
    print(f"Processing {len(items)} items in {len(batches)} batches...")
    
    # Process batches in parallel
    augmented_data = {}
    successful_g2p = 0
    
    if num_workers == 1:
        # Single process for debugging
        for batch in tqdm(batches, desc="Processing batches"):
            batch_results = process_batch(batch)
            for wav_file, augmented_item, success in batch_results:
                augmented_data[wav_file] = augmented_item
                if success:
                    successful_g2p += 1
    else:
        # Multiprocessing
        with mp.Pool(processes=num_workers) as pool:
            batch_results = list(tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ))
            
            for batch_result in batch_results:
                for wav_file, augmented_item, success in batch_result:
                    augmented_data[wav_file] = augmented_item
                    if success:
                        successful_g2p += 1
    
    total_samples = len(phoneme_data)
    print(f"G2P success rate: {successful_g2p}/{total_samples} ({successful_g2p/total_samples*100:.1f}%)")
    
    # Save augmented data
    print("Saving augmented data...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_samples = len(augmented_data)
    samples_with_error_labels = sum(1 for item in augmented_data.values() if 'error_labels' in item)
    samples_with_canonical = sum(1 for item in augmented_data.values() if 'canonical_aligned' in item)
    
    print(f"\nAugmentation completed!")
    print(f"Total samples: {total_samples}")
    print(f"Samples with canonical: {samples_with_canonical}")
    print(f"Samples with error labels: {samples_with_error_labels}")
    print(f"Error label coverage: {samples_with_error_labels/total_samples*100:.1f}%")
    print(f"Augmented data saved to: {output_path}")
    
    # Show sample
    sample_files = [f for f, item in augmented_data.items() if 'error_labels' in item]
    if sample_files:
        sample_key = sample_files[0]
        sample_item = augmented_data[sample_key]
        print(f"\nSample augmented data:")
        print(f"File: {sample_key}")
        if 'wrd' in sample_item:
            print(f"Text: {sample_item['wrd']}")
        if 'canonical_aligned' in sample_item:
            print(f"Canonical: {sample_item['canonical_aligned']}")
        if 'perceived_train_target' in sample_item:
            print(f"Perceived: {sample_item['perceived_train_target']}")
        if 'error_labels' in sample_item:
            print(f"Error Labels: {sample_item['error_labels']}")

def main():
    parser = argparse.ArgumentParser(description='Augment phoneme data with error labels using g2p-en (multiprocessing)')
    
    parser.add_argument('--phoneme_data', type=str, required=True,
                       help='Path to phoneme recognition data')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save augmented data')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    
    args = parser.parse_args()
    
    augment_phoneme_data_with_error_labels(
        args.phoneme_data, 
        args.output_path,
        args.num_workers,
        args.batch_size
    )

if __name__ == "__main__":
    main()