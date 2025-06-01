import json
import argparse

def merge_datasets(error_data_path, augmented_phoneme_data_path, output_path):
    """
    Merge error detection data with augmented phoneme data for complete multi-task dataset
    
    Args:
        error_data_path: Path to errors_train.json
        augmented_phoneme_data_path: Path to unified_train.json (phoneme data with error labels)
        output_path: Path to save complete multi-task dataset
    """
    
    # Load error detection data
    print("Loading error detection data...")
    with open(error_data_path, 'r', encoding='utf-8') as f:
        error_data = json.load(f)
    
    # Load augmented phoneme data  
    print("Loading augmented phoneme data...")
    with open(augmented_phoneme_data_path, 'r', encoding='utf-8') as f:
        phoneme_data = json.load(f)
    
    # Merge datasets
    print("Merging datasets...")
    merged_data = {}
    
    # Add all error detection samples
    error_count = 0
    for wav_file, item in error_data.items():
        merged_data[wav_file] = item.copy()
        error_count += 1
    
    # Add all phoneme samples (with error labels)
    phoneme_count = 0
    duplicates = 0
    
    for wav_file, item in phoneme_data.items():
        if wav_file in merged_data:
            # File exists in both datasets - merge the information
            print(f"Duplicate found: {wav_file}")
            existing_item = merged_data[wav_file]
            
            # Keep error data as primary, add phoneme info
            if 'perceived_train_target' in item:
                existing_item['perceived_train_target'] = item['perceived_train_target']
            if 'perceived_aligned' in item:
                existing_item['perceived_aligned'] = item['perceived_aligned']
            if 'canonical_aligned' in item and 'canonical_aligned' not in existing_item:
                existing_item['canonical_aligned'] = item['canonical_aligned']
                
            duplicates += 1
        else:
            # New file - add to merged dataset
            merged_data[wav_file] = item.copy()
            phoneme_count += 1
    
    # Save merged dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_samples = len(merged_data)
    samples_with_error_labels = sum(1 for item in merged_data.values() if 'error_labels' in item and item['error_labels'].strip())
    samples_with_phoneme_labels = sum(1 for item in merged_data.values() if 'perceived_train_target' in item and item['perceived_train_target'].strip())
    samples_with_both = sum(1 for item in merged_data.values() 
                           if 'error_labels' in item and item['error_labels'].strip() 
                           and 'perceived_train_target' in item and item['perceived_train_target'].strip())
    
    print(f"\nMerging completed!")
    print(f"Original error samples: {error_count}")
    print(f"Original phoneme samples: {len(phoneme_data)}")
    print(f"Duplicates found: {duplicates}")
    print(f"Final total samples: {total_samples}")
    print(f"")
    print(f"Samples with error labels: {samples_with_error_labels}")
    print(f"Samples with phoneme labels: {samples_with_phoneme_labels}")
    print(f"Samples with both labels: {samples_with_both}")
    print(f"")
    print(f"Error coverage: {samples_with_error_labels/total_samples*100:.1f}%")
    print(f"Phoneme coverage: {samples_with_phoneme_labels/total_samples*100:.1f}%")
    print(f"Multi-task coverage: {samples_with_both/total_samples*100:.1f}%")
    print(f"")
    print(f"Merged dataset saved to: {output_path}")
    
    # Show sample data types
    print(f"\nSample breakdown:")
    
    # Error-only sample
    error_only_files = [f for f, item in merged_data.items() 
                       if 'error_labels' in item and item['error_labels'].strip()
                       and ('perceived_train_target' not in item or not item['perceived_train_target'].strip())]
    if error_only_files:
        sample_key = error_only_files[0]
        sample_item = merged_data[sample_key]
        print(f"\n--- Error-only Sample ---")
        print(f"File: {sample_key}")
        if 'wrd' in sample_item:
            print(f"Text: {sample_item['wrd']}")
        if 'error_labels' in sample_item:
            print(f"Error Labels: {sample_item['error_labels']}")
    
    # Phoneme-only sample
    phoneme_only_files = [f for f, item in merged_data.items() 
                         if 'perceived_train_target' in item and item['perceived_train_target'].strip()
                         and ('error_labels' not in item or not item['error_labels'].strip())]
    if phoneme_only_files:
        sample_key = phoneme_only_files[0]
        sample_item = merged_data[sample_key]
        print(f"\n--- Phoneme-only Sample ---")
        print(f"File: {sample_key}")
        if 'wrd' in sample_item:
            print(f"Text: {sample_item['wrd']}")
        if 'perceived_train_target' in sample_item:
            print(f"Perceived: {sample_item['perceived_train_target']}")
    
    # Multi-task sample
    multi_task_files = [f for f, item in merged_data.items() 
                       if 'error_labels' in item and item['error_labels'].strip()
                       and 'perceived_train_target' in item and item['perceived_train_target'].strip()]
    if multi_task_files:
        sample_key = multi_task_files[0]
        sample_item = merged_data[sample_key]
        print(f"\n--- Multi-task Sample ---")
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
    parser = argparse.ArgumentParser(description='Merge error detection and augmented phoneme data')
    
    parser.add_argument('--error_data', type=str, required=True,
                       help='Path to error detection data (errors_train.json)')
    parser.add_argument('--phoneme_data', type=str, required=True,
                       help='Path to augmented phoneme data (unified_train.json)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save complete multi-task dataset')
    
    args = parser.parse_args()
    
    merge_datasets(
        args.error_data,
        args.phoneme_data, 
        args.output_path
    )

if __name__ == "__main__":
    main()