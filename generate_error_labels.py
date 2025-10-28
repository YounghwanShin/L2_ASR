"""Generate error labels for L2-ARCTIC pronunciation assessment."""

import json
from pathlib import Path


def needleman_wunsch_alignment(canonical, perceived):
    """Align canonical and perceived phoneme sequences using Needleman-Wunsch algorithm.
    
    Args:
        canonical: List of canonical phonemes
        perceived: List of perceived phonemes
    
    Returns:
        Tuple of (aligned_canonical, aligned_perceived) with gaps marked as None
    """
    n, m = len(canonical), len(perceived)
    
    # DP matrix initialization
    score = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Gap penalties
    for i in range(n + 1):
        score[i][0] = -i
    for j in range(m + 1):
        score[0][j] = -j
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i-1][j-1] + (0 if canonical[i-1] == perceived[j-1] else -1)
            delete = score[i-1][j] - 1
            insert = score[i][j-1] - 1
            score[i][j] = max(match, delete, insert)
    
    # Backtrack
    aligned_can, aligned_per = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (0 if canonical[i-1] == perceived[j-1] else -1):
            aligned_can.append(canonical[i-1])
            aligned_per.append(perceived[j-1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i-1][j] - 1:
            aligned_can.append(canonical[i-1])
            aligned_per.append(None)
            i -= 1
        else:
            aligned_can.append(None)
            aligned_per.append(perceived[j-1])
            j -= 1
    
    return aligned_can[::-1], aligned_per[::-1]


def generate_error_labels(canonical_str, perceived_str):
    """Generate error labels for perceived phonemes.
    
    Labels:
        C: Correct (match)
        S: Substitution (mismatch)
        I: Insertion (in perceived but not in canonical)
        D: Deletion (in canonical but not in perceived, marked on next perceived token)
        U: Unmatched (silence insertion)
    
    Args:
        canonical_str: Space-separated canonical phonemes
        perceived_str: Space-separated perceived phonemes
    
    Returns:
        Space-separated error labels (same length as perceived)
    """
    canonical = canonical_str.split()
    perceived = perceived_str.split()
    
    # Align sequences
    aligned_can, aligned_per = needleman_wunsch_alignment(canonical, perceived)
    
    # Generate error labels for each perceived token
    error_labels = []
    deletion_pending = False
    
    for can_phone, per_phone in zip(aligned_can, aligned_per):
        if per_phone is None:
            # Deletion: canonical exists but perceived doesn't
            deletion_pending = True
            continue
        
        # Now we have a perceived phone to label
        if deletion_pending:
            # Previous canonical phone was deleted, mark current perceived as D
            error_labels.append('D')
            deletion_pending = False
        elif can_phone is None:
            # Insertion: perceived exists but canonical doesn't
            if per_phone in ['sil', 'sp', 'spn']:
                error_labels.append('U')  # Unmatched (silence)
            else:
                error_labels.append('I')  # Insertion
        elif can_phone == per_phone:
            # Match
            error_labels.append('C')
        else:
            # Mismatch
            error_labels.append('S')
    
    # Verify length
    if len(error_labels) != len(perceived):
        print(f"Warning: Length mismatch! Expected {len(perceived)}, got {len(error_labels)}")
        print(f"  Canonical: {canonical_str}")
        print(f"  Perceived: {perceived_str}")
    
    return ' '.join(error_labels)


def process_json_with_error_labels(input_path, output_path):
    """Add error labels to preprocessed JSON."""
    print(f"Loading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} samples...")
    
    total = len(data)
    success = 0
    label_stats = {'C': 0, 'S': 0, 'I': 0, 'D': 0, 'U': 0}
    
    for i, (key, item) in enumerate(data.items()):
        canonical = item.get('canonical_train_target', '')
        perceived = item.get('perceived_train_target', '')
        
        if not canonical or not perceived:
            print(f"Warning: Missing canonical or perceived for {key}")
            continue
        
        # Generate error labels
        error_labels = generate_error_labels(canonical, perceived)
        
        # Verify length
        per_tokens = perceived.split()
        err_tokens = error_labels.split()
        
        if len(per_tokens) != len(err_tokens):
            print(f"Error: Length mismatch for {key}")
            print(f"  Perceived: {len(per_tokens)}, Error labels: {len(err_tokens)}")
            continue
        
        # Count labels
        for label in err_tokens:
            if label in label_stats:
                label_stats[label] += 1
        
        # Add error labels
        item['error_train_target'] = error_labels
        success += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} samples")
    
    print(f"\nSuccessfully processed {success}/{total} samples")
    print(f"\nLabel statistics:")
    print(f"  C (Correct):      {label_stats['C']}")
    print(f"  S (Substitution): {label_stats['S']}")
    print(f"  I (Insertion):    {label_stats['I']}")
    print(f"  D (Deletion):     {label_stats['D']}")
    print(f"  U (Unmatched):    {label_stats['U']}")
    
    # Save result
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Done!")


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    input_path = script_dir / 'data' / 'preprocessed.json'
    output_path = script_dir / 'data' / 'processed_with_error.json'
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    process_json_with_error_labels(input_path, output_path)


if __name__ == "__main__":
    main()
