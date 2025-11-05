"""Error label generation for pronunciation assessment.

This module generates error labels (D, I, S, C) for perceived phoneme sequences
by aligning them with canonical sequences using Needleman-Wunsch algorithm.
"""

import json
from pathlib import Path


def needleman_wunsch_alignment(canonical, perceived):
    """Aligns canonical and perceived phoneme sequences.
    
    Uses Needleman-Wunsch algorithm for global sequence alignment.
    
    Args:
        canonical: List of canonical phonemes.
        perceived: List of perceived phonemes.
    
    Returns:
        Tuple of (aligned_canonical, aligned_perceived) with gaps as None.
    """
    n, m = len(canonical), len(perceived)
    
    score = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        score[i][0] = -i
    for j in range(m + 1):
        score[0][j] = -j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i-1][j-1] + (
                0 if canonical[i-1] == perceived[j-1] else -1
            )
            delete = score[i-1][j] - 1
            insert = score[i][j-1] - 1
            score[i][j] = max(match, delete, insert)
    
    aligned_canonical, aligned_perceived = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and 
            score[i][j] == score[i-1][j-1] + (
                0 if canonical[i-1] == perceived[j-1] else -1
            )):
            aligned_canonical.append(canonical[i-1])
            aligned_perceived.append(perceived[j-1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i-1][j] - 1:
            aligned_canonical.append(canonical[i-1])
            aligned_perceived.append(None)
            i -= 1
        else:
            aligned_canonical.append(None)
            aligned_perceived.append(perceived[j-1])
            j -= 1
    
    return aligned_canonical[::-1], aligned_perceived[::-1]


def generate_error_labels(canonical_str, perceived_str):
    """Generates error labels for perceived phoneme sequence.
    
    Labels:
        C: Correct (match)
        S: Substitution (mismatch)
        I: Insertion (in perceived but not canonical)
        D: Deletion (in canonical but not perceived)
    
    Args:
        canonical_str: Space-separated canonical phonemes.
        perceived_str: Space-separated perceived phonemes.
    
    Returns:
        Space-separated error labels (same length as perceived).
    """
    canonical = canonical_str.split()
    perceived = perceived_str.split()
    
    aligned_canonical, aligned_perceived = needleman_wunsch_alignment(
        canonical, perceived
    )
    
    error_labels = []
    deletion_pending = False
    
    for canonical_phone, perceived_phone in zip(
        aligned_canonical, aligned_perceived
    ):
        if perceived_phone is None:
            deletion_pending = True
            continue
        
        if deletion_pending:
            error_labels.append('D')
            deletion_pending = False
        elif canonical_phone is None:
            error_labels.append('I')
        elif canonical_phone == perceived_phone:
            error_labels.append('C')
        else:
            error_labels.append('S')
    
    if len(error_labels) != len(perceived):
        print(f"Warning: Length mismatch! Expected {len(perceived)}, got {len(error_labels)}")
    
    return ' '.join(error_labels)


def add_error_labels_to_dataset(input_path, output_path):
    """Adds error labels to preprocessed dataset.
    
    Args:
        input_path: Path to input JSON file.
        output_path: Path to output JSON file.
    """
    print(f"Loading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} samples...")
    
    total = len(data)
    success = 0
    label_stats = {'C': 0, 'S': 0, 'I': 0, 'D': 0}
    
    for i, (key, item) in enumerate(data.items()):
        canonical = item.get('canonical_train_target', '')
        perceived = item.get('perceived_train_target', '')
        
        if not canonical or not perceived:
            continue
        
        error_labels = generate_error_labels(canonical, perceived)
        
        perceived_tokens = perceived.split()
        error_tokens = error_labels.split()
        
        if len(perceived_tokens) != len(error_tokens):
            print(f"Error: Length mismatch for {key}")
            continue
        
        for label in error_tokens:
            if label in label_stats:
                label_stats[label] += 1
        
        item['error_labels'] = error_labels
        success += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} samples")
    
    print(f"\nSuccessfully processed {success}/{total} samples")
    print(f"\nLabel statistics:")
    for label, count in label_stats.items():
        print(f"  {label}: {count}")
    
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Done!")
