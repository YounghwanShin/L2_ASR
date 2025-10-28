"""Setup script for NLTK data required by preprocessing.

Run this once before using preprocess_l2arctic.py:
    python setup_nltk.py
"""

import nltk


def setup_nltk_data():
    """Download required NLTK data."""
    print("Setting up NLTK data for L2-ARCTIC preprocessing...")
    
    resources = [
        ('corpora/cmudict', 'cmudict'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"✓ {name} already installed")
        except LookupError:
            print(f"Downloading {name}...")
            nltk.download(name, quiet=False)
            print(f"✓ {name} installed")
    
    print("\nSetup complete!")


if __name__ == "__main__":
    setup_nltk_data()