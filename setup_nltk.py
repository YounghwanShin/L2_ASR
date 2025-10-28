"""NLTK Data Setup Script.

This script downloads required NLTK data packages for the pronunciation
assessment system, including the CMU Pronouncing Dictionary.
"""

import nltk
import sys
import os


def download_nltk_data():
    """Downloads required NLTK data packages.
    
    Downloads:
        - cmudict: CMU Pronouncing Dictionary for phoneme generation
        - punkt: Sentence tokenizer
        - averaged_perceptron_tagger: POS tagger
    """
    print("="*60)
    print("NLTK Data Setup")
    print("="*60)
    print()
    
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of required packages
    packages = [
        ('cmudict', 'CMU Pronouncing Dictionary'),
        ('punkt', 'Punkt Tokenizer'),
        ('averaged_perceptron_tagger', 'POS Tagger')
    ]
    
    print("Downloading required NLTK packages...")
    print()
    
    success_count = 0
    failed_packages = []
    
    for package_name, description in packages:
        try:
            print(f"Downloading {description} ({package_name})...")
            nltk.download(package_name, quiet=False)
            success_count += 1
            print(f"✓ Successfully downloaded {package_name}")
            print()
        except Exception as e:
            print(f"✗ Failed to download {package_name}: {str(e)}")
            failed_packages.append(package_name)
            print()
    
    # Summary
    print("="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{len(packages)} packages")
    
    if failed_packages:
        print(f"Failed packages: {', '.join(failed_packages)}")
        print()
        print("Please try downloading failed packages manually:")
        for package in failed_packages:
            print(f"  python -c \"import nltk; nltk.download('{package}')\"")
        print()
        return False
    else:
        print("All packages downloaded successfully!")
        print()
        print("NLTK data location:", nltk_data_dir)
        print()
        print("Next steps:")
        print("  1. Download dataset: ./download_dataset.sh")
        print("  2. Preprocess data: python preprocess.py all")
        print("  3. Train model: python main.py train --training_mode phoneme_error")
        print()
        return True


def verify_installation():
    """Verifies that required NLTK packages are installed correctly."""
    print("Verifying installation...")
    print()
    
    try:
        # Test CMUdict
        from nltk.corpus import cmudict
        cmu_dict = cmudict.dict()
        print(f"✓ CMUdict loaded successfully ({len(cmu_dict)} entries)")
        
        # Test punkt
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"✓ Punkt tokenizer working correctly")
        
        # Test POS tagger
        from nltk import pos_tag
        tagged = pos_tag(tokens)
        print(f"✓ POS tagger working correctly")
        
        print()
        print("All packages verified successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {str(e)}")
        print()
        print("Some packages may not be installed correctly.")
        print("Please run this script again or install manually.")
        return False


def main():
    """Main function."""
    print()
    print("L2 Pronunciation Assessment - NLTK Setup")
    print()
    
    # Download packages
    download_success = download_nltk_data()
    
    if not download_success:
        print("Setup incomplete. Please resolve errors and try again.")
        sys.exit(1)
    
    # Verify installation
    print("="*60)
    verify_success = verify_installation()
    print("="*60)
    
    if verify_success:
        print()
        print("Setup completed successfully!")
        print()
        sys.exit(0)
    else:
        print()
        print("Setup completed with warnings. Please check the errors above.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()