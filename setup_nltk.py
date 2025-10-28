"""NLTK Data Setup Script.

This script downloads required NLTK data packages for the pronunciation
assessment system.
"""

import nltk
import sys
import os


def download_nltk_data():
    """Downloads required NLTK data packages."""
    print("="*60)
    print("NLTK Data Setup")
    print("="*60)
    print()
    
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
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
    
    print("="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{len(packages)} packages")
    
    if failed_packages:
        print(f"Failed packages: {', '.join(failed_packages)}")
        return False
    else:
        print("All packages downloaded successfully!")
        return True


def main():
    """Main function."""
    print()
    print("L2 Pronunciation Assessment - NLTK Setup")
    print()
    
    download_success = download_nltk_data()
    
    if not download_success:
        print("Setup incomplete. Please resolve errors and try again.")
        sys.exit(1)
    else:
        print("Setup completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
