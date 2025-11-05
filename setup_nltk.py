"""NLTK data setup script."""

import nltk
import os


def download_nltk_data():
  """Downloads required NLTK packages."""
  print("="*60)
  print("NLTK Data Setup")
  print("="*60)
  
  nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
  os.makedirs(nltk_data_dir, exist_ok=True)
  
  packages = [
      ('cmudict', 'CMU Pronouncing Dictionary'),
      ('punkt', 'Punkt Tokenizer'),
      ('averaged_perceptron_tagger_eng', 'POS Tagger')
  ]
  
  for package_name, description in packages:
    print(f"\nDownloading {description} ({package_name})...")
    try:
      nltk.download(package_name, quiet=False)
      print(f"✓ Successfully downloaded {package_name}")
    except Exception as e:
      print(f"✗ Failed: {e}")
  
  print("\n" + "="*60)
  print("Setup complete!")
  print("="*60)


if __name__ == "__main__":
  download_nltk_data()
