"""NLTK data setup script."""

import nltk


def download_nltk_data():
  """Downloads required NLTK packages for pronunciation assessment."""
  packages = [
      'cmudict',           # CMU Pronouncing Dictionary
      'punkt',             # Sentence tokenizer
      'averaged_perceptron_tagger_eng'  # POS tagger
  ]
  
  print("Downloading NLTK data packages...")
  for package in packages:
    print(f'  - {package}')
    try:
      nltk.download(package, quiet=True)
      print(f'    ✓ Downloaded')
    except Exception as e:
      print(f'    ✗ Error: {e}')
  
  print('\nSetup complete!')


if __name__ == '__main__':
  download_nltk_data()
