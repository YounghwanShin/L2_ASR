"""NLTK data setup."""

import nltk


def download_nltk_data():
  """Downloads required NLTK packages."""
  packages = ['cmudict', 'punkt', 'averaged_perceptron_tagger_eng']
  
  for package in packages:
    print(f'Downloading {package}...')
    nltk.download(package)
  
  print('Setup complete!')


if __name__ == '__main__':
  download_nltk_data()
