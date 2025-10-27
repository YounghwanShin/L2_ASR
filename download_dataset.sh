#!/bin/bash

echo "Installing gdown..."
pip install gdown

echo "Downloading dataset..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.zip

echo "Creating data directory..."
mkdir -p data

echo "Extracting dataset..."
unzip dataset.zip -d data/

echo "Cleaning up..."
rm dataset.zip

echo "Done! Dataset is in ./data/"