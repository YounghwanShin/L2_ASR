#!/bin/bash

echo "Installing gdown..."
pip install gdown

echo "Downloading dataset..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.tar.gz

echo "Creating data directory..."
mkdir -p data

echo "Extracting dataset..."
tar -xzf dataset.tar.gz -C data/

echo "Cleaning up..."
rm dataset.tar.gz

echo "Done! Dataset is in ./data/"