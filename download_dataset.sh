#!/bin/bash

set -e

echo "================================================"
echo "L2-ARCTIC Dataset Download Script"
echo "================================================"

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

echo "Downloading L2-ARCTIC dataset..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.tar.gz

echo "Creating data directory..."
mkdir -p data

echo "Extracting dataset..."
tar -xzf dataset.tar.gz -C data/

rm dataset.tar.gz

if [ -d "data/l2arctic" ]; then
    echo "================================================"
    echo "Dataset successfully downloaded!"
    echo "Location: ./data/l2arctic/"
    echo ""
    echo "Next steps:"
    echo "  1. python preprocess.py all"
    echo "  2. python main.py train --training_mode multitask"
    echo "================================================"
else
    echo "Error: Dataset directory not found!"
    exit 1
fi
