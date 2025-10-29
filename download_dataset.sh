#!/bin/bash

set -e

echo "========================================"
echo "L2-ARCTIC Dataset Download"
echo "========================================"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Create data directory
mkdir -p data

# Download L2-ARCTIC dataset
echo ""
echo "Downloading L2-ARCTIC dataset..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.tar.gz

echo "Extracting L2-ARCTIC dataset..."
tar -xzf dataset.tar.gz -C data/
rm dataset.tar.gz

# Download phoneme mapping
echo ""
echo "Downloading phoneme to ID mapping..."
gdown 1DbIckREiWy5aJ_uu3fNClZ-oKI75_pR0 -O data/phoneme_to_id.json

# Verify downloads
if [ -d "data/l2arctic" ] && [ -f "data/phoneme_to_id.json" ]; then
    echo ""
    echo "========================================"
    echo "Download completed successfully"
    echo "========================================"
    echo "Dataset location: ./data/l2arctic/"
    echo "Phoneme mapping: ./data/phoneme_to_id.json"
    echo ""
    echo "Next steps:"
    echo "  1. python preprocess.py all"
    echo "  2. python main.py train --training_mode multitask"
    echo "========================================"
else
    echo "Error: Required files not found"
    exit 1
fi