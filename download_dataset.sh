#!/bin/bash

set -e

echo "================================"
echo "L2-ARCTIC Dataset Download"
echo "================================"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Create data directory
mkdir -p data

echo ""
echo "Downloading L2-ARCTIC dataset..."
echo "Note: You may need to manually download from:"
echo "https://psi.engr.tamu.edu/l2-arctic-corpus/"
echo ""
echo "After downloading, extract to data/l2arctic/"
echo ""

# Alternative: If you have direct download link
# gdown YOUR_GOOGLE_DRIVE_ID -O dataset.tar.gz
# tar -xzf dataset.tar.gz -C data/
# rm dataset.tar.gz

echo "Please manually download the dataset and place it in data/l2arctic/"
