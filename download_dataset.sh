#!/bin/bash

# L2-ARCTIC Dataset Download Script

set -e

echo "================================================"
echo "L2-ARCTIC Dataset Download Script"
echo "================================================"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing gdown..."
    pip install gdown
    echo "gdown installed successfully!"
    echo ""
fi

# Download dataset
echo "Downloading L2-ARCTIC dataset..."
echo "This may take several minutes..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.tar.gz

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
    echo ""
else
    echo "Error: Download failed!"
    exit 1
fi

# Create data directory
echo "Creating data directory..."
mkdir -p data

# Extract dataset
echo "Extracting dataset to data/l2arctic/..."
tar -xzf dataset.tar.gz -C data/

if [ $? -eq 0 ]; then
    echo "Extraction completed successfully!"
    echo ""
else
    echo "Error: Extraction failed!"
    exit 1
fi

# Clean up
echo "Cleaning up temporary files..."
rm dataset.tar.gz
echo "Cleanup completed!"
echo ""

# Verify extraction
if [ -d "data/l2arctic" ]; then
    echo "================================================"
    echo "Dataset successfully downloaded and extracted!"
    echo "Location: ./data/l2arctic/"
    echo ""
    echo "Next steps:"
    echo "  1. Run preprocessing: python preprocess.py all"
    echo "  2. Or with CV: python preprocess.py all --use_cv --num_folds 5"
    echo "  3. Start training: python main.py train --training_mode multitask"
    echo "================================================"
else
    echo "Error: Dataset directory not found after extraction!"
    exit 1
fi
