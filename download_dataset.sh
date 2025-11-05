#!/bin/bash

set -e

echo "L2-ARCTIC Dataset Download"

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

mkdir -p data

echo "Downloading L2-ARCTIC dataset..."
gdown 1VzREuX7hP_-ksDewcbD1AGebSR5ASGiw -O dataset.tar.gz

echo "Extracting..."
tar -xzf dataset.tar.gz -C data/
rm dataset.tar.gz

echo "Downloading phoneme mapping..."
gdown 1DbIckREiWy5aJ_uu3fNClZ-oKI75_pR0 -O data/phoneme_to_id.json

echo "Download complete!"
