#!/bin/bash

set -e

if command -v conda &> /dev/null; then
    echo "Conda detected. Setting up a Conda environment..."
    conda create --prefix ./env python=3.12
    source activate env
elif command -v python3 &> /dev/null; then
    echo "Conda not found. Using venv to set up a virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
else
    echo "Neither Conda nor Python3 is installed. Please install one of them first."
    exit 1
fi

if [ -f "download.sh" ]; then
    echo "Running download.sh..."
    bash download.sh
else
    echo "download.sh not found. Please make sure it exists in the current directory."
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please make sure it exists in the current directory."
    exit 1
fi

echo "Setup complete."