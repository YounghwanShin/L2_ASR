#!/bin/bash

set -e

if command -v python3 &> /dev/null; then
    echo "Set up a virtual environment..."
    python3 -m venv .env
    source .env/bin/activate
else
    echo "Python3 is not installed. Please install it first."
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