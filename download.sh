#!/bin/bash

FOLDER_ID="1fV1QU16-XvhP7RUP0sZzCk5T4Q1c0Atx"
DEST_DIR="/dataset"

mkdir -p "$DEST_DIR"

if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

echo "Downloading folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$DEST_DIR"

echo "Download completed. Files are saved in $DEST_DIR."