#!/bin/bash

FOLDER_ID="your_google_drive_folder_id"
DEST_DIR="/dataset"

mkdir -p "$DEST_DIR"

if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

echo "Downloading folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$DEST_DIR"

echo "Download completed. Files are saved in $DEST_DIR."