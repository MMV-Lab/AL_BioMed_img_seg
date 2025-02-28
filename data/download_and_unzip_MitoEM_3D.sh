#!/bin/bash

# Set TARGET_DIR from the first argument, default to current directory if not provided
TARGET_DIR="${1:-$(pwd)}"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# List of URLs to download
URLS=(
    "https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-H-im-pad.zip"
    "https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip"
    "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-H-mito-train-val-v2.zip"
    "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-R-mito-train-val-v2.zip"
)

# Change to target directory
cd "$TARGET_DIR"

# Download and extract each file into specific folders
for url in "${URLS[@]}"; do
    filename=$(basename "$url")  # Get filename from URL

    # Define extraction directory based on filename
    extract_dir=""
    case "$filename" in
        "EM30-H-im-pad.zip") extract_dir="EM30-H-im-pad" ;;
        "EM30-R-im.zip") extract_dir="EM30-R-im" ;;
        "EM30-H-mito-train-val-v2.zip") extract_dir="EM30-H-mito-train-val-v2" ;;
        "EM30-R-mito-train-val-v2.zip") extract_dir="EM30-R-mito-train-val-v2" ;;
    esac

    # Ensure the extraction directory exists
    mkdir -p "$TARGET_DIR/$extract_dir"

    # Download the file
    wget -c "$url" -O "$filename"

    # Extract file to the specified directory
    unzip -o "$filename" -d "$TARGET_DIR/$extract_dir"

    # Remove the zip file after extraction
    rm "$filename"
done

echo "All files have been downloaded and extracted to $TARGET_DIR"
