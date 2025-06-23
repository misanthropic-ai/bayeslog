#!/bin/bash

# Script to set up libtorch for macOS

echo "Setting up libtorch for macOS..."

# Create directory for libtorch
mkdir -p ./libtorch_cache

# Check if libtorch is already downloaded
if [ ! -d "./libtorch_cache/libtorch" ]; then
    echo "Downloading libtorch..."
    cd libtorch_cache
    
    # Download libtorch for macOS (CPU version, which includes MPS support)
    curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip -o libtorch.zip
    
    # Extract
    unzip libtorch.zip
    rm libtorch.zip
    
    cd ..
    echo "Libtorch downloaded and extracted successfully!"
else
    echo "Libtorch already exists in cache."
fi

# Set environment variable
export LIBTORCH=$(pwd)/libtorch_cache/libtorch
echo "LIBTORCH environment variable set to: $LIBTORCH"

# Create a .env file for future use
echo "LIBTORCH=$(pwd)/libtorch_cache/libtorch" > .env
echo "Created .env file with LIBTORCH path"

echo "Setup complete! You can now build the project with:"
echo "  export LIBTORCH=$(pwd)/libtorch_cache/libtorch"
echo "  cargo build"