#!/bin/bash

# Script to run bayeslog with PyTorch support

# Set libtorch path
export LIBTORCH=$(pwd)/libtorch_cache/libtorch

# Set dynamic library path for macOS
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH

# Set library path for Linux (in case running on Linux)
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Run the command passed as arguments
"$@"