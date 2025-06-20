#!/bin/bash
# Runner script for cargo that sets up libtorch library paths

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up libtorch paths
export LIBTORCH="$SCRIPT_DIR/libtorch_cache/libtorch"
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"

# Run the actual binary
exec "$@"