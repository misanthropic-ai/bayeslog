#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./visualize.sh <scenario_name>"
  echo "Example: ./visualize.sh dating_simple"
  exit 1
fi

SCENARIO_NAME=$1

# Build and run the visualizer
RUST_BACKTRACE=1 RUST_LOG=info cargo run --bin visualize -- --scenario_name $SCENARIO_NAME

# Open the generated HTML file with the default browser
if [ -f "network.html" ]; then
  echo "Opening visualization in browser..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open network.html
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open network.html
  elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start network.html
  else
    echo "Please open network.html manually in your browser"
  fi
fi