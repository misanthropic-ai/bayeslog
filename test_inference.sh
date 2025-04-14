#!/bin/bash

# Assign variable names for better readability
SCENARIO_NAME=$1

# Check if the scenario name is provided
if [ -z "$SCENARIO_NAME" ] ; then
  echo "usage: ./test_inference.sh <SCENARIO_NAME>"
  echo "Available scenarios: one_var, dating_simple"
  exit 1
fi

# Execute the inference test with the provided scenario
RUST_BACKTRACE=1 RUST_LOG=info cargo run --bin test_inference -- --scenario_name=$SCENARIO_NAME