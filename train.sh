#!/bin/bash

# Assign variable names for better readability
SCENARIO_NAME=$1

# Check if the scenario name is provided
if [ -z "$SCENARIO_NAME" ] ; then
  echo "usage: ./train.sh <SCENARIO_NAME>"
  echo "Available scenarios: one_var, dating_simple"
  exit 1
fi

# Execute the training with the provided scenario
RUST_BACKTRACE=1 RUST_LOG=info cargo run --bin train -- --print_training_loss --scenario_name=$SCENARIO_NAME