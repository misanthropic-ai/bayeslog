#!/bin/bash
export ROCKET_ENV=development

if [ -z "$1" ]; then
  echo "Usage: ./explorer.sh <scenario_name>"
  echo "Example: ./explorer.sh dating_simple"
  exit 1
fi

SCENARIO_NAME=$1

RUST_BACKTRACE=1 cargo run --bin explorer_server -- --scenario_name $SCENARIO_NAME