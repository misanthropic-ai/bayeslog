#!/bin/bash

# Execute the dating inference test
RUST_BACKTRACE=1 RUST_LOG=info cargo run --bin test_dating