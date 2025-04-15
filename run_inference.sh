#!/bin/bash

# Get database path or use default
DB_PATH=${1:-dating_test.db}

# Check if the database file exists
if [ ! -f "$DB_PATH" ]; then
  echo "Error: Database file '$DB_PATH' does not exist"
  echo "You need to run the test_dating.sh script first to create and populate the database:"
  echo "  ./test_dating.sh persistent $DB_PATH"
  exit 1
fi

echo "Running inference on existing database: $DB_PATH"
echo "=============================================="

# Set environment variables and run the inference
RUST_BACKTRACE=1 DB_PATH=$DB_PATH cargo run --bin run_inference

# After running inference, query the database to see what's stored
echo ""
echo "Now querying the database to check its contents:"
./query_db.sh $DB_PATH