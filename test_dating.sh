#!/bin/bash

# Get storage type argument or default to in-memory
STORAGE_TYPE=${1:-in-memory}

# Get optional database path
DB_PATH=$2

# Display usage if help is requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: ./test_dating.sh [storage_type] [db_path]"
  echo "  storage_type: 'in-memory' (default) or 'persistent'"
  echo "  db_path: Path to SQLite database file (only used with persistent storage)"
  echo ""
  echo "Examples:"
  echo "  ./test_dating.sh"
  echo "  ./test_dating.sh in-memory"
  echo "  ./test_dating.sh persistent dating_db.sqlite"
  exit 0
fi

echo "Running dating test with storage type: $STORAGE_TYPE"
if [ -n "$DB_PATH" ]; then
  echo "Using database path: $DB_PATH"
  # Execute the dating inference test with specified storage type and DB path
  RUST_BACKTRACE=1 RUST_LOG=info STORAGE_TYPE=$STORAGE_TYPE DB_PATH=$DB_PATH cargo run --bin test_dating
else
  # Execute the dating inference test with specified storage type only
  RUST_BACKTRACE=1 RUST_LOG=info STORAGE_TYPE=$STORAGE_TYPE cargo run --bin test_dating
fi