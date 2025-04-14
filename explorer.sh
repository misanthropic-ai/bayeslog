#!/bin/bash
export ROCKET_ENV=development

if [ -z "$1" ]; then
  echo "Usage: ./explorer.sh <scenario_name> [storage_type] [db_path]"
  echo "  scenario_name: Name of the scenario to visualize"
  echo "  storage_type: 'in-memory' (default) or 'persistent'"
  echo "  db_path: Path to SQLite database file (only used with persistent storage)"
  echo ""
  echo "Examples:"
  echo "  ./explorer.sh dating_simple"
  echo "  ./explorer.sh dating_simple in-memory"
  echo "  ./explorer.sh dating_simple persistent my_database.db"
  exit 1
fi

SCENARIO_NAME=$1
STORAGE_TYPE=${2:-in-memory}  # Default to in-memory if not specified

# Check if a database path is provided as the third argument
if [ -n "$3" ]; then
  DB_PATH="--db_path $3"
else
  DB_PATH=""
fi

RUST_BACKTRACE=1 cargo run --bin explorer_server -- --scenario_name $SCENARIO_NAME --storage_type $STORAGE_TYPE $DB_PATH