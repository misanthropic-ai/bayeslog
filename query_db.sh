#!/bin/bash

# Check if database path is provided
if [ -z "$1" ]; then
  echo "Usage: ./query_db.sh <database_path>"
  echo "Example: ./query_db.sh dating_test.db"
  exit 1
fi

DB_PATH=$1

# Check if the file exists and is an SQLite database
if [ ! -f "$DB_PATH" ]; then
  echo "Error: File '$DB_PATH' does not exist"
  exit 1
fi

file_type=$(file "$DB_PATH" | grep -i "SQLite")
if [ -z "$file_type" ]; then
  echo "Error: '$DB_PATH' is not an SQLite database file"
  exit 1
fi

echo "===== DATABASE QUERY RESULTS ====="
echo "Database: $DB_PATH"
echo "================================="

# Query all tables
echo -e "\n=== Database Schema ==="
sqlite3 "$DB_PATH" ".schema"

# Count the number of nodes and edges
echo -e "\n=== Database Stats ==="
echo "Nodes count:" $(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM nodes;")
echo "Edges count:" $(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM edges;")

# Count nodes by label
echo -e "\n=== Node Types ==="
sqlite3 "$DB_PATH" "SELECT label, COUNT(*) FROM nodes GROUP BY label ORDER BY COUNT(*) DESC;"

# Show Hash nodes (used for evidence and other key-value storage)
echo -e "\n=== Hash Nodes (First 5) ==="
sqlite3 "$DB_PATH" "SELECT id, properties FROM nodes WHERE label='Hash' LIMIT 5;"

# Extract and show probabilities from the hash nodes (evidence)
echo -e "\n=== Proposition Probabilities ==="
sqlite3 "$DB_PATH" "
SELECT 
  json_extract(properties, '$.key') as namespace_key,
  json_extract(properties, '$.fields') as fields
FROM 
  nodes 
WHERE 
  label='Hash' AND
  json_extract(properties, '$.key') LIKE '%probabilities%';"

# Show SetMember nodes (used for collections)
echo -e "\n=== Set Members (First 5) ==="
sqlite3 "$DB_PATH" "SELECT json_extract(properties, '$.value') as value, json_extract(properties, '$.set_key') as set_key FROM nodes WHERE label='SetMember' LIMIT 5;"

echo -e "\n=== Relationship Query ==="
echo "Top source nodes with most outgoing edges:"
sqlite3 "$DB_PATH" "
SELECT 
  n.label as source_type, 
  e.label as relationship, 
  COUNT(*) as count 
FROM 
  edges e
  JOIN nodes n ON e.source_id = n.id
GROUP BY 
  n.label, e.label
ORDER BY 
  count DESC
LIMIT 5;"

echo -e "\n================================="
echo "Database query completed"