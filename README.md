# BayesLog

BayesLog is an embeddable knowledge representation and reasoning system that combines a graph database, a Quantified Boolean Bayesian Network (QBBN) for probabilistic reasoning, an ontology for type hierarchies, and advanced indexing for retrieval.

## Project Overview

BayesLog aims to modernize symbolic AI with probabilistic reasoning, inspired by ProbLog, OpenCyc, and contemporary hybrid approaches. The project is implemented in Rust for performance and safety, with SQLite as its backend storage.

## Current Status

This project is currently in development. The core graph database functionality has been implemented and integrated with a QBBN (Quantified Boolean Bayesian Network) implementation. You can now run training and inference on belief networks using the graph database backend.

## Getting Started

### Prerequisites

- Rust and Cargo (2021 Edition or newer)
- SQLite

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bayeslog.git
   cd bayeslog
   ```

2. Build the project:
   ```
   cargo build
   ```

3. Run the examples:
   ```
   # Run the main example
   cargo run
   
   # Train a QBBN model using our graph database adapter
   ./train.sh one_var
   
   # Run with debug logging for more information
   RUST_LOG=debug ./train.sh one_var
   ```

## Features

### Graph Database

The graph database component provides a flexible way to store and query nodes (entities) and relationships (edges) with properties. It supports:

- CRUD operations for nodes and edges
- Property-based storage using JSON serialization
- Graph traversal with directional filtering
- Connection pooling with r2d2 for efficient concurrent access
- Transaction support for atomic operations

#### Usage Example

```rust
use bayeslog::graph::database::GraphDatabase;
use bayeslog::graph::models::{Direction, Value};
use std::collections::HashMap;

// Create an in-memory graph database
let db = GraphDatabase::new_in_memory()?;

// Add nodes with properties
let alice_props = HashMap::from([
    ("name".to_string(), Value::String("Alice Smith".to_string())),
    ("age".to_string(), Value::Integer(30)),
]);
let alice_id = db.add_node("Person", alice_props)?;

let bob_props = HashMap::from([
    ("name".to_string(), Value::String("Bob Johnson".to_string())),
    ("age".to_string(), Value::Integer(35)),
]);
let bob_id = db.add_node("Person", bob_props)?;

// Create a relationship
let knows_props = HashMap::from([
    ("since".to_string(), Value::String("2020".to_string())),
]);
let relationship_id = db.add_edge(&alice_id, "KNOWS", &bob_id, knows_props)?;

// Query the graph
if let Some(alice) = db.get_node(&alice_id)? {
    println!("Found Alice: {:#?}", alice);
}

// Get Alice's neighbors (people she knows)
let alices_connections = db.get_neighbors(&alice_id, Direction::Outgoing)?;
for (person, relationship) in alices_connections {
    println!("Alice knows {} since {}", 
        person.properties.get("name").unwrap(),
        relationship.properties.get("since").unwrap()
    );
}

// Use transactions for atomic operations
db.with_transaction(|tx| {
    // Multiple operations that either all succeed or all fail
    // For example: Update a node and its relationships in one atomic operation
    
    // Execute direct SQL if needed
    tx.execute(
        "UPDATE nodes SET properties = ?1 WHERE id = ?2",
        rusqlite::params!["{\"name\":\"Alice Updated\",\"age\":31}", alice_id],
    )?;
    
    Ok(())
})?;
```

#### API Documentation

- `GraphDatabase::new_in_memory()` - Create a new in-memory graph database
- `GraphDatabase::new(path: &str)` - Create a new file-based graph database
- `with_transaction<F, T>(&self, f: F) -> Result<T>` - Execute operations in a transaction
- `add_node(label: &str, properties: HashMap<String, Value>) -> String` - Add a node and return its ID
- `get_node(id: &str) -> Option<Node>` - Retrieve a node by ID
- `update_node(id: &str, properties: HashMap<String, Value>) -> bool` - Update a node's properties
- `delete_node(id: &str) -> bool` - Delete a node and its connected edges
- `add_edge(source_id: &str, label: &str, target_id: &str, properties: HashMap<String, Value>) -> String` - Add an edge between nodes
- `get_edge(id: &str) -> Option<Edge>` - Retrieve an edge by ID
- `update_edge(id: &str, properties: HashMap<String, Value>) -> bool` - Update an edge's properties
- `delete_edge(id: &str) -> bool` - Delete an edge
- `get_neighbors(id: &str, direction: Direction) -> Vec<(Node, Edge)>` - Get a node's neighbors with connecting edges
- `get_node_edges(id: &str, direction: Direction) -> Vec<Edge>` - Get all edges connected to a node

## Project Structure

- `src/graph/models.rs` - Core data structures for the graph database
- `src/graph/database.rs` - Implementation of the graph database with SQLite
- `src/belief/` - Belief network structures and operations
- `src/qbbn/` - Quantified Boolean Bayesian Network implementation
- `src/qbbn/graphdb/` - Graph database adapter for QBBN
- `src/bin/` - Executable programs including training and inference tools

## Future Development

BayesLog is being developed in a series of milestones:

1. ✅ Core Graph Database
2. ✅ Belief Network Foundation
3. ✅ Inference Engine
4. ✅ QBBN Graph Database Integration (Current)
5. Ontology System
6. REPL Interface
7. Indexing and Search
8. LLM Integration
9. Integration and Polish

See the [TODO.md](TODO.md) file for details on the current development status.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by ProbLog and OpenCyc
- Thanks to the Rust community for excellent libraries