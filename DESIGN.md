# BayesLog Design Documentation

This document outlines the internal design decisions and data structures for the BayesLog project, focusing on the implemented Graph Database component (Milestone 1).

## Graph Database Design

### Storage Strategy

The graph database component uses SQLite as its storage backend with the following characteristics:

1. **SQLite Tables**:
   - `nodes`: Stores entities with `id` (primary key), `label` (node type), and `properties` (JSON-serialized key-value pairs)
   - `edges`: Stores relationships with `id`, `source_id`, `target_id`, `label` (relationship type), and `properties`

2. **Connection Pooling**:
   - Uses `r2d2` and `r2d2_sqlite` for efficient connection management
   - Pool size configurable (default: 10 connections)
   - Prevents connection bottlenecks in concurrent scenarios

3. **Schema Optimization**:
   - Indices on frequently queried columns (`source_id`, `target_id`, `label`)
   - Write-Ahead Logging (WAL) mode for improved concurrency
   - Foreign key constraints to maintain referential integrity

### Core Data Structures

1. **Value Enum**:
   - Represents property values of different types
   - Supports String, Integer, Float, Boolean, Null, Array, and Object (nested key-value)
   - Bidirectional conversion with `serde_json::Value`
   - Helper methods (`as_string`, `as_integer`, etc.) for type-safe access

2. **Node Struct**:
   - Represents entities in the graph
   - Contains id, label, and properties (HashMap<String, Value>)
   - Factory methods: `new()` (auto-generates UUID) and `with_id()` (explicit ID)

3. **Edge Struct**:
   - Represents relationships between nodes
   - Contains id, source_id, target_id, label, and properties
   - Factory methods: `new()` and `with_id()`

4. **Direction Enum**:
   - Controls traversal direction in graph operations
   - Values: Outgoing, Incoming, Both

### Transaction Model

The database implements a closure-based transaction API with the following characteristics:

1. **Transaction Handling**:
   - `with_transaction<F, T>(&self, f: F) -> Result<T>` method that accepts a closure
   - Automatically manages transaction lifecycle (begin, commit, rollback)
   - Commits automatically on success, rolls back on error
   - Ensures atomicity for multi-step operations

2. **Advantages**:
   - Prevents transaction leakage
   - Simplifies error handling
   - Avoids lifetime complexities common in other approaches
   - Maintains ACID properties across multiple operations

### Core Operations

1. **Node Management**:
   - Creation with auto-generated UUIDs
   - Retrieval by ID with property deserialization
   - Updates that preserve node identity
   - Cascading deletion (nodes and connected edges)

2. **Edge Management**:
   - Creation with source/target validation
   - Edge properties for relationship metadata
   - Bidirectional traversal
   - Single-edge removal

3. **Graph Traversal**:
   - Neighbor retrieval (nodes connected by edges)
   - Direction-specific traversal (outgoing/incoming/both)
   - Edge retrieval for specific nodes

4. **Search Operations**:
   - Find nodes by label
   - Find edges by label
   - Find nodes by property value (with selective filtering)

### Optimizations

1. **Performance Enhancements**:
   - Strategic indices on frequently queried columns:
     - `idx_edges_source_id`: Optimizes outgoing edge queries
     - `idx_edges_target_id`: Optimizes incoming edge queries
     - `idx_edges_label`: Optimizes relationship type queries
     - `idx_nodes_label`: Optimizes node type queries
   
   - SQLite Configuration:
     - WAL journal mode: Improves concurrent read/write operations
     - `PRAGMA synchronous = NORMAL`: Balances durability and performance
     - `PRAGMA foreign_keys = ON`: Ensures data integrity

2. **Memory Management**:
   - Connection pooling to reuse connections
   - Prepared statements to minimize query compilation overhead
   - Bounded result sets to prevent memory exhaustion

### Serialization Strategy

1. **Property Storage**:
   - Properties stored as JSON strings in the database
   - Custom `Value` enum for type-safe property access
   - Bidirectional conversion with standard JSON types

2. **Type Conversion**:
   - `From<serde_json::Value>` implementation for seamless imports
   - Helper methods for intuitive type access and conversion
   - Graceful handling of type mismatches

### Error Handling

1. **Result-Based API**:
   - All operations return `Result<T>` types
   - Context-rich errors via `anyhow`
   - Transparent error propagation with `?` operator

2. **Error Categories**:
   - Database connection errors
   - Schema initialization errors
   - Query execution errors
   - Serialization/deserialization errors
   - Foreign key constraint violations

## Design Decisions and Rationale

### Why SQLite?

1. **Embedded Database**:
   - Zero-configuration setup
   - No separate server process
   - Consistent cross-platform behavior

2. **Performance Characteristics**:
   - Excellent for medium-sized graphs
   - Efficient single-file storage
   - Well-optimized query planner

3. **Integration Benefits**:
   - Easily embeddable in applications
   - Low operational overhead
   - Mature Rust ecosystem support

### Why r2d2 Connection Pooling?

1. **Concurrency Benefits**:
   - Efficient connection reuse
   - Connection lifetime management
   - Thread-safe access to database resources

2. **Alternatives Considered**:
   - Single mutex-locked connection: Limits concurrency
   - Thread-local connections: Complicates cross-thread operations
   - Manual connection management: Error-prone lifetime handling

### Why JSON for Properties?

1. **Flexibility**:
   - Schema-free property storage
   - Support for complex nested structures
   - Straightforward serialization/deserialization

2. **Efficiency Tradeoffs**:
   - Pro: Simplified schema evolution
   - Pro: Handles heterogeneous property sets
   - Con: Limited query capabilities on properties
   - Con: Requires deserialization for property-based filtering

### Transaction API Design Choice

1. **Closure-Based API**:
   - Automatic resource management
   - Clear ownership semantics
   - Prevents transaction leakage

2. **Alternatives Considered**:
   - Method-based API (begin, commit, rollback): Error-prone
   - RAII transaction objects: Complex lifetime management
   - Global transaction context: Thread-safety challenges

## Future Considerations

1. **Scaling Strategies**:
   - Sharding for larger graphs
   - Separate indices for optimized property queries
   - Read replicas for query-heavy workloads

2. **Performance Optimizations**:
   - Batch operations for bulk imports
   - Custom indices for frequently queried properties
   - Caching frequently accessed nodes and edges

3. **Feature Expansion**:
   - Streaming query results for memory efficiency
   - Graph algorithms (shortest path, community detection)
   - Change notification system for reactive applications