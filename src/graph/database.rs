use crate::graph::models::{Direction, Edge, Node, Value};
use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::collections::HashMap;

/// GraphDatabase handles storage and retrieval of nodes and edges using SQLite.
pub struct GraphDatabase {
    /// Connection pool for SQLite
    pool: Pool<SqliteConnectionManager>,
}

impl GraphDatabase {
    /// Create a new graph database with an in-memory SQLite database
    pub fn new_in_memory() -> Result<Self> {
        let manager = SqliteConnectionManager::memory();
        let pool = Pool::builder()
            .max_size(10) // Maximum connections in the pool
            .build(manager)
            .context("Failed to create connection pool")?;
        
        let db = Self { pool };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create a new graph database with a file-based SQLite database
    pub fn new(path: &str) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool = Pool::builder()
            .max_size(10) // Maximum connections in the pool
            .build(manager)
            .context("Failed to create connection pool")?;
        
        let db = Self { pool };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema with tables for nodes and edges
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT NOT NULL
            )",
            [],
        )
        .context("Failed to create nodes table")?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes (id),
                FOREIGN KEY (target_id) REFERENCES nodes (id)
            )",
            [],
        )
        .context("Failed to create edges table")?;

        // Create indices for faster lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges (source_id)",
            [],
        )
        .context("Failed to create source_id index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges (target_id)",
            [],
        )
        .context("Failed to create target_id index")?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_label ON edges (label)",
            [],
        )
        .context("Failed to create edge label index")?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes (label)",
            [],
        )
        .context("Failed to create node label index")?;
        
        // Enable WAL mode for better concurrent access (only for file-based databases)
        let _ = conn.pragma_update(None, "journal_mode", "WAL")
            .context("Failed to enable WAL mode");
            
        // Other performance optimizations
        let _ = conn.pragma_update(None, "synchronous", "NORMAL")
            .context("Failed to set synchronous mode");
            
        // Always ensure foreign keys are enabled
        conn.pragma_update(None, "foreign_keys", "ON")
            .context("Failed to enable foreign keys")?;

        Ok(())
    }

    /// Execute a function within a transaction
    /// 
    /// This method takes a closure that receives a transaction as an argument
    /// and executes it within that transaction. The transaction will be committed
    /// if the closure returns Ok, or rolled back if it returns Err.
    /// 
    /// # Example
    /// ```no_run
    /// # use anyhow::Result;
    /// # use bayeslog::graph::database::GraphDatabase;
    /// # fn main() -> Result<()> {
    /// # let db = GraphDatabase::new_in_memory()?;
    /// db.with_transaction(|tx| {
    ///     // Perform operations using the transaction
    ///     // The transaction will be committed if this closure returns Ok
    ///     Ok(())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Transaction) -> Result<T>,
    {
        let mut conn = self.pool.get()
            .context("Failed to get connection from pool")?;
        
        let tx = conn.transaction()?;
        
        // Execute the provided function with the transaction
        let result = f(&tx);
        
        // Commit or rollback the transaction based on the result
        match result {
            Ok(value) => {
                tx.commit()?;
                Ok(value)
            }
            Err(e) => {
                // Transaction will be rolled back automatically when dropped
                Err(e)
            }
        }
    }

    /// Add a node to the graph database
    pub fn add_node(&self, label: &str, properties: HashMap<String, Value>) -> Result<String> {
        let node = Node::new(label, properties);
        let node_id = node.id.clone();

        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        let properties_json = serde_json::to_string(&node.properties)
            .context("Failed to serialize node properties")?;

        conn.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            params![node.id, node.label, properties_json],
        )
        .context("Failed to insert node")?;

        Ok(node_id)
    }

    /// Get a node by its ID
    pub fn get_node(&self, id: &str) -> Result<Option<Node>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        let mut stmt = conn.prepare("SELECT id, label, properties FROM nodes WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let label: String = row.get(1)?;
            let properties_json: String = row.get(2)?;

            let properties: HashMap<String, Value> = serde_json::from_str(&properties_json)
                .context("Failed to deserialize node properties")?;

            Ok(Some(Node::with_id(&id, &label, properties)))
        } else {
            Ok(None)
        }
    }

    /// Update a node's properties
    pub fn update_node(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        // First check if the node exists
        let exists: bool = conn.query_row(
            "SELECT 1 FROM nodes WHERE id = ?1",
            params![id],
            |_| Ok(true),
        ).unwrap_or(false);

        if !exists {
            return Ok(false);
        }

        let properties_json = serde_json::to_string(&properties)
            .context("Failed to serialize updated properties")?;

        conn.execute(
            "UPDATE nodes SET properties = ?1 WHERE id = ?2",
            params![properties_json, id],
        )
        .context("Failed to update node properties")?;

        Ok(true)
    }

    /// Delete a node and all its connected edges
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        self.with_transaction(|tx| {
            // Check if node exists
            let exists: bool = tx
                .query_row("SELECT 1 FROM nodes WHERE id = ?1", params![id], |_| {
                    Ok(true)
                })
                .unwrap_or(false);

            if !exists {
                return Ok(false);
            }

            // Delete all connected edges first (both incoming and outgoing)
            tx.execute(
                "DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1",
                params![id],
            )
            .context("Failed to delete connected edges")?;

            // Delete the node
            tx.execute("DELETE FROM nodes WHERE id = ?1", params![id])
                .context("Failed to delete node")?;

            Ok(true)
        })
    }

    /// Add an edge connecting two nodes
    pub fn add_edge(
        &self,
        source_id: &str,
        label: &str,
        target_id: &str,
        properties: HashMap<String, Value>,
    ) -> Result<String> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        // Verify that both source and target nodes exist
        let source_exists: bool = conn
            .query_row(
                "SELECT 1 FROM nodes WHERE id = ?1",
                params![source_id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        let target_exists: bool = conn
            .query_row(
                "SELECT 1 FROM nodes WHERE id = ?1",
                params![target_id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !source_exists {
            return Err(anyhow::anyhow!("Source node with ID '{}' does not exist", source_id));
        }

        if !target_exists {
            return Err(anyhow::anyhow!("Target node with ID '{}' does not exist", target_id));
        }

        let edge = Edge::new(source_id, label, target_id, properties);
        let edge_id = edge.id.clone();

        let properties_json = serde_json::to_string(&edge.properties)
            .context("Failed to serialize edge properties")?;

        conn.execute(
            "INSERT INTO edges (id, source_id, target_id, label, properties)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![edge.id, edge.source_id, edge.target_id, edge.label, properties_json],
        )
        .context("Failed to insert edge")?;

        Ok(edge_id)
    }

    /// Get an edge by its ID
    pub fn get_edge(&self, id: &str) -> Result<Option<Edge>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, label, properties FROM edges WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let source_id: String = row.get(1)?;
            let target_id: String = row.get(2)?;
            let label: String = row.get(3)?;
            let properties_json: String = row.get(4)?;

            let properties: HashMap<String, Value> = serde_json::from_str(&properties_json)
                .context("Failed to deserialize edge properties")?;

            Ok(Some(Edge::with_id(
                &id, &source_id, &label, &target_id, properties,
            )))
        } else {
            Ok(None)
        }
    }

    /// Update an edge's properties
    pub fn update_edge(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        // First check if the edge exists
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM edges WHERE id = ?1",
                params![id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            return Ok(false);
        }

        let properties_json = serde_json::to_string(&properties)
            .context("Failed to serialize updated properties")?;

        conn.execute(
            "UPDATE edges SET properties = ?1 WHERE id = ?2",
            params![properties_json, id],
        )
        .context("Failed to update edge properties")?;

        Ok(true)
    }

    /// Delete an edge by its ID
    pub fn delete_edge(&self, id: &str) -> Result<bool> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        // Check if edge exists
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM edges WHERE id = ?1",
                params![id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            return Ok(false);
        }

        conn.execute("DELETE FROM edges WHERE id = ?1", params![id])
            .context("Failed to delete edge")?;

        Ok(true)
    }

    /// Get all neighbors of a node along with the connecting edges
    pub fn get_neighbors(&self, id: &str, direction: Direction) -> Result<Vec<(Node, Edge)>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        let mut neighbors = Vec::new();

        // For outgoing edges (this node -> others)
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            let mut stmt = conn.prepare(
                "SELECT e.id, e.source_id, e.target_id, e.label, e.properties,
                        n.id, n.label, n.properties
                 FROM edges e
                 JOIN nodes n ON e.target_id = n.id
                 WHERE e.source_id = ?1",
            )?;

            let rows = stmt.query_map(params![id], |row| {
                let edge_id: String = row.get(0)?;
                let source_id: String = row.get(1)?;
                let target_id: String = row.get(2)?;
                let edge_label: String = row.get(3)?;
                let edge_props_json: String = row.get(4)?;

                let node_id: String = row.get(5)?;
                let node_label: String = row.get(6)?;
                let node_props_json: String = row.get(7)?;

                let edge_props: HashMap<String, Value> =
                    serde_json::from_str(&edge_props_json).unwrap_or_default();
                let node_props: HashMap<String, Value> =
                    serde_json::from_str(&node_props_json).unwrap_or_default();

                let edge = Edge::with_id(&edge_id, &source_id, &edge_label, &target_id, edge_props);
                let node = Node::with_id(&node_id, &node_label, node_props);

                Ok((node, edge))
            })?;

            for row_result in rows {
                neighbors.push(row_result?);
            }
        }

        // For incoming edges (others -> this node)
        if matches!(direction, Direction::Incoming | Direction::Both) {
            let mut stmt = conn.prepare(
                "SELECT e.id, e.source_id, e.target_id, e.label, e.properties,
                        n.id, n.label, n.properties
                 FROM edges e
                 JOIN nodes n ON e.source_id = n.id
                 WHERE e.target_id = ?1",
            )?;

            let rows = stmt.query_map(params![id], |row| {
                let edge_id: String = row.get(0)?;
                let source_id: String = row.get(1)?;
                let target_id: String = row.get(2)?;
                let edge_label: String = row.get(3)?;
                let edge_props_json: String = row.get(4)?;

                let node_id: String = row.get(5)?;
                let node_label: String = row.get(6)?;
                let node_props_json: String = row.get(7)?;

                let edge_props: HashMap<String, Value> =
                    serde_json::from_str(&edge_props_json).unwrap_or_default();
                let node_props: HashMap<String, Value> =
                    serde_json::from_str(&node_props_json).unwrap_or_default();

                let edge = Edge::with_id(&edge_id, &source_id, &edge_label, &target_id, edge_props);
                let node = Node::with_id(&node_id, &node_label, node_props);

                Ok((node, edge))
            })?;

            for row_result in rows {
                neighbors.push(row_result?);
            }
        }

        Ok(neighbors)
    }

    /// Get all edges connected to a node
    pub fn get_node_edges(&self, id: &str, direction: Direction) -> Result<Vec<Edge>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;

        let mut edges = Vec::new();

        // For outgoing edges (this node -> others)
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            let mut stmt = conn.prepare(
                "SELECT id, source_id, target_id, label, properties
                 FROM edges
                 WHERE source_id = ?1",
            )?;

            let rows = stmt.query_map(params![id], |row| {
                let edge_id: String = row.get(0)?;
                let source_id: String = row.get(1)?;
                let target_id: String = row.get(2)?;
                let label: String = row.get(3)?;
                let properties_json: String = row.get(4)?;

                let properties: HashMap<String, Value> =
                    serde_json::from_str(&properties_json).unwrap_or_default();

                Ok(Edge::with_id(
                    &edge_id, &source_id, &label, &target_id, properties,
                ))
            })?;

            for row_result in rows {
                edges.push(row_result?);
            }
        }

        // For incoming edges (others -> this node)
        if matches!(direction, Direction::Incoming | Direction::Both) {
            let mut stmt = conn.prepare(
                "SELECT id, source_id, target_id, label, properties
                 FROM edges
                 WHERE target_id = ?1",
            )?;

            let rows = stmt.query_map(params![id], |row| {
                let edge_id: String = row.get(0)?;
                let source_id: String = row.get(1)?;
                let target_id: String = row.get(2)?;
                let label: String = row.get(3)?;
                let properties_json: String = row.get(4)?;

                let properties: HashMap<String, Value> =
                    serde_json::from_str(&properties_json).unwrap_or_default();

                Ok(Edge::with_id(
                    &edge_id, &source_id, &label, &target_id, properties,
                ))
            })?;

            for row_result in rows {
                edges.push(row_result?);
            }
        }

        Ok(edges)
    }
    
    /// Find nodes by label
    pub fn find_nodes_by_label(&self, label: &str) -> Result<Vec<Node>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;
            
        let mut stmt = conn.prepare("SELECT id, label, properties FROM nodes WHERE label = ?1")?;
        let rows = stmt.query_map(params![label], |row| {
            let id: String = row.get(0)?;
            let label: String = row.get(1)?;
            let properties_json: String = row.get(2)?;

            let properties: HashMap<String, Value> = serde_json::from_str(&properties_json)
                .unwrap_or_default();

            Ok(Node::with_id(&id, &label, properties))
        })?;
        
        let mut nodes = Vec::new();
        for row_result in rows {
            nodes.push(row_result?);
        }
        
        Ok(nodes)
    }
    
    /// Find edges by label
    pub fn find_edges_by_label(&self, label: &str) -> Result<Vec<Edge>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;
            
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, label, properties FROM edges WHERE label = ?1"
        )?;
        
        let rows = stmt.query_map(params![label], |row| {
            let edge_id: String = row.get(0)?;
            let source_id: String = row.get(1)?;
            let target_id: String = row.get(2)?;
            let label: String = row.get(3)?;
            let properties_json: String = row.get(4)?;

            let properties: HashMap<String, Value> = serde_json::from_str(&properties_json)
                .unwrap_or_default();

            Ok(Edge::with_id(
                &edge_id, &source_id, &label, &target_id, properties,
            ))
        })?;
        
        let mut edges = Vec::new();
        for row_result in rows {
            edges.push(row_result?);
        }
        
        Ok(edges)
    }
    
    /// Find nodes by property value
    pub fn find_nodes_by_property(&self, property_name: &str, property_value: &str) -> Result<Vec<Node>> {
        let conn = self.pool.get()
            .context("Failed to get connection from pool")?;
            
        // This is less efficient as it requires deserializing all properties but is necessary
        // since we're storing properties as JSON
        let mut stmt = conn.prepare("SELECT id, label, properties FROM nodes")?;
        
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let label: String = row.get(1)?;
            let properties_json: String = row.get(2)?;

            let properties: HashMap<String, Value> = serde_json::from_str(&properties_json)
                .unwrap_or_default();

            Ok(Node::with_id(&id, &label, properties))
        })?;
        
        let mut nodes = Vec::new();
        for row_result in rows {
            let node = row_result?;
            
            // Filter nodes based on property value
            if let Some(value) = node.properties.get(property_name) {
                // Compare string representation for simplicity
                // A more robust implementation would handle different value types
                if value.to_string().contains(property_value) {
                    nodes.push(node);
                }
            }
        }
        
        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_node() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add a node
        let props = HashMap::from([
            ("name".to_string(), Value::String("Test Node".to_string())),
            ("value".to_string(), Value::Integer(42)),
        ]);
        
        let node_id = db.add_node("TestLabel", props).unwrap();
        
        // Retrieve the node
        let node = db.get_node(&node_id).unwrap().unwrap();
        
        assert_eq!(node.id, node_id);
        assert_eq!(node.label, "TestLabel");
        
        if let Some(Value::String(name)) = node.properties.get("name") {
            assert_eq!(name, "Test Node");
        } else {
            panic!("Expected 'name' property to be a string");
        }
        
        if let Some(Value::Integer(value)) = node.properties.get("value") {
            assert_eq!(*value, 42);
        } else {
            panic!("Expected 'value' property to be an integer");
        }
    }
    
    #[test]
    fn test_update_node() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add a node
        let props = HashMap::from([
            ("name".to_string(), Value::String("Original Name".to_string())),
            ("value".to_string(), Value::Integer(100)),
        ]);
        
        let node_id = db.add_node("TestLabel", props).unwrap();
        
        // Update the node
        let updated_props = HashMap::from([
            ("name".to_string(), Value::String("Updated Name".to_string())),
            ("value".to_string(), Value::Integer(200)),
            ("new_prop".to_string(), Value::Boolean(true)),
        ]);
        
        let result = db.update_node(&node_id, updated_props).unwrap();
        assert!(result);
        
        // Retrieve the updated node
        let node = db.get_node(&node_id).unwrap().unwrap();
        
        if let Some(Value::String(name)) = node.properties.get("name") {
            assert_eq!(name, "Updated Name");
        } else {
            panic!("Expected 'name' property to be a string");
        }
        
        if let Some(Value::Integer(value)) = node.properties.get("value") {
            assert_eq!(*value, 200);
        } else {
            panic!("Expected 'value' property to be an integer");
        }
        
        if let Some(Value::Boolean(new_prop)) = node.properties.get("new_prop") {
            assert!(*new_prop);
        } else {
            panic!("Expected 'new_prop' property to be a boolean");
        }
        
        // Test updating a non-existent node
        let result = db.update_node("non-existent", HashMap::new()).unwrap();
        assert!(!result);
    }
    
    #[test]
    fn test_delete_node() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add a node
        let props = HashMap::from([
            ("name".to_string(), Value::String("Test Node".to_string())),
        ]);
        
        let node_id = db.add_node("TestLabel", props).unwrap();
        
        // Verify it exists
        assert!(db.get_node(&node_id).unwrap().is_some());
        
        // Delete the node
        let result = db.delete_node(&node_id).unwrap();
        assert!(result);
        
        // Verify it's gone
        assert!(db.get_node(&node_id).unwrap().is_none());
        
        // Try to delete a non-existent node
        let result = db.delete_node("non-existent").unwrap();
        assert!(!result);
    }
    
    #[test]
    fn test_add_and_get_edge() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add two nodes
        let source_props = HashMap::from([
            ("name".to_string(), Value::String("Source".to_string())),
        ]);
        let target_props = HashMap::from([
            ("name".to_string(), Value::String("Target".to_string())),
        ]);
        
        let source_id = db.add_node("Source", source_props).unwrap();
        let target_id = db.add_node("Target", target_props).unwrap();
        
        // Add an edge
        let edge_props = HashMap::from([
            ("weight".to_string(), Value::Float(1.5)),
            ("active".to_string(), Value::Boolean(true)),
        ]);
        
        let edge_id = db.add_edge(&source_id, "CONNECTS_TO", &target_id, edge_props).unwrap();
        
        // Retrieve the edge
        let edge = db.get_edge(&edge_id).unwrap().unwrap();
        
        assert_eq!(edge.id, edge_id);
        assert_eq!(edge.source_id, source_id);
        assert_eq!(edge.target_id, target_id);
        assert_eq!(edge.label, "CONNECTS_TO");
        
        if let Some(Value::Float(weight)) = edge.properties.get("weight") {
            assert_eq!(*weight, 1.5);
        } else {
            panic!("Expected 'weight' property to be a float");
        }
        
        if let Some(Value::Boolean(active)) = edge.properties.get("active") {
            assert!(*active);
        } else {
            panic!("Expected 'active' property to be a boolean");
        }
        
        // Test adding an edge with a non-existent source
        let result = db.add_edge("non-existent", "CONNECTS_TO", &target_id, HashMap::new());
        assert!(result.is_err());
        
        // Test adding an edge with a non-existent target
        let result = db.add_edge(&source_id, "CONNECTS_TO", "non-existent", HashMap::new());
        assert!(result.is_err());
        
        // Test getting a non-existent edge
        let result = db.get_edge("non-existent").unwrap();
        assert!(result.is_none());
    }
    
    #[test]
    fn test_get_node_edges() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes
        let alice_id = db.add_node("Person", HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
        ])).unwrap();
        
        let bob_id = db.add_node("Person", HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
        ])).unwrap();
        
        let charlie_id = db.add_node("Person", HashMap::from([
            ("name".to_string(), Value::String("Charlie".to_string())),
        ])).unwrap();
        
        // Add edges
        let props = HashMap::new();
        db.add_edge(&alice_id, "KNOWS", &bob_id, props.clone()).unwrap();
        db.add_edge(&bob_id, "KNOWS", &charlie_id, props.clone()).unwrap();
        db.add_edge(&charlie_id, "KNOWS", &alice_id, props.clone()).unwrap();
        
        // Test getting all edges
        let alice_outgoing = db.get_node_edges(&alice_id, Direction::Outgoing).unwrap();
        let alice_incoming = db.get_node_edges(&alice_id, Direction::Incoming).unwrap();
        let alice_both = db.get_node_edges(&alice_id, Direction::Both).unwrap();
        
        assert_eq!(alice_outgoing.len(), 1);
        assert_eq!(alice_incoming.len(), 1);
        assert_eq!(alice_both.len(), 2);
        
        // Test non-existent node
        let non_existent = db.get_node_edges("non-existent", Direction::Both).unwrap();
        assert_eq!(non_existent.len(), 0);
    }
    
    #[test]
    fn test_get_neighbors() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes
        let alice_props = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
        ]);
        let bob_props = HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
        ]);
        let charlie_props = HashMap::from([
            ("name".to_string(), Value::String("Charlie".to_string())),
        ]);
        
        let alice_id = db.add_node("Person", alice_props).unwrap();
        let bob_id = db.add_node("Person", bob_props).unwrap();
        let charlie_id = db.add_node("Person", charlie_props).unwrap();
        
        // Add edges
        let knows_props = HashMap::from([
            ("since".to_string(), Value::String("2020".to_string())),
        ]);
        
        db.add_edge(&alice_id, "KNOWS", &bob_id, knows_props.clone()).unwrap();
        db.add_edge(&bob_id, "KNOWS", &charlie_id, knows_props.clone()).unwrap();
        db.add_edge(&charlie_id, "KNOWS", &alice_id, knows_props.clone()).unwrap();
        
        // Test outgoing neighbors
        let alice_outgoing = db.get_neighbors(&alice_id, Direction::Outgoing).unwrap();
        assert_eq!(alice_outgoing.len(), 1);
        assert_eq!(alice_outgoing[0].0.id, bob_id);
        
        // Test incoming neighbors
        let alice_incoming = db.get_neighbors(&alice_id, Direction::Incoming).unwrap();
        assert_eq!(alice_incoming.len(), 1);
        assert_eq!(alice_incoming[0].0.id, charlie_id);
        
        // Test both directions
        let alice_both = db.get_neighbors(&alice_id, Direction::Both).unwrap();
        assert_eq!(alice_both.len(), 2);
        
        // Test non-existent node
        let non_existent = db.get_neighbors("non-existent", Direction::Both).unwrap();
        assert_eq!(non_existent.len(), 0);
    }
    
    #[test]
    fn test_update_edge() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes first
        let src_id = db.add_node("Source", HashMap::new()).unwrap();
        let dst_id = db.add_node("Target", HashMap::new()).unwrap();
        
        // Add an edge
        let edge_props = HashMap::from([
            ("weight".to_string(), Value::Float(1.0)),
        ]);
        
        let edge_id = db.add_edge(&src_id, "CONNECTS", &dst_id, edge_props).unwrap();
        
        // Update the edge
        let updated_props = HashMap::from([
            ("weight".to_string(), Value::Float(2.0)),
            ("priority".to_string(), Value::Integer(1)),
        ]);
        
        let result = db.update_edge(&edge_id, updated_props).unwrap();
        assert!(result);
        
        // Verify the update
        let edge = db.get_edge(&edge_id).unwrap().unwrap();
        assert_eq!(edge.properties.len(), 2);
        assert!(edge.properties.contains_key("weight"));
        assert!(edge.properties.contains_key("priority"));
        
        // Test updating a non-existent edge
        let result = db.update_edge("non-existent", HashMap::new()).unwrap();
        assert!(!result);
    }
    
    #[test]
    fn test_delete_edge() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add two nodes
        let source_props = HashMap::from([
            ("name".to_string(), Value::String("Source".to_string())),
        ]);
        let target_props = HashMap::from([
            ("name".to_string(), Value::String("Target".to_string())),
        ]);
        
        let source_id = db.add_node("Source", source_props).unwrap();
        let target_id = db.add_node("Target", target_props).unwrap();
        
        // Add an edge
        let edge_props = HashMap::new();
        let edge_id = db.add_edge(&source_id, "CONNECTS_TO", &target_id, edge_props).unwrap();
        
        // Verify it exists
        assert!(db.get_edge(&edge_id).unwrap().is_some());
        
        // Delete the edge
        let result = db.delete_edge(&edge_id).unwrap();
        assert!(result);
        
        // Verify it's gone
        assert!(db.get_edge(&edge_id).unwrap().is_none());
        
        // Try to delete a non-existent edge
        let result = db.delete_edge("non-existent").unwrap();
        assert!(!result);
    }
    
    #[test]
    fn test_node_deletion_cascade_to_edges() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes
        let node1_props = HashMap::from([
            ("name".to_string(), Value::String("Node 1".to_string())),
        ]);
        let node2_props = HashMap::from([
            ("name".to_string(), Value::String("Node 2".to_string())),
        ]);
        
        let node1_id = db.add_node("Node", node1_props).unwrap();
        let node2_id = db.add_node("Node", node2_props).unwrap();
        
        // Add edges in both directions
        let edge_props = HashMap::new();
        let edge1_id = db.add_edge(&node1_id, "CONNECTS_TO", &node2_id, edge_props.clone()).unwrap();
        let edge2_id = db.add_edge(&node2_id, "CONNECTS_TO", &node1_id, edge_props.clone()).unwrap();
        
        // Verify edges exist
        assert!(db.get_edge(&edge1_id).unwrap().is_some());
        assert!(db.get_edge(&edge2_id).unwrap().is_some());
        
        // Delete node1
        db.delete_node(&node1_id).unwrap();
        
        // Verify both edges are gone
        assert!(db.get_edge(&edge1_id).unwrap().is_none());
        assert!(db.get_edge(&edge2_id).unwrap().is_none());
        
        // Verify node2 still exists
        assert!(db.get_node(&node2_id).unwrap().is_some());
    }
    
    #[test]
    fn test_find_nodes_by_label() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes with different labels
        let alice_props = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
        ]);
        let bob_props = HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
        ]);
        let acme_props = HashMap::from([
            ("name".to_string(), Value::String("ACME Corp".to_string())),
        ]);
        
        db.add_node("Person", alice_props).unwrap();
        db.add_node("Person", bob_props).unwrap();
        db.add_node("Company", acme_props).unwrap();
        
        // Find nodes by label
        let persons = db.find_nodes_by_label("Person").unwrap();
        let companies = db.find_nodes_by_label("Company").unwrap();
        let nonexistent = db.find_nodes_by_label("NonExistent").unwrap();
        
        // Verify counts
        assert_eq!(persons.len(), 2);
        assert_eq!(companies.len(), 1);
        assert_eq!(nonexistent.len(), 0);
        
        // Verify labels
        for person in persons {
            assert_eq!(person.label, "Person");
        }
        
        for company in companies {
            assert_eq!(company.label, "Company");
        }
    }
    
    #[test]
    fn test_find_edges_by_label() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes
        let alice_props = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
        ]);
        let bob_props = HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
        ]);
        let charlie_props = HashMap::from([
            ("name".to_string(), Value::String("Charlie".to_string())),
        ]);
        
        let alice_id = db.add_node("Person", alice_props).unwrap();
        let bob_id = db.add_node("Person", bob_props).unwrap();
        let charlie_id = db.add_node("Person", charlie_props).unwrap();
        
        // Add edges with different labels
        let knows_props = HashMap::from([
            ("since".to_string(), Value::String("2020".to_string())),
        ]);
        
        let works_with_props = HashMap::from([
            ("project".to_string(), Value::String("Project X".to_string())),
        ]);
        
        db.add_edge(&alice_id, "KNOWS", &bob_id, knows_props.clone()).unwrap();
        db.add_edge(&bob_id, "KNOWS", &charlie_id, knows_props.clone()).unwrap();
        db.add_edge(&alice_id, "WORKS_WITH", &charlie_id, works_with_props.clone()).unwrap();
        
        // Find edges by label
        let knows_edges = db.find_edges_by_label("KNOWS").unwrap();
        let works_with_edges = db.find_edges_by_label("WORKS_WITH").unwrap();
        let nonexistent = db.find_edges_by_label("NONEXISTENT").unwrap();
        
        // Verify counts
        assert_eq!(knows_edges.len(), 2);
        assert_eq!(works_with_edges.len(), 1);
        assert_eq!(nonexistent.len(), 0);
        
        // Verify labels
        for edge in knows_edges {
            assert_eq!(edge.label, "KNOWS");
        }
        
        for edge in works_with_edges {
            assert_eq!(edge.label, "WORKS_WITH");
        }
    }
    
    #[test]
    fn test_find_nodes_by_property() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Add nodes with different properties
        let alice_props = HashMap::from([
            ("name".to_string(), Value::String("Alice Smith".to_string())),
            ("age".to_string(), Value::Integer(30)),
            ("city".to_string(), Value::String("New York".to_string())),
        ]);
        
        let bob_props = HashMap::from([
            ("name".to_string(), Value::String("Bob Johnson".to_string())),
            ("age".to_string(), Value::Integer(25)),
            ("city".to_string(), Value::String("Boston".to_string())),
        ]);
        
        let charlie_props = HashMap::from([
            ("name".to_string(), Value::String("Charlie Brown".to_string())),
            ("age".to_string(), Value::Integer(35)),
            ("city".to_string(), Value::String("New York".to_string())),
        ]);
        
        db.add_node("Person", alice_props).unwrap();
        db.add_node("Person", bob_props).unwrap();
        db.add_node("Person", charlie_props).unwrap();
        
        // Find nodes by property value
        let new_york_residents = db.find_nodes_by_property("city", "New York").unwrap();
        let name_with_smith = db.find_nodes_by_property("name", "Smith").unwrap();
        let nonexistent = db.find_nodes_by_property("country", "USA").unwrap();
        
        // Verify counts
        assert_eq!(new_york_residents.len(), 2);
        assert_eq!(name_with_smith.len(), 1);
        assert_eq!(nonexistent.len(), 0);
    }
    
    #[test]
    fn test_transaction_commit() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Execute operations in a transaction that should commit
        let result: Result<(), anyhow::Error> = db.with_transaction(|tx| {
            // Add a node
            let person_props = serde_json::to_string(&HashMap::from([
                ("name".to_string(), Value::String("Transaction Test".to_string())),
            ])).unwrap();
            
            tx.execute(
                "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
                params!["transaction-test", "Person", person_props],
            )?;
            
            Ok(())
        });
        
        // Check that the transaction was committed
        assert!(result.is_ok());
        let node = db.get_node("transaction-test").unwrap();
        assert!(node.is_some());
        assert_eq!(node.unwrap().label, "Person");
    }
    
    #[test]
    fn test_transaction_rollback() {
        let db = GraphDatabase::new_in_memory().unwrap();
        
        // Execute operations in a transaction that should roll back
        let result: Result<(), anyhow::Error> = db.with_transaction(|tx| {
            // Add a node
            let person_props = serde_json::to_string(&HashMap::from([
                ("name".to_string(), Value::String("Will Rollback".to_string())),
            ])).unwrap();
            
            tx.execute(
                "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
                params!["rollback-test", "Person", person_props],
            )?;
            
            // Force a rollback by returning an error
            Err(anyhow::anyhow!("Forced rollback for testing"))
        });
        
        // Check that the transaction was rolled back
        assert!(result.is_err());
        let node = db.get_node("rollback-test").unwrap();
        assert!(node.is_none());
    }
    
    #[test]
    fn test_new_file_database() {
        // Test the functionality with a temporary file
        use std::fs;
        use tempfile::tempdir;
        
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db_path_str = db_path.to_str().unwrap();
        
        // Create a database
        {
            let db = GraphDatabase::new(db_path_str).unwrap();
            
            // Add a test node
            let props = HashMap::from([
                ("name".to_string(), Value::String("Test".to_string())),
            ]);
            
            db.add_node("Test", props).unwrap();
            
            // Let db go out of scope and close
        }
        
        // Verify the file exists
        assert!(db_path.exists());
        
        // Open it again and check the data is there
        {
            let db = GraphDatabase::new(db_path_str).unwrap();
            let nodes = db.find_nodes_by_label("Test").unwrap();
            assert_eq!(nodes.len(), 1);
        }
        
        // Clean up
        fs::remove_file(db_path).ok();
        dir.close().unwrap();
    }
}