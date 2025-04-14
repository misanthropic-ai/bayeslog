use std::error::Error;
use crate::qbbn::graphdb::GraphDBAdapter;

// Re-export the GraphDBAdapter as RedisManager
pub use crate::qbbn::graphdb::GraphDBAdapter as RedisManager;

// Create a connection-like wrapper around GraphDBAdapter
pub struct MockConnection {
    pub adapter: GraphDBAdapter,
}

impl MockConnection {
    pub fn new(adapter: GraphDBAdapter) -> Self {
        Self { adapter }
    }
    
    pub fn new_in_memory() -> Result<Self, Box<dyn Error>> {
        // Create a new in-memory GraphDBAdapter with "default" namespace
        let adapter = GraphDBAdapter::new_in_memory("default")?;
        Ok(Self { adapter })
    }
}

// This is just a utility function we're keeping around, 
// but it's implemented directly in the GraphDBAdapter's schema
pub fn namespace_qualified_key(namespace: &str, key: &str) -> String {
    format!("bayes-star:{namespace}:{key}")
}

// Below we redefine the Redis functions to work with our MockConnection wrapper
// These delegate to the implementation in GraphDBAdapter

pub fn set_value(
    conn: &mut MockConnection,
    namespace: &str,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    conn.adapter.set_value(namespace, key, value)
}

pub fn get_value(
    conn: &mut MockConnection,
    namespace: &str,
    key: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    conn.adapter.get_value(namespace, key)
}

pub fn map_insert(
    conn: &mut MockConnection,
    namespace: &str,
    key: &str,
    field: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    conn.adapter.map_insert(namespace, key, field, value)
}

pub fn map_get(
    conn: &mut MockConnection,
    namespace: &str,
    key: &str,
    field: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    conn.adapter.map_get(namespace, key, field)
}

pub fn set_add(conn: &mut MockConnection, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
    conn.adapter.set_add(namespace, key, member)
}

pub fn set_members(conn: &mut MockConnection, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
    conn.adapter.set_members(namespace, key)
}

pub fn is_member(conn: &mut MockConnection, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
    conn.adapter.is_member(namespace, key, member)
}

pub fn seq_push(conn: &mut MockConnection, namespace: &str, key: &str, value: &str) -> Result<i64, Box<dyn Error>> {
    conn.adapter.seq_push(namespace, key, value)
}

pub fn seq_get_all(conn: &mut MockConnection, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
    conn.adapter.seq_get_all(namespace, key)
}
