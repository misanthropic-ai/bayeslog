use crate::graph::database::GraphDatabase;
use crate::graph::models::{Direction, Value};
use crate::qbbn::graphdb::schema::{NodeLabel, EdgeLabel};
use crate::qbbn::graphdb::schema::redis_property;
use crate::qbbn::graphdb::schema::namespace;
use anyhow::Result;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

/// GraphDBAdapter provides a Redis-compatible interface using our graph database
/// This adapter maintains compatibility with the existing Bayes Star implementation
/// but stores all data in the graph database instead of Redis.
pub struct GraphDBAdapter {
    /// Connection to the graph database
    pub graph_db: Arc<GraphDatabase>,
    /// Namespace for this network (used to separate multiple networks)
    pub namespace: String,
}

impl GraphDBAdapter {
    /// Creates a new GraphDBAdapter with the given database and namespace
    pub fn new(graph_db: Arc<GraphDatabase>, namespace: &str) -> Self {
        Self {
            graph_db,
            namespace: namespace.to_string(),
        }
    }

    /// Creates a new GraphDBAdapter instance (RedisManager equivalent)
    pub fn new_default() -> Result<Self, Box<dyn Error>> {
        Self::new_in_memory("default")
    }

    /// Creates a new GraphDBAdapter with an in-memory database
    pub fn new_in_memory(namespace: &str) -> Result<Self, Box<dyn Error>> {
        let graph_db = Arc::new(GraphDatabase::new_in_memory()?);
        Ok(Self::new(graph_db, namespace))
    }

    /// Creates a new GraphDBAdapter with a file-based database
    pub fn new_with_file(path: &str, namespace: &str) -> Result<Self, Box<dyn Error>> {
        let graph_db = Arc::new(GraphDatabase::new(path)?);
        Ok(Self::new(graph_db, namespace))
    }

    /// Creates a RefCell containing a MockConnection
    /// This matches the RefCell<Connection> pattern used in Bayes Star
    pub fn get_connection(&self) -> RefCell<crate::qbbn::common::redis::MockConnection> {
        use crate::qbbn::common::redis::MockConnection;
        let adapter = Self::new(Arc::clone(&self.graph_db), &self.namespace);
        let mock_conn = MockConnection::new(adapter);
        RefCell::new(mock_conn)
    }
    
    /// Returns a mutex-guarded connection
    pub fn get_mutex_guarded_connection(&self) -> Result<Mutex<crate::qbbn::common::redis::MockConnection>, Box<dyn Error>> {
        use crate::qbbn::common::redis::MockConnection;
        let adapter = Self::new(Arc::clone(&self.graph_db), &self.namespace);
        let mock_conn = MockConnection::new(adapter);
        Ok(Mutex::new(mock_conn))
    }
    
    /// Returns an Arc-wrapped mutex-guarded connection
    pub fn get_arc_mutex_guarded_connection(&self) -> Result<Arc<Mutex<crate::qbbn::common::redis::MockConnection>>, Box<dyn Error>> {
        use crate::qbbn::common::redis::MockConnection;
        let adapter = Self::new(Arc::clone(&self.graph_db), &self.namespace);
        let mock_conn = MockConnection::new(adapter);
        Ok(Arc::new(Mutex::new(mock_conn)))
    }

    //
    // Key-Value Storage (Redis String Equivalents)
    //

    /// Sets a value in the graph database (Redis SET equivalent)
    /// 
    /// Implemented by creating or updating a "KeyValue" node:
    /// - label: "KeyValue"
    /// - properties:
    ///   - key: The namespaced key
    ///   - value: The stored value
    pub fn set_value(&mut self, namespace: &str, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Check if key exists
        let props = HashMap::from([
            (redis_property::KEY.to_string(), Value::String(nskey.clone())),
            (redis_property::VALUE.to_string(), Value::String(value.to_string())),
        ]);
        
        // Find existing node
        let existing_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        if let Some(node) = existing_nodes.first() {
            // Update existing node
            self.graph_db.update_node(&node.id, props)?;
        } else {
            // Create new node
            self.graph_db.add_node(NodeLabel::KeyValue.as_str(), props)?;
        }
        
        Ok(())
    }

    /// Gets a value from the graph database (Redis GET equivalent)
    pub fn get_value(&mut self, namespace: &str, key: &str) -> Result<Option<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find node with matching key
        let nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        if let Some(node) = nodes.first() {
            if let Some(Value::String(value)) = node.properties.get(redis_property::VALUE) {
                return Ok(Some(value.clone()));
            }
        }
        
        Ok(None)
    }

    //
    // Hash Operations (Redis Hash Equivalents)
    //

    /// Sets a field in a hash (Redis HSET equivalent)
    /// 
    /// Implemented by creating or updating a "Hash" node with nested properties:
    /// - label: "Hash"
    /// - properties:
    ///   - key: The namespaced key
    ///   - fields: A HashMap stored as a JSON object
    pub fn map_insert(&mut self, namespace: &str, key: &str, field: &str, value: &str) -> Result<(), Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find existing hash
        let hash_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        if let Some(node) = hash_nodes.first() {
            // Update existing hash
            let mut props = node.properties.clone();
            
            // Extract fields or create new HashMap
            let mut fields = if let Some(Value::Object(fields_map)) = props.get(redis_property::FIELDS) {
                fields_map.clone()
            } else {
                HashMap::new()
            };
            
            // Update field
            fields.insert(field.to_string(), Value::String(value.to_string()));
            
            // Update properties
            props.insert(redis_property::FIELDS.to_string(), Value::Object(fields));
            
            // Update node
            self.graph_db.update_node(&node.id, props)?;
        } else {
            // Create new hash
            let mut fields = HashMap::new();
            fields.insert(field.to_string(), Value::String(value.to_string()));
            
            let props = HashMap::from([
                (redis_property::KEY.to_string(), Value::String(nskey)),
                (redis_property::FIELDS.to_string(), Value::Object(fields)),
            ]);
            
            self.graph_db.add_node(NodeLabel::Hash.as_str(), props)?;
        }
        
        Ok(())
    }

    /// Gets a field from a hash (Redis HGET equivalent)
    pub fn map_get(&mut self, namespace: &str, key: &str, field: &str) -> Result<Option<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find hash
        let hash_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        if let Some(node) = hash_nodes.first() {
            // Extract fields
            if let Some(Value::Object(fields)) = node.properties.get(redis_property::FIELDS) {
                // Get field value
                if let Some(Value::String(value)) = fields.get(field) {
                    return Ok(Some(value.clone()));
                }
            }
        }
        
        Ok(None)
    }

    //
    // Set Operations (Redis Set Equivalents)
    //

    /// Adds a member to a set (Redis SADD equivalent)
    /// 
    /// Implemented by creating or updating a "Set" node and connecting members:
    /// - Set node:
    ///   - label: "Set"
    ///   - properties:
    ///     - key: The namespaced key
    /// - Member nodes:
    ///   - label: "SetMember" 
    ///   - properties:
    ///     - value: The member value
    /// - Edges:
    ///   - from Set node to each Member node with label "CONTAINS"
    pub fn set_add(&mut self, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create set node
        let set_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        let set_id = if let Some(node) = set_nodes.first() {
            node.id.clone()
        } else {
            // Create new set
            let props = HashMap::from([
                (redis_property::KEY.to_string(), Value::String(nskey.clone())),
            ]);
            
            self.graph_db.add_node(NodeLabel::Set.as_str(), props)?
        };
        
        // Check if member already exists by checking edges
        let _set_node = self.graph_db.get_node(&set_id)?.unwrap();
        let edges = self.graph_db.get_node_edges(&set_id, Direction::Outgoing)?;
        
        // Check each edge to see if it points to a member with our value
        for edge in &edges {
            if edge.label == EdgeLabel::Contains.as_str() {
                if let Some(target_node) = self.graph_db.get_node(&edge.target_id)? {
                    if let Some(Value::String(val)) = target_node.properties.get(redis_property::VALUE) {
                        if val == member {
                            // Member already exists
                            return Ok(false);
                        }
                    }
                }
            }
        }
        
        // Create new member and connection
        let member_props = HashMap::from([
            (redis_property::VALUE.to_string(), Value::String(member.to_string())),
            ("set_key".to_string(), Value::String(nskey.clone())),
        ]);
        
        let member_id = self.graph_db.add_node(NodeLabel::SetMember.as_str(), member_props)?;
        
        // Connect set to member
        let edge_props = HashMap::new();
        self.graph_db.add_edge(&set_id, EdgeLabel::Contains.as_str(), &member_id, edge_props)?;
        
        Ok(true)
    }

    /// Gets all members of a set (Redis SMEMBERS equivalent)
    pub fn set_members(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find set node
        let set_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        let mut members = Vec::new();
        
        if let Some(set_node) = set_nodes.first() {
            // Get all outgoing edges with label CONTAINS
            let edges = self.graph_db.get_node_edges(&set_node.id, Direction::Outgoing)?;
            
            // For each edge, get the target node's value
            for edge in edges {
                if edge.label == EdgeLabel::Contains.as_str() {
                    if let Some(member_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(value)) = member_node.properties.get(redis_property::VALUE) {
                            members.push(value.clone());
                        }
                    }
                }
            }
        }
        
        Ok(members)
    }

    /// Checks if a member is in a set (Redis SISMEMBER equivalent)
    pub fn is_member(&mut self, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find set node
        let set_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        if let Some(set_node) = set_nodes.first() {
            // Get all outgoing edges with label CONTAINS
            let edges = self.graph_db.get_node_edges(&set_node.id, Direction::Outgoing)?;
            
            // For each edge, check if the target node's value matches
            for edge in edges {
                if edge.label == EdgeLabel::Contains.as_str() {
                    if let Some(member_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(value)) = member_node.properties.get(redis_property::VALUE) {
                            if value == member {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(false)
    }

    //
    // List Operations (Redis List Equivalents)
    //

    /// Pushes a value to a list (Redis RPUSH equivalent)
    /// 
    /// Implemented using a "List" node and ordered "ListItem" nodes:
    /// - List node:
    ///   - label: "List"
    ///   - properties:
    ///     - key: The namespaced key
    ///     - length: The current length
    /// - Item nodes:
    ///   - label: "ListItem"
    ///   - properties:
    ///     - value: The item value
    ///     - position: The position in the list (0-based)
    /// - Edges:
    ///   - from List to Items with label "HAS_ITEM" and position property
    pub fn seq_push(&mut self, namespace: &str, key: &str, value: &str) -> Result<i64, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create list node
        let list_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        let (list_id, current_length) = if let Some(node) = list_nodes.first() {
            let length = if let Some(Value::Integer(len)) = node.properties.get(redis_property::LENGTH) {
                *len
            } else {
                0
            };
            
            (node.id.clone(), length)
        } else {
            // Create new list
            let props = HashMap::from([
                (redis_property::KEY.to_string(), Value::String(nskey.clone())),
                (redis_property::LENGTH.to_string(), Value::Integer(0)),
            ]);
            
            let id = self.graph_db.add_node(NodeLabel::List.as_str(), props)?;
            (id, 0)
        };
        
        // Create new item
        let item_props = HashMap::from([
            (redis_property::VALUE.to_string(), Value::String(value.to_string())),
            (redis_property::POSITION.to_string(), Value::Integer(current_length)),
            ("list_key".to_string(), Value::String(nskey.clone())),
        ]);
        
        let item_id = self.graph_db.add_node(NodeLabel::ListItem.as_str(), item_props)?;
        
        // Connect list to item
        let edge_props = HashMap::from([
            (redis_property::POSITION.to_string(), Value::Integer(current_length)),
        ]);
        
        self.graph_db.add_edge(&list_id, EdgeLabel::HasItem.as_str(), &item_id, edge_props)?;
        
        // Update list length
        let new_length = current_length + 1;
        let updated_props = HashMap::from([
            (redis_property::KEY.to_string(), Value::String(nskey.clone())),
            (redis_property::LENGTH.to_string(), Value::Integer(new_length)),
        ]);
        
        self.graph_db.update_node(&list_id, updated_props)?;
        
        Ok(new_length)
    }

    /// Gets all items in a list (Redis LRANGE equivalent with 0 to -1)
    pub fn seq_get_all(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find list node
        let list_nodes = self.graph_db.find_nodes_by_property(redis_property::KEY, &nskey)?;
        
        let mut items = Vec::new();
        
        if let Some(list_node) = list_nodes.first() {
            // Get all outgoing edges with label HAS_ITEM
            let edges = self.graph_db.get_node_edges(&list_node.id, Direction::Outgoing)?;
            
            // Create a vector to store item values with their positions
            let mut positioned_items = Vec::new();
            
            // For each edge, get the target node's value and position
            for edge in edges {
                if edge.label == EdgeLabel::HasItem.as_str() {
                    if let Some(item_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(value)) = item_node.properties.get(redis_property::VALUE) {
                            if let Some(Value::Integer(position)) = item_node.properties.get(redis_property::POSITION) {
                                positioned_items.push((*position, value.clone()));
                            }
                        }
                    }
                }
            }
            
            // Sort by position
            positioned_items.sort_by_key(|(pos, _)| *pos);
            
            // Extract values in order
            items = positioned_items.into_iter().map(|(_, value)| value).collect();
        }
        
        Ok(items)
    }

    //
    // QBBN-specific Operations
    //

    // These operations would be implemented to directly work with 
    // the graph database for more efficient operations, but we'll start
    // with the Redis-compatible operations to ensure compatibility.
    // 
    // Future optimizations would include methods like:
    // - store_proposition() - Store a proposition as a proper graph node
    // - find_factors_with_premise() - Find factors having a given proposition as premise
    // - get_connected_propositions() - Get all propositions connected to a factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        assert_eq!(adapter.namespace, "test");
    }

    #[test]
    fn test_key_value_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Set and get a value
        adapter.set_value("test", "mykey", "myvalue").unwrap();
        let value = adapter.get_value("test", "mykey").unwrap();
        
        assert_eq!(value, Some("myvalue".to_string()));
        
        // Get a non-existent value
        let none_value = adapter.get_value("test", "nonexistent").unwrap();
        assert_eq!(none_value, None);
    }

    #[test]
    fn test_hash_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Insert hash fields
        adapter.map_insert("test", "myhash", "field1", "value1").unwrap();
        adapter.map_insert("test", "myhash", "field2", "value2").unwrap();
        
        // Get hash fields
        let value1 = adapter.map_get("test", "myhash", "field1").unwrap();
        let value2 = adapter.map_get("test", "myhash", "field2").unwrap();
        let none_value = adapter.map_get("test", "myhash", "nonexistent").unwrap();
        
        assert_eq!(value1, Some("value1".to_string()));
        assert_eq!(value2, Some("value2".to_string()));
        assert_eq!(none_value, None);
    }

    #[test]
    fn test_set_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Add set members
        let added1 = adapter.set_add("test", "myset", "member1").unwrap();
        let added2 = adapter.set_add("test", "myset", "member2").unwrap();
        let added_again = adapter.set_add("test", "myset", "member1").unwrap();
        
        assert!(added1);
        assert!(added2);
        assert!(!added_again); // Adding same member again returns false
        
        // Check membership
        let is_member1 = adapter.is_member("test", "myset", "member1").unwrap();
        let is_member2 = adapter.is_member("test", "myset", "member2").unwrap();
        let is_member3 = adapter.is_member("test", "myset", "nonexistent").unwrap();
        
        assert!(is_member1);
        assert!(is_member2);
        assert!(!is_member3);
        
        // Get all members
        let members = adapter.set_members("test", "myset").unwrap();
        
        assert_eq!(members.len(), 2);
        assert!(members.contains(&"member1".to_string()));
        assert!(members.contains(&"member2".to_string()));
    }

    #[test]
    fn test_list_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Push items
        let len1 = adapter.seq_push("test", "mylist", "item1").unwrap();
        let len2 = adapter.seq_push("test", "mylist", "item2").unwrap();
        let len3 = adapter.seq_push("test", "mylist", "item3").unwrap();
        
        assert_eq!(len1, 1);
        assert_eq!(len2, 2);
        assert_eq!(len3, 3);
        
        // Get all items
        let items = adapter.seq_get_all("test", "mylist").unwrap();
        
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], "item1");
        assert_eq!(items[1], "item2");
        assert_eq!(items[2], "item3");
    }
}