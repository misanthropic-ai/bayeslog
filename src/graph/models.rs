use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use serde_json;

/// Direction enum for specifying edge traversal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Outgoing edges (from source node to target nodes)
    Outgoing,
    /// Incoming edges (from other nodes to this node)
    Incoming,
    /// Both incoming and outgoing edges
    Both,
}

/// Property value for node and edge properties
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl From<serde_json::Value> for Value {
    fn from(json_value: serde_json::Value) -> Self {
        match json_value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Boolean(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    // Fallback to string if number doesn't fit in i64 or f64
                    Value::String(n.to_string())
                }
            },
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(arr) => {
                Value::Array(arr.into_iter().map(Value::from).collect())
            },
            serde_json::Value::Object(obj) => {
                Value::Object(obj.into_iter().map(|(k, v)| (k, Value::from(v))).collect())
            },
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => write!(f, "{}", s),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Null => write!(f, "null"),
            Value::Array(arr) => write!(f, "{:?}", arr),
            Value::Object(obj) => write!(f, "{:?}", obj),
        }
    }
}

impl Value {
    
    /// Try to get the value as a string
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Try to get the value as an integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }
    
    /// Try to get the value as a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Integer(i) => Some(*i as f64), // Automatic conversion
            _ => None,
        }
    }
    
    /// Try to get the value as a boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Node in the graph database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for the node
    pub id: String,
    /// Label describing the node type
    pub label: String,
    /// Additional properties as key-value pairs
    pub properties: HashMap<String, Value>,
}

impl Node {
    /// Create a new node with the given label and properties
    pub fn new(label: &str, properties: HashMap<String, Value>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            label: label.to_string(),
            properties,
        }
    }

    /// Create a new node with a specific ID, label, and properties
    pub fn with_id(id: &str, label: &str, properties: HashMap<String, Value>) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            properties,
        }
    }
}

/// Edge connecting two nodes in the graph database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier for the edge
    pub id: String,
    /// ID of the source node
    pub source_id: String,
    /// ID of the target node
    pub target_id: String,
    /// Label describing the edge type/relationship
    pub label: String,
    /// Additional properties as key-value pairs
    pub properties: HashMap<String, Value>,
}

impl Edge {
    /// Create a new edge connecting source_id to target_id with the given label and properties
    pub fn new(
        source_id: &str,
        label: &str,
        target_id: &str,
        properties: HashMap<String, Value>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            label: label.to_string(),
            properties,
        }
    }

    /// Create a new edge with a specific ID, source, target, label, and properties
    pub fn with_id(
        id: &str,
        source_id: &str,
        label: &str,
        target_id: &str,
        properties: HashMap<String, Value>,
    ) -> Self {
        Self {
            id: id.to_string(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            label: label.to_string(),
            properties,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_value_conversion() {
        // Test From<serde_json::Value>
        let json_string = serde_json::json!("test string");
        let json_int = serde_json::json!(42);
        let json_float = serde_json::json!(3.14);
        let json_bool = serde_json::json!(true);
        let json_null = serde_json::json!(null);
        let json_array = serde_json::json!([1, 2, 3]);
        let json_object = serde_json::json!({"key": "value"});
        
        // Check conversions
        assert!(matches!(Value::from(json_string), Value::String(s) if s == "test string"));
        assert!(matches!(Value::from(json_int), Value::Integer(i) if i == 42));
        assert!(matches!(Value::from(json_float), Value::Float(f) if f == 3.14));
        assert!(matches!(Value::from(json_bool), Value::Boolean(b) if b));
        assert!(matches!(Value::from(json_null), Value::Null));
        assert!(matches!(Value::from(json_array), Value::Array(_)));
        assert!(matches!(Value::from(json_object), Value::Object(_)));
    }
    
    #[test]
    fn test_value_accessors() {
        // Create values
        let string_val = Value::String("test".to_string());
        let int_val = Value::Integer(42);
        let float_val = Value::Float(3.14);
        let bool_val = Value::Boolean(true);
        
        // Test as_string
        assert_eq!(string_val.as_string(), Some(&"test".to_string()));
        assert_eq!(int_val.as_string(), None);
        
        // Test as_integer
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(string_val.as_integer(), None);
        
        // Test as_float
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(int_val.as_float(), Some(42.0)); // Integer converts to float
        assert_eq!(string_val.as_float(), None);
        
        // Test as_boolean
        assert_eq!(bool_val.as_boolean(), Some(true));
        assert_eq!(string_val.as_boolean(), None);
    }
    
    #[test]
    fn test_value_to_string() {
        let string_val = Value::String("test".to_string());
        let int_val = Value::Integer(42);
        let float_val = Value::Float(3.14);
        let bool_val = Value::Boolean(true);
        let null_val = Value::Null;
        let array_val = Value::Array(vec![int_val.clone(), string_val.clone()]);
        let mut obj_map = HashMap::new();
        obj_map.insert("key".to_string(), string_val.clone());
        let obj_val = Value::Object(obj_map);
        
        assert_eq!(string_val.to_string(), "test");
        assert_eq!(int_val.to_string(), "42");
        assert_eq!(float_val.to_string(), "3.14");
        assert_eq!(bool_val.to_string(), "true");
        assert_eq!(null_val.to_string(), "null");
        assert!(array_val.to_string().contains("Integer(42)"));
        assert!(obj_val.to_string().contains("key"));
    }
    
    #[test]
    fn test_node_creation() {
        let props = HashMap::from([
            ("name".to_string(), Value::String("Test Node".to_string())),
            ("value".to_string(), Value::Integer(42)),
        ]);
        
        // Test new
        let node = Node::new("TestLabel", props.clone());
        assert_eq!(node.label, "TestLabel");
        assert_eq!(node.properties.len(), 2);
        
        // Test with_id
        let node_id = "custom-id-123";
        let node = Node::with_id(node_id, "TestLabel", props);
        assert_eq!(node.id, node_id);
        assert_eq!(node.label, "TestLabel");
    }
    
    #[test]
    fn test_edge_creation() {
        let props = HashMap::from([
            ("weight".to_string(), Value::Float(1.5)),
        ]);
        
        // Test new
        let edge = Edge::new("source-123", "CONNECTS_TO", "target-456", props.clone());
        assert_eq!(edge.source_id, "source-123");
        assert_eq!(edge.target_id, "target-456");
        assert_eq!(edge.label, "CONNECTS_TO");
        
        // Test with_id
        let edge_id = "custom-edge-id";
        let edge = Edge::with_id(edge_id, "source-123", "CONNECTS_TO", "target-456", props);
        assert_eq!(edge.id, edge_id);
        assert_eq!(edge.source_id, "source-123");
        assert_eq!(edge.target_id, "target-456");
        assert_eq!(edge.label, "CONNECTS_TO");
    }
}