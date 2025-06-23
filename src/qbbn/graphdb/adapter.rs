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

/// GraphDBAdapter provides a native graph-based implementation for the QBBN belief network
/// This adapter uses our built-in graph database to represent QBBN components (Propositions,
/// Predicates, Factors, etc.) as proper graph nodes with meaningful relationships between them,
/// rather than using generic storage nodes.
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

    /// Sets a field value for a property in a mapping
    /// 
    /// This version stores properties on appropriate graph nodes based on the key:
    /// - For "probabilities" keys: Creates/updates Proposition nodes with belief values
    /// - For other mappings: Creates/updates dedicated nodes based on their semantic meaning
    pub fn map_insert(&mut self, namespace: &str, key: &str, field: &str, value: &str) -> Result<(), Box<dyn Error>> {
        // Handle special cases based on key type
        if key == "probabilities" {
            // In this case, field is a proposition hash, value is a probability
            return self.store_proposition_probability(namespace, field, value);
        }
        
        // For now, keep a limited fallback implementation
        let nskey = namespace::qualified_key(namespace, key);
        
        // Look for an existing node that represents this mapping
        let key_nodes = self.graph_db.find_nodes_by_property("mapping_key", &nskey)?;
        
        if let Some(node) = key_nodes.first() {
            // Update existing node
            let mut props = node.properties.clone();
            props.insert(field.to_string(), Value::String(value.to_string()));
            self.graph_db.update_node(&node.id, props)?;
        } else {
            // Create a new node of the appropriate type based on the key
            let node_label = match key {
                "features" => NodeLabel::Feature,
                "weights" => NodeLabel::Weight,
                _ => NodeLabel::KeyValue, // Fallback for unknown keys
            };
            
            // Create properties including the field being set
            let mut props = HashMap::new();
            props.insert("mapping_key".to_string(), Value::String(nskey));
            props.insert(field.to_string(), Value::String(value.to_string()));
            
            self.graph_db.add_node(node_label.as_str(), props)?;
        }
        
        Ok(())
    }
    
    /// Store a probability value for a proposition
    fn store_proposition_probability(&mut self, namespace: &str, prop_hash: &str, prob_value: &str) -> Result<(), Box<dyn Error>> {
        // Find or create the Proposition node
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", prop_hash)?;
        
        if let Some(node) = prop_nodes.first() {
            // Update existing proposition
            let mut props = node.properties.clone();
            props.insert("belief".to_string(), Value::String(prob_value.to_string()));
            self.graph_db.update_node(&node.id, props)?;
        } else {
            // Create new proposition node
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(prop_hash.to_string())),
                ("belief".to_string(), Value::String(prob_value.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
        }
        
        Ok(())
    }

    /// Gets a field value from a mapping
    /// 
    /// This version retrieves properties from appropriate graph nodes based on the key:
    /// - For "probabilities" keys: Gets belief values from Proposition nodes
    /// - For other mappings: Gets properties from dedicated nodes based on semantic meaning
    pub fn map_get(&mut self, namespace: &str, key: &str, field: &str) -> Result<Option<String>, Box<dyn Error>> {
        // Handle special cases based on key type
        if key == "probabilities" {
            // In this case, field is a proposition hash, and we want to return its belief value
            return self.get_proposition_probability(namespace, field);
        }
        
        // For other keys, find the node with the mapping_key property
        let nskey = namespace::qualified_key(namespace, key);
        let nodes = self.graph_db.find_nodes_by_property("mapping_key", &nskey)?;
        
        if let Some(node) = nodes.first() {
            // Look for the requested field directly as a property
            if let Some(Value::String(value)) = node.properties.get(field) {
                return Ok(Some(value.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Get a probability value for a proposition
    fn get_proposition_probability(&mut self, _namespace: &str, prop_hash: &str) -> Result<Option<String>, Box<dyn Error>> {
        // Find the Proposition node
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", prop_hash)?;
        
        if let Some(node) = prop_nodes.first() {
            // Get the belief property
            if let Some(Value::String(belief)) = node.properties.get("belief") {
                return Ok(Some(belief.clone()));
            }
        }
        
        Ok(None)
    }

    //
    // Set Operations (Redis Set Equivalents)
    //

    /// Adds a member to a set, creating appropriate graph relationships based on the set type
    /// 
    /// This implementation uses graph-appropriate structures:
    /// - For premise sets: Creates Factor nodes with connections to Proposition nodes
    /// - For feature sets: Creates Feature nodes with connections to relevant entities
    /// - For general sets: Uses a more semantic approach based on the set meaning
    pub fn set_add(&mut self, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
        // Handle specific set types based on the key
        if key.starts_with("premises:") {
            return self.add_premise_to_factor(namespace, key, member);
        } else if key.starts_with("features:") {
            return self.add_feature_to_set(namespace, key, member);
        } else if key.starts_with("evidence:") {
            return self.add_evidence(namespace, key, member);
        }
        
        // Generic fallback implementation for backward compatibility
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create semantic collection node based on key
        let collection_label = self.determine_collection_type(key);
        let collection_nodes = self.graph_db.find_nodes_by_property("collection_key", &nskey)?;
        
        let collection_id = if let Some(node) = collection_nodes.first() {
            node.id.clone()
        } else {
            // Create new collection node
            let props = HashMap::from([
                ("collection_key".to_string(), Value::String(nskey.clone())),
                ("collection_type".to_string(), Value::String(key.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            // Use the string representation of the enum
            self.graph_db.add_node(&collection_label.as_str(), props)?
        };
        
        // Check if member already exists in this collection
        let edges = self.graph_db.get_node_edges(&collection_id, Direction::Outgoing)?;
        
        for edge in &edges {
            if edge.label == EdgeLabel::Contains.as_str() {
                if let Some(target_node) = self.graph_db.get_node(&edge.target_id)? {
                    if let Some(Value::String(val)) = target_node.properties.get("value") {
                        if val == member {
                            // Member already exists
                            return Ok(false);
                        }
                    }
                }
            }
        }
        
        // Create appropriate member node type
        let member_type = self.determine_member_type(key);
        let member_props = HashMap::from([
            ("value".to_string(), Value::String(member.to_string())),
            ("collection_key".to_string(), Value::String(nskey.clone())),
            ("namespace".to_string(), Value::String(namespace.to_string())),
        ]);
        
        let member_id = self.graph_db.add_node(&member_type.as_str(), member_props)?;
        
        // Connect collection to member with appropriate relationship
        let edge_props = HashMap::new();
        let edge_type = self.determine_edge_type(key);
        self.graph_db.add_edge(&collection_id, edge_type.as_str(), &member_id, edge_props)?;
        
        Ok(true)
    }
    
    /// Adds a premise to a factor, creating appropriate graph relationships
    fn add_premise_to_factor(&mut self, namespace: &str, key: &str, proposition_hash: &str) -> Result<bool, Box<dyn Error>> {
        // Extract factor ID from the key (format: "premises:factor_id")
        let factor_id = key.trim_start_matches("premises:");
        
        // Find the factor node
        let factor_nodes = self.graph_db.find_nodes_by_property("factor_id", factor_id)?;
        let factor_id = if let Some(node) = factor_nodes.first() {
            node.id.clone()
        } else {
            // If factor doesn't exist yet, create it
            let props = HashMap::from([
                ("factor_id".to_string(), Value::String(factor_id.to_string())),
                ("factor_type".to_string(), Value::String("implication".to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Factor.as_str(), props)?
        };
        
        // Find the proposition node
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", proposition_hash)?;
        let prop_id = if let Some(node) = prop_nodes.first() {
            node.id.clone()
        } else {
            // If proposition doesn't exist yet, create it
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(proposition_hash.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?
        };
        
        // Check if the premise relationship already exists
        let edges = self.graph_db.get_node_edges(&factor_id, Direction::Outgoing)?;
        for edge in &edges {
            if edge.label == EdgeLabel::HasPremise.as_str() && edge.target_id == prop_id {
                return Ok(false); // Premise already exists
            }
        }
        
        // Connect factor to proposition as premise
        let edge_props = HashMap::new();
        self.graph_db.add_edge(&factor_id, EdgeLabel::HasPremise.as_str(), &prop_id, edge_props)?;
        
        Ok(true)
    }
    
    /// Adds a feature to a set, creating appropriate graph relationships
    fn add_feature_to_set(&mut self, namespace: &str, key: &str, feature_id: &str) -> Result<bool, Box<dyn Error>> {
        // For now, just create feature nodes and connect them
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create feature set
        let set_nodes = self.graph_db.find_nodes_by_property("collection_key", &nskey)?;
        let set_id = if let Some(node) = set_nodes.first() {
            node.id.clone()
        } else {
            // Create feature set
            let props = HashMap::from([
                ("collection_key".to_string(), Value::String(nskey.clone())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(NodeLabel::Set.as_str(), props)?
        };
        
        // Find or create feature
        let feature_nodes = self.graph_db.find_nodes_by_property("feature_id", feature_id)?;
        let feature_id_node = if let Some(node) = feature_nodes.first() {
            node.id.clone()
        } else {
            // Create feature
            let props = HashMap::from([
                ("feature_id".to_string(), Value::String(feature_id.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(NodeLabel::Feature.as_str(), props)?
        };
        
        // Check if connection already exists
        let edges = self.graph_db.get_node_edges(&set_id, Direction::Outgoing)?;
        for edge in &edges {
            if edge.label == EdgeLabel::HasFeature.as_str() && edge.target_id == feature_id_node {
                return Ok(false); // Feature already in set
            }
        }
        
        // Connect set to feature
        let edge_props = HashMap::new();
        self.graph_db.add_edge(&set_id, EdgeLabel::HasFeature.as_str(), &feature_id_node, edge_props)?;
        
        Ok(true)
    }
    
    /// Adds evidence to the belief network
    fn add_evidence(&mut self, namespace: &str, _key: &str, proposition_hash: &str) -> Result<bool, Box<dyn Error>> {
        // Find the proposition node
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", proposition_hash)?;
        if let Some(node) = prop_nodes.first() {
            // Update existing proposition to mark as evidence
            let mut props = node.properties.clone();
            props.insert("evidence".to_string(), Value::Boolean(true));
            self.graph_db.update_node(&node.id, props)?;
        } else {
            // Create new proposition node marked as evidence
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(proposition_hash.to_string())),
                ("evidence".to_string(), Value::Boolean(true)),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
        };
        
        Ok(true)
    }
    
    /// Determines the appropriate collection node type based on the key
    fn determine_collection_type(&self, key: &str) -> NodeLabel {
        match key {
            k if k.starts_with("features:") => NodeLabel::Feature,
            k if k.starts_with("weights:") => NodeLabel::Weight,
            _ => NodeLabel::Set // Default
        }
    }
    
    /// Determines the appropriate member node type based on the key
    fn determine_member_type(&self, key: &str) -> NodeLabel {
        match key {
            k if k.starts_with("features:") => NodeLabel::Feature,
            k if k.starts_with("weights:") => NodeLabel::Weight,
            _ => NodeLabel::SetMember // Default
        }
    }
    
    /// Determines the appropriate edge type based on the key
    fn determine_edge_type(&self, key: &str) -> EdgeLabel {
        match key {
            k if k.starts_with("features:") => EdgeLabel::HasFeature,
            k if k.starts_with("weights:") => EdgeLabel::HasWeight,
            _ => EdgeLabel::Contains // Default
        }
    }

    /// Gets all members of a set, retrieving appropriate information based on set type
    pub fn set_members(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Handle specific set types based on the key
        if key.starts_with("premises:") {
            return self.get_premises_of_factor(namespace, key);
        } else if key.starts_with("features:") {
            return self.get_features_of_set(namespace, key);
        } else if key.starts_with("evidence:") {
            return self.get_evidence(namespace);
        }
        
        // Generic fallback implementation
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find collection node
        let collection_nodes = self.graph_db.find_nodes_by_property("collection_key", &nskey)?;
        
        let mut members = Vec::new();
        
        if let Some(collection_node) = collection_nodes.first() {
            // Get all outgoing edges with appropriate label
            let edge_type = self.determine_edge_type(key);
            let edges = self.graph_db.get_node_edges(&collection_node.id, Direction::Outgoing)?;
            
            // For each edge, get the target node's value based on type
            for edge in edges {
                if edge.label == edge_type.as_str() {
                    if let Some(member_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(value)) = member_node.properties.get("value") {
                            members.push(value.clone());
                        } else if let Some(Value::String(id)) = member_node.properties.get("feature_id") {
                            members.push(id.clone());
                        } else if let Some(Value::String(id)) = member_node.properties.get("predicate_hash") {
                            members.push(id.clone());
                        }
                    }
                }
            }
        }
        
        Ok(members)
    }
    
    /// Gets all premises of a factor
    fn get_premises_of_factor(&mut self, _namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Extract factor ID from the key (format: "premises:factor_id")
        let factor_id = key.trim_start_matches("premises:");
        
        // Find the factor node
        let factor_nodes = self.graph_db.find_nodes_by_property("factor_id", factor_id)?;
        
        let mut premises = Vec::new();
        
        if let Some(factor_node) = factor_nodes.first() {
            // Get all outgoing HasPremise edges
            let edges = self.graph_db.get_node_edges(&factor_node.id, Direction::Outgoing)?;
            
            for edge in edges {
                if edge.label == EdgeLabel::HasPremise.as_str() {
                    if let Some(prop_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(hash)) = prop_node.properties.get("predicate_hash") {
                            premises.push(hash.clone());
                        }
                    }
                }
            }
        }
        
        Ok(premises)
    }
    
    /// Gets all features of a set
    fn get_features_of_set(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find the feature set
        let set_nodes = self.graph_db.find_nodes_by_property("collection_key", &nskey)?;
        
        let mut features = Vec::new();
        
        if let Some(set_node) = set_nodes.first() {
            // Get all outgoing HasFeature edges
            let edges = self.graph_db.get_node_edges(&set_node.id, Direction::Outgoing)?;
            
            for edge in edges {
                if edge.label == EdgeLabel::HasFeature.as_str() {
                    if let Some(feature_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(id)) = feature_node.properties.get("feature_id") {
                            features.push(id.clone());
                        }
                    }
                }
            }
        }
        
        Ok(features)
    }
    
    /// Gets all evidence
    fn get_evidence(&mut self, namespace: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Find proposition nodes marked as evidence
        let mut evidence = Vec::new();
        
        // Get all Proposition nodes
        let props = self.graph_db.find_nodes_by_label(NodeLabel::Proposition.as_str())?;
        
        for prop in props {
            if let Some(Value::Boolean(is_evidence)) = prop.properties.get("evidence") {
                if *is_evidence {
                    if let Some(Value::String(hash)) = prop.properties.get("predicate_hash") {
                        if let Some(Value::String(ns)) = prop.properties.get("namespace") {
                            if ns == namespace {
                                evidence.push(hash.clone());
                            }
                        }
                    }
                }
            }
        }
        
        Ok(evidence)
    }

    /// Checks if a member is in a set, checking for the appropriate relationship type
    pub fn is_member(&mut self, namespace: &str, key: &str, member: &str) -> Result<bool, Box<dyn Error>> {
        // For specific types, get all members and check if the target is included
        // This is less efficient but ensures consistency with the refactored code
        if key.starts_with("premises:") || key.starts_with("features:") || key.starts_with("evidence:") {
            let members = self.set_members(namespace, key)?;
            return Ok(members.contains(&member.to_string()));
        }
        
        // Generic fallback implementation
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find collection node
        let collection_nodes = self.graph_db.find_nodes_by_property("collection_key", &nskey)?;
        
        if let Some(collection_node) = collection_nodes.first() {
            // Get all outgoing edges with appropriate label
            let edge_type = self.determine_edge_type(key);
            let edges = self.graph_db.get_node_edges(&collection_node.id, Direction::Outgoing)?;
            
            // For each edge, check if the target node's value matches
            for edge in edges {
                if edge.label == edge_type.as_str() {
                    if let Some(member_node) = self.graph_db.get_node(&edge.target_id)? {
                        // Check multiple possible property names
                        if let Some(Value::String(value)) = member_node.properties.get("value") {
                            if value == member {
                                return Ok(true);
                            }
                        } else if let Some(Value::String(id)) = member_node.properties.get("feature_id") {
                            if id == member {
                                return Ok(true);
                            }
                        } else if let Some(Value::String(id)) = member_node.properties.get("predicate_hash") {
                            if id == member {
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

    /// Pushes a value to a sequence, creating an ordered relationship
    /// 
    /// This implementation uses a more semantic graph structure based on the sequence type:
    /// - For training sequences: Creates ordered connections to appropriate training examples
    /// - For general sequences: Maintains order with position properties on edges
    pub fn seq_push(&mut self, namespace: &str, key: &str, value: &str) -> Result<i64, Box<dyn Error>> {
        // Handle special cases based on key type
        if key.starts_with("training:") {
            return self.add_training_example(namespace, key, value);
        }
        
        // Generic implementation with semantic collection type
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create sequence node
        let collection_label = self.determine_sequence_type(key);
        let seq_nodes = self.graph_db.find_nodes_by_property("sequence_key", &nskey)?;
        
        let (seq_id, current_length) = if let Some(node) = seq_nodes.first() {
            let length = if let Some(Value::Integer(len)) = node.properties.get("length") {
                *len
            } else {
                0
            };
            
            (node.id.clone(), length)
        } else {
            // Create new sequence node
            let props = HashMap::from([
                ("sequence_key".to_string(), Value::String(nskey.clone())),
                ("length".to_string(), Value::Integer(0)),
                ("sequence_type".to_string(), Value::String(key.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            let id = self.graph_db.add_node(&collection_label, props)?;
            (id, 0)
        };
        
        // Create new item node with appropriate type
        let item_type = self.determine_item_type(key);
        let item_props = HashMap::from([
            ("value".to_string(), Value::String(value.to_string())),
            ("position".to_string(), Value::Integer(current_length)),
            ("sequence_key".to_string(), Value::String(nskey.clone())),
            ("namespace".to_string(), Value::String(namespace.to_string())),
        ]);
        
        let item_id = self.graph_db.add_node(&item_type.as_str(), item_props)?;
        
        // Connect sequence to item with appropriate relationship
        let edge_type = EdgeLabel::HasItem;
        let edge_props = HashMap::from([
            ("position".to_string(), Value::Integer(current_length)),
        ]);
        
        self.graph_db.add_edge(&seq_id, edge_type.as_str(), &item_id, edge_props)?;
        
        // Update sequence length
        let new_length = current_length + 1;
        let mut updated_props = HashMap::new();
        for (k, v) in self.graph_db.get_node(&seq_id)?.unwrap().properties {
            updated_props.insert(k, v);
        }
        updated_props.insert("length".to_string(), Value::Integer(new_length));
        
        self.graph_db.update_node(&seq_id, updated_props)?;
        
        Ok(new_length)
    }
    
    /// Add a training example to the appropriate collection
    fn add_training_example(&mut self, namespace: &str, key: &str, example_hash: &str) -> Result<i64, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find or create training collection
        let training_nodes = self.graph_db.find_nodes_by_property("training_key", &nskey)?;
        
        let (training_id, current_length) = if let Some(node) = training_nodes.first() {
            let length = if let Some(Value::Integer(len)) = node.properties.get("length") {
                *len
            } else {
                0
            };
            
            (node.id.clone(), length)
        } else {
            // Create new training collection
            let props = HashMap::from([
                ("training_key".to_string(), Value::String(nskey.clone())),
                ("length".to_string(), Value::Integer(0)),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            let id = self.graph_db.add_node("TrainingSet", props)?;
            (id, 0)
        };
        
        // Find or create proposition for this example
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", example_hash)?;
        let prop_id = if let Some(node) = prop_nodes.first() {
            node.id.clone()
        } else {
            // Create placeholder proposition
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(example_hash.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?
        };
        
        // Check if example already exists in this training set
        let edges = self.graph_db.get_node_edges(&training_id, Direction::Outgoing)?;
        for edge in &edges {
            if edge.label == "INCLUDES_EXAMPLE" && edge.target_id == prop_id {
                // Example already in training set
                return Ok(current_length); // Don't increment length
            }
        }
        
        // Connect training set to example
        let edge_props = HashMap::from([
            ("position".to_string(), Value::Integer(current_length)),
        ]);
        
        self.graph_db.add_edge(&training_id, "INCLUDES_EXAMPLE", &prop_id, edge_props)?;
        
        // Update training set length
        let new_length = current_length + 1;
        let mut updated_props = HashMap::new();
        for (k, v) in self.graph_db.get_node(&training_id)?.unwrap().properties {
            updated_props.insert(k, v);
        }
        updated_props.insert("length".to_string(), Value::Integer(new_length));
        
        self.graph_db.update_node(&training_id, updated_props)?;
        
        Ok(new_length)
    }

    /// Gets all items in a sequence, retrieving appropriate objects based on sequence type
    pub fn seq_get_all(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Handle special cases based on key type
        if key.starts_with("training:") {
            return self.get_training_examples(namespace, key);
        }
        
        // Generic implementation
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find sequence node
        let seq_nodes = self.graph_db.find_nodes_by_property("sequence_key", &nskey)?;
        
        let mut items = Vec::new();
        
        if let Some(seq_node) = seq_nodes.first() {
            // Get all outgoing edges with appropriate label
            let edges = self.graph_db.get_node_edges(&seq_node.id, Direction::Outgoing)?;
            
            // Create a vector to store item values with their positions
            let mut positioned_items = Vec::new();
            
            // For each edge, get the target node's value and position
            for edge in edges {
                if edge.label == EdgeLabel::HasItem.as_str() {
                    if let Some(item_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(value)) = item_node.properties.get("value") {
                            if let Some(Value::Integer(position)) = item_node.properties.get("position") {
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
    
    /// Gets all training examples from a training set
    fn get_training_examples(&mut self, namespace: &str, key: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let nskey = namespace::qualified_key(namespace, key);
        
        // Find training set
        let training_nodes = self.graph_db.find_nodes_by_property("training_key", &nskey)?;
        
        let mut examples = Vec::new();
        
        if let Some(training_node) = training_nodes.first() {
            // Get all outgoing edges to examples
            let edges = self.graph_db.get_node_edges(&training_node.id, Direction::Outgoing)?;
            
            // Create a vector to store examples with their positions
            let mut positioned_examples = Vec::new();
            
            for edge in edges {
                if edge.label == "INCLUDES_EXAMPLE" {
                    if let Some(prop_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(hash)) = prop_node.properties.get("predicate_hash") {
                            if let Some(Value::Integer(position)) = edge.properties.get("position") {
                                positioned_examples.push((*position, hash.clone()));
                            }
                        }
                    }
                }
            }
            
            // Sort by position
            positioned_examples.sort_by_key(|(pos, _)| *pos);
            
            // Extract values in order
            examples = positioned_examples.into_iter().map(|(_, hash)| hash).collect();
        }
        
        Ok(examples)
    }
    
    /// Determines the appropriate sequence node type based on the key
    fn determine_sequence_type(&self, key: &str) -> String {
        match key {
            k if k.starts_with("training:") => "TrainingSet".to_string(),
            k if k.starts_with("examples:") => "ExampleSet".to_string(),
            _ => NodeLabel::List.as_str().to_string() // Default
        }
    }
    
    /// Determines the appropriate item node type based on the key
    fn determine_item_type(&self, key: &str) -> NodeLabel {
        match key {
            k if k.starts_with("training:") => NodeLabel::Proposition,
            k if k.starts_with("examples:") => NodeLabel::Proposition,
            _ => NodeLabel::ListItem // Default
        }
    }

    //
    // QBBN-specific Graph Operations
    //
    
    /// Store a proposition with its full details 
    pub fn store_proposition(&mut self, namespace: &str, proposition: &crate::qbbn::model::objects::Proposition) -> Result<String, Box<dyn Error>> {
        // Create a proposition node
        let predicate_hash = proposition.hash_string();
        
        let props = HashMap::from([
            ("predicate_hash".to_string(), Value::String(predicate_hash.clone())),
            ("namespace".to_string(), Value::String(namespace.to_string())),
            ("relation_name".to_string(), Value::String(proposition.predicate.relation.relation_name.clone())),
            ("belief".to_string(), Value::String("0.5".to_string())), // Default belief
            ("pi".to_string(), Value::String("0.5".to_string())),     // Default pi
            ("lambda".to_string(), Value::String("0.5".to_string())), // Default lambda
        ]);
        
        // Check if proposition already exists
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", &predicate_hash)?;
        
        let prop_id = if let Some(node) = prop_nodes.first() {
            // Update existing proposition
            self.graph_db.update_node(&node.id, props)?;
            node.id.clone()
        } else {
            // Create new proposition
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?
        };
        
        // Create or update the predicate node
        let predicate_props = HashMap::from([
            ("predicate_hash".to_string(), Value::String(predicate_hash.clone())),
            ("relation_name".to_string(), Value::String(proposition.predicate.relation.relation_name.clone())),
            ("namespace".to_string(), Value::String(namespace.to_string())),
        ]);
        
        let predicate_nodes = self.graph_db.find_nodes_by_property("predicate_hash", &predicate_hash)?;
        
        let predicate_id = if let Some(node) = predicate_nodes.first() {
            // Update existing predicate
            self.graph_db.update_node(&node.id, predicate_props)?;
            node.id.clone()
        } else {
            // Create new predicate
            self.graph_db.add_node(NodeLabel::Predicate.as_str(), predicate_props)?
        };
        
        // Connect proposition to predicate if not already connected
        let has_predicate_edges = self.graph_db.get_node_edges(&prop_id, Direction::Outgoing)?;
        let mut predicate_connected = false;
        
        for edge in &has_predicate_edges {
            if edge.label == "HAS_PREDICATE" && edge.target_id == predicate_id {
                predicate_connected = true;
                break;
            }
        }
        
        if !predicate_connected {
            self.graph_db.add_edge(&prop_id, "HAS_PREDICATE", &predicate_id, HashMap::new())?;
        }
        
        // For each role in the predicate, create or update argument nodes
        for role in &proposition.predicate.roles {
            let role_name = &role.role_name;
            let arg = &role.argument;
            
            // Create appropriate argument properties
            let mut arg_props = HashMap::new();
            arg_props.insert("role_name".to_string(), Value::String(role_name.clone()));
            arg_props.insert("namespace".to_string(), Value::String(namespace.to_string()));
            
            match arg {
                crate::qbbn::model::objects::Argument::Constant(const_arg) => {
                    arg_props.insert("arg_type".to_string(), Value::String("constant".to_string()));
                    arg_props.insert("domain".to_string(), Value::String(const_arg.domain.clone()));
                    arg_props.insert("entity_id".to_string(), Value::String(const_arg.entity_id.clone()));
                    
                    // Find or create entity node
                    let entity_props = HashMap::from([
                        ("entity_id".to_string(), Value::String(const_arg.entity_id.clone())),
                        ("domain".to_string(), Value::String(const_arg.domain.clone())),
                        ("namespace".to_string(), Value::String(namespace.to_string())),
                    ]);
                    
                    let entity_nodes = self.graph_db.find_nodes_by_property("entity_id", &const_arg.entity_id)?;
                    
                    let entity_id = if let Some(node) = entity_nodes.first() {
                        node.id.clone()
                    } else {
                        self.graph_db.add_node(NodeLabel::Entity.as_str(), entity_props)?
                    };
                    
                    // Find or create domain node
                    let domain_props = HashMap::from([
                        ("domain_name".to_string(), Value::String(const_arg.domain.clone())),
                        ("namespace".to_string(), Value::String(namespace.to_string())),
                    ]);
                    
                    let domain_nodes = self.graph_db.find_nodes_by_property("domain_name", &const_arg.domain)?;
                    
                    let domain_id = if let Some(node) = domain_nodes.first() {
                        node.id.clone()
                    } else {
                        self.graph_db.add_node(NodeLabel::Domain.as_str(), domain_props)?
                    };
                    
                    // Connect entity to domain if not already connected
                    let domain_edges = self.graph_db.get_node_edges(&entity_id, Direction::Outgoing)?;
                    let mut domain_connected = false;
                    
                    for edge in &domain_edges {
                        if edge.label == EdgeLabel::BelongsToDomain.as_str() && edge.target_id == domain_id {
                            domain_connected = true;
                            break;
                        }
                    }
                    
                    if !domain_connected {
                        self.graph_db.add_edge(&entity_id, EdgeLabel::BelongsToDomain.as_str(), &domain_id, HashMap::new())?;
                    }
                },
                crate::qbbn::model::objects::Argument::Variable(var_arg) => {
                    arg_props.insert("arg_type".to_string(), Value::String("variable".to_string()));
                    arg_props.insert("domain".to_string(), Value::String(var_arg.domain.clone()));
                }
            }
            
            // Find or create argument node
            let arg_hash = format!("{}:{}", predicate_hash, role_name);
            arg_props.insert("arg_hash".to_string(), Value::String(arg_hash.clone()));
            
            let arg_nodes = self.graph_db.find_nodes_by_property("arg_hash", &arg_hash)?;
            
            let arg_id = if let Some(node) = arg_nodes.first() {
                self.graph_db.update_node(&node.id, arg_props)?;
                node.id.clone()
            } else {
                self.graph_db.add_node(NodeLabel::Argument.as_str(), arg_props)?
            };
            
            // Connect predicate to argument if not already connected
            let has_arg_edges = self.graph_db.get_node_edges(&predicate_id, Direction::Outgoing)?;
            let mut arg_connected = false;
            
            for edge in &has_arg_edges {
                if edge.label == EdgeLabel::HasArgument.as_str() && edge.target_id == arg_id {
                    arg_connected = true;
                    break;
                }
            }
            
            if !arg_connected {
                let edge_props = HashMap::from([
                    ("role_name".to_string(), Value::String(role_name.clone())),
                ]);
                
                self.graph_db.add_edge(&predicate_id, EdgeLabel::HasArgument.as_str(), &arg_id, edge_props)?;
            }
        }
        
        Ok(prop_id)
    }
    
    /// Find all factors that have a specific proposition as a premise
    pub fn find_factors_with_premise(&mut self, _namespace: &str, proposition_hash: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Find the proposition node
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", proposition_hash)?;
        
        let mut factor_ids = Vec::new();
        
        if let Some(prop_node) = prop_nodes.first() {
            // Find all incoming edges with HasPremise label
            let edges = self.graph_db.get_node_edges(&prop_node.id, Direction::Incoming)?;
            
            for edge in edges {
                if edge.label == EdgeLabel::HasPremise.as_str() {
                    // This is a factor with this proposition as premise
                    if let Some(factor_node) = self.graph_db.get_node(&edge.source_id)? {
                        if let Some(Value::String(factor_id)) = factor_node.properties.get("factor_id") {
                            factor_ids.push(factor_id.clone());
                        }
                    }
                }
            }
        }
        
        Ok(factor_ids)
    }
    
    /// Find all propositions connected to a factor (premises and conclusions)
    pub fn get_connected_propositions(&mut self, _namespace: &str, factor_id: &str) -> Result<(Vec<String>, Vec<String>), Box<dyn Error>> {
        // Find the factor node
        let factor_nodes = self.graph_db.find_nodes_by_property("factor_id", factor_id)?;
        
        let mut premises = Vec::new();
        let mut conclusions = Vec::new();
        
        if let Some(factor_node) = factor_nodes.first() {
            // Find all outgoing edges to premises
            let edges = self.graph_db.get_node_edges(&factor_node.id, Direction::Outgoing)?;
            
            for edge in edges {
                if edge.label == EdgeLabel::HasPremise.as_str() {
                    if let Some(prop_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(hash)) = prop_node.properties.get("predicate_hash") {
                            premises.push(hash.clone());
                        }
                    }
                } else if edge.label == EdgeLabel::HasConclusion.as_str() {
                    if let Some(prop_node) = self.graph_db.get_node(&edge.target_id)? {
                        if let Some(Value::String(hash)) = prop_node.properties.get("predicate_hash") {
                            conclusions.push(hash.clone());
                        }
                    }
                }
            }
        }
        
        Ok((premises, conclusions))
    }
    
    /// Create a factor with premises and a conclusion
    pub fn create_factor(&mut self, namespace: &str, 
                         premises: &[String], 
                         conclusion: &str, 
                         factor_type: &str) -> Result<String, Box<dyn Error>> {
        // Create factor node
        let factor_id = format!("factor-{}", uuid::Uuid::new_v4());
        
        let factor_props = HashMap::from([
            ("factor_id".to_string(), Value::String(factor_id.clone())),
            ("factor_type".to_string(), Value::String(factor_type.to_string())),
            ("namespace".to_string(), Value::String(namespace.to_string())),
        ]);
        
        let factor_node_id = self.graph_db.add_node(&NodeLabel::Factor.as_str(), factor_props)?;
        
        // Connect to all premises
        for premise_hash in premises {
            // Find the premise proposition
            let premise_nodes = self.graph_db.find_nodes_by_property("predicate_hash", premise_hash)?;
            
            if let Some(premise_node) = premise_nodes.first() {
                // Connect factor to premise
                self.graph_db.add_edge(&factor_node_id, EdgeLabel::HasPremise.as_str(), &premise_node.id, HashMap::new())?;
            } else {
                // Create placeholder proposition
                let props = HashMap::from([
                    ("predicate_hash".to_string(), Value::String(premise_hash.clone())),
                    ("namespace".to_string(), Value::String(namespace.to_string())),
                ]);
                
                let premise_id = self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
                
                // Connect factor to premise
                self.graph_db.add_edge(&factor_node_id, EdgeLabel::HasPremise.as_str(), &premise_id, HashMap::new())?;
            }
        }
        
        // Connect to conclusion
        let conclusion_nodes = self.graph_db.find_nodes_by_property("predicate_hash", conclusion)?;
        
        if let Some(conclusion_node) = conclusion_nodes.first() {
            // Connect factor to conclusion
            self.graph_db.add_edge(&factor_node_id, EdgeLabel::HasConclusion.as_str(), &conclusion_node.id, HashMap::new())?;
        } else {
            // Create placeholder proposition
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(conclusion.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
            ]);
            
            let conclusion_id = self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
            
            // Connect factor to conclusion
            self.graph_db.add_edge(&factor_node_id, EdgeLabel::HasConclusion.as_str(), &conclusion_id, HashMap::new())?;
        }
        
        Ok(factor_id)
    }
    
    /// Set evidence on a proposition
    pub fn set_evidence(&mut self, namespace: &str, proposition_hash: &str, value: bool, confidence: f64) -> Result<(), Box<dyn Error>> {
        // Find the proposition
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", proposition_hash)?;
        
        if let Some(prop_node) = prop_nodes.first() {
            // Update the proposition properties
            let mut props = prop_node.properties.clone();
            
            // Set evidence properties
            props.insert("evidence".to_string(), Value::Boolean(true));
            props.insert("evidence_value".to_string(), Value::Boolean(value));
            props.insert("confidence".to_string(), Value::String(confidence.to_string()));
            
            // Set appropriate belief based on evidence value
            if value {
                props.insert("belief".to_string(), Value::String("1.0".to_string()));
                props.insert("lambda".to_string(), Value::String("1.0".to_string()));
                props.insert("pi".to_string(), Value::String("1.0".to_string()));
            } else {
                props.insert("belief".to_string(), Value::String("0.0".to_string()));
                props.insert("lambda".to_string(), Value::String("0.0".to_string()));
                props.insert("pi".to_string(), Value::String("0.0".to_string()));
            }
            
            self.graph_db.update_node(&prop_node.id, props)?;
        } else {
            // Create new proposition with evidence
            let mut props = HashMap::new();
            props.insert("predicate_hash".to_string(), Value::String(proposition_hash.to_string()));
            props.insert("namespace".to_string(), Value::String(namespace.to_string()));
            props.insert("evidence".to_string(), Value::Boolean(true));
            props.insert("evidence_value".to_string(), Value::Boolean(value));
            props.insert("confidence".to_string(), Value::String(confidence.to_string()));
            
            // Set appropriate belief based on evidence value
            if value {
                props.insert("belief".to_string(), Value::String("1.0".to_string()));
                props.insert("lambda".to_string(), Value::String("1.0".to_string()));
                props.insert("pi".to_string(), Value::String("1.0".to_string()));
            } else {
                props.insert("belief".to_string(), Value::String("0.0".to_string()));
                props.insert("lambda".to_string(), Value::String("0.0".to_string()));
                props.insert("pi".to_string(), Value::String("0.0".to_string()));
            }
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
        }
        
        Ok(())
    }
    
    /// Update belief values for a proposition
    pub fn update_belief(&mut self, namespace: &str, proposition_hash: &str, 
                      pi: f64, lambda: f64, belief: f64) -> Result<(), Box<dyn Error>> {
        // Find the proposition
        let prop_nodes = self.graph_db.find_nodes_by_property("predicate_hash", proposition_hash)?;
        
        if let Some(prop_node) = prop_nodes.first() {
            // Update the proposition properties
            let mut props = prop_node.properties.clone();
            
            // Don't update if this is evidence
            if let Some(Value::Boolean(is_evidence)) = props.get("evidence") {
                if *is_evidence {
                    return Ok(());
                }
            }
            
            // Set belief values
            props.insert("pi".to_string(), Value::String(pi.to_string()));
            props.insert("lambda".to_string(), Value::String(lambda.to_string()));
            props.insert("belief".to_string(), Value::String(belief.to_string()));
            
            self.graph_db.update_node(&prop_node.id, props)?;
        } else {
            // Create new proposition with these belief values
            let props = HashMap::from([
                ("predicate_hash".to_string(), Value::String(proposition_hash.to_string())),
                ("namespace".to_string(), Value::String(namespace.to_string())),
                ("pi".to_string(), Value::String(pi.to_string())),
                ("lambda".to_string(), Value::String(lambda.to_string())),
                ("belief".to_string(), Value::String(belief.to_string())),
            ]);
            
            self.graph_db.add_node(&NodeLabel::Proposition.as_str(), props)?;
        }
        
        Ok(())
    }
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
    fn test_probabilities_hash() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Test storing a probability in the probabilities hash
        adapter.map_insert("test", "probabilities", "prop1", "0.75").unwrap();
        
        // Verify we can retrieve it
        let value = adapter.map_get("test", "probabilities", "prop1").unwrap();
        assert_eq!(value, Some("0.75".to_string()));
        
        // Ensure it created a Proposition node
        let prop_nodes = adapter.graph_db.find_nodes_by_property("predicate_hash", "prop1").unwrap();
        assert!(!prop_nodes.is_empty());
        
        // Check the belief value is stored on the node
        if let Some(node) = prop_nodes.first() {
            if let Some(Value::String(belief)) = node.properties.get("belief") {
                assert_eq!(belief, "0.75");
            } else {
                panic!("Belief property not found or not a string");
            }
        }
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
    fn test_premise_set_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Add premises to a factor
        let factor_key = "premises:factor1";
        
        let added1 = adapter.set_add("test", factor_key, "premise1").unwrap();
        let added2 = adapter.set_add("test", factor_key, "premise2").unwrap();
        let added_again = adapter.set_add("test", factor_key, "premise1").unwrap();
        
        assert!(added1);
        assert!(added2);
        assert!(!added_again); // Adding same premise again returns false
        
        // Check membership
        let is_member1 = adapter.is_member("test", factor_key, "premise1").unwrap();
        let is_member2 = adapter.is_member("test", factor_key, "premise2").unwrap();
        let is_member3 = adapter.is_member("test", factor_key, "nonexistent").unwrap();
        
        assert!(is_member1);
        assert!(is_member2);
        assert!(!is_member3);
        
        // Get all premises
        let premises = adapter.set_members("test", factor_key).unwrap();
        
        assert_eq!(premises.len(), 2);
        assert!(premises.contains(&"premise1".to_string()));
        assert!(premises.contains(&"premise2".to_string()));
        
        // Verify Factor node was created with proper connections
        let factor_nodes = adapter.graph_db.find_nodes_by_property("factor_id", "factor1").unwrap();
        assert!(!factor_nodes.is_empty());
        
        if let Some(factor_node) = factor_nodes.first() {
            // Check connections to premises
            let edges = adapter.graph_db.get_node_edges(&factor_node.id, Direction::Outgoing).unwrap();
            let premise_edges = edges.iter().filter(|e| e.label == EdgeLabel::HasPremise.as_str()).count();
            assert_eq!(premise_edges, 2);
        }
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
    
    #[test]
    fn test_training_list_operations() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Push training examples
        let len1 = adapter.seq_push("test", "training:examples", "example1").unwrap();
        let len2 = adapter.seq_push("test", "training:examples", "example2").unwrap();
        
        assert_eq!(len1, 1);
        assert_eq!(len2, 2);
        
        // Get all training examples
        let examples = adapter.seq_get_all("test", "training:examples").unwrap();
        
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0], "example1");
        assert_eq!(examples[1], "example2");
        
        // Verify proper nodes were created
        let training_nodes = adapter.graph_db.find_nodes_by_property("training_key", 
            &namespace::qualified_key("test", "training:examples")).unwrap();
        assert!(!training_nodes.is_empty());
    }
    
    #[test]
    fn test_evidence_setting() {
        let mut adapter = GraphDBAdapter::new_in_memory("test").unwrap();
        
        // Set evidence directly with the QBBN-specific method
        adapter.set_evidence("test", "proposition1", true, 1.0).unwrap();
        
        // Verify evidence through the standard interface (through set membership)
        let is_member = adapter.is_member("test", "evidence:test", "proposition1").unwrap();
        assert!(is_member);
        
        // Verify the proposition node was created with evidence properties
        let prop_nodes = adapter.graph_db.find_nodes_by_property("predicate_hash", "proposition1").unwrap();
        assert!(!prop_nodes.is_empty());
        
        if let Some(node) = prop_nodes.first() {
            if let Some(Value::Boolean(is_evidence)) = node.properties.get("evidence") {
                assert!(*is_evidence);
            } else {
                panic!("Evidence property not found or not a boolean");
            }
            
            if let Some(Value::Boolean(value)) = node.properties.get("evidence_value") {
                assert!(*value);
            } else {
                panic!("Evidence value property not found or not a boolean");
            }
            
            if let Some(Value::String(belief)) = node.properties.get("belief") {
                assert_eq!(belief, "1.0");
            } else {
                panic!("Belief property not found or not a string");
            }
        }
    }
}