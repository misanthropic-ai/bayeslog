pub mod objects;
pub mod creators;
pub mod choose;
pub mod ops;
pub mod weights;
pub mod exponential;
pub mod config;
pub mod device;
pub mod torch_weights;
pub mod torch_exponential;
pub mod unified;
pub mod delta_weights;
pub mod weight_manager;

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
// use std::path::Path;
use chrono::{DateTime, Utc};

/// Common weight storage format that can be used by both ExponentialModel and TorchExponentialModel
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelWeights {
    /// Maps feature names to their weight values
    pub weights: HashMap<String, f64>,
    /// Maps feature names to their indices (for tensor-based models)
    pub feature_indices: HashMap<String, i64>,
    /// Namespace for the model
    pub namespace: String,
    /// Model type identifier
    pub model_type: String,
    /// Version for format compatibility
    pub version: u32,
    
    // Additional metadata
    /// Timestamp when the model was saved
    pub timestamp: DateTime<Utc>,
    /// Number of training steps completed
    pub training_steps: Option<i64>,
    /// Training loss at save time
    pub training_loss: Option<f64>,
    /// Graph node IDs for related entities (for linking with knowledge graph)
    pub entity_node_ids: Option<HashMap<String, String>>,
    /// Graph node IDs for related propositions
    pub proposition_node_ids: Option<HashMap<String, String>>,
}

impl ModelWeights {
    pub const CURRENT_VERSION: u32 = 1;
    
    /// Create a new ModelWeights instance
    pub fn new(namespace: String, model_type: String) -> Self {
        ModelWeights {
            weights: HashMap::new(),
            feature_indices: HashMap::new(),
            namespace,
            model_type,
            version: Self::CURRENT_VERSION,
            timestamp: Utc::now(),
            training_steps: None,
            training_loss: None,
            entity_node_ids: None,
            proposition_node_ids: None,
        }
    }
    
    /// Save weights to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    /// Load weights from a file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let json = fs::read_to_string(path)?;
        let weights: ModelWeights = serde_json::from_str(&json)?;
        
        // Check version compatibility
        if weights.version > Self::CURRENT_VERSION {
            return Err(format!(
                "Model file version {} is newer than supported version {}",
                weights.version, Self::CURRENT_VERSION
            ).into());
        }
        
        Ok(weights)
    }
    
    /// Create ModelWeights from a weight vector (for ExponentialModel)
    pub fn from_weight_map(weights: HashMap<String, f64>, namespace: String) -> Self {
        let mut feature_indices = HashMap::new();
        for (idx, feature) in weights.keys().enumerate() {
            feature_indices.insert(feature.clone(), idx as i64);
        }
        
        ModelWeights {
            weights,
            feature_indices,
            namespace,
            model_type: "exponential".to_string(),
            version: Self::CURRENT_VERSION,
            timestamp: Utc::now(),
            training_steps: None,
            training_loss: None,
            entity_node_ids: None,
            proposition_node_ids: None,
        }
    }
    
    /// Create ModelWeights from tensor data (for TorchExponentialModel)
    pub fn from_tensor_data(
        weights: HashMap<String, f64>,
        feature_indices: HashMap<String, i64>,
        namespace: String,
    ) -> Self {
        ModelWeights {
            weights,
            feature_indices,
            namespace,
            model_type: "torch_exponential".to_string(),
            version: Self::CURRENT_VERSION,
            timestamp: Utc::now(),
            training_steps: None,
            training_loss: None,
            entity_node_ids: None,
            proposition_node_ids: None,
        }
    }
}