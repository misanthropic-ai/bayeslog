use crate::qbbn::common::redis::MockConnection as Connection;
use crate::qbbn::model::objects::ImplicationFactor;
use crate::qbbn::model::weights::{positive_feature, negative_feature, CLASS_LABELS};
use crate::qbbn::model::ModelWeights;
use log::trace;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use tch::{Device, Kind, Tensor};

/// GPU-accelerated weight storage using PyTorch tensors
pub struct TorchWeights {
    namespace: String,
    device: Device,
    /// Maps feature names to indices in the weight tensor
    feature_indices: Arc<Mutex<HashMap<String, i64>>>,
    /// The actual weight tensor
    weights: Arc<Mutex<Option<Tensor>>>,
    /// Next available index for new features
    next_index: Arc<Mutex<i64>>,
}

impl TorchWeights {
    pub fn new(namespace: String, device: Device) -> Result<Self, Box<dyn Error>> {
        Ok(TorchWeights {
            namespace,
            device,
            feature_indices: Arc::new(Mutex::new(HashMap::new())),
            weights: Arc::new(Mutex::new(None)),
            next_index: Arc::new(Mutex::new(0)),
        })
    }

    /// Initialize weights for a new implication
    pub fn initialize_weights(
        &mut self,
        _connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        trace!("TorchWeights::initialize_weights - Start: {:?}", implication);
        
        let feature = implication.unique_key();
        let mut feature_indices = self.feature_indices.lock().unwrap();
        let mut next_index = self.next_index.lock().unwrap();
        let mut weights = self.weights.lock().unwrap();
        
        for class_label in CLASS_LABELS {
            let posf = positive_feature(&feature, class_label);
            let negf = negative_feature(&feature, class_label);
            
            // Add features to index if not already present
            if !feature_indices.contains_key(&posf) {
                feature_indices.insert(posf.clone(), *next_index);
                *next_index += 1;
            }
            if !feature_indices.contains_key(&negf) {
                feature_indices.insert(negf.clone(), *next_index);
                *next_index += 1;
            }
        }
        
        // Resize weight tensor if needed
        let required_size = *next_index;
        match weights.as_mut() {
            Some(w) => {
                let current_size = w.size()[0];
                if current_size < required_size {
                    // Extend the weight tensor
                    let new_weights = Tensor::randn(&[required_size], (Kind::Float, self.device)) * 0.2;
                    new_weights.narrow(0, 0, current_size).copy_(w);
                    *w = new_weights;
                }
            }
            None => {
                // Initialize weight tensor
                *weights = Some(Tensor::randn(&[required_size], (Kind::Float, self.device)) * 0.2);
            }
        }
        
        trace!("TorchWeights::initialize_weights - End");
        Ok(())
    }

    /// Read weights for specific features
    pub fn read_weight_vector(
        &self,
        _connection: &mut Connection,
        features: &[String],
    ) -> Result<HashMap<String, f64>, Box<dyn Error>> {
        trace!("TorchWeights::read_weight_vector - Start");
        
        let feature_indices = self.feature_indices.lock().unwrap();
        let weights = self.weights.lock().unwrap();
        let mut result = HashMap::new();
        
        if let Some(weight_tensor) = weights.as_ref() {
            for feature in features {
                if let Some(&idx) = feature_indices.get(feature) {
                    let weight_value = weight_tensor.double_value(&[idx]);
                    result.insert(feature.clone(), weight_value);
                } else {
                    // Feature not found, use default weight of 0.0
                    result.insert(feature.clone(), 0.0);
                }
            }
        } else {
            // No weights initialized yet, return zeros
            for feature in features {
                result.insert(feature.clone(), 0.0);
            }
        }
        
        trace!("TorchWeights::read_weight_vector - End");
        Ok(result)
    }

    /// Save updated weights
    pub fn save_weight_vector(
        &mut self,
        _connection: &mut Connection,
        weight_updates: &HashMap<String, f64>,
    ) -> Result<(), Box<dyn Error>> {
        trace!("TorchWeights::save_weight_vector - Start");
        
        let feature_indices = self.feature_indices.lock().unwrap();
        let mut weights = self.weights.lock().unwrap();
        
        if let Some(weight_tensor) = weights.as_mut() {
            for (feature, &value) in weight_updates {
                if let Some(&idx) = feature_indices.get(feature) {
                    // Update the weight at the specific index
                    let _ = weight_tensor.narrow(0, idx, 1).fill_(value);
                }
            }
        }
        
        trace!("TorchWeights::save_weight_vector - End");
        Ok(())
    }

    /// Get all weights as a tensor for batch operations
    pub fn get_weight_tensor(&self) -> Option<Tensor> {
        self.weights.lock().unwrap().as_ref().map(|t| t.shallow_clone())
    }

    /// Update weights from a tensor (for optimizer updates)
    pub fn update_from_tensor(&mut self, new_weights: &Tensor) -> Result<(), Box<dyn Error>> {
        let mut weights = self.weights.lock().unwrap();
        *weights = Some(new_weights.shallow_clone());
        Ok(())
    }

    /// Get indices for a batch of features
    pub fn get_feature_indices(&self, features: &[String]) -> Vec<i64> {
        let feature_indices = self.feature_indices.lock().unwrap();
        features.iter()
            .map(|f| feature_indices.get(f).copied().unwrap_or(-1))
            .collect()
    }

    /// Convert features to a sparse tensor representation
    pub fn features_to_tensor(&self, features: &HashMap<String, f64>) -> Result<Tensor, Box<dyn Error>> {
        let feature_indices = self.feature_indices.lock().unwrap();
        let weights = self.weights.lock().unwrap();
        
        // Get the actual size of the weight tensor
        let num_features = if let Some(ref weight_tensor) = *weights {
            weight_tensor.size()[0]
        } else {
            // If no weights initialized yet, use next_index
            *self.next_index.lock().unwrap()
        };
        
        // If no features, return zeros
        if num_features == 0 {
            return Ok(Tensor::zeros(&[1], (Kind::Float, self.device)));
        }
        
        // Create a dense tensor initialized with zeros
        let mut values = vec![0.0f32; num_features as usize];
        
        // Fill in the feature values
        for (feature, &value) in features {
            if let Some(&idx) = feature_indices.get(feature) {
                if idx < num_features {
                    values[idx as usize] = value as f32;
                }
            }
        }
        
        Ok(Tensor::from_slice(&values).to_device(self.device))
    }
    
    /// Convert weights to ModelWeights format
    pub fn to_model_weights(&self) -> Result<ModelWeights, Box<dyn Error>> {
        trace!("TorchWeights::to_model_weights - Start");
        
        let feature_indices = self.feature_indices.lock().unwrap();
        let weights_tensor = self.weights.lock().unwrap();
        
        let mut weight_map = HashMap::new();
        
        if let Some(tensor) = weights_tensor.as_ref() {
            // Convert tensor weights to HashMap
            for (feature, &idx) in feature_indices.iter() {
                if idx >= 0 && idx < tensor.size()[0] {
                    let weight_value = tensor.double_value(&[idx]);
                    weight_map.insert(feature.clone(), weight_value);
                }
            }
        }
        
        let model_weights = ModelWeights::from_tensor_data(
            weight_map,
            feature_indices.clone(),
            self.namespace.clone(),
        );
        
        trace!("TorchWeights::to_model_weights - End");
        Ok(model_weights)
    }
    
    /// Load weights from ModelWeights format
    pub fn load_from_model_weights(&mut self, model_weights: &ModelWeights) -> Result<(), Box<dyn Error>> {
        trace!("TorchWeights::load_from_model_weights - Start");
        
        let mut feature_indices = self.feature_indices.lock().unwrap();
        let mut weights = self.weights.lock().unwrap();
        let mut next_index = self.next_index.lock().unwrap();
        
        // Clear existing data
        feature_indices.clear();
        *weights = None;
        *next_index = 0;
        
        // Load feature indices
        *feature_indices = model_weights.feature_indices.clone();
        
        // Find the maximum index
        let max_index = feature_indices.values().max().copied().unwrap_or(-1);
        if max_index >= 0 {
            *next_index = max_index + 1;
            
            // Create weight tensor
            let mut weight_values = vec![0.0f32; (*next_index) as usize];
            
            // Fill in the weights
            for (feature, &idx) in feature_indices.iter() {
                if let Some(&weight) = model_weights.weights.get(feature) {
                    if idx >= 0 && (idx as usize) < weight_values.len() {
                        weight_values[idx as usize] = weight as f32;
                    }
                }
            }
            
            // Create tensor from values
            *weights = Some(Tensor::from_slice(&weight_values).to_device(self.device));
        }
        
        trace!("TorchWeights::load_from_model_weights - End");
        Ok(())
    }
}