use super::{
    delta_weights::{DeltaWeights, DeltaWeightStats},
    weights::ExponentialWeights,
    objects::ImplicationFactor,
    ModelWeights,
};
use crate::qbbn::common::redis::MockConnection as Connection;
use chrono::Duration;
use log::{debug, info, warn};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};

/// Manages base weights and delta weights for efficient online learning
/// Inspired by LoRA (Low-Rank Adaptation) approach in LLMs
pub struct WeightManager {
    /// Base weights (from training)
    base_weights: ExponentialWeights,
    
    /// Delta weights (from online learning)
    delta_weights: Arc<Mutex<DeltaWeights>>,
    
    /// Configuration
    config: WeightManagerConfig,
    
    /// Namespace
    namespace: String,
}

#[derive(Debug, Clone)]
pub struct WeightManagerConfig {
    /// Consolidate when delta size exceeds this fraction of base weights
    pub consolidation_threshold: f64,
    
    /// Maximum age for delta weights before consolidation
    pub max_delta_age: Duration,
    
    /// Maximum number of features in delta before consolidation
    pub max_delta_size: usize,
    
    /// Minimum updates before a feature is considered "hot"
    pub hot_feature_threshold: u32,
    
    /// Learning rate decay for old deltas
    pub decay_factor: f64,
}

impl Default for WeightManagerConfig {
    fn default() -> Self {
        WeightManagerConfig {
            consolidation_threshold: 0.1,  // 10% of base size
            max_delta_age: Duration::hours(24),
            max_delta_size: 10000,
            hot_feature_threshold: 10,
            decay_factor: 0.99,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConsolidationTrigger {
    DeltaSizeThreshold,
    TimeBasedSchedule,
    UpdateCountThreshold,
    MemoryPressure,
    Manual,
}

#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub trigger: ConsolidationTrigger,
    pub features_consolidated: usize,
    pub hot_features_kept: usize,
    pub duration_ms: u128,
}

impl WeightManager {
    /// Create a new WeightManager with base weights
    pub fn new(base_weights: ExponentialWeights, namespace: String) -> Self {
        Self::with_config(base_weights, namespace, WeightManagerConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(
        base_weights: ExponentialWeights,
        namespace: String,
        config: WeightManagerConfig,
    ) -> Self {
        let delta_weights = Arc::new(Mutex::new(DeltaWeights::new(namespace.clone())));
        
        WeightManager {
            base_weights,
            delta_weights,
            config,
            namespace,
        }
    }
    
    /// Read effective weight (base + delta)
    pub fn read_weight(
        &self,
        connection: &mut Connection,
        feature: &str,
    ) -> Result<f64, Box<dyn Error>> {
        // Get base weight
        let base_weight = self.base_weights.read_single_weight(connection, feature)?;
        
        // Get delta weight
        let delta_weight = {
            let delta = self.delta_weights.lock().unwrap();
            delta.get_delta(feature)
        };
        
        // Return combined weight
        Ok(base_weight + delta_weight)
    }
    
    /// Read effective weight vector (base + delta)
    pub fn read_weight_vector(
        &self,
        connection: &mut Connection,
        features: &[String],
    ) -> Result<HashMap<String, f64>, Box<dyn Error>> {
        // Get base weights
        let mut weights = self.base_weights.read_weight_vector(connection, features)?;
        
        // Apply deltas
        {
            let delta = self.delta_weights.lock().unwrap();
            for feature in features {
                if let Some(weight) = weights.get_mut(feature) {
                    *weight += delta.get_delta(feature);
                }
            }
        }
        
        Ok(weights)
    }
    
    /// Update a single weight (adds to delta)
    pub fn update_weight(&mut self, feature: &str, delta: f64) {
        let mut delta_weights = self.delta_weights.lock().unwrap();
        delta_weights.update_weight(feature, delta);
        
        // Check if consolidation is needed
        if self.should_consolidate(&delta_weights) {
            drop(delta_weights); // Release lock before consolidation
            if let Err(e) = self.consolidate(ConsolidationTrigger::DeltaSizeThreshold) {
                warn!("Auto-consolidation failed: {}", e);
            }
        }
    }
    
    /// Update multiple weights at once
    pub fn update_weights(&mut self, updates: &HashMap<String, f64>) {
        let mut delta_weights = self.delta_weights.lock().unwrap();
        for (feature, delta) in updates {
            delta_weights.update_weight(feature, *delta);
        }
        
        if self.should_consolidate(&delta_weights) {
            drop(delta_weights);
            if let Err(e) = self.consolidate(ConsolidationTrigger::DeltaSizeThreshold) {
                warn!("Auto-consolidation failed: {}", e);
            }
        }
    }
    
    /// Check if consolidation is needed
    fn should_consolidate(&self, delta_weights: &DeltaWeights) -> bool {
        // Check size threshold
        if delta_weights.size() > self.config.max_delta_size {
            return true;
        }
        
        // Check age threshold
        if let Some(oldest) = delta_weights.get_old_features(self.config.max_delta_age).first() {
            debug!("Found old delta features, oldest: {}", oldest);
            return true;
        }
        
        // TODO: Add memory pressure check
        
        false
    }
    
    /// Consolidate delta weights into base weights
    pub fn consolidate(
        &mut self,
        trigger: ConsolidationTrigger,
    ) -> Result<ConsolidationResult, Box<dyn Error>> {
        // Create a temporary connection for consolidation
        let mut temp_conn = Connection::new_in_memory()?;
        self.consolidate_with_connection(&mut temp_conn, trigger)
    }
    
    /// Consolidate delta weights into base weights using provided connection
    pub fn consolidate_with_connection(
        &mut self,
        connection: &mut Connection,
        trigger: ConsolidationTrigger,
    ) -> Result<ConsolidationResult, Box<dyn Error>> {
        let start = std::time::Instant::now();
        info!("Starting weight consolidation, trigger: {:?}", trigger);
        
        // Get features to consolidate
        let (features_to_consolidate, hot_features) = {
            let delta = self.delta_weights.lock().unwrap();
            let hot = delta.get_hot_features(self.config.hot_feature_threshold);
            let all_features: Vec<String> = delta.get_features()
                .into_iter()
                .cloned()
                .collect();
            (all_features, hot)
        };
        
        // Read current weights - only consolidate non-hot features
        let mut consolidated_weights = HashMap::new();
        for feature in &features_to_consolidate {
            if !hot_features.contains(feature) {
                let effective_weight = self.read_weight(connection, feature)?;
                consolidated_weights.insert(feature.clone(), effective_weight);
            }
        }
        
        // Save consolidated weights to base
        self.base_weights.save_weight_vector(connection, &consolidated_weights)?;
        
        // Clear consolidated features from delta (keep hot features with reduced weight)
        {
            let mut delta = self.delta_weights.lock().unwrap();
            
            // Clear non-hot features
            let features_to_clear: Vec<String> = features_to_consolidate
                .iter()
                .filter(|f| !hot_features.contains(f))
                .cloned()
                .collect();
            
            delta.clear_features(&features_to_clear);
            
            // Decay hot features instead of clearing
            for hot_feature in &hot_features {
                if let Some(delta_value) = delta.deltas.get_mut(hot_feature) {
                    *delta_value *= self.config.decay_factor;
                }
            }
        }
        
        let duration_ms = start.elapsed().as_millis();
        let result = ConsolidationResult {
            trigger,
            features_consolidated: features_to_consolidate.len() - hot_features.len(),
            hot_features_kept: hot_features.len(),
            duration_ms,
        };
        
        info!("Consolidation complete: {:?}", result);
        Ok(result)
    }
    
    /// Get delta weight statistics
    pub fn get_delta_stats(&self) -> DeltaWeightStats {
        self.delta_weights.lock().unwrap().get_stats()
    }
    
    /// Save both base and delta weights
    pub fn save_to_files(
        &self,
        connection: &mut Connection,
        base_path: &str,
        delta_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Save base weights
        let base_model_weights = self.base_weights.to_model_weights(connection)?;
        base_model_weights.save_to_file(base_path)?;
        
        // Save delta weights
        self.delta_weights.lock().unwrap().save_to_file(delta_path)?;
        
        Ok(())
    }
    
    /// Load from saved files
    pub fn load_from_files(
        namespace: String,
        base_path: &str,
        delta_path: &str,
        connection: &mut Connection,
    ) -> Result<Self, Box<dyn Error>> {
        // Load base weights
        let base_model_weights = ModelWeights::load_from_file(base_path)?;
        let mut base_weights = ExponentialWeights::new(namespace.clone())?;
        base_weights.load_from_model_weights(connection, &base_model_weights)?;
        
        // Load delta weights
        let delta_weights = DeltaWeights::load_from_file(delta_path)?;
        
        Ok(WeightManager {
            base_weights,
            delta_weights: Arc::new(Mutex::new(delta_weights)),
            config: WeightManagerConfig::default(),
            namespace,
        })
    }
    
    /// Initialize weights for a new implication
    pub fn initialize_weights(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        self.base_weights.initialize_weights(connection, implication)
    }
    
    /// Export all weights (base + delta) as ModelWeights
    pub fn export_weights(&self, connection: &mut Connection) -> Result<ModelWeights, Box<dyn Error>> {
        // Get all features from base weights
        let base_model = self.base_weights.to_model_weights(connection)?;
        let mut all_weights = base_model.weights.clone();
        
        // Apply deltas
        {
            let delta = self.delta_weights.lock().unwrap();
            for (feature, base_weight) in &mut all_weights {
                let delta_weight = delta.get_delta(feature);
                *base_weight += delta_weight;
            }
            
            // Add any features that only exist in delta
            for feature in delta.get_features() {
                if !all_weights.contains_key(feature) {
                    all_weights.insert(feature.clone(), delta.get_delta(feature));
                }
            }
        }
        
        Ok(ModelWeights {
            weights: all_weights,
            feature_indices: base_model.feature_indices,
            namespace: self.namespace.clone(),
            model_type: base_model.model_type,
            version: base_model.version,
            timestamp: base_model.timestamp,
            training_steps: base_model.training_steps,
            training_loss: base_model.training_loss,
            entity_node_ids: base_model.entity_node_ids,
            proposition_node_ids: base_model.proposition_node_ids,
        })
    }
}