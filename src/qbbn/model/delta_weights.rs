use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use log::{debug, info, trace};

/// Sparse weight updates inspired by LoRA (Low-Rank Adaptation)
/// Only stores weights that have been updated, not the entire weight matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaWeights {
    /// Namespace for these weights (matches the model namespace)
    pub namespace: String,
    
    /// Sparse storage - only changed weights
    /// Maps feature name to delta value (not absolute value)
    pub deltas: HashMap<String, f64>,
    
    /// Track update counts per feature
    pub update_counts: HashMap<String, u32>,
    
    /// Track last update time per feature
    pub last_updated: HashMap<String, DateTime<Utc>>,
    
    /// Global statistics
    pub total_updates: u64,
    pub created_at: DateTime<Utc>,
    pub last_consolidation: Option<DateTime<Utc>>,
}

impl DeltaWeights {
    /// Create a new empty DeltaWeights instance
    pub fn new(namespace: String) -> Self {
        DeltaWeights {
            namespace,
            deltas: HashMap::new(),
            update_counts: HashMap::new(),
            last_updated: HashMap::new(),
            total_updates: 0,
            created_at: Utc::now(),
            last_consolidation: None,
        }
    }
    
    /// Add or update a delta weight
    pub fn update_weight(&mut self, feature: &str, delta: f64) {
        trace!("DeltaWeights::update_weight - feature: {}, delta: {}", feature, delta);
        
        // Update or insert the delta
        self.deltas.entry(feature.to_string())
            .and_modify(|d| *d += delta)
            .or_insert(delta);
        
        // Update count
        self.update_counts.entry(feature.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        
        // Update timestamp
        self.last_updated.insert(feature.to_string(), Utc::now());
        
        // Update global counter
        self.total_updates += 1;
    }
    
    /// Get the delta for a specific feature (returns 0.0 if not present)
    pub fn get_delta(&self, feature: &str) -> f64 {
        self.deltas.get(feature).copied().unwrap_or(0.0)
    }
    
    /// Get all features that have deltas
    pub fn get_features(&self) -> Vec<&String> {
        self.deltas.keys().collect()
    }
    
    /// Get the number of features with deltas
    pub fn size(&self) -> usize {
        self.deltas.len()
    }
    
    /// Get the age of a specific delta
    pub fn get_delta_age(&self, feature: &str) -> Option<Duration> {
        self.last_updated.get(feature)
            .map(|timestamp| Utc::now().signed_duration_since(*timestamp))
    }
    
    /// Get features older than a specified duration
    pub fn get_old_features(&self, max_age: Duration) -> Vec<String> {
        let cutoff = Utc::now() - max_age;
        self.last_updated.iter()
            .filter(|(_, timestamp)| **timestamp < cutoff)
            .map(|(feature, _)| feature.clone())
            .collect()
    }
    
    /// Get features that have been updated more than a threshold
    pub fn get_hot_features(&self, update_threshold: u32) -> Vec<String> {
        self.update_counts.iter()
            .filter(|(_, count)| **count > update_threshold)
            .map(|(feature, _)| feature.clone())
            .collect()
    }
    
    /// Clear specific features (used after consolidation)
    pub fn clear_features(&mut self, features: &[String]) {
        for feature in features {
            self.deltas.remove(feature);
            self.update_counts.remove(feature);
            self.last_updated.remove(feature);
        }
    }
    
    /// Clear all deltas (used after full consolidation)
    pub fn clear_all(&mut self) {
        debug!("DeltaWeights::clear_all - Clearing {} deltas", self.deltas.len());
        self.deltas.clear();
        self.update_counts.clear();
        self.last_updated.clear();
        self.last_consolidation = Some(Utc::now());
    }
    
    /// Get statistics about the delta weights
    pub fn get_stats(&self) -> DeltaWeightStats {
        let avg_update_count = if self.update_counts.is_empty() {
            0.0
        } else {
            self.update_counts.values().sum::<u32>() as f64 / self.update_counts.len() as f64
        };
        
        let max_update_count = self.update_counts.values().max().copied().unwrap_or(0);
        
        let oldest_update = self.last_updated.values()
            .min()
            .map(|t| Utc::now().signed_duration_since(*t));
        
        DeltaWeightStats {
            num_features: self.deltas.len(),
            total_updates: self.total_updates,
            avg_update_count,
            max_update_count,
            oldest_update_age: oldest_update,
            time_since_consolidation: self.last_consolidation
                .map(|t| Utc::now().signed_duration_since(t)),
        }
    }
    
    /// Save delta weights to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        info!("Saved delta weights to: {}", path);
        Ok(())
    }
    
    /// Load delta weights from a file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let json = std::fs::read_to_string(path)?;
        let delta_weights = serde_json::from_str(&json)?;
        info!("Loaded delta weights from: {}", path);
        Ok(delta_weights)
    }
}

#[derive(Debug, Clone)]
pub struct DeltaWeightStats {
    pub num_features: usize,
    pub total_updates: u64,
    pub avg_update_count: f64,
    pub max_update_count: u32,
    pub oldest_update_age: Option<Duration>,
    pub time_since_consolidation: Option<Duration>,
}