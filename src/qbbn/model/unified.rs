use super::{ModelWeights, exponential::ExponentialModel, torch_exponential::TorchExponentialModel};
use crate::qbbn::common::interface::{PredictStatistics, TrainStatistics};
use crate::qbbn::common::model::{FactorContext, FactorModel};
use crate::qbbn::common::redis::MockConnection as Connection;
use crate::qbbn::model::objects::ImplicationFactor;
use log::info;
use std::error::Error;
// use std::sync::Arc;
use std::env;

/// Unified model that can switch between CPU (ExponentialModel) and GPU (TorchExponentialModel)
/// based on workload characteristics and configuration
pub struct UnifiedExponentialModel {
    /// The underlying model implementation
    model: Box<dyn FactorModel>,
    /// Namespace for the model
    namespace: String,
    /// Force a specific backend (if set)
    force_backend: Option<ModelBackend>,
    /// Threshold for switching to GPU (number of entities)
    gpu_threshold: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelBackend {
    Cpu,
    Gpu,
}

impl UnifiedExponentialModel {
    /// Create a new unified model with automatic backend selection
    pub fn new(namespace: String) -> Result<Self, Box<dyn Error>> {
        let force_backend = Self::get_forced_backend();
        let gpu_threshold = Self::get_gpu_threshold();
        
        // Start with CPU model by default
        let model = ExponentialModel::new_mutable(namespace.clone())?;
        
        Ok(UnifiedExponentialModel {
            model,
            namespace,
            force_backend,
            gpu_threshold,
        })
    }
    
    /// Create from a saved file, detecting the model type
    pub fn from_file(namespace: String, path: &str) -> Result<Self, Box<dyn Error>> {
        info!("Loading UnifiedExponentialModel from file: {}", path);
        
        // Load the weights to check the model type
        let model_weights = ModelWeights::load_from_file(path)?;
        
        // Create the appropriate model based on saved type
        let model: Box<dyn FactorModel> = match model_weights.model_type.as_str() {
            "exponential" => ExponentialModel::from_file(namespace.clone(), path)?,
            "torch_exponential" => TorchExponentialModel::from_file(namespace.clone(), path)?,
            _ => return Err(format!("Unknown model type: {}", model_weights.model_type).into()),
        };
        
        let force_backend = Self::get_forced_backend();
        let gpu_threshold = Self::get_gpu_threshold();
        
        Ok(UnifiedExponentialModel {
            model,
            namespace,
            force_backend,
            gpu_threshold,
        })
    }
    
    /// Save the current model to a file
    pub fn save_to_file(&self, connection: &mut Connection, path: &str) -> Result<(), Box<dyn Error>> {
        info!("Saving UnifiedExponentialModel to file: {}", path);
        
        // Delegate to the underlying model's save method
        self.model.save_to_file(connection, path)
    }
    
    /// Switch to a specific backend
    pub fn switch_backend(&mut self, backend: ModelBackend, _connection: &mut Connection) -> Result<(), Box<dyn Error>> {
        info!("Switching to {:?} backend", backend);
        
        // First, we need to extract the current weights
        // This is challenging without a common interface
        // For now, we'll need to add this functionality to the trait
        
        match backend {
            ModelBackend::Cpu => {
                self.model = ExponentialModel::new_mutable(self.namespace.clone())?;
            }
            ModelBackend::Gpu => {
                self.model = TorchExponentialModel::new_mutable(self.namespace.clone())?;
            }
        }
        
        Ok(())
    }
    
    /// Automatically select the best backend based on workload
    pub fn auto_select_backend(&mut self, num_entities: usize, connection: &mut Connection) -> Result<(), Box<dyn Error>> {
        if let Some(_forced) = self.force_backend {
            // Respect forced backend
            return Ok(());
        }
        
        let should_use_gpu = num_entities >= self.gpu_threshold;
        let current_is_gpu = self.is_using_gpu();
        
        if should_use_gpu && !current_is_gpu {
            info!("Auto-switching to GPU backend (entities: {})", num_entities);
            self.switch_backend(ModelBackend::Gpu, connection)?;
        } else if !should_use_gpu && current_is_gpu {
            info!("Auto-switching to CPU backend (entities: {})", num_entities);
            self.switch_backend(ModelBackend::Cpu, connection)?;
        }
        
        Ok(())
    }
    
    /// Check if currently using GPU backend
    fn is_using_gpu(&self) -> bool {
        self.model.model_type() == "torch_exponential"
    }
    
    /// Get forced backend from environment
    fn get_forced_backend() -> Option<ModelBackend> {
        if env::var("BAYESLOG_FORCE_CPU").is_ok() {
            Some(ModelBackend::Cpu)
        } else if env::var("BAYESLOG_FORCE_GPU").is_ok() {
            Some(ModelBackend::Gpu)
        } else {
            None
        }
    }
    
    /// Get GPU threshold from environment (default: 10)
    fn get_gpu_threshold() -> usize {
        env::var("BAYESLOG_GPU_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
    }
}

impl FactorModel for UnifiedExponentialModel {
    fn initialize_connection(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        self.model.initialize_connection(connection, implication)
    }

    fn train(
        &mut self,
        connection: &mut Connection,
        factor: &FactorContext,
        gold_probability: f64,
    ) -> Result<TrainStatistics, Box<dyn Error>> {
        // Auto-select backend based on factor complexity
        // (This is a simplified heuristic)
        let num_entities = factor.factor.len();
        self.auto_select_backend(num_entities, connection)?;
        
        self.model.train(connection, factor, gold_probability)
    }

    fn predict(
        &self,
        connection: &mut Connection,
        factor: &FactorContext,
    ) -> Result<PredictStatistics, Box<dyn Error>> {
        self.model.predict(connection, factor)
    }
}