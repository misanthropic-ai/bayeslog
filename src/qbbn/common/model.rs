use crate::qbbn::{
    inference::graph::PropositionFactor,
    model::{
        exponential::ExponentialModel,
        objects::ImplicationFactor,
    },
};
use crate::qbbn::common::redis::MockConnection as Connection;
use std::{error::Error, sync::Arc};

use super::{
    graph::InferenceGraph,
    interface::{PredictStatistics, TrainStatistics},
};

pub struct InferenceModel {
    pub graph: Arc<InferenceGraph>,
    pub model: Arc<dyn FactorModel>,
}

impl InferenceModel {
    pub fn new_shared(namespace: String) -> Result<Arc<Self>, Box<dyn Error>> {
        let graph = InferenceGraph::new_shared(namespace.clone())?;
        let model = ExponentialModel::new_shared(namespace.clone())?;
        Ok(Arc::new(InferenceModel { graph, model }))
    }
}

#[derive(Debug)]
pub struct FactorContext {
    pub factor: Vec<PropositionFactor>,
    pub probabilities: Vec<f64>,
}

/// FactorModel defines a trait for models that can predict probabilities for factors
/// Implements Send + Sync to allow thread-safe sharing of models using Arc
pub trait FactorModel: Send + Sync {
    fn initialize_connection(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>>;

    fn train(
        &mut self,
        connection: &mut Connection,
        factor: &FactorContext,
        probability: f64,
    ) -> Result<TrainStatistics, Box<dyn Error>>;

    fn predict(
        &self,
        connection: &mut Connection,
        factor: &FactorContext,
    ) -> Result<PredictStatistics, Box<dyn Error>>;
    
    /// Export model weights to a file (optional implementation)
    fn save_to_file(&self, _connection: &mut Connection, _path: &str) -> Result<(), Box<dyn Error>> {
        Err("Model does not support saving to file".into())
    }
    
    /// Get the model type identifier
    fn model_type(&self) -> &str {
        "unknown"
    }
}
