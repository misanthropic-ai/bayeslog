use super::objects::ImplicationFactor;
use super::torch_weights::TorchWeights;
use super::device::{TorchConfig, OptimizerType};
use crate::qbbn::common::interface::{PredictStatistics, TrainStatistics};
use crate::qbbn::common::model::{FactorContext, FactorModel};
use crate::qbbn::common::redis::MockConnection as Connection;
use crate::qbbn::model::exponential::features_from_factor;
use crate::qbbn::model::weights::CLASS_LABELS;
use log::{info, trace};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

pub struct TorchExponentialModel {
    config: TorchConfig,
    weights: TorchWeights,
    var_store: Arc<Mutex<Option<nn::VarStore>>>,
    optimizer: Arc<Mutex<Option<nn::Optimizer>>>,
    training_step: Arc<Mutex<i64>>,
}

impl TorchExponentialModel {
    pub fn new_mutable(namespace: String) -> Result<Box<dyn FactorModel>, Box<dyn Error>> {
        let config = TorchConfig::from_env();
        info!("Initializing TorchExponentialModel with config: {:?}", config);
        
        let weights = TorchWeights::new(namespace.clone(), config.device)?;
        
        Ok(Box::new(TorchExponentialModel {
            config,
            weights,
            var_store: Arc::new(Mutex::new(None)),
            optimizer: Arc::new(Mutex::new(None)),
            training_step: Arc::new(Mutex::new(0)),
        }))
    }

    pub fn new_shared(namespace: String) -> Result<Arc<dyn FactorModel>, Box<dyn Error>> {
        let config = TorchConfig::from_env();
        info!("Initializing shared TorchExponentialModel with config: {:?}", config);
        
        let weights = TorchWeights::new(namespace.clone(), config.device)?;
        
        Ok(Arc::new(TorchExponentialModel {
            config,
            weights,
            var_store: Arc::new(Mutex::new(None)),
            optimizer: Arc::new(Mutex::new(None)),
            training_step: Arc::new(Mutex::new(0)),
        }))
    }

    /// Initialize optimizer when we have weights
    fn ensure_optimizer(&self) -> Result<(), Box<dyn Error>> {
        let mut var_store_guard = self.var_store.lock().unwrap();
        let mut opt_guard = self.optimizer.lock().unwrap();
        
        if opt_guard.is_none() {
            // Create a new VarStore
            let vars = nn::VarStore::new(self.config.device);
            
            // Get current weights or initialize new ones
            let weight_size = if let Some(weight_tensor) = self.weights.get_weight_tensor() {
                let size = weight_tensor.size()[0];
                vars.root().var_copy("weights", &weight_tensor);
                size
            } else {
                // Start with space for 100 features
                vars.root().randn("weights", &[100], 0.0, 0.1);
                100
            };
            
            trace!("Initialized VarStore with {} weights", weight_size);
            
            // Create optimizer
            let optimizer = match self.config.optimizer {
                OptimizerType::SGD { momentum } => {
                    nn::Sgd {
                        momentum,
                        dampening: 0.0,
                        wd: 0.0,
                        nesterov: false,
                    }.build(&vars, self.config.learning_rate)?
                }
                OptimizerType::Adam { beta1, beta2, epsilon } => {
                    nn::Adam {
                        beta1,
                        beta2,
                        eps: epsilon,
                        wd: 0.0,
                        amsgrad: false,
                    }.build(&vars, self.config.learning_rate)?
                }
                OptimizerType::AdamW { beta1, beta2, epsilon, weight_decay } => {
                    nn::AdamW {
                        beta1,
                        beta2,
                        eps: epsilon,
                        wd: weight_decay,
                        amsgrad: false,
                    }.build(&vars, self.config.learning_rate)?
                }
            };
            
            *var_store_guard = Some(vars);
            *opt_guard = Some(optimizer);
        }
        Ok(())
    }

    /// Compute potential using tensor operations
    fn compute_potential_tensor(&self, features: &HashMap<String, f64>) -> Result<f64, Box<dyn Error>> {
        let feature_tensor = self.weights.features_to_tensor(features)?;
        let weight_tensor = self.weights.get_weight_tensor()
            .ok_or("Weights not initialized")?;
        
        // Ensure dimensions match
        let feature_size = feature_tensor.size()[0];
        let weight_size = weight_tensor.size()[0];
        
        let dot_product = if feature_size <= weight_size {
            // Use only the relevant portion of weights
            let relevant_weights = weight_tensor.narrow(0, 0, feature_size);
            feature_tensor.dot(&relevant_weights)
        } else {
            // Pad weights with zeros if needed
            let padded_weights = Tensor::zeros(&[feature_size], (Kind::Float, self.config.device));
            padded_weights.narrow(0, 0, weight_size).copy_(&weight_tensor);
            feature_tensor.dot(&padded_weights)
        };
        
        let potential = dot_product.exp().double_value(&[]);
        Ok(potential)
    }

    /// Batch compute potentials for efficiency
    fn compute_potentials_batch(&self, feature_list: &[HashMap<String, f64>]) -> Result<Vec<f64>, Box<dyn Error>> {
        if feature_list.is_empty() {
            return Ok(vec![]);
        }

        let weight_tensor = self.weights.get_weight_tensor()
            .ok_or("Weights not initialized")?;
        
        // Convert all features to tensors
        let feature_tensors: Result<Vec<_>, _> = feature_list.iter()
            .map(|f| self.weights.features_to_tensor(f))
            .collect();
        let feature_tensors = feature_tensors?;
        
        // Stack into a batch
        let batch_features = Tensor::stack(&feature_tensors, 0);
        
        // Ensure dimensions match
        let feature_size = batch_features.size()[1];
        let weight_size = weight_tensor.size()[0];
        
        let dot_products = if feature_size <= weight_size {
            let relevant_weights = weight_tensor.narrow(0, 0, feature_size);
            batch_features.matmul(&relevant_weights.unsqueeze(1)).squeeze_dim(1)
        } else {
            let padded_weights = Tensor::zeros(&[feature_size], (Kind::Float, self.config.device));
            padded_weights.narrow(0, 0, weight_size).copy_(&weight_tensor);
            batch_features.matmul(&padded_weights.unsqueeze(1)).squeeze_dim(1)
        };
        
        let potentials = dot_products.exp();
        
        // Convert back to Vec<f64>
        let potentials_vec: Vec<f64> = (0..potentials.size()[0])
            .map(|i| potentials.double_value(&[i]))
            .collect();
        
        Ok(potentials_vec)
    }
}

impl FactorModel for TorchExponentialModel {
    fn initialize_connection(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        self.weights.initialize_weights(connection, implication)?;
        Ok(())
    }

    fn train(
        &mut self,
        _connection: &mut Connection,
        factor: &FactorContext,
        gold_probability: f64,
    ) -> Result<TrainStatistics, Box<dyn Error>> {
        trace!("TorchExponentialModel::train - Start");
        
        // Get features
        let features = features_from_factor(factor)?;
        
        // Ensure optimizer is initialized
        self.ensure_optimizer()?;
        
        let loss_value = {
            let var_store_guard = self.var_store.lock().unwrap();
            let mut opt_guard = self.optimizer.lock().unwrap();
            
            if let (Some(vars), Some(optimizer)) = (var_store_guard.as_ref(), opt_guard.as_mut()) {
                // Get the weight tensor from VarStore
                let vars_variables = vars.variables();
                let weights = vars_variables.get("weights")
                    .ok_or("Weights not found in VarStore")?;
                
                // Convert features to tensors and compute potentials using the VarStore weights
                let mut potentials = Vec::new();
                for &class_label in &CLASS_LABELS {
                    let feature_tensor = self.weights.features_to_tensor(&features[class_label])?;
                    
                    // Ensure dimensions match
                    let feature_size = feature_tensor.size()[0];
                    let weight_size = weights.size()[0];
                    
                    let dot_product = if feature_size <= weight_size {
                        feature_tensor.dot(&weights.narrow(0, 0, feature_size))
                    } else {
                        // Pad weights if needed
                        let padded_weights = Tensor::zeros(&[feature_size], (Kind::Float, self.config.device));
                        padded_weights.narrow(0, 0, weight_size).copy_(weights);
                        feature_tensor.dot(&padded_weights)
                    };
                    
                    potentials.push(dot_product.exp());
                }
                
                // Stack potentials and compute softmax
                let potentials_tensor = Tensor::stack(&potentials, 0);
                let probs = potentials_tensor.softmax(0, Kind::Float);
                
                // Extract probability for class 1
                let predicted_prob = probs.double_value(&[1]);
                
                // Compute negative log likelihood loss
                let loss_tensor = if gold_probability > 0.5 {
                    -(probs.get(1).log())
                } else {
                    -(probs.get(0).log())
                };
                
                // Backward pass
                optimizer.zero_grad();
                loss_tensor.backward();
                optimizer.step();
                
                // Get loss value
                let loss = loss_tensor.double_value(&[]);
                
                // Update our weight storage with the new weights
                self.weights.update_from_tensor(weights)?;
                
                loss
            } else {
                return Err("Optimizer not initialized".into());
            }
        };
        
        // Increment training step
        let mut step_guard = self.training_step.lock().unwrap();
        *step_guard += 1;
        
        if *step_guard % 100 == 0 {
            info!("Training step {}: loss = {:.6}", *step_guard, loss_value);
        }
        
        trace!("TorchExponentialModel::train - End");
        Ok(TrainStatistics { loss: loss_value })
    }

    fn predict(
        &self,
        _connection: &mut Connection,
        factor: &FactorContext,
    ) -> Result<PredictStatistics, Box<dyn Error>> {
        trace!("TorchExponentialModel::predict - Start");
        
        let features = features_from_factor(factor)?;
        
        // Use batch computation even for single prediction for consistency
        let feature_list: Vec<_> = CLASS_LABELS.iter()
            .map(|&cl| features[cl].clone())
            .collect();
        
        let potentials = self.compute_potentials_batch(&feature_list)?;
        
        let normalization = potentials[0] + potentials[1];
        let probability = potentials[1] / normalization;
        
        trace!("TorchExponentialModel::predict - probability: {}", probability);
        Ok(PredictStatistics { probability })
    }
}