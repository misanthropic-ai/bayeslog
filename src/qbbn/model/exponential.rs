use super::objects::ImplicationFactor;
use super::weights::{negative_feature, positive_feature, ExponentialWeights, CLASS_LABELS};
use super::weight_manager::{WeightManager, WeightManagerConfig};
use super::ModelWeights;
use crate::qbbn::common::interface::{PredictStatistics, TrainStatistics};
use crate::qbbn::common::model::{FactorContext, FactorModel};
use crate::qbbn::common::redis::MockConnection as Connection;
use log::{debug, info, trace};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
pub struct ExponentialModel {
    print_training_loss: bool,
    weights: Arc<RwLock<WeightManager>>,
    /// Whether to enable online learning with delta weights
    online_learning: bool,
}

impl ExponentialModel {
    pub fn new_mutable(namespace: String) -> Result<Box<dyn FactorModel>, Box<dyn Error>> {
        let base_weights = ExponentialWeights::new(namespace.clone())?;
        let weights = Arc::new(RwLock::new(WeightManager::new(base_weights, namespace.clone())));
        Ok(Box::new(ExponentialModel {
            print_training_loss: false,
            weights,
            online_learning: true, // Enable by default
        }))
    }
    
    pub fn new_shared(namespace: String) -> Result<Arc<dyn FactorModel>, Box<dyn Error>> {
        let base_weights = ExponentialWeights::new(namespace.clone())?;
        let weights = Arc::new(RwLock::new(WeightManager::new(base_weights, namespace.clone())));
        Ok(Arc::new(ExponentialModel {
            print_training_loss: false,
            weights,
            online_learning: true,
        }))
    }
    
    /// Create with custom configuration
    pub fn new_with_config(
        namespace: String,
        config: WeightManagerConfig,
        online_learning: bool,
    ) -> Result<Box<dyn FactorModel>, Box<dyn Error>> {
        let base_weights = ExponentialWeights::new(namespace.clone())?;
        let weights = Arc::new(RwLock::new(WeightManager::with_config(base_weights, namespace.clone(), config)));
        Ok(Box::new(ExponentialModel {
            print_training_loss: false,
            weights,
            online_learning,
        }))
    }
    
    /// Create a new ExponentialModel from a saved file
    pub fn from_file(namespace: String, path: &str) -> Result<Box<dyn FactorModel>, Box<dyn Error>> {
        info!("Loading ExponentialModel from file: {}", path);
        let model_weights = ModelWeights::load_from_file(path)?;
        
        // Verify namespace matches
        if model_weights.namespace != namespace {
            return Err(format!(
                "Namespace mismatch: file contains '{}', expected '{}'",
                model_weights.namespace, namespace
            ).into());
        }
        
        let mut base_weights = ExponentialWeights::new(namespace.clone())?;
        // Create a temporary connection for loading weights
        let mut temp_conn = Connection::new_in_memory()?;
        base_weights.load_from_model_weights(&mut temp_conn, &model_weights)?;
        
        let weights = Arc::new(RwLock::new(WeightManager::new(base_weights, namespace)));
        
        Ok(Box::new(ExponentialModel {
            print_training_loss: false,
            weights,
            online_learning: true,
        }))
    }
    
    /// Save the model weights to a file
    pub fn save_to_file(&self, connection: &mut Connection, path: &str) -> Result<(), Box<dyn Error>> {
        info!("Saving ExponentialModel to file: {}", path);
        
        // Optionally consolidate before saving
        if self.online_learning {
            let stats = self.weights.read().unwrap().get_delta_stats();
            if stats.num_features > 0 {
                info!("Consolidating {} delta weights before save", stats.num_features);
                self.weights.write().unwrap().consolidate(super::weight_manager::ConsolidationTrigger::Manual)?;
            }
        }
        
        // Export all weights (base + delta)
        let model_weights = self.weights.read().unwrap().export_weights(connection)?;
        model_weights.save_to_file(path)?;
        
        Ok(())
    }
}

fn dot_product(dict1: &HashMap<String, f64>, dict2: &HashMap<String, f64>) -> f64 {
    let mut result = 0.0;
    for (key, &v1) in dict1 {
        if let Some(&v2) = dict2.get(key) {
            let product = v1 * v2;
            trace!(
                "dot_product: key {}, v1 {}, v2 {}, product {}",
                key,
                v1,
                v2,
                product
            );
            result += product;
        }
        // In case of null (None), we skip the key as per the original JavaScript logic.
    }
    result
}

pub fn compute_potential(weights: &HashMap<String, f64>, features: &HashMap<String, f64>) -> f64 {
    let dot = dot_product(weights, features);
    dot.exp()
}

pub fn features_from_factor(
    factor: &FactorContext,
) -> Result<Vec<HashMap<String, f64>>, Box<dyn Error>> {
    let mut vec_result = vec![];
    for class_label in CLASS_LABELS {
        let mut result = HashMap::new();
        for (i, premise) in factor.factor.iter().enumerate() {
            debug!("Processing backimplication {}", i);
            let feature = premise.inference.unique_key();
            debug!("Generated unique key for feature: {}", feature);
            let probability = factor.probabilities[i];
            debug!(
                "Conjunction probability for backimplication {}: {}",
                i, probability
            );
            let posf = positive_feature(&feature, class_label);
            let negf = negative_feature(&feature, class_label);
            result.insert(posf.clone(), probability);
            result.insert(negf.clone(), 1.0 - probability);
            debug!(
                "Inserted features for backimplication {}: positive - {}, negative - {}",
                i, posf, negf
            );
        }
        vec_result.push(result);
    }
    trace!("features_from_backimplications completed successfully");
    Ok(vec_result)
}

pub fn compute_expected_features(
    probability: f64,
    features: &HashMap<String, f64>,
) -> HashMap<String, f64> {
    let mut result = HashMap::new();
    for (key, &value) in features {
        result.insert(key.clone(), value * probability);
    }
    result
}

const LEARNING_RATE: f64 = 0.05;

pub fn do_sgd_update(
    weights: &HashMap<String, f64>,
    gold_features: &HashMap<String, f64>,
    expected_features: &HashMap<String, f64>,
    print_training_loss: bool,
) -> HashMap<String, f64> {
    let mut new_weights = HashMap::new();
    for (feature, &wv) in weights {
        let gv = gold_features.get(feature).unwrap_or(&0.0);
        let ev = expected_features.get(feature).unwrap_or(&0.0);
        let new_weight = wv + LEARNING_RATE * (gv - ev);
        let loss = (gv - ev).abs();
        if print_training_loss {
            trace!(
                "feature: {}, gv: {}, ev: {}, loss: {}, old_weight: {}, new_weight: {}",
                feature,
                gv,
                ev,
                loss,
                wv,
                new_weight
            );
        }
        new_weights.insert(feature.clone(), new_weight);
    }
    new_weights
}

impl FactorModel for ExponentialModel {
    fn initialize_connection(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        self.weights.write().unwrap().initialize_weights(connection, implication)?;
        Ok(())
    }

    fn train(
        &mut self,
        connection: &mut Connection,
        factor: &FactorContext,
        gold_probability: f64,
    ) -> Result<TrainStatistics, Box<dyn Error>> {
        trace!("train_on_example - Getting features from backimplications");
        let features = match features_from_factor(factor) {
            Ok(f) => f,
            Err(e) => {
                trace!(
                    "train_on_example - Error in features_from_backimplications: {:?}",
                    e
                );
                return Err(e);
            }
        };
        let mut weight_vectors = vec![];
        let mut potentials = vec![];
        for class_label in CLASS_LABELS {
            for (feature, weight) in &features[class_label] {
                trace!("feature {:?} {}", feature, weight);
            }
            trace!(
                "train_on_example - Reading weights for class {}",
                class_label
            );
            let weight_vector = match self.weights.read().unwrap().read_weight_vector(
                connection,
                &features[class_label].keys().cloned().collect::<Vec<_>>(),
            ) {
                Ok(w) => w,
                Err(e) => {
                    trace!("train_on_example - Error in read_weights: {:?}", e);
                    return Err(e);
                }
            };
            trace!("train_on_example - Computing probability");
            let potential = compute_potential(&weight_vector, &features[class_label]);
            trace!("train_on_example - Computed probability: {}", potential);
            potentials.push(potential);
            weight_vectors.push(weight_vector);
        }
        let normalization = potentials[0] + potentials[1];
        for class_label in CLASS_LABELS {
            let probability = potentials[class_label] / normalization;
            trace!("train_on_example - Computing expected features");
            let this_true_prob = if class_label == 0 {
                1f64 - gold_probability
            } else {
                gold_probability
            };
            let gold = compute_expected_features(this_true_prob, &features[class_label]);
            let expected = compute_expected_features(probability, &features[class_label]);
            trace!("train_on_example - Performing SGD update");
            let new_weights = do_sgd_update(
                &weight_vectors[class_label],
                &gold,
                &expected,
                self.print_training_loss,
            );
            
            if self.online_learning {
                // For online learning, compute deltas and update
                trace!("train_on_example - Updating delta weights");
                let mut weight_deltas = HashMap::new();
                for (feature, new_weight) in &new_weights {
                    if let Some(old_weight) = weight_vectors[class_label].get(feature) {
                        let delta = new_weight - old_weight;
                        if delta.abs() > 1e-8 { // Only update if meaningful change
                            weight_deltas.insert(feature.clone(), delta);
                        }
                    }
                }
                self.weights.write().unwrap().update_weights(&weight_deltas);
            } else {
                // For batch training, save directly (this won't work with WeightManager yet)
                // TODO: Add batch training mode to WeightManager
                return Err("Batch training mode not yet implemented with WeightManager".into());
            }
        }
        trace!("train_on_example - End");
        Ok(TrainStatistics { loss: 1f64 })
    }
    fn predict(
        &self,
        connection: &mut Connection,
        factor: &FactorContext,
    ) -> Result<PredictStatistics, Box<dyn Error>> {
        let features = match features_from_factor(factor) {
            Ok(f) => f,
            Err(e) => {
                trace!(
                    "inference_probability - Error in features_from_backimplications: {:?}",
                    e
                );
                return Err(e);
            }
        };
        let mut potentials = vec![];
        for class_label in CLASS_LABELS {
            let this_features = &features[class_label];
            for (feature, weight) in this_features.iter() {
                trace!("feature {:?} {}", &feature, weight);
            }
            trace!("inference_probability - Reading weights");
            let weight_vector = match self.weights.read().unwrap().read_weight_vector(
                connection,
                &this_features.keys().cloned().collect::<Vec<_>>(),
            ) {
                Ok(w) => w,
                Err(e) => {
                    trace!("inference_probability - Error in read_weights: {:?}", e);
                    return Err(e);
                }
            };
            for (feature, weight) in weight_vector.iter() {
                trace!("weight {:?} {}", &feature, weight);
            }
            let potential = compute_potential(&weight_vector, this_features);
            trace!("potential for {} {} {:?}", class_label, potential, &factor);
            potentials.push(potential);
        }
        let normalization = potentials[0] + potentials[1];
        let probability = potentials[1] / normalization;
        trace!(
            "dot_product: normalization {}, marginal {}",
            normalization,
            probability
        );
        Ok(PredictStatistics { probability })
    }
    
    fn save_to_file(&self, connection: &mut Connection, path: &str) -> Result<(), Box<dyn Error>> {
        self.save_to_file(connection, path)
    }
    
    fn model_type(&self) -> &str {
        "exponential"
    }
}
