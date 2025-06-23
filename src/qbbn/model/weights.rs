use crate::qbbn::{
    common::redis::{map_get, map_insert},
    model::objects::ImplicationFactor,
    model::ModelWeights,
};
use log::trace;
use rand::Rng;
use crate::qbbn::common::redis::MockConnection as Connection;
use std::error::Error;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub const CLASS_LABELS: [usize; 2] = [0, 1];

fn random_weight() -> f64 {
    let mut rng = rand::thread_rng();
    (rng.r#gen::<f64>() - rng.r#gen::<f64>()) / 5.0
}

fn sign_char(value: usize) -> String {
    if value == 0 {
        '-'.to_string()
    } else {
        "+".to_string()
    }
}

pub fn positive_feature(feature: &str, class_label: usize) -> String {
    format!("+>{} {}", sign_char(class_label), feature)
}

pub fn negative_feature(feature: &str, class_label: usize) -> String {
    format!("->{} {}", sign_char(class_label), feature)
}

pub struct ExponentialWeights {
    namespace: String,
    /// Track all features that have been initialized or accessed
    /// Arc<Mutex<>> allows thread-safe updates during concurrent training
    known_features: Arc<Mutex<HashSet<String>>>,
}

impl ExponentialWeights {
    pub fn new(namespace: String) -> Result<ExponentialWeights, Box<dyn Error>> {
        Ok(ExponentialWeights { 
            namespace,
            known_features: Arc::new(Mutex::new(HashSet::new())),
        })
    }
}

impl ExponentialWeights {
    pub const WEIGHTS_KEY: &'static str = "weights";

    pub fn initialize_weights(
        &mut self,
        connection: &mut Connection,
        implication: &ImplicationFactor,
    ) -> Result<(), Box<dyn Error>> {
        trace!("initialize_weights - Start: {:?}", implication);
        let feature = implication.unique_key();
        trace!("initialize_weights - Unique key: {}", feature);
        for class_label in CLASS_LABELS {
            let posf = positive_feature(&feature, class_label);
            let negf = negative_feature(&feature, class_label);
            trace!(
                "initialize_weights - Positive feature: {}, Negative feature: {}",
                posf,
                negf
            );
            let weight1 = random_weight();
            let weight2 = random_weight();
            trace!(
                "initialize_weights - Generated weights: {}, {}",
                weight1,
                weight2
            );
            // Track the features
            {
                let mut features = self.known_features.lock().unwrap();
                features.insert(posf.clone());
                features.insert(negf.clone());
            }
            
            map_insert(
                connection,
                &self.namespace,
                Self::WEIGHTS_KEY,
                &posf,
                &weight1.to_string(),
            )?;
            map_insert(
                connection,
                &self.namespace,
                Self::WEIGHTS_KEY,
                &negf,
                &weight2.to_string(),
            )?;
        }
        trace!("initialize_weights - End");
        Ok(())
    }

    pub fn read_single_weight(
        &self,
        connection: &mut Connection,
        feature: &str,
    ) -> Result<f64, Box<dyn Error>> {
        trace!("read_weights - Start");
        trace!("read_weights - Reading weight for feature: {}", feature);
        let weight_record = map_get(connection, &self.namespace, Self::WEIGHTS_KEY, feature)?
        .unwrap_or("0.0".to_string());
            // .expect("should be there");
        let weight = weight_record.parse::<f64>().map_err(|e| {
            trace!("read_weights - Error parsing weight: {:?}", e);
            Box::new(e) as Box<dyn Error>
        })?;
        trace!("read_weights - End");
        Ok(weight)
    }

    pub fn read_weight_vector(
        &self,
        connection: &mut Connection,
        features: &[String],
    ) -> Result<HashMap<String, f64>, Box<dyn Error>> {
        trace!("read_weights - Start");
        let mut weights = HashMap::new();
        
        // Track all features we're reading
        {
            let mut known = self.known_features.lock().unwrap();
            for feature in features {
                known.insert(feature.clone());
            }
        }
        
        for feature in features {
            trace!("read_weights - Reading weight for feature: {}", feature);
            let weight_record = map_get(connection, &self.namespace, Self::WEIGHTS_KEY, feature)?
                .unwrap_or("0.0".to_string());
            let weight = weight_record.parse::<f64>().map_err(|e| {
                trace!("read_weights - Error parsing weight: {:?}", e);
                Box::new(e) as Box<dyn Error>
            })?;
            weights.insert(feature.clone(), weight);
        }
        trace!("read_weights - End");
        Ok(weights)
    }

    pub fn save_weight_vector(
        &mut self,
        connection: &mut Connection,
        weights: &HashMap<String, f64>,
    ) -> Result<(), Box<dyn Error>> {
        trace!("save_weights - Start");
        
        // Track all features we're saving
        {
            let mut known = self.known_features.lock().unwrap();
            for feature in weights.keys() {
                known.insert(feature.clone());
            }
        }
        
        for (feature, &value) in weights {
            trace!(
                "save_weights - Saving weight for feature {}: {}",
                feature,
                value
            );
            map_insert(
                connection,
                &self.namespace,
                Self::WEIGHTS_KEY,
                feature,
                &value.to_string(),
            )?;
        }
        trace!("save_weights - End");
        Ok(())
    }
    
    /// Convert weights to ModelWeights format
    pub fn to_model_weights(&self, connection: &mut Connection) -> Result<ModelWeights, Box<dyn Error>> {
        trace!("to_model_weights - Start");
        
        let known_features = self.known_features.lock().unwrap();
        let mut weights = HashMap::new();
        let mut feature_indices = HashMap::new();
        
        // Read all known features
        for (idx, feature) in known_features.iter().enumerate() {
            trace!("to_model_weights - Reading weight for feature: {}", feature);
            
            // Get the weight value
            if let Some(weight_str) = map_get(connection, &self.namespace, Self::WEIGHTS_KEY, feature)? {
                if let Ok(weight) = weight_str.parse::<f64>() {
                    weights.insert(feature.clone(), weight);
                    feature_indices.insert(feature.clone(), idx as i64);
                }
            }
        }
        
        let model_weights = ModelWeights::from_weight_map(weights, self.namespace.clone());
        
        trace!("to_model_weights - End");
        Ok(model_weights)
    }
    
    /// Load weights from ModelWeights format
    pub fn load_from_model_weights(&mut self, connection: &mut Connection, model_weights: &ModelWeights) -> Result<(), Box<dyn Error>> {
        trace!("load_from_model_weights - Start");
        
        // Clear and update known features
        {
            let mut known = self.known_features.lock().unwrap();
            known.clear();
            for feature in model_weights.weights.keys() {
                known.insert(feature.clone());
            }
        }
        
        // Store all weights
        for (feature, &weight) in &model_weights.weights {
            map_insert(
                connection,
                &self.namespace,
                Self::WEIGHTS_KEY,
                feature,
                &weight.to_string(),
            )?;
        }
        
        trace!("load_from_model_weights - End");
        Ok(())
    }
}
