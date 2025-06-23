use crate::qbbn::{
    common::{
        interface::BeliefTable,
        model::FactorContext,
        graph::InferenceGraph,
        proposition_db::RedisBeliefTable,
    },
    model::{
        choose::extract_backimplications_from_proposition,
        objects::{Proposition, PropositionGroup},
    },
};
use crate::qbbn::common::redis::MockConnection as Connection;
use log::info;
use std::error::Error;
use super::train::TrainingPlan;
use super::resources::ResourceContext;

/// Batch training support for GPU acceleration
pub struct BatchTrainer {
    batch_size: usize,
}

impl BatchTrainer {
    pub fn new(batch_size: usize) -> Self {
        BatchTrainer { batch_size }
    }

    /// Extract training data in batches
    pub fn prepare_training_batches(
        &self,
        connection: &mut Connection,
        proposition_db: &dyn BeliefTable,
        graph: &InferenceGraph,
        propositions: &[Proposition],
    ) -> Result<Vec<(Vec<FactorContext>, Vec<f64>)>, Box<dyn Error>> {
        let mut batches = Vec::new();
        let mut current_factors = Vec::new();
        let mut current_probabilities = Vec::new();

        for proposition in propositions {
            // Extract factor context
            let factor = extract_factor_for_training(
                connection,
                proposition_db,
                graph,
                proposition.clone(),
            )?;
            
            // Get gold probability
            let probability = proposition_db
                .get_proposition_probability(connection, proposition)?
                .expect("Probability should exist");

            current_factors.push(factor);
            current_probabilities.push(probability);

            // Create batch when size is reached
            if current_factors.len() >= self.batch_size {
                batches.push((
                    std::mem::take(&mut current_factors),
                    std::mem::take(&mut current_probabilities),
                ));
            }
        }

        // Add remaining items as final batch
        if !current_factors.is_empty() {
            batches.push((current_factors, current_probabilities));
        }

        Ok(batches)
    }
}

/// Extract factor context for a single proposition
fn extract_factor_for_training(
    connection: &mut Connection,
    proposition_db: &dyn BeliefTable,
    graph: &InferenceGraph,
    conclusion: Proposition,
) -> Result<FactorContext, Box<dyn Error>> {
    let factors = extract_backimplications_from_proposition(connection, graph, &conclusion)?;
    let mut probabilities = vec![];
    
    for factor in &factors {
        let probability = extract_group_probability_for_training(
            connection,
            proposition_db,
            &factor.premise,
        )?;
        probabilities.push(probability);
    }
    
    Ok(FactorContext {
        factor: factors,
        probabilities,
    })
}

/// Extract probability for a proposition group
fn extract_group_probability_for_training(
    connection: &mut Connection,
    proposition_db: &dyn BeliefTable,
    premise: &PropositionGroup,
) -> Result<f64, Box<dyn Error>> {
    let mut product = 1.0;
    for term in &premise.terms {
        let part = proposition_db
            .get_proposition_probability(connection, term)?
            .unwrap_or(0.0);
        product *= part;
    }
    Ok(product)
}

/// Batch training function that can utilize GPU acceleration
pub fn do_batch_training(
    resources: &ResourceContext,
    namespace: String,
    use_torch: bool,
) -> Result<(), Box<dyn Error>> {
    let mut connection = resources.connection.lock().unwrap();
    let graph = InferenceGraph::new_mutable(namespace.clone())?;
    let proposition_db = RedisBeliefTable::new_mutable(namespace.clone())?;
    let plan = TrainingPlan::new(namespace.clone())?;
    
    // Choose model based on configuration
    let mut factor_model: Box<dyn crate::qbbn::common::model::FactorModel> = if use_torch {
        info!("Using GPU-accelerated TorchExponentialModel");
        crate::qbbn::model::torch_exponential::TorchExponentialModel::new_mutable(namespace.clone())?
    } else {
        info!("Using standard ExponentialModel");
        crate::qbbn::model::exponential::ExponentialModel::new_mutable(namespace.clone())?
    };
    
    // Initialize weights for all implications
    let implications = graph.get_all_implications(&mut connection)?;
    for implication in implications {
        factor_model.initialize_connection(&mut connection, &implication)?;
    }
    
    // Get training data
    let training_questions = plan.get_training_questions(&mut connection)?;
    info!("Processing {} training examples", training_questions.len());
    
    if use_torch {
        // Use batch training for GPU
        let batch_trainer = BatchTrainer::new(32); // TODO: Make configurable
        let batches = batch_trainer.prepare_training_batches(
            &mut connection,
            proposition_db.as_ref(),
            &graph,
            &training_questions,
        )?;
        
        info!("Created {} batches for training", batches.len());
        
        for (batch_idx, (factors, probabilities)) in batches.iter().enumerate() {
            // For now, train examples individually even in torch mode
            // TODO: Implement true batch training in TorchExponentialModel
            for (factor, &probability) in factors.iter().zip(probabilities.iter()) {
                let _stats = factor_model.train(&mut connection, factor, probability)?;
            }
            
            if batch_idx % 10 == 0 {
                info!("Processed batch {}/{}", batch_idx + 1, batches.len());
            }
        }
    } else {
        // Standard training loop
        for (idx, proposition) in training_questions.iter().enumerate() {
            let factor = extract_factor_for_training(
                &mut connection,
                proposition_db.as_ref(),
                &graph,
                proposition.clone(),
            )?;
            
            let probability = proposition_db
                .get_proposition_probability(&mut connection, proposition)?
                .expect("Probability should exist");
            
            let _stats = factor_model.train(&mut connection, &factor, probability)?;
            
            if idx % 100 == 0 {
                info!("Processed {}/{} examples", idx + 1, training_questions.len());
            }
        }
    }
    
    info!("Training complete");
    Ok(())
}