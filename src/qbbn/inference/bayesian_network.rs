use std::error::Error;

use crate::qbbn::common::{
    // interface::BeliefTable, 
    model::InferenceModel,
    proposition_db::EmptyBeliefTable,
    redis::MockConnection,
};

use crate::qbbn::model::objects::Proposition;
use super::{
    graph::PropositionGraph, 
    engine::{Inferencer, MarginalTable},
};

/// BayesianNetwork provides a simplified interface to the QBBN implementation
/// This serves as a user-friendly wrapper around the more complex Inferencer
pub struct BayesianNetwork {
    /// The underlying Inferencer implementation
    pub inferencer: Box<Inferencer>,
    /// The connection to the storage backend
    pub connection: MockConnection,
}

impl BayesianNetwork {
    /// Creates a new BayesianNetwork for the given scenario
    pub fn new(scenario_name: &str) -> Result<Self, Box<dyn Error>> {
        // Set up the model
        let model = InferenceModel::new_shared(scenario_name.to_string())?;
        
        // Create the storage connection
        let mut connection = MockConnection::new_in_memory()?;
        
        // Set up the belief table
        let fact_memory = EmptyBeliefTable::new_shared(scenario_name)?;
        
        // Get the target proposition
        let target = model.graph.get_target(&mut connection)?;
        
        // Create the proposition graph
        let proposition_graph = PropositionGraph::new_shared(
            &mut connection, 
            &model.graph, 
            target
        )?;
        
        // Create the inferencer
        let mut inferencer = Inferencer::new_mutable(
            model.clone(), 
            proposition_graph.clone(), 
            fact_memory
        )?;
        
        // Initialize the inference chart
        inferencer.initialize_chart(&mut connection)?;
        
        Ok(BayesianNetwork {
            inferencer,
            connection,
        })
    }
    
    /// Runs inference and returns the marginal probability table
    pub fn run_inference(&mut self) -> Result<MarginalTable, Box<dyn Error>> {
        self.inferencer.do_full_forward_and_backward(&mut self.connection)?;
        self.inferencer.log_table_to_file()
    }
    
    /// Updates the belief for a given proposition
    pub fn update_belief(&mut self, proposition: &Proposition, belief: f64) -> Result<(), Box<dyn Error>> {
        // Store the proposition belief
        self.inferencer.fact_memory.store_proposition_probability(
            &mut self.connection,
            proposition,
            belief
        )?;
        
        // Run inference
        self.run_inference()?;
        
        Ok(())
    }
    
    /// Gets the belief for a given proposition
    pub fn get_belief(&mut self, proposition: &Proposition) -> Result<Option<f64>, Box<dyn Error>> {
        self.inferencer.fact_memory.get_proposition_probability(
            &mut self.connection,
            proposition
        )
    }
}