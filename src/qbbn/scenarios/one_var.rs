use crate::qbbn::common::graph::InferenceGraph;
use crate::qbbn::common::interface::ScenarioMaker;
use crate::qbbn::common::proposition_db::RedisBeliefTable;
use crate::qbbn::common::resources::ResourceContext;
use crate::qbbn::common::train::TrainingPlan;
use crate::qbbn::model::creators::{
    constant, proposition, relation, sub, variable_argument,
};
use crate::qbbn::model::objects::{Domain, Entity};
use rand::Rng; // Import Rng trait
use std::error::Error;

#[allow(dead_code)]
fn cointoss() -> f64 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    if rng.r#gen::<f64>() < 0.5 {
        1.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn weighted_cointoss(threshold: f64) -> f64 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    if rng.r#gen::<f64>() < threshold {
        1.0
    } else {
        0.0
    }
}

pub struct OneVariable {}

impl ScenarioMaker for OneVariable {
    fn setup_scenario(&self, resources: &ResourceContext) -> Result<(), Box<dyn Error>> {
        let mut connection = resources.connection.lock().unwrap();
        let namespace = "one_var".to_string();
        let mut graph = InferenceGraph::new_mutable(namespace.clone())?;
        let proposition_db = RedisBeliefTable::new_mutable(namespace.clone())?;
        let mut plan = TrainingPlan::new(namespace.clone())?;
        let total_members_each_class = 16; // Using a smaller number for testing
        
        let man_domain = Domain::MAN.to_string();
        graph.register_domain(&mut connection, &man_domain)?;
        
        let exciting_relation = relation(
            "exciting".to_string(),
            vec![variable_argument(man_domain.clone())],
        );
        graph.register_relation(&mut connection, &exciting_relation)?;
        
        for i in 0..total_members_each_class {
            let is_test = i % 4 == 3;  // Every 4th item is test data
            let is_training = !is_test;
            let domain = Domain::MAN.to_string();
            let prefix = if is_test { "test" } else { "train" };
            let name = format!("{}_{}{}", &prefix, domain, i);
            
            let entity = Entity {
                domain: domain.clone(),
                name: name.clone(),
            };
            graph.store_entity(&mut connection, &entity)?;
            
            // Generate random probability (biased toward 0.3)
            let probability = weighted_cointoss(0.3f64);
            
            // Create proposition: exciting(entity)
            let entity_constant = constant(entity.domain, entity.name.clone());
            let entity_exciting = proposition(exciting_relation.clone(), vec![sub(entity_constant)]);
            
            // Ensure proposition existence
            graph.ensure_existence_backlinks_for_proposition(&mut connection, &entity_exciting)?;
            
            // Store probability
            proposition_db.store_proposition_probability(&mut connection, &entity_exciting, probability)?;
            
            // Add to training or test queue
            plan.maybe_add_to_training(&mut connection, is_training, &entity_exciting)?;
            plan.maybe_add_to_test(&mut connection, is_test, &entity_exciting)?;
        }
        
        Ok(())
    }
}
