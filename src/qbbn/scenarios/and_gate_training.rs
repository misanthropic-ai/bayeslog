use crate::qbbn::common::graph::InferenceGraph;
use crate::qbbn::common::proposition_db::RedisBeliefTable;
use crate::qbbn::common::resources::ResourceContext;
use crate::qbbn::common::train::TrainingPlan;
use crate::qbbn::model::creators::{predicate, relation, variable_argument};
use crate::qbbn::{
    common::interface::ScenarioMaker,
    model::{
        creators::{conjunction, constant, implication, proposition, sub, variable},
        objects::Entity,
    },
};
use std::error::Error;

/// Training scenario specifically designed to train AND gate behavior
pub struct AndGateTraining {}

impl ScenarioMaker for AndGateTraining {
    fn setup_scenario(&self, resources: &ResourceContext) -> Result<(), Box<dyn Error>> {
        let mut connection = resources.connection.lock().unwrap();
        let namespace = "and_gate_training".to_string();
        let mut graph = InferenceGraph::new_mutable(namespace.clone())?;
        let proposition_db = RedisBeliefTable::new_mutable(namespace.clone())?;
        let mut plan = TrainingPlan::new(namespace.clone())?;
        
        // Create a simple domain for testing
        let test_domain = "TestDomain".to_string();
        
        // Create test entities
        let _entities = vec![
            Entity { domain: test_domain.clone(), name: "A".to_string() },
            Entity { domain: test_domain.clone(), name: "B".to_string() },
            Entity { domain: test_domain.clone(), name: "C".to_string() },
            Entity { domain: test_domain.clone(), name: "D".to_string() },
        ];
        
        // Create test relations
        let input_relation = relation("input".to_string(), vec![variable_argument(test_domain.clone())]);
        let output_relation = relation("output".to_string(), vec![variable_argument(test_domain.clone())]);
        
        // Create constants for our entities
        let const_a = constant(test_domain.clone(), "A".to_string());
        let const_b = constant(test_domain.clone(), "B".to_string());
        let const_c = constant(test_domain.clone(), "C".to_string());
        let const_d = constant(test_domain.clone(), "D".to_string());
        
        // Create variables
        let _x = variable(test_domain.clone());
        
        // Training pattern 1: Two-input AND gate
        // If input(A) AND input(B), then output(A)
        let and_2_implication = implication(
            conjunction(vec![
                predicate(input_relation.clone(), vec![sub(const_a.clone())]),
                predicate(input_relation.clone(), vec![sub(const_b.clone())]),
            ]),
            predicate(output_relation.clone(), vec![sub(const_a.clone())]),
            vec![],
        );
        graph.store_predicate_implication(&mut connection, &and_2_implication)?;
        
        // Training pattern 2: Three-input AND gate
        // If input(A) AND input(B) AND input(C), then output(B)
        let and_3_implication = implication(
            conjunction(vec![
                predicate(input_relation.clone(), vec![sub(const_a.clone())]),
                predicate(input_relation.clone(), vec![sub(const_b.clone())]),
                predicate(input_relation.clone(), vec![sub(const_c.clone())]),
            ]),
            predicate(output_relation.clone(), vec![sub(const_b.clone())]),
            vec![],
        );
        graph.store_predicate_implication(&mut connection, &and_3_implication)?;
        
        // Training pattern 3: Four-input AND gate
        // If input(A) AND input(B) AND input(C) AND input(D), then output(C)
        let and_4_implication = implication(
            conjunction(vec![
                predicate(input_relation.clone(), vec![sub(const_a.clone())]),
                predicate(input_relation.clone(), vec![sub(const_b.clone())]),
                predicate(input_relation.clone(), vec![sub(const_c.clone())]),
                predicate(input_relation.clone(), vec![sub(const_d.clone())]),
            ]),
            predicate(output_relation.clone(), vec![sub(const_c.clone())]),
            vec![],
        );
        graph.store_predicate_implication(&mut connection, &and_4_implication)?;
        
        // Generate training examples with different input combinations
        let test_cases = vec![
            // For 2-input AND gate
            (vec![true, true], true, 0.9),      // All true -> high probability
            (vec![true, false], false, 0.1),    // One false -> low probability
            (vec![false, true], false, 0.1),    // One false -> low probability
            (vec![false, false], false, 0.05),  // All false -> very low probability
            
            // For 3-input AND gate
            (vec![true, true, true], true, 0.9),     // All true -> high probability
            (vec![true, true, false], false, 0.1),   // One false -> low probability
            (vec![true, false, true], false, 0.1),   // One false -> low probability
            (vec![false, true, true], false, 0.1),   // One false -> low probability
            (vec![false, false, false], false, 0.05), // All false -> very low probability
            
            // For 4-input AND gate
            (vec![true, true, true, true], true, 0.9),      // All true -> high probability
            (vec![true, true, true, false], false, 0.1),    // One false -> low probability
            (vec![true, false, true, true], false, 0.1),    // One false -> low probability
            (vec![false, false, false, false], false, 0.05), // All false -> very low probability
        ];
        
        let num_test_cases = test_cases.len();
        
        // Add training examples to the proposition database
        let input_entities = vec![const_a, const_b, const_c, const_d];
        let output_entities = vec![
            constant(test_domain.clone(), "A".to_string()),
            constant(test_domain.clone(), "B".to_string()),
            constant(test_domain.clone(), "C".to_string()),
        ];
        
        // Generate propositions for all combinations
        for (input_pattern, expected_output, _probability) in test_cases {
            // Set input propositions
            for (i, &value) in input_pattern.iter().enumerate() {
                if i < input_entities.len() {
                    let prop = proposition(
                        input_relation.clone(),
                        vec![sub(input_entities[i].clone())],
                    );
                    
                    proposition_db.store_proposition_probability(
                        &mut connection,
                        &prop,
                        if value { 1.0 } else { 0.0 },
                    )?;
                    
                    // Add to training queue
                    plan.add_proposition_to_queue(&mut connection, "training_queue", &prop)?;
                }
            }
            
            // Set expected output (for the first three patterns)
            if input_pattern.len() <= 4 {
                let output_idx = match input_pattern.len() {
                    2 => 0, // output(A)
                    3 => 1, // output(B)
                    4 => 2, // output(C)
                    _ => 0,
                };
                
                let output_prop = proposition(
                    output_relation.clone(),
                    vec![sub(output_entities[output_idx].clone())],
                );
                
                proposition_db.store_proposition_probability(
                    &mut connection,
                    &output_prop,
                    if expected_output { 0.9 } else { 0.1 },
                )?;
                
                // Add to training queue
                plan.add_proposition_to_queue(&mut connection, "training_queue", &output_prop)?;
            }
        }
        
        // The graph is automatically saved via the connection
        
        println!("AND gate training scenario setup complete");
        println!("Created {} training examples for AND gates", num_test_cases);
        
        Ok(())
    }
}