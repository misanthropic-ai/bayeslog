use bayeslog::qbbn::model::exponential::ExponentialModel;
use bayeslog::qbbn::model::torch_exponential::TorchExponentialModel;
use bayeslog::qbbn::model::unified::UnifiedExponentialModel;
use bayeslog::qbbn::common::model::FactorModel;
use bayeslog::qbbn::common::redis::MockConnection;
use bayeslog::qbbn::model::objects::{ImplicationFactor, Predicate, RoleLabel, Argument, Variable, TypeName};
use bayeslog::qbbn::model::creators::make_implication;
use std::collections::HashMap;
use tempfile::NamedTempFile;

#[test]
fn test_weight_compatibility_between_models() -> Result<(), Box<dyn std::error::Error>> {
    let namespace = "test_compat";
    
    // Create a simple implication for testing
    let premise = Predicate {
        function_name: "likes".to_string(),
        role_arguments: HashMap::from([
            (RoleLabel { name: "subject".to_string() }, Argument::Variable(Variable {
                type_name: TypeName { name: "Person".to_string() },
                name: "X".to_string(),
            })),
            (RoleLabel { name: "object".to_string() }, Argument::Variable(Variable {
                type_name: TypeName { name: "Food".to_string() },
                name: "Y".to_string(),
            })),
        ]),
    };
    
    let conclusion = Predicate {
        function_name: "eats".to_string(),
        role_arguments: HashMap::from([
            (RoleLabel { name: "subject".to_string() }, Argument::Variable(Variable {
                type_name: TypeName { name: "Person".to_string() },
                name: "X".to_string(),
            })),
            (RoleLabel { name: "object".to_string() }, Argument::Variable(Variable {
                type_name: TypeName { name: "Food".to_string() },
                name: "Y".to_string(),
            })),
        ]),
    };
    
    let implication = make_implication(vec![premise], conclusion);
    
    // Step 1: Train with ExponentialModel (CPU)
    {
        let mut cpu_model = ExponentialModel::new_mutable(namespace.to_string())?;
        let mut conn = MockConnection::new_in_memory()?;
        
        // Initialize weights
        cpu_model.initialize_connection(&mut conn, &implication)?;
        
        // TODO: Add some training examples here
        
        // Save the model
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_str().unwrap();
        cpu_model.save_to_file(&mut conn, path)?;
        
        println!("Saved CPU model to: {}", path);
        
        // Step 2: Load into TorchExponentialModel (GPU)
        let gpu_model = TorchExponentialModel::from_file(namespace.to_string(), path)?;
        
        // TODO: Verify predictions are the same
        
        println!("Successfully loaded CPU model into GPU model!");
    }
    
    // Step 3: Train with TorchExponentialModel and load back to CPU
    {
        let mut gpu_model = TorchExponentialModel::new_mutable(namespace.to_string())?;
        let mut conn = MockConnection::new_in_memory()?;
        
        // Initialize weights
        gpu_model.initialize_connection(&mut conn, &implication)?;
        
        // TODO: Add some training examples here
        
        // Save the model
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_str().unwrap();
        gpu_model.save_to_file(&mut conn, path)?;
        
        println!("Saved GPU model to: {}", path);
        
        // Load into CPU model
        let cpu_model = ExponentialModel::from_file(namespace.to_string(), path)?;
        
        println!("Successfully loaded GPU model into CPU model!");
    }
    
    // Step 4: Test UnifiedExponentialModel auto-switching
    {
        let mut unified_model = UnifiedExponentialModel::new(namespace.to_string())?;
        let mut conn = MockConnection::new_in_memory()?;
        
        // Initialize
        unified_model.initialize_connection(&mut conn, &implication)?;
        
        // TODO: Add training with different batch sizes to trigger auto-switching
        
        // Save and reload
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_str().unwrap();
        unified_model.save_to_file(&mut conn, path)?;
        
        // Load back
        let loaded_model = UnifiedExponentialModel::from_file(namespace.to_string(), path)?;
        
        println!("Successfully saved and loaded UnifiedExponentialModel!");
    }
    
    Ok(())
}

#[test]
fn test_belief_memory_integration() -> Result<(), Box<dyn std::error::Error>> {
    use bayeslog::{GraphDatabase, BeliefMemory};
    use std::sync::Arc;
    
    // Create an in-memory graph database
    let graph_db = Arc::new(GraphDatabase::new_in_memory()?);
    let namespace = "test_memory";
    
    // Create BeliefMemory
    let mut memory = BeliefMemory::new(graph_db, namespace)?;
    
    // Add a proposition with prior belief
    let mut args = HashMap::new();
    args.insert("subject".to_string(), "Alice".to_string());
    args.insert("object".to_string(), "Pizza".to_string());
    
    let prop_id = memory.add_proposition_with_prior("likes", args, 0.7)?;
    println!("Added proposition: {}", prop_id);
    
    // Update belief based on observation
    memory.update_belief_from_observation(&prop_id, true, 0.9)?;
    
    // Query beliefs about Alice
    let beliefs = memory.query_beliefs_about("Alice")?;
    println!("Beliefs about Alice: {:?}", beliefs);
    
    // Save the model
    let temp_file = NamedTempFile::new()?;
    let path = temp_file.path().to_str().unwrap();
    memory.save_to_file(path)?;
    
    println!("BeliefMemory test completed successfully!");
    
    Ok(())
}