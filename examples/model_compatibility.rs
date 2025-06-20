use bayeslog::qbbn::model::exponential::ExponentialModel;
use bayeslog::qbbn::model::torch_exponential::TorchExponentialModel;
use bayeslog::qbbn::common::model::FactorModel;
use bayeslog::qbbn::graphdb::adapter::GraphDBAdapter;
use bayeslog::qbbn::model::objects::{
    Argument, ConstantArgument, LabeledArgument, Predicate, Relation,
};
use bayeslog::qbbn::model::creators::{implication, conjunction, variable_argument};
use std::collections::HashMap;
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up logging
    env_logger::init();
    
    println!("=== Model Compatibility Demo ===\n");
    
    let namespace = "compatibility_test";
    
    // Create a simple implication: likes(X, Y) => eats(X, Y)
    let premise_relation = Relation::new("likes".to_string(), vec![
        variable_argument("Person".to_string()),
        variable_argument("Food".to_string()),
    ]);
    
    let premise = Predicate::new_from_relation(premise_relation, vec![
        LabeledArgument::new("subject".to_string(), Argument::Variable(variable_argument("Person".to_string()))),
        LabeledArgument::new("object".to_string(), Argument::Variable(variable_argument("Food".to_string()))),
    ]);
    
    let conclusion_relation = Relation::new("eats".to_string(), vec![
        variable_argument("Person".to_string()),
        variable_argument("Food".to_string()),
    ]);
    
    let conclusion = Predicate::new_from_relation(conclusion_relation, vec![
        LabeledArgument::new("subject".to_string(), Argument::Variable(variable_argument("Person".to_string()))),
        LabeledArgument::new("object".to_string(), Argument::Variable(variable_argument("Food".to_string()))),
    ]);
    
    // Create role mappings
    let mut role_map = HashMap::new();
    role_map.insert("subject".to_string(), "subject".to_string());
    role_map.insert("object".to_string(), "object".to_string());
    let role_maps = vec![bayeslog::qbbn::model::objects::RoleMap::new(role_map)];
    
    let implication_factor = implication(conjunction(vec![premise]), conclusion, role_maps);
    
    // Step 1: Train with CPU model
    println!("Step 1: Training with CPU model (ExponentialModel)");
    {
        let mut cpu_model = ExponentialModel::new_mutable(namespace.to_string())?;
        let adapter = GraphDBAdapter::new_in_memory(namespace)?;
        let mut conn = adapter.get_connection().into_inner();
        
        // Initialize weights
        cpu_model.initialize_connection(&mut conn, &implication_factor)?;
        println!("✓ CPU model initialized");
        
        // TODO: Add training examples here
        // For now, just save the initialized weights
        
        // Save the model
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_str().unwrap();
        cpu_model.save_to_file(&mut conn, path)?;
        println!("✓ Saved CPU model to: {}", path);
        
        // Step 2: Load into GPU model
        println!("\nStep 2: Loading CPU model into GPU model (TorchExponentialModel)");
        let gpu_model = TorchExponentialModel::from_file(namespace.to_string(), path)?;
        println!("✓ Successfully loaded CPU model into GPU model!");
        
        // Keep temp file from being deleted
        temp_file.into_temp_path().keep()?;
    }
    
    // Step 3: Train with GPU model and load back to CPU
    println!("\nStep 3: Training with GPU model and loading back to CPU");
    {
        let mut gpu_model = TorchExponentialModel::new_mutable(namespace.to_string())?;
        let adapter = GraphDBAdapter::new_in_memory(namespace)?;
        let mut conn = adapter.get_connection().into_inner();
        
        // Initialize weights
        gpu_model.initialize_connection(&mut conn, &implication_factor)?;
        println!("✓ GPU model initialized");
        
        // Save the model
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_str().unwrap();
        gpu_model.save_to_file(&mut conn, path)?;
        println!("✓ Saved GPU model to: {}", path);
        
        // Load into CPU model
        let cpu_model = ExponentialModel::from_file(namespace.to_string(), path)?;
        println!("✓ Successfully loaded GPU model into CPU model!");
        
        // Keep temp file from being deleted
        temp_file.into_temp_path().keep()?;
    }
    
    println!("\n✨ Model compatibility demonstrated successfully!");
    println!("\nKey takeaways:");
    println!("- Models can be saved and loaded across CPU/GPU implementations");
    println!("- Train on GPU for large datasets, then deploy on CPU for inference");
    println!("- Perfect for LLM agent memory scenarios");
    
    Ok(())
}