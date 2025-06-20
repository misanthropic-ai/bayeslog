use bayeslog::qbbn::{
    common::{
        model::{FactorModel, FactorContext},
    },
    model::{
        torch_exponential::TorchExponentialModel,
        exponential::ExponentialModel,
        objects::{ImplicationFactor, Proposition, Argument, PropositionFactor},
    },
};
use log::info;
use std::time::Instant;

fn create_test_factor() -> (ImplicationFactor, FactorContext) {
    use bayeslog::qbbn::model::objects::{
        ConstantArgument, VariableArgument, Relation, Predicate, 
        PredicateGroup, GroupRoleMap
    };
    use bayeslog::qbbn::model::creators::{sub, obj};
    
    // Create a simple test implication: likes(X, Y) => dates(X, Y)
    let person_domain = "person".to_string();
    
    // Create constants
    let alice = Argument::Constant(ConstantArgument::new(person_domain.clone(), "alice".to_string()));
    let bob = Argument::Constant(ConstantArgument::new(person_domain.clone(), "bob".to_string()));
    
    // Create relations with variable arguments
    let x_var = VariableArgument::new(person_domain.clone());
    let y_var = VariableArgument::new(person_domain.clone());
    
    let likes_rel = Relation::new("likes".to_string(), vec![x_var.clone(), y_var.clone()]);
    let dates_rel = Relation::new("dates".to_string(), vec![x_var.clone(), y_var.clone()]);
    
    // Create predicates
    let likes_pred = Predicate::new_from_relation(
        likes_rel.clone(),
        vec![sub(alice.clone()), obj(bob.clone())]
    );
    
    let dates_pred = Predicate::new_from_relation(
        dates_rel.clone(),
        vec![sub(alice.clone()), obj(bob.clone())]
    );
    
    // Create propositions
    let likes_prop = Proposition::from(likes_pred.clone());
    let dates_prop = Proposition::from(dates_pred.clone());
    
    // Create implication factor
    let implication = ImplicationFactor {
        premise: PredicateGroup {
            terms: vec![likes_pred]
        },
        conclusion: dates_pred,
        role_maps: GroupRoleMap {
            role_maps: vec![] // No role mapping needed for this simple example
        }
    };
    
    // Create factor context with some probabilities
    let factor_context = FactorContext {
        factor: vec![implication.clone()],
        probabilities: vec![0.8], // likes(alice, bob) has 0.8 probability
    };
    
    (implication, factor_context)
}

fn test_model_operations<M: FactorModel>(model: &mut M, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing {} model...", name);
    
    // Create mock connection
    let mut connection = bayeslog::qbbn::common::redis::MockConnection::new();
    
    // Create test data
    let (implication, factor_context) = create_test_factor();
    
    // Initialize model with implication
    info!("Initializing model...");
    let init_start = Instant::now();
    model.initialize_connection(&mut connection, &implication)?;
    info!("Initialization took: {:?}", init_start.elapsed());
    
    // Test prediction before training
    info!("Testing prediction before training...");
    let pred_start = Instant::now();
    let initial_pred = model.predict(&mut connection, &factor_context)?;
    info!("Initial prediction: {:.4}, took: {:?}", initial_pred.probability, pred_start.elapsed());
    
    // Train the model
    info!("Training model...");
    let mut total_loss = 0.0;
    let train_start = Instant::now();
    
    for epoch in 0..100 {
        let gold_probability = 0.9; // We want dates(alice, bob) to have high probability
        let stats = model.train(&mut connection, &factor_context, gold_probability)?;
        total_loss += stats.loss;
        
        if epoch % 20 == 0 {
            let pred = model.predict(&mut connection, &factor_context)?;
            info!("Epoch {}: loss = {:.6}, prediction = {:.4}", epoch, stats.loss, pred.probability);
        }
    }
    
    let train_elapsed = train_start.elapsed();
    info!("Training completed in: {:?}, avg loss: {:.6}", train_elapsed, total_loss / 100.0);
    
    // Test prediction after training
    info!("Testing prediction after training...");
    let final_pred = model.predict(&mut connection, &factor_context)?;
    info!("Final prediction: {:.4}", final_pred.probability);
    
    // The prediction should be closer to 0.9 after training
    let improvement = (final_pred.probability - initial_pred.probability).abs();
    info!("Prediction improved by: {:.4}", improvement);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    info!("=== Testing PyTorch-based Exponential Model ===");
    
    // Test device selection
    let device = bayeslog::qbbn::model::device::select_device();
    info!("Selected device: {:?}", device);
    
    // Test TorchExponentialModel
    info!("\n--- Testing TorchExponentialModel ---");
    let mut torch_model = TorchExponentialModel::new_mutable("test_torch".to_string())?;
    test_model_operations(torch_model.as_mut(), "TorchExponential")?;
    
    // Test standard ExponentialModel for comparison
    info!("\n--- Testing Standard ExponentialModel (for comparison) ---");
    let mut standard_model = ExponentialModel::new_mutable("test_standard".to_string())?;
    test_model_operations(standard_model.as_mut(), "StandardExponential")?;
    
    info!("\n=== All tests completed successfully! ===");
    Ok(())
}