use bayeslog::qbbn::{
    common::{
        batch_train::do_batch_training,
        setup::{CommandLineOptions, StorageType},
        resources::ResourceContext,
        train::do_training,
    },
    scenarios::factory::ScenarioMakerFactory,
};
use log::info;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    info!("=== Testing Training with Standard vs Torch Models ===");
    
    // Create configuration for dating scenario
    let config = CommandLineOptions {
        scenario_name: "dating_simple".to_string(),
        test_scenario: None,
        entities_per_domain: 100, // Use 100 entities for meaningful test
        print_training_loss: false,
        test_example: None,
        marginal_output_file: None,
        storage_type: StorageType::InMemory,
        db_path: None,
    };
    
    // Setup resources
    let resources = ResourceContext::new(&config)?;
    
    // Create scenario
    let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name)?;
    
    // Setup scenario data (this generates the synthetic data)
    info!("Setting up dating scenario with {} entities per domain...", config.entities_per_domain);
    let setup_start = Instant::now();
    scenario_maker.setup_scenario(&resources)?;
    info!("Scenario setup completed in {:?}", setup_start.elapsed());
    
    // Test standard model training
    info!("\n--- Training with Standard ExponentialModel ---");
    let standard_start = Instant::now();
    do_training(&resources, "dating_simple".to_string())?;
    let standard_time = standard_start.elapsed();
    info!("Standard model training completed in {:?}", standard_time);
    
    // Test torch model training
    info!("\n--- Training with Torch ExponentialModel ---");
    let torch_start = Instant::now();
    do_batch_training(&resources, "dating_simple".to_string(), true)?;
    let torch_time = torch_start.elapsed();
    info!("Torch model training completed in {:?}", torch_time);
    
    // Compare results
    info!("\n=== Training Time Comparison ===");
    info!("Standard model: {:?}", standard_time);
    info!("Torch model: {:?}", torch_time);
    let speedup = standard_time.as_secs_f64() / torch_time.as_secs_f64();
    info!("Speedup: {:.2}x", speedup);
    
    Ok(())
}