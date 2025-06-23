use std::borrow::Borrow;
use std::env;

use bayeslog::qbbn::common::setup::parse_configuration_options;
use bayeslog::qbbn::common::{resources::ResourceContext, train::setup_and_train};
use bayeslog::qbbn::scenarios::factory::ScenarioMakerFactory;

fn main() {
    // Logging is already initialized in setup.rs
    
    // Check if we should use torch model
    let use_torch = env::var("BAYESLOG_USE_TORCH").unwrap_or_else(|_| "0".to_string()) == "1";
    
    if use_torch {
        println!("Starting training process with Torch-based ExponentialModel");
    } else {
        println!("Starting training process with Standard ExponentialModel");
    }
    
    let config = parse_configuration_options();
    println!("Scenario: {}", &config.scenario_name);
    
    let resources = ResourceContext::new(&config).expect("Couldn't create resources.");
    println!("Created resources");
    
    let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name).unwrap();
    println!("Created scenario maker for {}", &config.scenario_name);
    
    println!("Running setup_and_train");
    
    if use_torch {
        // Use batch training with torch model
        use bayeslog::qbbn::common::batch_train::do_batch_training;
        
        // First setup the scenario
        scenario_maker.setup_scenario(&resources).expect("Error setting up scenario");
        
        // Then do batch training
        do_batch_training(&resources, config.scenario_name.clone(), true).expect("Error in torch training");
    } else {
        // Use standard training
        setup_and_train(&resources, scenario_maker.borrow(), &config.scenario_name).expect("Error in training.");
    }
    
    println!("Training completed successfully!");
}
