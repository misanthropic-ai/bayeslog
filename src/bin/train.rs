use std::borrow::Borrow;

use bayeslog::qbbn::common::setup::parse_configuration_options;
use bayeslog::qbbn::common::{resources::ResourceContext, train::setup_and_train};
use bayeslog::qbbn::scenarios::factory::ScenarioMakerFactory;

fn main() {
    // Logging is already initialized in setup.rs
    
    println!("Starting training process");
    let config = parse_configuration_options();
    println!("Scenario: {}", &config.scenario_name);
    
    let resources = ResourceContext::new(&config).expect("Couldn't create resources.");
    println!("Created resources");
    
    let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name).unwrap();
    println!("Created scenario maker for {}", &config.scenario_name);
    
    println!("Running setup_and_train");
    setup_and_train(&resources, scenario_maker.borrow(), &config.scenario_name).expect("Error in training.");
    
    println!("Training completed successfully!");
}
