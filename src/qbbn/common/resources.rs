use std::{error::Error, sync::{Arc, Mutex}};
use super::{redis::{RedisManager, MockConnection}, setup::CommandLineOptions};

/// The ResourceContext provides access to the shared database connection and config
/// In the reference implementation, this would be a Redis connection
/// In our implementation, it's a MockConnection wrapping our GraphDBAdapter
pub struct ResourceContext {
    pub connection: Arc<Mutex<MockConnection>>,
    pub config: CommandLineOptions,
}

impl ResourceContext {
    /// Creates a new ResourceContext with a GraphDB-backed connection
    pub fn new(options: &CommandLineOptions) -> Result<ResourceContext, Box<dyn Error>> {
        // Create a RedisManager (our GraphDBAdapter)
        let manager = RedisManager::new_default()?;
        
        // Get a MockConnection wrapped in Arc<Mutex>
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        Ok(ResourceContext {
            connection,
            config: options.clone(),
        })
    }
    
    /// Creates a new ResourceContext with an in-memory database
    pub fn new_in_memory(namespace: &str) -> Result<ResourceContext, Box<dyn Error>> {
        let manager = RedisManager::new_in_memory(namespace)?;
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        // Create a default config for in-memory databases
        let config = CommandLineOptions {
            scenario_name: namespace.to_string(),
            test_scenario: None,
            entities_per_domain: 20,
            print_training_loss: false,
            test_example: None,
            marginal_output_file: None,
        };
        
        Ok(ResourceContext {
            connection,
            config,
        })
    }
    
    /// Creates a new ResourceContext with a file-based database
    pub fn new_with_file(path: &str, namespace: &str) -> Result<ResourceContext, Box<dyn Error>> {
        let manager = RedisManager::new_with_file(path, namespace)?;
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        // Create a default config for file-based databases
        let config = CommandLineOptions {
            scenario_name: namespace.to_string(),
            test_scenario: None,
            entities_per_domain: 20,
            print_training_loss: false,
            test_example: None,
            marginal_output_file: None,
        };
        
        Ok(ResourceContext {
            connection,
            config,
        })
    }
}