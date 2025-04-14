use std::{error::Error, sync::{Arc, Mutex}};
use super::{redis::{RedisManager, MockConnection}, setup::CommandLineOptions};

/// The ResourceContext provides access to the shared database connection
/// In the reference implementation, this would be a Redis connection
/// In our implementation, it's a MockConnection wrapping our GraphDBAdapter
pub struct ResourceContext {
    pub connection: Arc<Mutex<MockConnection>>,
}

impl ResourceContext {
    /// Creates a new ResourceContext with a GraphDB-backed connection
    pub fn new(_options: &CommandLineOptions) -> Result<ResourceContext, Box<dyn Error>> {
        // Create a RedisManager (our GraphDBAdapter)
        let manager = RedisManager::new_default()?;
        
        // Get a MockConnection wrapped in Arc<Mutex>
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        Ok(ResourceContext {
            connection,
        })
    }
    
    /// Creates a new ResourceContext with an in-memory database
    pub fn new_in_memory(namespace: &str) -> Result<ResourceContext, Box<dyn Error>> {
        let manager = RedisManager::new_in_memory(namespace)?;
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        Ok(ResourceContext {
            connection,
        })
    }
    
    /// Creates a new ResourceContext with a file-based database
    pub fn new_with_file(path: &str, namespace: &str) -> Result<ResourceContext, Box<dyn Error>> {
        let manager = RedisManager::new_with_file(path, namespace)?;
        let connection = manager.get_arc_mutex_guarded_connection()?;
        
        Ok(ResourceContext {
            connection,
        })
    }
}