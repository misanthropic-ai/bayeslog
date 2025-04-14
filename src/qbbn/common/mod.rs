pub mod graph;
pub mod interface;
#[macro_use]
pub mod logging;
pub mod model;
pub mod proposition_db;
pub mod redis;
pub mod resources;
pub mod setup;
pub mod test;
pub mod train;

// Re-export key types
pub use interface::BeliefTable;
pub use model::InferenceModel;
pub use resources::ResourceContext;

// Add a shim for GraphDBAdapter for compatibility
pub use crate::qbbn::graphdb::GraphDBAdapter as RedisManager;