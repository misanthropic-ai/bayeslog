pub mod common;
pub mod inference;
pub mod model;
pub mod graphdb;  // This will be our new module replacing Redis
pub mod scenarios; // Expose the scenarios module for testing

// Re-export color printing macros
pub use crate::print_blue;
pub use crate::print_green;
pub use crate::print_red;
pub use crate::print_yellow;

/// Exports the main types for easy access
pub use common::interface::BeliefTable;
pub use inference::BayesianNetwork;
pub use model::objects::{Proposition, Predicate, Argument, ConstantArgument, VariableArgument};