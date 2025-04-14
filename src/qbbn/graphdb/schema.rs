/// This module defines the schema for storing QBBN components in the graph database
/// It provides consistent labels and relationship types to ensure that
/// the mapping between QBBN concepts and graph entities is consistent.
/// Node labels for different types of entities in the QBBN graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLabel {
    // Storage entities for Redis compatibility
    KeyValue,    // Simple key-value pair storage
    Hash,        // Hash map storage
    Set,         // Set container
    SetMember,   // Individual member of a set
    List,        // List container
    ListItem,    // Individual item in a list
    
    // QBBN domain entities
    Relation,    // Type of relationship
    Domain,      // Domain of discourse for entities
    Entity,      // Instance of a domain
    
    // QBBN core concepts
    Proposition, // Grounded predicate with truth value
    Predicate,   // Relation with role arguments
    Argument,    // Role argument (constant or variable)
    Factor,      // Logical operation connecting propositions
    Feature,     // Feature for the exponential model
    Weight,      // Weight associated with a feature
}

use std::str::FromStr;

impl NodeLabel {
    /// Convert to string representation for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            // Storage entities
            NodeLabel::KeyValue => "KeyValue",
            NodeLabel::Hash => "Hash",
            NodeLabel::Set => "Set",
            NodeLabel::SetMember => "SetMember",
            NodeLabel::List => "List",
            NodeLabel::ListItem => "ListItem",
            
            // Domain entities
            NodeLabel::Relation => "Relation",
            NodeLabel::Domain => "Domain",
            NodeLabel::Entity => "Entity",
            
            // Core concepts
            NodeLabel::Proposition => "Proposition",
            NodeLabel::Predicate => "Predicate",
            NodeLabel::Argument => "Argument",
            NodeLabel::Factor => "Factor",
            NodeLabel::Feature => "Feature",
            NodeLabel::Weight => "Weight",
        }
    }
}

impl FromStr for NodeLabel {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // Storage entities
            "KeyValue" => Ok(NodeLabel::KeyValue),
            "Hash" => Ok(NodeLabel::Hash),
            "Set" => Ok(NodeLabel::Set),
            "SetMember" => Ok(NodeLabel::SetMember),
            "List" => Ok(NodeLabel::List),
            "ListItem" => Ok(NodeLabel::ListItem),
            
            // Domain entities
            "Relation" => Ok(NodeLabel::Relation),
            "Domain" => Ok(NodeLabel::Domain),
            "Entity" => Ok(NodeLabel::Entity),
            
            // Core concepts
            "Proposition" => Ok(NodeLabel::Proposition),
            "Predicate" => Ok(NodeLabel::Predicate),
            "Argument" => Ok(NodeLabel::Argument),
            "Factor" => Ok(NodeLabel::Factor),
            "Feature" => Ok(NodeLabel::Feature),
            "Weight" => Ok(NodeLabel::Weight),
            
            _ => Err(format!("Unknown NodeLabel: {}", s)),
        }
    }
}

/// Edge labels for relationships between nodes in the QBBN graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeLabel {
    // Storage relationships for Redis compatibility
    Contains,     // Set contains member
    HasItem,      // List has item at position
    
    // Domain relationships
    BelongsToDomain, // Entity belongs to domain
    
    // Core concept relationships
    HasArgument,     // Predicate has argument with role
    HasPremise,      // Factor has premise
    HasConclusion,   // Factor has conclusion
    HasFeature,      // Factor has feature
    HasWeight,       // Feature has weight
    FactorInput,     // Proposition is input to factor
    FactorOutput,    // Proposition is output from factor
    ImpliedBy,       // Proposition is implied by another
    
    // Logic relationships
    Negation,        // Negation relationship
    Conjunction,     // Conjunction relationship
    Disjunction,     // Disjunction relationship
}

impl EdgeLabel {
    /// Convert to string representation for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            // Storage relationships
            EdgeLabel::Contains => "CONTAINS",
            EdgeLabel::HasItem => "HAS_ITEM",
            
            // Domain relationships
            EdgeLabel::BelongsToDomain => "BELONGS_TO_DOMAIN",
            
            // Core concept relationships
            EdgeLabel::HasArgument => "HAS_ARGUMENT",
            EdgeLabel::HasPremise => "HAS_PREMISE",
            EdgeLabel::HasConclusion => "HAS_CONCLUSION",
            EdgeLabel::HasFeature => "HAS_FEATURE",
            EdgeLabel::HasWeight => "HAS_WEIGHT",
            EdgeLabel::FactorInput => "FACTOR_INPUT",
            EdgeLabel::FactorOutput => "FACTOR_OUTPUT",
            EdgeLabel::ImpliedBy => "IMPLIED_BY",
            
            // Logic relationships
            EdgeLabel::Negation => "NEGATION",
            EdgeLabel::Conjunction => "CONJUNCTION",
            EdgeLabel::Disjunction => "DISJUNCTION",
        }
    }
}

impl FromStr for EdgeLabel {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // Storage relationships
            "CONTAINS" => Ok(EdgeLabel::Contains),
            "HAS_ITEM" => Ok(EdgeLabel::HasItem),
            
            // Domain relationships
            "BELONGS_TO_DOMAIN" => Ok(EdgeLabel::BelongsToDomain),
            
            // Core concept relationships
            "HAS_ARGUMENT" => Ok(EdgeLabel::HasArgument),
            "HAS_PREMISE" => Ok(EdgeLabel::HasPremise),
            "HAS_CONCLUSION" => Ok(EdgeLabel::HasConclusion),
            "HAS_FEATURE" => Ok(EdgeLabel::HasFeature),
            "HAS_WEIGHT" => Ok(EdgeLabel::HasWeight),
            "FACTOR_INPUT" => Ok(EdgeLabel::FactorInput),
            "FACTOR_OUTPUT" => Ok(EdgeLabel::FactorOutput),
            "IMPLIED_BY" => Ok(EdgeLabel::ImpliedBy),
            
            // Logic relationships
            "NEGATION" => Ok(EdgeLabel::Negation),
            "CONJUNCTION" => Ok(EdgeLabel::Conjunction),
            "DISJUNCTION" => Ok(EdgeLabel::Disjunction),
            
            _ => Err(format!("Unknown EdgeLabel: {}", s)),
        }
    }
}

/// Factor types in the QBBN
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Implication,    // Logical implication
    Conjunction,    // Logical AND
    Disjunction,    // Logical OR
    Negation,       // Logical NOT
    WeightedSum,    // Weighted sum for exponential model
    ExponentialModel, // Full exponential model
}

impl FactorType {
    /// Convert to string representation for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            FactorType::Implication => "IMPLICATION",
            FactorType::Conjunction => "CONJUNCTION",
            FactorType::Disjunction => "DISJUNCTION",
            FactorType::Negation => "NEGATION",
            FactorType::WeightedSum => "WEIGHTED_SUM",
            FactorType::ExponentialModel => "EXPONENTIAL_MODEL",
        }
    }
}

impl FromStr for FactorType {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "IMPLICATION" => Ok(FactorType::Implication),
            "CONJUNCTION" => Ok(FactorType::Conjunction),
            "DISJUNCTION" => Ok(FactorType::Disjunction),
            "NEGATION" => Ok(FactorType::Negation),
            "WEIGHTED_SUM" => Ok(FactorType::WeightedSum),
            "EXPONENTIAL_MODEL" => Ok(FactorType::ExponentialModel),
            _ => Err(format!("Unknown FactorType: {}", s)),
        }
    }
}

/// Proposition property names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropProperty {
    Id,            // Unique identifier
    PredicateHash, // Hash of the predicate
    Belief,        // Current belief value
    Evidence,      // Whether this is observed evidence
    Lambda,        // Lambda message value
    Pi,            // Pi message value
    Timestamp,     // Temporal information
}

impl PropProperty {
    /// Convert to string representation for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            PropProperty::Id => "id",
            PropProperty::PredicateHash => "predicate_hash",
            PropProperty::Belief => "belief",
            PropProperty::Evidence => "evidence",
            PropProperty::Lambda => "lambda",
            PropProperty::Pi => "pi",
            PropProperty::Timestamp => "timestamp",
        }
    }
}

/// Standard storage property names for Redis-compatible storage
pub mod redis_property {
    pub const KEY: &str = "key";
    pub const VALUE: &str = "value";
    pub const FIELDS: &str = "fields";
    pub const POSITION: &str = "position";
    pub const LENGTH: &str = "length";
}

/// Constants for namespacing in the graph database
pub mod namespace {
    pub const PREFIX: &str = "bayes-star";
    
    /// Create a namespaced key as used in Bayes Star
    pub fn qualified_key(namespace: &str, key: &str) -> String {
        format!("{}:{}:{}", PREFIX, namespace, key)
    }
}