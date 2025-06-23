use crate::graph::database::GraphDatabase;
use crate::graph::models::{Direction, Value};
use crate::qbbn::model::unified::UnifiedExponentialModel;
// use crate::qbbn::model::ModelWeights;
use crate::qbbn::graphdb::adapter::GraphDBAdapter;
use crate::qbbn::graphdb::schema::{NodeLabel, EdgeLabel};
// use crate::qbbn::common::model::FactorModel;
use crate::qbbn::model::objects::{Predicate, Proposition, Argument, ConstantArgument, LabeledArgument};
use crate::qbbn::model::creators::{proposition, relation};
// use crate::qbbn::common::redis::MockConnection as Connection;
use log::{info, debug};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use chrono::Utc;

/// High-level interface for using the belief network as agent memory for LLMs
/// 
/// This provides a user-friendly API that:
/// - Automatically manages CPU/GPU model switching
/// - Integrates with the graph database for entity linking
/// - Supports both batch training and single proposition updates
/// - Provides natural language interfaces for LLM integration
pub struct BeliefMemory {
    /// The unified model that handles CPU/GPU switching
    model: UnifiedExponentialModel,
    /// Connection to the graph database
    graph_db: Arc<GraphDatabase>,
    /// Namespace for this belief network
    namespace: String,
    /// GraphDB adapter for QBBN operations
    adapter: GraphDBAdapter,
    /// Cache of entity name to node ID mappings
    entity_cache: HashMap<String, String>,
    /// Cache of proposition hash to node ID mappings
    proposition_cache: HashMap<String, String>,
}

impl BeliefMemory {
    /// Create a new BeliefMemory instance
    pub fn new(graph_db: Arc<GraphDatabase>, namespace: &str) -> Result<Self, Box<dyn Error>> {
        info!("Creating new BeliefMemory with namespace: {}", namespace);
        
        let model = UnifiedExponentialModel::new(namespace.to_string())?;
        let adapter = GraphDBAdapter::new(Arc::clone(&graph_db), namespace);
        
        Ok(BeliefMemory {
            model,
            graph_db,
            namespace: namespace.to_string(),
            adapter,
            entity_cache: HashMap::new(),
            proposition_cache: HashMap::new(),
        })
    }
    
    /// Load BeliefMemory from a saved model file
    pub fn from_file(graph_db: Arc<GraphDatabase>, namespace: &str, path: &str) -> Result<Self, Box<dyn Error>> {
        info!("Loading BeliefMemory from file: {}", path);
        
        let model = UnifiedExponentialModel::from_file(namespace.to_string(), path)?;
        let adapter = GraphDBAdapter::new(Arc::clone(&graph_db), namespace);
        
        let mut memory = BeliefMemory {
            model,
            graph_db,
            namespace: namespace.to_string(),
            adapter,
            entity_cache: HashMap::new(),
            proposition_cache: HashMap::new(),
        };
        
        // Rebuild caches from graph
        memory.rebuild_caches()?;
        
        Ok(memory)
    }
    
    /// Save the current model to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        info!("Saving BeliefMemory to file: {}", path);
        
        // Get a connection for the save operation
        let mut conn = self.adapter.get_connection().into_inner();
        
        // Save the model with graph references
        // TODO: Implement proper save with entity/proposition mappings
        self.model.save_to_file(&mut conn, path)?;
        
        Ok(())
    }
    
    /// Add a proposition with an initial belief probability
    /// 
    /// This is the primary interface for LLM agents to add new beliefs
    /// without requiring full training.
    pub fn add_proposition_with_prior(
        &mut self,
        predicate_name: &str,
        arguments: HashMap<String, String>,
        initial_belief: f64,
    ) -> Result<String, Box<dyn Error>> {
        info!("Adding proposition {} with prior belief {}", predicate_name, initial_belief);
        
        // Ensure entities exist in the graph
        let mut roles = Vec::new();
        for (role, entity_name) in arguments {
            let _entity_id = self.ensure_entity(&entity_name)?;
            
            let arg = Argument::Constant(ConstantArgument {
                domain: "Entity".to_string(),
                entity_id: entity_name.clone(),
            });
            
            roles.push(LabeledArgument {
                role_name: role,
                argument: arg,
            });
        }
        
        // Create the predicate and proposition
        let rel = relation(predicate_name.to_string(), vec![]);
        let predicate = Predicate::new_from_relation(rel.clone(), roles);
        
        let prop = proposition(rel, predicate.roles.clone());
        let prop_hash = prop.debug_string();
        
        // Store in graph database
        self.store_proposition_in_graph(&prop, initial_belief)?;
        
        // Quick update without full training
        self.quick_belief_update(&prop_hash, initial_belief)?;
        
        Ok(prop_hash)
    }
    
    /// Update belief based on a single observation
    /// 
    /// This allows quick updates without full batch training, suitable for
    /// interactive LLM agent scenarios.
    pub fn update_belief_from_observation(
        &mut self,
        proposition_id: &str,
        observed: bool,
        confidence: f64,
    ) -> Result<(), Box<dyn Error>> {
        info!("Updating belief for {} based on observation: {} (confidence: {})", 
              proposition_id, observed, confidence);
        
        // Convert observation to probability
        let new_belief = if observed {
            confidence
        } else {
            1.0 - confidence
        };
        
        // Perform quick update
        self.quick_belief_update(proposition_id, new_belief)?;
        
        Ok(())
    }
    
    /// Query beliefs about a specific entity
    /// 
    /// Returns all propositions involving the entity along with their belief values
    pub fn query_beliefs_about(&self, entity_name: &str) -> Result<Vec<(String, f64)>, Box<dyn Error>> {
        debug!("Querying beliefs about entity: {}", entity_name);
        
        // Find the entity node
        let entity_nodes = self.graph_db.find_nodes_by_property("name", entity_name)?;
        if entity_nodes.is_empty() {
            return Ok(Vec::new());
        }
        
        let entity_node = &entity_nodes[0];
        
        // Find all propositions connected to this entity
        let neighbors = self.graph_db.get_neighbors(&entity_node.id, Direction::Incoming)?;
        
        let mut results = Vec::new();
        for (prop_node, _edge) in neighbors {
            if prop_node.label == NodeLabel::Proposition.as_str() {
                if let Some(Value::String(prop_text)) = prop_node.properties.get("text") {
                    if let Some(Value::String(belief_str)) = prop_node.properties.get("belief") {
                        if let Ok(belief) = belief_str.parse::<f64>() {
                            results.push((prop_text.clone(), belief));
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Find related beliefs using semantic search
    /// 
    /// This uses vector embeddings to find semantically similar propositions
    pub fn find_related_beliefs(&self, query_text: &str, _limit: usize) -> Result<Vec<(String, f64)>, Box<dyn Error>> {
        debug!("Finding beliefs related to: {}", query_text);
        
        // TODO: Implement vector search once embeddings are integrated
        // For now, return empty results
        Ok(Vec::new())
    }
    
    /// Batch train the model on multiple examples
    /// 
    /// This switches to GPU mode automatically for large batches
    pub fn batch_train(&mut self, examples: Vec<(String, f64)>) -> Result<(), Box<dyn Error>> {
        info!("Batch training on {} examples", examples.len());
        
        // Get a connection
        let _conn = self.adapter.get_connection().into_inner();
        
        // The model will auto-switch to GPU if the batch is large enough
        for (_prop_id, _gold_prob) in examples {
            // TODO: Implement training based on proposition IDs
            // This requires building the FactorContext from the proposition
        }
        
        Ok(())
    }
    
    // Helper methods
    
    /// Ensure an entity exists in the graph, creating it if necessary
    fn ensure_entity(&mut self, entity_name: &str) -> Result<String, Box<dyn Error>> {
        // Check cache first
        if let Some(id) = self.entity_cache.get(entity_name) {
            return Ok(id.clone());
        }
        
        // Check graph database
        let nodes = self.graph_db.find_nodes_by_property("name", entity_name)?;
        if let Some(node) = nodes.first() {
            self.entity_cache.insert(entity_name.to_string(), node.id.clone());
            return Ok(node.id.clone());
        }
        
        // Create new entity node
        let props = HashMap::from([
            ("name".to_string(), Value::String(entity_name.to_string())),
            ("namespace".to_string(), Value::String(self.namespace.clone())),
            ("created_at".to_string(), Value::String(Utc::now().to_rfc3339())),
        ]);
        
        let node_id = self.graph_db.add_node(NodeLabel::Entity.as_str(), props)?;
        self.entity_cache.insert(entity_name.to_string(), node_id.clone());
        
        Ok(node_id)
    }
    
    /// Store a proposition in the graph database
    fn store_proposition_in_graph(&mut self, proposition: &Proposition, belief: f64) -> Result<String, Box<dyn Error>> {
        let prop_hash = proposition.debug_string();
        
        // Create proposition node
        let props = HashMap::from([
            ("predicate_hash".to_string(), Value::String(prop_hash.clone())),
            ("text".to_string(), Value::String(format!("{:?}", proposition.predicate))),
            ("belief".to_string(), Value::String(belief.to_string())),
            ("namespace".to_string(), Value::String(self.namespace.clone())),
            ("created_at".to_string(), Value::String(Utc::now().to_rfc3339())),
        ]);
        
        let prop_node_id = self.graph_db.add_node(NodeLabel::Proposition.as_str(), props)?;
        
        // Link to entities
        for role in &proposition.predicate.roles {
            if let Argument::Constant(constant) = &role.argument {
                if let Some(entity_node_id) = self.entity_cache.get(&constant.entity_id) {
                    self.graph_db.add_edge(
                        &prop_node_id,
                        EdgeLabel::HasArgument.as_str(),
                        entity_node_id,
                        HashMap::new(),
                    )?;
                }
            }
        }
        
        self.proposition_cache.insert(prop_hash, prop_node_id.clone());
        
        Ok(prop_node_id)
    }
    
    /// Perform a quick belief update without full training
    fn quick_belief_update(&mut self, proposition_id: &str, new_belief: f64) -> Result<(), Box<dyn Error>> {
        // Update in graph database
        if let Some(node_id) = self.proposition_cache.get(proposition_id) {
            let mut node = self.graph_db.get_node(node_id)?
                .ok_or("Proposition node not found")?;
            
            node.properties.insert("belief".to_string(), Value::String(new_belief.to_string()));
            node.properties.insert("updated_at".to_string(), Value::String(Utc::now().to_rfc3339()));
            
            self.graph_db.update_node(node_id, node.properties)?;
        }
        
        // TODO: Update model weights using a lightweight update mechanism
        // For now, this just updates the graph storage
        
        Ok(())
    }
    
    /// Rebuild internal caches from the graph database
    fn rebuild_caches(&mut self) -> Result<(), Box<dyn Error>> {
        // Rebuild entity cache
        let entity_nodes = self.graph_db.find_nodes_by_label(NodeLabel::Entity.as_str())?;
        for node in entity_nodes {
            if let Some(Value::String(name)) = node.properties.get("name") {
                self.entity_cache.insert(name.clone(), node.id.clone());
            }
        }
        
        // Rebuild proposition cache
        let prop_nodes = self.graph_db.find_nodes_by_label(NodeLabel::Proposition.as_str())?;
        for node in prop_nodes {
            if let Some(Value::String(hash)) = node.properties.get("predicate_hash") {
                self.proposition_cache.insert(hash.clone(), node.id.clone());
            }
        }
        
        info!("Rebuilt caches: {} entities, {} propositions", 
              self.entity_cache.len(), self.proposition_cache.len());
        
        Ok(())
    }
}

/// Extension methods for LLM integration
impl BeliefMemory {
    /// Extract and add propositions from natural language text
    /// 
    /// This would use an LLM to parse the text and extract structured propositions
    pub fn extract_and_add_propositions(&mut self, text: &str) -> Result<Vec<String>, Box<dyn Error>> {
        info!("Extracting propositions from text: {}", text);
        
        // TODO: Integrate with LLM client to extract propositions
        // For now, return empty list
        Ok(Vec::new())
    }
    
    /// Generate natural language explanation for a belief
    pub fn explain_belief_in_natural_language(&self, proposition_id: &str) -> Result<String, Box<dyn Error>> {
        // TODO: Use LLM to generate explanation based on belief network structure
        Ok(format!("Belief {} is based on the current evidence in the network.", proposition_id))
    }
    
    /// Suggest questions to reduce uncertainty about an entity
    pub fn suggest_questions_for_uncertainty(&self, entity_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // TODO: Analyze belief network to find high-uncertainty propositions
        Ok(vec![
            format!("What is the relationship between {} and other entities?", entity_name),
            format!("Can you provide more information about {}?", entity_name),
        ])
    }
}