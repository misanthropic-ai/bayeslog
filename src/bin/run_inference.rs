use bayeslog::qbbn::common::setup::{CommandLineOptions, StorageType};
use bayeslog::qbbn::common::resources::ResourceContext;
use bayeslog::qbbn::common::proposition_db::RedisBeliefTable;
use bayeslog::qbbn::common::model::InferenceModel;
use bayeslog::qbbn::common::graph::InferenceGraph;
use bayeslog::qbbn::inference::graph::PropositionGraph;
use bayeslog::qbbn::inference::engine::Inferencer;
use bayeslog::qbbn::model::creators::{constant, relation, proposition, sub, obj, variable_argument};
use bayeslog::qbbn::model::objects::Domain;
use std::env;
use std::error::Error;
use std::sync::Arc;
use bayeslog::qbbn::model::exponential::ExponentialModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting BayesLog Dating Inference (No Training)");
    
    // Create a default configuration with dating_simple scenario
    let entities_per_domain = 10; // Small value for faster execution
    let scenario_name = "dating_simple".to_string();
    
    // Get storage type from environment variable or default to persistent
    let storage_type_str = env::var("STORAGE_TYPE").unwrap_or_else(|_| "persistent".to_string());
    let storage_type = match storage_type_str.as_str() {
        "in-memory" => StorageType::InMemory,
        _ => StorageType::Persistent,
    };
    
    // Get database path from environment variable or use default
    let db_path = env::var("DB_PATH").unwrap_or_else(|_| "dating_test.db".to_string());
    
    println!("===== CONFIGURATION =====");
    println!("Scenario: {}", scenario_name);
    println!("Storage Type: {:?}", storage_type);
    println!("Database Path: {}", db_path);
    println!("=========================");
    
    let config = CommandLineOptions {
        scenario_name,
        test_scenario: None,
        entities_per_domain,
        print_training_loss: false,
        test_example: None,
        marginal_output_file: None,
        storage_type,
        db_path: Some(db_path),
    };
    
    // Create resource context using the persistent database
    let resources = ResourceContext::new(&config)?;
    println!("Created resource context using existing database");
    
    // Get a connection
    let mut connection = resources.connection.lock().unwrap();
    
    // For dating_simple, we'll create a dating target proposition
    println!("\n===== SETTING UP INFERENCE =====");
    println!("Creating target proposition for dating inference...");
    
    let man_domain = Domain::MAN.to_string();
    let woman_domain = Domain::WOMAN.to_string();
    
    // In our scenario, we'll use the test entities from position 0
    println!("Using test entities: test_Man0 and test_Woman0");
    let jack = constant(man_domain.clone(), "test_Man0".to_string());
    let jill = constant(woman_domain.clone(), "test_Woman0".to_string());
    
    println!("Creating dating relation for inference target");
    let dating_relation = relation(
        "date".to_string(),
        vec![
            variable_argument(man_domain.clone()),
            variable_argument(woman_domain.clone()),
        ],
    );
    
    // Create the target proposition: date(test_Man0, test_Woman0)
    println!("Creating target proposition: date(test_Man0, test_Woman0)");
    let target = proposition(
        dating_relation.clone(),
        vec![sub(jack.clone()), obj(jill.clone())]
    );
    println!("Target proposition created: {:?}", target);
    println!("==============================");
    
    // Create a mutable InferenceGraph and register the target
    println!("\n===== INITIALIZING INFERENCE ENGINE =====");
    println!("Creating inference graph and registering target...");
    let mut graph = InferenceGraph::new_mutable(config.scenario_name.clone())?;
    graph.register_target(&mut connection, &target)?;
    
    // Now create the InferenceModel with our graph
    println!("Creating inference model with target...");
    let graph_arc = Arc::new(*graph);
    let model_arc = ExponentialModel::new_shared(config.scenario_name.clone())?;
    let model = Arc::new(InferenceModel {
        graph: graph_arc.clone(),
        model: model_arc,
    });
    
    // Create the proposition graph
    println!("Creating proposition graph for belief propagation...");
    let proposition_graph = PropositionGraph::new_shared(&mut connection, &graph_arc, target.clone())?;
    
    // Create the belief table
    println!("Creating belief table for storing probabilities...");
    let fact_memory = RedisBeliefTable::new_shared(config.scenario_name.clone())?;
    
    // Create the inferencer
    println!("Creating inferencer with model, proposition graph, and belief table...");
    let mut inferencer = Inferencer::new_mutable(
        model.clone(),
        proposition_graph.clone(),
        fact_memory.clone(),
    )?;
    
    // Initialize the inference chart
    println!("Initializing inference chart...");
    inferencer.initialize_chart(&mut connection)?;
    println!("Inference chart initialized successfully");
    println!("======================================\n");
    
    // Run initial inference
    println!("\n===== RUNNING BASELINE INFERENCE =====");
    println!("Running initial inference (checking for existing evidence)...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    println!("Updating marginal probabilities...");
    let initial_marginals = inferencer.update_marginals()?;
    println!("Initial inference completed");
    println!("=====================================");
    
    // Print initial marginals
    println!("\nInitial marginals (proposition = probability):");
    println!("{}", initial_marginals.render_marginal_table());
    println!("\nThese are the marginal probabilities based on any existing evidence.");
    
    // Create relations for evidence checking
    println!("\n===== CHECKING EXISTING EVIDENCE =====");
    
    // Create exciting relation
    let exciting_relation = relation(
        "exciting".to_string(),
        vec![variable_argument(woman_domain.clone())],
    );
    
    // Create lonely relation
    let lonely_relation = relation(
        "lonely".to_string(),
        vec![variable_argument(man_domain.clone())],
    );
    
    // Create like relation
    let woman_likes_man_relation = relation(
        "like".to_string(),
        vec![
            variable_argument(woman_domain.clone()),
            variable_argument(man_domain.clone()),
        ],
    );
    
    // Create evidence propositions
    let jill_exciting = proposition(
        exciting_relation.clone(),
        vec![sub(jill.clone())]
    );
    
    let jack_lonely = proposition(
        lonely_relation.clone(),
        vec![sub(jack.clone())]
    );
    
    let jill_likes_jack = proposition(
        woman_likes_man_relation.clone(),
        vec![sub(jill.clone()), obj(jack.clone())]
    );
    
    // Check evidence values
    let exciting_prob = fact_memory.get_proposition_probability(&mut connection, &jill_exciting)?;
    let lonely_prob = fact_memory.get_proposition_probability(&mut connection, &jack_lonely)?;
    let likes_prob = fact_memory.get_proposition_probability(&mut connection, &jill_likes_jack)?;
    
    println!("Evidence check results:");
    println!("  exciting(test_Woman0) = {:?}", exciting_prob);
    println!("  lonely(test_Man0) = {:?}", lonely_prob);
    println!("  like(test_Woman0, test_Man0) = {:?}", likes_prob);
    
    println!("\nEach value of 'Some(1.0)' indicates that evidence was previously set");
    println!("and has been persisted in the database.");
    println!("=======================================");
    
    // Run one more inference to get final results
    println!("\n===== RUNNING FINAL INFERENCE =====");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    let final_marginals = inferencer.update_marginals()?;
    
    println!("\nFinal marginals with all persisted evidence:");
    println!("{}", final_marginals.render_marginal_table());
    println!("\nThese results reflect the evidence that was previously set and");
    println!("persisted in the database from earlier runs.");
    println!("======================================");
    
    Ok(())
}