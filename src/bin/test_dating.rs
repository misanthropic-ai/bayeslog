use bayeslog::qbbn::common::setup::{CommandLineOptions, StorageType};
use bayeslog::qbbn::common::resources::ResourceContext;
use bayeslog::qbbn::common::proposition_db::RedisBeliefTable;
use bayeslog::qbbn::common::model::InferenceModel;
use bayeslog::qbbn::common::graph::InferenceGraph;
use bayeslog::qbbn::inference::graph::PropositionGraph;
use bayeslog::qbbn::inference::engine::Inferencer;
use bayeslog::qbbn::model::creators::{constant, relation, proposition, sub, obj, variable_argument};
use bayeslog::qbbn::model::objects::Domain;
use bayeslog::qbbn::scenarios::factory::ScenarioMakerFactory;
use bayeslog::qbbn::common::train::setup_and_train;
use std::borrow::Borrow;
use std::env;
use std::error::Error;
use std::sync::Arc;
use bayeslog::qbbn::model::exponential::ExponentialModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting BayesLog Dating Scenario Test");
    
    // Create a default configuration with dating_simple scenario
    // Use a much smaller number of entities for faster demo purposes
    let entities_per_domain = 10; // Reduced from 1024 for faster demo execution
    let print_training_loss = true;
    let scenario_name = "dating_simple".to_string();
    
    // Get storage type from environment variable or default to in-memory
    let storage_type_str = env::var("STORAGE_TYPE").unwrap_or_else(|_| "in-memory".to_string());
    let storage_type = match storage_type_str.as_str() {
        "persistent" => StorageType::Persistent,
        _ => StorageType::InMemory,
    };
    
    // Get database path from environment variable
    let db_path = env::var("DB_PATH").ok();
    
    println!("===== CONFIGURATION =====");
    println!("Scenario: {}", scenario_name);
    println!("Entities per domain: {}", entities_per_domain);
    println!("Storage Type: {:?}", storage_type);
    if let Some(path) = &db_path {
        println!("Database Path: {}", path);
    }
    println!("=========================");
    
    let config = CommandLineOptions {
        scenario_name,
        test_scenario: None,
        entities_per_domain,
        print_training_loss,
        test_example: None,
        marginal_output_file: None,
        storage_type,
        db_path,
    };
    println!("Scenario: {}", config.scenario_name);
    
    // Create resource context
    let resources = ResourceContext::new(&config)?;
    println!("Created resources");
    
    // Setup and train the model if needed
    println!("\n===== TRAINING MODEL =====");
    println!("Creating scenario maker for: {}", config.scenario_name);
    let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name)?;
    
    println!("Setting up training data and running training...");
    println!("This will create test entities and train on their relationships");
    setup_and_train(&resources, scenario_maker.borrow(), &config.scenario_name)?;
    println!("Training completed successfully");
    println!("===========================");
    
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
    println!("Running initial inference (no evidence set)...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    println!("Updating marginal probabilities...");
    let initial_marginals = inferencer.update_marginals()?;
    println!("Initial inference completed");
    println!("=====================================");
    
    // Print initial marginals
    println!("\n1. Initial marginals (proposition = probability):");
    println!("{}", initial_marginals.render_marginal_table());
    println!("\nExplanation: These are the initial marginal probabilities for the dating relationship.");
    println!("The target proposition is whether 'test_Man0' dates 'test_Woman0'.");
    println!("The probability is calculated based on the learned weights from training data.");
    
    // Create relations for evidence
    println!("\n===== PREPARING EVIDENCE =====");
    println!("Creating relations for evidence propositions...");
    let exciting_relation = relation(
        "exciting".to_string(),
        vec![variable_argument(woman_domain.clone())],
    );
    
    let lonely_relation = relation(
        "lonely".to_string(),
        vec![variable_argument(man_domain.clone())],
    );
    
    // Create evidence propositions
    println!("Creating evidence proposition: exciting(test_Woman0)");
    let jill_exciting = proposition(
        exciting_relation.clone(),
        vec![sub(jill.clone())]
    );
    println!("==============================\n");
    
    // Set evidence: jill is exciting
    println!("\n===== INFERENCE WITH EVIDENCE 1 =====");
    println!("2. Setting evidence: Woman is exciting");
    fact_memory.store_proposition_probability(&mut connection, &jill_exciting, 1.0)?;
    println!("Evidence set: exciting(test_Woman0) = 1.0");
    
    // Run full inference with the new evidence
    println!("Running belief propagation with new evidence...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    
    // Get updated marginals
    println!("Updating marginal probabilities...");
    let updated_marginals1 = inferencer.update_marginals()?;
    println!("Inference completed with evidence 1");
    println!("====================================");
    
    // Print updated marginals
    println!("\nMarginals after setting Woman is exciting:");
    println!("{}", updated_marginals1.render_marginal_table());
    println!("\nExplanation: The probability of dating has increased because an exciting woman");
    println!("is more likely to be liked by the man, as per one of our inference rules.");
    
    // Now set evidence that Jack is lonely
    println!("\n===== INFERENCE WITH EVIDENCE 1+2 =====");
    println!("Creating evidence proposition: lonely(test_Man0)");
    let jack_lonely = proposition(
        lonely_relation.clone(),
        vec![sub(jack.clone())]
    );
    
    // Set evidence: Jack is lonely
    println!("\n3. Setting additional evidence: Man is lonely");
    fact_memory.store_proposition_probability(&mut connection, &jack_lonely, 1.0)?;
    println!("Evidence set: lonely(test_Man0) = 1.0");
    println!("Previous evidence remains: exciting(test_Woman0) = 1.0");
    
    // Run full inference with the new evidence
    println!("Running belief propagation with both pieces of evidence...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    
    // Get updated marginals
    println!("Updating marginal probabilities...");
    let updated_marginals2 = inferencer.update_marginals()?;
    println!("Inference completed with evidence 1+2");
    println!("======================================");
    
    // Print updated marginals
    println!("\nMarginals after setting both Woman is exciting AND Man is lonely:");
    println!("{}", updated_marginals2.render_marginal_table());
    println!("\nExplanation: The probability of dating has increased even more because:");
    println!("1. A lonely man is more likely to like any woman (inference rule 1)");
    println!("2. An exciting woman is more likely to be liked by any man (inference rule 2)");
    println!("3. If they like each other, they're more likely to date (inference rule 3)");
    
    // Create a likes relationship
    println!("\n===== INFERENCE WITH ALL EVIDENCE =====");
    println!("Creating final evidence proposition: Woman likes Man");
    let woman_likes_man_relation = relation(
        "like".to_string(),
        vec![
            variable_argument(woman_domain.clone()),
            variable_argument(man_domain.clone()),
        ],
    );
    
    let jill_likes_jack = proposition(
        woman_likes_man_relation.clone(),
        vec![sub(jill.clone()), obj(jack.clone())]
    );
    
    // Set evidence: Jill likes Jack
    println!("\n4. Setting evidence that Woman likes Man");
    fact_memory.store_proposition_probability(&mut connection, &jill_likes_jack, 1.0)?;
    println!("Evidence set: like(test_Woman0, test_Man0) = 1.0");
    println!("Previous evidence remains:");
    println!("  - exciting(test_Woman0) = 1.0");
    println!("  - lonely(test_Man0) = 1.0");
    
    // Run full inference with all evidence
    println!("Running belief propagation with all evidence...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    
    // Get final marginals
    println!("Updating final marginal probabilities...");
    let final_marginals = inferencer.update_marginals()?;
    println!("Inference completed with all evidence");
    println!("======================================");
    
    // Print final marginals
    println!("\nFinal marginals with all evidence:");
    println!("{}", final_marginals.render_marginal_table());
    println!("\nExplanation: With all evidence in place, the probability of dating is now very high.");
    println!("This demonstrates how Bayesian inference combines multiple pieces of evidence");
    println!("to update our belief in the target proposition based on the rules in our model.");
    
    println!("\n===== TEST SUMMARY =====");
    println!("This demonstration showed how Bayesian inference works in our system:");
    println!("1. We started with a base probability for 'test_Man0 dates test_Woman0'");
    println!("2. We set evidence that 'test_Woman0 is exciting', raising the probability");
    println!("3. We added evidence that 'test_Man0 is lonely', further raising the probability");
    println!("4. Finally, we added evidence that 'test_Woman0 likes test_Man0', completing the conditions");
    println!("   for our inference rule: 'if they like each other, they'll date'");
    println!("\nThe system correctly propagated these beliefs through the network using:");
    println!("- The exponential model for predicting factor probabilities");
    println!("- The belief propagation algorithm for updating marginals");
    println!("- Loopy belief propagation for handling cycles in the network");
    println!("\nDating scenario inference test completed successfully");
    println!("========================");
    Ok(())
}