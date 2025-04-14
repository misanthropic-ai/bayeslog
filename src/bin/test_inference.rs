use bayeslog::qbbn::common::setup::parse_configuration_options;
use bayeslog::qbbn::common::resources::ResourceContext;
use bayeslog::qbbn::common::proposition_db::RedisBeliefTable;
use bayeslog::qbbn::common::model::InferenceModel;
use bayeslog::qbbn::common::graph::InferenceGraph;
use bayeslog::qbbn::inference::graph::PropositionGraph;
use bayeslog::qbbn::inference::engine::Inferencer;
use bayeslog::qbbn::model::creators::{constant, relation, proposition, sub, variable_argument};
use bayeslog::qbbn::model::objects::Domain;
use bayeslog::qbbn::scenarios::factory::ScenarioMakerFactory;
use bayeslog::qbbn::common::train::setup_and_train;
use std::borrow::Borrow;
use std::error::Error;
use std::sync::Arc;
use bayeslog::qbbn::model::exponential::ExponentialModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting BayesLog Inference Test");
    
    // Parse command line options
    let config = parse_configuration_options();
    let scenario_name = &config.scenario_name;
    println!("Scenario: {}", scenario_name);
    
    // Create resource context
    let resources = ResourceContext::new(&config)?;
    println!("Created resources");
    
    // Setup and train the model if needed
    println!("Setting up and training the model...");
    let scenario_maker = ScenarioMakerFactory::new_shared(scenario_name)?;
    setup_and_train(&resources, scenario_maker.borrow(), scenario_name)?;
    println!("Training completed");
    
    // Get a connection
    let mut connection = resources.connection.lock().unwrap();
    
    // Create and register the target proposition before creating the Arc-wrapped model
    println!("Creating target proposition...");
    
    // Create the relation
    let man_domain = Domain::MAN.to_string();
    let exciting_relation = relation(
        "exciting".to_string(),
        vec![variable_argument(man_domain.clone())],
    );
    
    // Create a proposition for a test entity
    let entity = constant(man_domain.clone(), "test_Man0".to_string());
    
    // Create the target proposition: exciting(test_Man0)
    let target = proposition(
        exciting_relation.clone(),
        vec![sub(entity.clone())]
    );
    println!("Target proposition: {:?}", target);
    
    // Create a mutable InferenceGraph and register the target
    let mut graph = InferenceGraph::new_mutable(scenario_name.to_string())?;
    graph.register_target(&mut connection, &target)?;
    
    // Now create the InferenceModel manually with our pre-registered graph
    println!("Creating model with pre-registered target...");
    let graph_arc = Arc::new(*graph);
    let model_arc = ExponentialModel::new_shared(scenario_name.to_string())?;
    let model = Arc::new(InferenceModel {
        graph: graph_arc.clone(),
        model: model_arc,
    });
    
    // Create the proposition graph
    println!("Creating proposition graph...");
    let proposition_graph = PropositionGraph::new_shared(&mut connection, &graph_arc, target.clone())?;
    
    // Create the belief table
    let fact_memory = RedisBeliefTable::new_shared(scenario_name.to_string())?;
    
    // Create the inferencer
    let mut inferencer = Inferencer::new_mutable(
        model.clone(),
        proposition_graph.clone(),
        fact_memory.clone(),
    )?;
    
    // Initialize the inference chart
    inferencer.initialize_chart(&mut connection)?;
    println!("Inference chart initialized");
    
    // Run initial inference
    println!("Running initial inference...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    let initial_marginals = inferencer.update_marginals()?;
    
    // Print initial marginals
    println!("Initial marginals (proposition = probability):");
    println!("{}", initial_marginals.render_marginal_table());
    
    // Set evidence for a proposition if this is the one_var scenario
    if scenario_name == "one_var" {
        println!("\nSetting evidence for a test proposition...");
        
        // Create a proposition for a different test entity (Man3)
        let test_entity = constant(man_domain.clone(), "test_Man3".to_string());
        
        // Create the proposition: exciting(test_Man3)
        let test_prop = proposition(
            exciting_relation.clone(),
            vec![sub(test_entity)]
        );
        
        // Set the evidence to true (1.0)
        fact_memory.store_proposition_probability(&mut connection, &test_prop, 1.0)?;
        println!("Set evidence: exciting(test_Man3) = 1.0");
        
        // Instead of trying to find the node directly, we'll run full inference
        println!("\nRunning full inference after setting evidence...");
        inferencer.do_full_forward_and_backward(&mut connection)?;
        
        // Get updated marginals
        let updated_marginals = inferencer.update_marginals()?;
        
        // Print updated marginals
        println!("Updated marginals after setting evidence:");
        println!("{}", updated_marginals.render_marginal_table());
    }
    
    println!("\nInference test completed successfully");
    Ok(())
}