use bayeslog::qbbn::{
    common::{
        graph::InferenceGraph,
        model::InferenceModel,
        proposition_db::RedisBeliefTable,
        resources::ResourceContext,
        setup::parse_configuration_options,
    },
    explorer::diagram_utils::diagram_proposition,
    inference::{
        graph::PropositionGraph,
        engine::{Inferencer, MarginalTable},
    },
};
use std::fs;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("BayesLog Network Visualization");
    
    // Parse command line options
    let config = parse_configuration_options();
    let scenario_name = &config.scenario_name;
    println!("Scenario: {}", scenario_name);
    
    // Create resource context
    let resources = ResourceContext::new(&config)?;
    
    // Get a connection
    let mut connection = resources.connection.lock().unwrap();
    
    // Load the model
    println!("Loading model...");
    let graph = InferenceGraph::new_shared(scenario_name.to_string())?;
    let target = graph.get_target(&mut connection)?;
    
    println!("Target proposition: {:?}", target);
    
    // Create a visualization
    let model = InferenceModel::new_shared(scenario_name.to_string())?;
    let proposition_graph = PropositionGraph::new_shared(&mut connection, &graph, target.clone())?;
    
    // Create a mock belief table with evidence
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
    
    // Run inference
    println!("Running inference...");
    inferencer.do_full_forward_and_backward(&mut connection)?;
    let marginals = inferencer.update_marginals()?;
    
    // Generate visualization HTML
    let html = generate_network_html(&target, &marginals)?;
    
    // Save to file
    fs::write("network.html", html)?;
    println!("Visualization saved to network.html");
    
    Ok(())
}

fn generate_network_html(target: &bayeslog::qbbn::model::objects::Proposition, marginals: &MarginalTable) -> Result<String, Box<dyn Error>> {
    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BayesLog Network</title>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            height: 100%;
            font-family: Arial, sans-serif;
        }}
        
        header {{
            background-color: #333;
            width: 100%;
            padding: 20px 0;
            text-align: center;
            color: white;
        }}
        
        main {{
            padding: 20px;
        }}
        
        .relation {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            border: 2px solid #333;
            border-radius: 5px;
        }}
        
        .relation_name {{
            font-weight: bold;
            background-color: #007bff;
            color: white;
            padding: 5px 10px;
            margin-right: 10px;
        }}
        
        .marginal {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <header>
        <h1>BayesLog Network Visualization</h1>
    </header>
    <main>
        <h2>Target Proposition</h2>
        <div class="proposition">
            {visualization}
        </div>
        
        <h2>Marginals</h2>
        <table border="1">
            <tr>
                <th>Proposition</th>
                <th>Probability</th>
            </tr>
            {marginals_table}
        </table>
    </main>
</body>
</html>
        "#,
        visualization = diagram_proposition(target, Some(marginals)),
        marginals_table = generate_marginals_table(marginals),
    );
    
    Ok(html)
}

fn generate_marginals_table(marginals: &MarginalTable) -> String {
    let mut table = String::new();
    
    for (prop_str, prob) in &marginals.entries {
        table.push_str(&format!(
            "<tr><td>{}</td><td>{:.6}</td></tr>",
            prop_str,
            prob
        ));
    }
    
    table
}