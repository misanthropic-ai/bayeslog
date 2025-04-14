//! Basic usage examples for the graph database component

use anyhow::Result;
use bayeslog::graph::database::GraphDatabase;
use bayeslog::graph::models::{Direction, Value};
use std::collections::HashMap;

fn main() -> Result<()> {
    // Create a new in-memory graph database
    let db = GraphDatabase::new_in_memory()?;
    
    // Add nodes with properties
    let person_props = HashMap::from([
        ("name".to_string(), Value::String("Alice".to_string())),
        ("age".to_string(), Value::Integer(30)),
    ]);
    
    let company_props = HashMap::from([
        ("name".to_string(), Value::String("ACME Corp".to_string())),
        ("founded".to_string(), Value::Integer(1985)),
    ]);
    
    // Add nodes and get their IDs
    let alice_id = db.add_node("Person", person_props)?;
    let company_id = db.add_node("Company", company_props)?;
    
    println!("Added Person node with ID: {}", alice_id);
    println!("Added Company node with ID: {}", company_id);
    
    // Add an edge connecting the nodes
    let works_at_props = HashMap::from([
        ("since".to_string(), Value::Integer(2020)),
        ("role".to_string(), Value::String("Software Engineer".to_string())),
    ]);
    
    let edge_id = db.add_edge(&alice_id, "WORKS_AT", &company_id, works_at_props)?;
    println!("Added WORKS_AT edge with ID: {}", edge_id);
    
    // Retrieve the nodes
    let alice = db.get_node(&alice_id)?.unwrap();
    let company = db.get_node(&company_id)?.unwrap();
    
    println!("\nRetrieved person: {}", alice.properties.get("name").unwrap().to_string());
    println!("Retrieved company: {}", company.properties.get("name").unwrap().to_string());
    
    // Update a node's properties
    let mut updated_props = HashMap::new();
    updated_props.insert("name".to_string(), Value::String("Alice Smith".to_string()));
    updated_props.insert("age".to_string(), Value::Integer(31));
    updated_props.insert("department".to_string(), Value::String("Engineering".to_string()));
    
    db.update_node(&alice_id, updated_props)?;
    println!("\nUpdated person's properties");
    
    // Get the updated node
    let updated_alice = db.get_node(&alice_id)?.unwrap();
    println!("Updated name: {}", updated_alice.properties.get("name").unwrap().to_string());
    println!("Updated age: {}", updated_alice.properties.get("age").unwrap().as_integer().unwrap());
    println!("New property - department: {}", updated_alice.properties.get("department").unwrap().to_string());
    
    // Find Alice's neighbors (outgoing relationships)
    println!("\nFinding neighbors (outgoing relationships):");
    let neighbors = db.get_neighbors(&alice_id, Direction::Outgoing)?;
    for (node, edge) in neighbors {
        println!("  - {} --[{}]--> {}", 
                 updated_alice.properties.get("name").unwrap().to_string(),
                 edge.label,
                 node.properties.get("name").unwrap().to_string());
        
        println!("    Role: {}", edge.properties.get("role").unwrap().to_string());
    }
    
    // Find nodes by label
    println!("\nFinding nodes by label 'Person':");
    let persons = db.find_nodes_by_label("Person")?;
    for person in persons {
        println!("  - {} (id: {})", person.properties.get("name").unwrap().to_string(), person.id);
    }
    
    // Find nodes by property
    println!("\nFinding nodes with 'Engineering' in properties:");
    let engineering_nodes = db.find_nodes_by_property("department", "Engineering")?;
    for node in engineering_nodes {
        println!("  - {} (label: {})", node.properties.get("name").unwrap().to_string(), node.label);
    }
    
    Ok(())
}
