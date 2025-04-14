use anyhow::Result;
use bayeslog::graph::database::GraphDatabase;
use bayeslog::graph::models::{Direction, Value};
use std::collections::HashMap;

fn main() -> Result<()> {
    // Create an in-memory graph database
    let db = GraphDatabase::new_in_memory()?;
    
    // Create nodes
    let alice_props = HashMap::from([
        ("name".to_string(), Value::String("Alice Smith".to_string())),
        ("age".to_string(), Value::Integer(30)),
    ]);
    let alice_id = db.add_node("Person", alice_props)?;
    println!("Added Alice with ID: {}", alice_id);
    
    let bob_props = HashMap::from([
        ("name".to_string(), Value::String("Bob Johnson".to_string())),
        ("age".to_string(), Value::Integer(35)),
    ]);
    let bob_id = db.add_node("Person", bob_props)?;
    println!("Added Bob with ID: {}", bob_id);
    
    let acme_props = HashMap::from([
        ("name".to_string(), Value::String("ACME Corporation".to_string())),
        ("industry".to_string(), Value::String("Software".to_string())),
    ]);
    let acme_id = db.add_node("Company", acme_props)?;
    println!("Added ACME with ID: {}", acme_id);
    
    // Create relationships
    let works_at_props = HashMap::from([
        ("role".to_string(), Value::String("Software Engineer".to_string())),
        ("start_date".to_string(), Value::String("2022-01-15".to_string())),
    ]);
    let alice_works_at_id = db.add_edge(&alice_id, "WORKS_AT", &acme_id, works_at_props)?;
    println!("Added relationship: Alice WORKS_AT ACME with ID: {}", alice_works_at_id);
    
    let knows_props = HashMap::from([
        ("since".to_string(), Value::String("2020".to_string())),
    ]);
    let alice_knows_bob_id = db.add_edge(&alice_id, "KNOWS", &bob_id, knows_props)?;
    println!("Added relationship: Alice KNOWS Bob with ID: {}", alice_knows_bob_id);
    
    // Retrieve and display node
    if let Some(alice) = db.get_node(&alice_id)? {
        println!("\nRetrieved Alice: {:#?}", alice);
    }
    
    // Get Alice's neighbors
    let alice_neighbors = db.get_neighbors(&alice_id, Direction::Outgoing)?;
    println!("\nAlice's connections:");
    for (node, edge) in alice_neighbors {
        println!("- {} -[{}]-> {}", alice_id, edge.label, node.id);
        println!("  Node label: {}", node.label);
        println!("  Relationship properties: {:#?}", edge.properties);
    }
    
    // Update Alice's age
    let mut alice_update = HashMap::new();
    alice_update.insert("name".to_string(), Value::String("Alice Smith".to_string()));
    alice_update.insert("age".to_string(), Value::Integer(31));
    db.update_node(&alice_id, alice_update)?;
    println!("\nUpdated Alice's age to 31");
    
    // Retrieve and display the updated node
    if let Some(alice) = db.get_node(&alice_id)? {
        println!("Updated Alice: {:#?}", alice);
    }
    
    Ok(())
}