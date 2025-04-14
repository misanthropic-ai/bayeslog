use anyhow::Result;
use bayeslog::graph::database::GraphDatabase;
use bayeslog::graph::models::Value;
use std::collections::HashMap;

fn main() -> Result<()> {
    // Create an in-memory graph database
    let db = GraphDatabase::new_in_memory()?;
    
    // Use a transaction to add multiple nodes and edges atomically
    db.with_transaction(|tx| {
        // Add Alice node directly using the transaction
        let alice_props = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Integer(30)),
        ]);
        
        let properties_json = serde_json::to_string(&alice_props)?;
        let alice_id = "alice-123".to_string();
        let alice_label = "Person";
        
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            rusqlite::params![alice_id, alice_label, properties_json],
        )?;
        
        // Add Bob node
        let bob_props = HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
            ("age".to_string(), Value::Integer(35)),
        ]);
        
        let properties_json = serde_json::to_string(&bob_props)?;
        let bob_id = "bob-456".to_string();
        let bob_label = "Person";
        
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            rusqlite::params![bob_id, bob_label, properties_json],
        )?;
        
        // Add a KNOWS relationship between Alice and Bob
        let knows_props = HashMap::from([
            ("since".to_string(), Value::String("2020".to_string())),
        ]);
        
        let properties_json = serde_json::to_string(&knows_props)?;
        let edge_id = "knows-789".to_string();
        let edge_label = "KNOWS";
        
        tx.execute(
            "INSERT INTO edges (id, source_id, target_id, label, properties) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![edge_id, alice_id, bob_id, edge_label, properties_json],
        )?;
        
        // This entire operation (adding both nodes and the relationship) will be committed
        // atomically or rolled back completely if any part fails
        Ok(())
    })?;
    
    // Query the nodes to verify they were added
    match db.get_node("alice-123")? {
        Some(node) => println!("Found Alice: {:#?}", node),
        None => println!("Alice not found"),
    }
    
    match db.get_node("bob-456")? {
        Some(node) => println!("Found Bob: {:#?}", node),
        None => println!("Bob not found"),
    }
    
    // Let's try to demonstrate a transaction that fails
    let result = db.with_transaction(|tx| {
        // Try to add an edge between nodes that don't exist
        // This should cause the transaction to roll back
        let invalid_props: HashMap<String, Value> = HashMap::new();
        let properties_json = serde_json::to_string(&invalid_props)?;
        
        tx.execute(
            "INSERT INTO edges (id, source_id, target_id, label, properties) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params!["invalid-edge", "does-not-exist", "also-not-exist", "INVALID", properties_json],
        )?;
        
        // This will fail because of the foreign key constraint
        Ok(())
    });
    
    match result {
        Ok(_) => println!("Transaction succeeded (should not happen)"),
        Err(e) => println!("Transaction failed as expected: {}", e),
    }
    
    Ok(())
}