//! Examples demonstrating transaction usage in the graph database

use anyhow::{anyhow, Result};
use bayeslog::graph::database::GraphDatabase;
use bayeslog::graph::models::Value;
use rusqlite::params;
use std::collections::HashMap;

fn main() -> Result<()> {
    // Create a new in-memory graph database
    let db = GraphDatabase::new_in_memory()?;
    
    println!("Transaction Example - Successful Commit");
    println!("--------------------------------------");
    
    // Example 1: Successful transaction that commits
    let result: Result<String> = db.with_transaction(|tx| {
        // Add a node within the transaction
        let person_props = serde_json::to_string(&HashMap::from([
            ("name".to_string(), Value::String("Transaction Test".to_string())),
            ("age".to_string(), Value::Integer(42)),
        ]))?;
        
        let id = "transaction-1";
        
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            params![id, "Person", person_props],
        )?;
        
        println!("- Added node within transaction");
        
        // This transaction will commit automatically when the closure returns Ok
        Ok(id.to_string())
    });
    
    match result {
        Ok(id) => {
            println!("- Transaction committed successfully");
            
            // Verify the node exists after commit
            let node = db.get_node(&id)?;
            println!("- Node exists after commit: {}", node.is_some());
            
            if let Some(node) = node {
                println!("- Node label: {}", node.label);
                println!("- Node name: {}", node.properties.get("name").unwrap().to_string());
            }
        },
        Err(e) => println!("- Transaction failed: {}", e),
    }
    
    println!("
Transaction Example - Forced Rollback");
    println!("------------------------------------");
    
    // Example 2: Transaction that rolls back due to an error
    let result: Result<()> = db.with_transaction(|tx| {
        // Add a node that would be rolled back
        let person_props = serde_json::to_string(&HashMap::from([
            ("name".to_string(), Value::String("Will Rollback".to_string())),
        ]))?;
        
        let id = "rollback-test";
        
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            params![id, "Person", person_props],
        )?;
        
        println!("- Added node that will be rolled back");
        
        // Force a rollback by returning an error
        Err(anyhow!("Forced rollback for demonstration"))
    });
    
    match result {
        Ok(_) => println!("- Transaction unexpectedly committed"),
        Err(e) => {
            println!("- Transaction rolled back as expected: {}", e);
            
            // Verify the node doesn't exist after rollback
            let node = db.get_node("rollback-test")?;
            println!("- Node exists after rollback: {}", node.is_some());
        }
    }
    
    println!("
Transaction Example - Multiple Operations");
    println!("---------------------------------------");
    
    // Example 3: Transaction with multiple operations
    let result: Result<()> = db.with_transaction(|tx| {
        // Create multiple nodes and an edge in a single transaction
        let alice_props = serde_json::to_string(&HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
        ]))?;
        
        let bob_props = serde_json::to_string(&HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
        ]))?;
        
        let alice_id = "alice-123";
        let bob_id = "bob-456";
        
        // Add nodes
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            params![alice_id, "Person", alice_props],
        )?;
        
        tx.execute(
            "INSERT INTO nodes (id, label, properties) VALUES (?1, ?2, ?3)",
            params![bob_id, "Person", bob_props],
        )?;
        
        // Add edge
        let edge_props = serde_json::to_string(&HashMap::from([
            ("since".to_string(), Value::Integer(2023)),
        ]))?;
        
        tx.execute(
            "INSERT INTO edges (id, source_id, target_id, label, properties) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params!["friendship-1", alice_id, bob_id, "FRIENDS", edge_props],
        )?;
        
        println!("- Added 2 nodes and 1 edge in transaction");
        
        // All operations succeed, transaction will commit
        Ok(())
    });
    
    match result {
        Ok(_) => {
            println!("- Multi-operation transaction committed");
            
            // Verify both nodes and the edge exist
            let alice = db.get_node("alice-123")?;
            let bob = db.get_node("bob-456")?;
            let edge = db.get_edge("friendship-1")?;
            
            println!("- Alice node exists: {}", alice.is_some());
            println!("- Bob node exists: {}", bob.is_some());
            println!("- Friendship edge exists: {}", edge.is_some());
        },
        Err(e) => println!("- Transaction failed: {}", e),
    }
    
    Ok(())
}
