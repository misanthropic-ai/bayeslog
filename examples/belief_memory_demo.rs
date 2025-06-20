use bayeslog::{GraphDatabase, BeliefMemory};
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("=== BeliefMemory Demo: LLM Agent Memory ===\n");
    
    // Create an in-memory graph database
    let graph_db = Arc::new(GraphDatabase::new_in_memory()?);
    let namespace = "agent_memory";
    
    // Create BeliefMemory instance
    let mut memory = BeliefMemory::new(graph_db, namespace)?;
    
    println!("📝 Adding beliefs about entities...\n");
    
    // Scenario: An LLM agent learning about people and their preferences
    
    // Add belief: Alice likes Pizza (high confidence)
    let mut args1 = HashMap::new();
    args1.insert("subject".to_string(), "Alice".to_string());
    args1.insert("object".to_string(), "Pizza".to_string());
    let prop1 = memory.add_proposition_with_prior("likes", args1, 0.8)?;
    println!("✓ Added: Alice likes Pizza (belief: 0.8)");
    
    // Add belief: Bob likes Sushi (medium confidence)
    let mut args2 = HashMap::new();
    args2.insert("subject".to_string(), "Bob".to_string());
    args2.insert("object".to_string(), "Sushi".to_string());
    let prop2 = memory.add_proposition_with_prior("likes", args2, 0.6)?;
    println!("✓ Added: Bob likes Sushi (belief: 0.6)");
    
    // Add belief: Alice works_at TechCorp (high confidence)
    let mut args3 = HashMap::new();
    args3.insert("person".to_string(), "Alice".to_string());
    args3.insert("company".to_string(), "TechCorp".to_string());
    let prop3 = memory.add_proposition_with_prior("works_at", args3, 0.9)?;
    println!("✓ Added: Alice works_at TechCorp (belief: 0.9)");
    
    println!("\n🔄 Updating beliefs based on new observations...\n");
    
    // Update: We observe Alice eating Pizza (confirms our belief)
    memory.update_belief_from_observation(&prop1, true, 0.95)?;
    println!("✓ Updated: Observed Alice eating Pizza (belief increased)");
    
    // Update: We observe Bob not eating Sushi (contradicts our belief)
    memory.update_belief_from_observation(&prop2, false, 0.7)?;
    println!("✓ Updated: Observed Bob NOT eating Sushi (belief decreased)");
    
    println!("\n🔍 Querying beliefs about entities...\n");
    
    // Query all beliefs about Alice
    let alice_beliefs = memory.query_beliefs_about("Alice")?;
    println!("Beliefs about Alice:");
    for (prop_text, belief) in alice_beliefs {
        println!("  - {} (belief: {:.2})", prop_text, belief);
    }
    
    // Query all beliefs about Bob
    let bob_beliefs = memory.query_beliefs_about("Bob")?;
    println!("\nBeliefs about Bob:");
    for (prop_text, belief) in bob_beliefs {
        println!("  - {} (belief: {:.2})", prop_text, belief);
    }
    
    println!("\n💾 Saving belief memory for later use...\n");
    
    // Save the belief memory
    let save_path = "/tmp/agent_memory.json";
    match memory.save_to_file(save_path) {
        Ok(_) => println!("✓ Saved belief memory to: {}", save_path),
        Err(e) => println!("⚠️  Could not save (not fully implemented): {}", e),
    }
    
    println!("\n✨ BeliefMemory demo complete!");
    println!("\nKey features demonstrated:");
    println!("- Adding propositions with prior beliefs");
    println!("- Updating beliefs based on observations");
    println!("- Querying beliefs about specific entities");
    println!("- Graph-based storage with entity linking");
    println!("\nThis provides a perfect foundation for LLM agent memory!");
    
    Ok(())
}