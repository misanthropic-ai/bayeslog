use crate::qbbn::model::objects::{Proposition, PropositionGroup};
use log::trace;
use std::collections::HashMap;

use colored::*;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum GenericNodeType {
    Single(Proposition),
    Group(PropositionGroup),
}

#[derive(PartialEq, Eq, Clone)]
pub struct PropositionNode {
    pub node: GenericNodeType,
    pub underlying_hash: u64,
}

fn hash_proposition(proposition: &Proposition) -> u64 {
    let mut hasher = DefaultHasher::new();
    proposition.hash(&mut hasher);
    hasher.finish() // This returns the hash as u64
}

fn hash_group(group: &PropositionGroup) -> u64 {
    let mut hasher = DefaultHasher::new();
    group.hash(&mut hasher);
    hasher.finish() // This returns the hash as u64
}

impl Hash for PropositionNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.underlying_hash.hash(state);
    }
}

impl PropositionNode {
    pub fn from_single(proposition: &Proposition) -> PropositionNode {
        let underlying_hash = hash_proposition(proposition);
        PropositionNode {
            node: GenericNodeType::Single(proposition.clone()),
            underlying_hash,
        }
    }

    pub fn from_group(group: &PropositionGroup) -> PropositionNode {
        let underlying_hash = hash_group(group);
        trace!("got hash {} {:?}", underlying_hash, group);
        PropositionNode {
            node: GenericNodeType::Group(group.clone()),
            underlying_hash,
        }
    }

    pub fn debug_string(&self) -> String {
        let string_part = match &self.node {
            GenericNodeType::Single(proposition) => proposition.debug_string(),
            GenericNodeType::Group(group) => group.debug_string(),
        };
        string_part.to_string()
    }

    pub fn is_single(&self) -> bool {
        matches!(self.node, GenericNodeType::Single(_))
    }

    pub fn is_group(&self) -> bool {
        matches!(self.node, GenericNodeType::Group(_))
    }

    pub fn extract_single(&self) -> Proposition {
        match &self.node {
            GenericNodeType::Single(proposition) => proposition.clone(),
            _ => panic!("This is not a single."),
        }
    }

    pub fn extract_group(&self) -> PropositionGroup {
        match &self.node {
            GenericNodeType::Group(group) => group.clone(),
            _ => panic!("This is not a group."),
        }
    }
}
impl fmt::Debug for PropositionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.debug_string())
    }
}

#[derive(Debug, Clone)]

pub struct HashMapBeliefTable {
    pi_values: HashMap<(PropositionNode, usize), f64>,
    lambda_values: HashMap<(PropositionNode, usize), f64>,
    pi_messages: HashMap<(PropositionNode, PropositionNode, usize), f64>,
    lambda_messages: HashMap<(PropositionNode, PropositionNode, usize), f64>,
    bfs_order: Vec<PropositionNode>,
}

fn print_sorted_map(
    map: &HashMap<(PropositionNode, usize), f64>,
    bfs_order: &Vec<PropositionNode>,
) {
    for proposition in bfs_order {
        let key = (proposition.clone(), 1);
        let prob_true = map.get(&key).unwrap();
        let prob_false = 1.0 - prob_true;
        let formatted_prob_true = format!("{:.8}", prob_true);
        let formatted_prob_false = format!("{:.8}", prob_false);
        println!(
            "{:<12} {:<12} {}",
            formatted_prob_true.green(),
            formatted_prob_false.red(),
            proposition.debug_string()
        );
    }
}

fn print_sorted_messages(
    map: &HashMap<(PropositionNode, PropositionNode, usize), f64>,
    bfs_order: &Vec<PropositionNode>,
) {
    for from in bfs_order {
        for to in bfs_order {
            let key = (from.clone(), to.clone(), 1);
            if let Some(&prob_true) = map.get(&key) {
                let prob_false = 1.0 - prob_true;
                let formatted_prob_true = format!("{:.8}", prob_true);
                let formatted_prob_false = format!("{:.8}", prob_false);
                println!(
                    "{:<12} {:<12} {:<20} {}",
                    formatted_prob_true.green(),
                    formatted_prob_false.red(),
                    from.debug_string(),
                    to.debug_string()
                );
            }
        }
    }
}

impl HashMapBeliefTable {
    pub fn print_table(&self, table_name: &str) {
        match table_name {
            "pv" => {
                println!("PI VALUES");
                print_sorted_map(&self.pi_values, &self.bfs_order);
            }
            "lv" => {
                println!("LAMBDA VALUES");
                print_sorted_map(&self.lambda_values, &self.bfs_order);
            }
            "pm" => {
                println!("PI MESSAGES");
                print_sorted_messages(&self.pi_messages, &self.bfs_order);
            }
            "lm" => {
                println!("LAMBDA MESSAGES");
                print_sorted_messages(&self.lambda_messages, &self.bfs_order);
            }
            _ => println!("Table not recognized."),
        };
    }
}

impl HashMapBeliefTable {
    // Constructor to create a new instance
    pub fn new(bfs_order: Vec<PropositionNode>) -> Self {
        HashMapBeliefTable {
            pi_values: HashMap::new(),
            lambda_values: HashMap::new(),
            pi_messages: HashMap::new(),
            lambda_messages: HashMap::new(),
            bfs_order,
        }
    }

    // Getter for pi values
    pub fn get_pi_value(&self, node: &PropositionNode, outcome: usize) -> Option<f64> {
        let key = (node.clone(), outcome);
        self.pi_values.get(&key).cloned()
    }

    // Setter for pi values
    pub fn set_pi_value(&mut self, node: &PropositionNode, outcome: usize, value: f64) {
        let key = (node.clone(), outcome);
        self.pi_values.insert(key, value);
    }

    // Getter for lambda values
    pub fn get_lambda_value(&self, node: &PropositionNode, outcome: usize) -> Option<f64> {
        let key = (node.clone(), outcome);
        self.lambda_values.get(&key).cloned()
    }

    // Setter for lambda values
    pub fn set_lambda_value(&mut self, node: &PropositionNode, outcome: usize, value: f64) {
        let key = (node.clone(), outcome);
        self.lambda_values.insert(key, value);
    }

    // Getter for pi messages
    pub fn get_pi_message(
        &self,
        from: &PropositionNode,
        to: &PropositionNode,
        outcome: usize,
    ) -> Option<f64> {
        let key = (from.clone(), to.clone(), outcome);
        self.pi_messages.get(&key).cloned()
    }

    // Setter for pi messages
    pub fn set_pi_message(
        &mut self,
        from: &PropositionNode,
        to: &PropositionNode,
        outcome: usize,
        value: f64,
    ) {
        let key = (from.clone(), to.clone(), outcome);
        self.pi_messages.insert(key, value);
    }

    // Getter for lambda messages
    pub fn get_lambda_message(
        &self,
        from: &PropositionNode,
        to: &PropositionNode,
        outcome: usize,
    ) -> Option<f64> {
        let key = (from.clone(), to.clone(), outcome);
        self.lambda_messages.get(&key).cloned()
    }

    // Setter for lambda messages
    pub fn set_lambda_message(
        &mut self,
        from: &PropositionNode,
        to: &PropositionNode,
        outcome: usize,
        value: f64,
    ) {
        let key = (from.clone(), to.clone(), outcome);
        self.lambda_messages.insert(key, value);
    }
}

pub struct VariableAssignment {
    pub assignment_map: HashMap<PropositionNode, bool>,
}

impl VariableAssignment {
    pub fn new(assignment_map: HashMap<PropositionNode, bool>) -> VariableAssignment {
        VariableAssignment { assignment_map }
    }
}

pub struct FactorProbabilityTable {
    pub pairs: Vec<(VariableAssignment, f64)>,
}

impl FactorProbabilityTable {
    pub fn new(pairs: Vec<(VariableAssignment, f64)>) -> FactorProbabilityTable {
        FactorProbabilityTable { pairs }
    }
}
