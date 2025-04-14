use std::{collections::HashMap, sync::Arc};
use log::trace;

// use colored::Colorize; // This can be re-enabled if needed
use crate::qbbn::common::redis::MockConnection as Connection;

use crate::qbbn::{
    common::proposition_db::HashMapBeliefTable,
    inference::{
        graph::PropositionGraph,
        engine::Inferencer,
        table::{PropositionNode},
    },
};

use super::interface::BeliefTable;

pub struct ReplState {
    pub inferencer: Box<Inferencer>,
    pub fact_memory: Arc<HashMapBeliefTable>,
    /// Relative set by the `print_ordering` last time it serialized an ordering.
    pub question_index: HashMap<u64, PropositionNode>,
    pub proposition_index: HashMap<String, PropositionNode>,
}

impl ReplState {
    pub fn new(mut inferencer: Box<Inferencer>) -> ReplState {
        let fact_memory = HashMapBeliefTable::new();
        inferencer.fact_memory = fact_memory.clone();
        let proposition_index = make_proposition_map(&inferencer.proposition_graph);
        ReplState {
            inferencer,
            fact_memory,
            question_index: HashMap::new(),
            proposition_index,
        }
    }

    pub fn set_pairs_by_name(
        &mut self,
        connection: &mut Connection,
        pairs: &Vec<(&str, f64)>,
    ) -> Option<PropositionNode> {
        assert!(pairs.len() <= 1);
        if let Some(pair) = pairs.iter().next() {
            let key = pair.0.to_string();
            trace!("key {key}");
            let node = self.proposition_index.get(&key).unwrap();
            let prop = node.extract_single();
            trace!("setting {} to {}", &key, pair.1);
            self.fact_memory
                .store_proposition_probability(connection, &prop, pair.1)
                .unwrap();
            self.inferencer
                .do_fan_out_from_node(connection, node)
                .unwrap();
            return Some(node.clone());
        }
        None
    }
}

fn make_proposition_map(graph: &PropositionGraph) -> HashMap<String, PropositionNode> {
    let bfs = graph.get_bfs_order();
    let mut result = HashMap::new();
    for node in bfs.iter() {
        let name = node.debug_string();
        trace!("name_key: {}", &name);
        result.insert(name, node.clone());
    }
    result
}
