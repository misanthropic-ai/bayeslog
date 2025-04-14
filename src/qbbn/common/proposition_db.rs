use crate::qbbn::{
    common::{interface::BeliefTable, redis::map_insert},
    inference::table::PropositionNode,
    model::{
        objects::{
            Proposition,
            existence_predicate_name,
        },
    },
};
use log::trace;
use crate::qbbn::common::redis::MockConnection as Connection;
use std::{collections::HashMap, error::Error, sync::{Arc, Mutex}};

use super::redis::map_get;

pub struct RedisBeliefTable {
    namespace: String,
}

impl RedisBeliefTable {
    pub fn new_mutable(namespace: String) -> Result<Box<dyn BeliefTable>, Box<dyn Error>> {
        Ok(Box::new(RedisBeliefTable { namespace }))
    }
    pub fn new_shared(namespace: String) -> Result<Arc<dyn BeliefTable>, Box<dyn Error>> {
        Ok(Arc::new(RedisBeliefTable { namespace }))
    }
    pub const PROBABILITIES_KEY: &'static str = "probabilities";
}

impl BeliefTable for RedisBeliefTable {
    // Return Some if the probability exists in the table, or else None.
    fn get_proposition_probability(
        &self,
        connection: &mut Connection,
        proposition: &Proposition,
    ) -> Result<Option<f64>, Box<dyn Error>> {
        if proposition.predicate.relation.relation_name == existence_predicate_name() {
            return Ok(Some(1f64));
        }
        let hash_string = proposition.predicate.hash_string();
        let probability_opt = map_get(
             connection,
            &self.namespace,
            Self::PROBABILITIES_KEY,
            &hash_string,
        )?;
        
        match probability_opt {
            Some(probability_record) => {
                let probability = probability_record
                    .parse::<f64>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                Ok(Some(probability))
            },
            None => Ok(None)
        }
    }

    fn store_proposition_probability(
        &self,
        connection: &mut Connection,
        proposition: &Proposition,
        probability: f64,
    ) -> Result<(), Box<dyn Error>> {
        trace!("GraphicalModel::store_proposition_probability - Start. Input proposition: {:?}, probability: {}", proposition, probability);
        let hash_string = proposition.predicate.hash_string();
        map_insert(
            connection,
            &self.namespace,
            Self::PROBABILITIES_KEY,
            &hash_string,
            &probability.to_string(),
        )?;
        Ok(())
    }
}

pub struct EmptyBeliefTable;

impl EmptyBeliefTable {
    pub fn new_shared(_namespace: &str) -> Result<Arc<dyn BeliefTable>, Box<dyn Error>> {
        Ok(Arc::new(EmptyBeliefTable {}))
    }
}

impl BeliefTable for EmptyBeliefTable {
    // Return Some if the probability exists in the table, or else None.
    fn get_proposition_probability(
        &self,
        _connection: &mut Connection,
        proposition: &Proposition,
    ) -> Result<Option<f64>, Box<dyn Error>> {
        if proposition.predicate.relation.relation_name == existence_predicate_name() {
            return Ok(Some(1f64));
        }
        Ok(None)
    }

    fn store_proposition_probability(
        &self,
        _connection: &mut Connection,
        _proposition: &Proposition,
        _probability: f64,
    ) -> Result<(), Box<dyn Error>> {
        panic!("Shouldn't call this.")
    }
}

/// HashMapBeliefTable provides thread-safe storage for proposition probabilities
/// It uses a Mutex to ensure safe concurrent access from multiple threads
pub struct HashMapBeliefTable {
    /// Thread-safe storage for proposition probabilities
    evidence: Mutex<HashMap<PropositionNode, f64>>,
}

impl HashMapBeliefTable {
    pub fn new() -> Arc<HashMapBeliefTable> {
        Arc::new(HashMapBeliefTable {
            evidence: Mutex::new(HashMap::new()),
        })
    }

    pub fn clear(&self, node: &PropositionNode) {
        if let Ok(mut evidence) = self.evidence.lock() {
            evidence.remove(node);
        }
    }
}

impl BeliefTable for HashMapBeliefTable {
    fn get_proposition_probability(
        &self,
        _connection: &mut Connection,
        proposition: &Proposition,
    ) -> Result<Option<f64>, Box<dyn Error>> {
        if proposition.predicate.relation.relation_name == existence_predicate_name() {
            return Ok(Some(1f64));
        }
        let node = PropositionNode::from_single(proposition);
        if let Ok(map) = self.evidence.lock() {
            let result = map.get(&node);
            Ok(result.copied())
        } else {
            // Return None if we can't acquire the lock
            Ok(None)
        }
    }

    fn store_proposition_probability(
        &self,
        _connection: &mut Connection,
        proposition: &Proposition,
        probability: f64,
    ) -> Result<(), Box<dyn Error>> {
        let node = PropositionNode::from_single(proposition);
        if let Ok(mut map) = self.evidence.lock() {
            map.insert(node, probability);
            Ok(())
        } else {
            Err("Could not acquire lock for HashMapBeliefTable".into())
        }
    }
}
