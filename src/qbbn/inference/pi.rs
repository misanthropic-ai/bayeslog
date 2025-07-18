use crate::qbbn::common::redis::MockConnection as Connection;
use log::trace;

use super::{
    engine::{compute_each_combination, Inferencer},
    table::PropositionNode,
};
use crate::qbbn::model::weights::CLASS_LABELS;
use std::error::Error;

impl Inferencer {
    pub fn do_pi_traversal(&mut self, connection: &mut Connection) -> Result<(), Box<dyn Error>> {
        let bfs_order = self.bfs_order.clone();
        for node in &bfs_order {
            self.pi_visit_node(connection, node)?;
        }
        Ok(())
    }

    pub fn pi_visit_node(
        &mut self,
        connection: &mut Connection,
        from_node: &PropositionNode,
    ) -> Result<(), Box<dyn Error>> {
        if !self.is_root(from_node) {
            let is_observed = self.is_observed(connection, from_node)?;
            if is_observed {
                self.pi_set_from_evidence(connection, from_node)?;
            } else {
                self.pi_compute_value(connection, from_node)?;
            }
        } else {
            self.pi_compute_root(from_node)?;
        }
        self.pi_send_messages(from_node)?;
        Ok(())
    }

    fn pi_compute_root(&mut self, node: &PropositionNode) -> Result<(), Box<dyn Error>> {
        let root = node.extract_single();
        self.data
            .set_pi_value(&PropositionNode::from_single(&root), 1, 1.0f64);
        self.data
            .set_pi_value(&PropositionNode::from_single(&root), 0, 0.0f64);
        Ok(())
    }

    pub fn pi_set_from_evidence(
        &mut self,
        connection: &mut Connection,
        node: &PropositionNode,
    ) -> Result<(), Box<dyn Error>> {
        let as_single = node.extract_single();
        let probability = self
            .fact_memory
            .get_proposition_probability(connection, &as_single)?
            .unwrap();
        self.data.set_pi_value(node, 1, probability);
        self.data.set_pi_value(node, 0, 1f64 - probability);
        Ok(())
    }

    pub fn pi_compute_value(
        &mut self,
        connection: &mut Connection,
        node: &PropositionNode,
    ) -> Result<(), Box<dyn Error>> {
        let is_observed = self.is_observed(connection, node)?;
        assert!(!is_observed);
        let parent_nodes = self.proposition_graph.get_all_backward(node);
        let all_combinations = compute_each_combination(&parent_nodes);
        let mut sum_true = 0f64;
        let mut sum_false = 0f64;
        for combination in &all_combinations {
            let mut product = 1f64;
            for parent_node in parent_nodes.iter() {
                let boolean_outcome = combination.get(parent_node).unwrap();
                let usize_outcome = if *boolean_outcome { 1 } else { 0 };
                let pi_x_z = self
                    .data
                    .get_pi_message(parent_node, node, usize_outcome)
                    .unwrap();
                trace!(
                    "getting pi message parent_node {:?}, node {:?}, usize_outcome {}, pi_x_z {}",
                    &parent_node,
                    &node,
                    usize_outcome,
                    pi_x_z,
                );
                product *= pi_x_z;
            }
            let true_marginal = self.score_factor_assignment(connection, &parent_nodes, combination, node)?;
            let false_marginal = 1f64 - true_marginal;
            sum_true += true_marginal * product;
            sum_false += false_marginal * product;
        }
        self.data.set_pi_value(node, 1, sum_true);
        self.data.set_pi_value(node, 0, sum_false);
        Ok(())
    }

    pub fn pi_send_messages(&mut self, node: &PropositionNode) -> Result<(), Box<dyn Error>> {
        let forward_groups = self.proposition_graph.get_all_forward(node);
        for (this_index, to_node) in forward_groups.iter().enumerate() {
            for class_label in &CLASS_LABELS {
                let mut lambda_part = 1f64;
                for (other_index, other_child) in forward_groups.iter().enumerate() {
                    if other_index != this_index {
                        let this_lambda = self
                            .data
                            .get_lambda_message(other_child, node, *class_label)
                            .unwrap();
                        lambda_part *= this_lambda;
                    }
                }
                let pi_part = self.data.get_pi_value(node, *class_label).unwrap();
                let message = pi_part * lambda_part;
                self.data
                    .set_pi_message(node, to_node, *class_label, message);
            }
        }
        Ok(())
    }
}
