use std::error::Error;

use crate::qbbn::common::redis::MockConnection as Connection;

use crate::qbbn::common::{model::InferenceModel, proposition_db::EmptyBeliefTable, test::ReplState};

use super::{graph::PropositionGraph, engine::{Inferencer, MarginalTable}, table::PropositionNode};

fn setup_test_scenario(
    connection: &mut Connection,
    scenario_name: &str,
    test_scenario: &str,
    repl_state: &mut ReplState,
) -> Result<Option<PropositionNode>, Box<dyn Error>> {
    let pairs = match (scenario_name, test_scenario) {
        ("dating_simple", "prior") => vec![],
        ("dating_simple", "jack_lonely") => vec![("lonely[sub=test_Man0]", 1f64)],
        ("dating_simple", "they_date") => vec![("date[obj=test_Woman0,sub=test_Man0]", 1f64)],
        ("dating_simple", "jack_likes") => vec![("like[obj=test_Woman0,sub=test_Man0]", 1f64)],
        ("dating_simple", "jill_likes") => vec![("like[obj=test_Man0,sub=test_Woman0]", 1f64)],
        ("dating_triangle", "prior") => vec![("charming[sub=test_Man0]", 1f64)],
        ("dating_triangle", "charming") => vec![("charming[sub=test_Man0]", 1f64)],
        ("dating_triangle", "baller") => vec![("baller[sub=test_Man0]", 1f64)],
        ("long_chain", "prior") => vec![],
        ("long_chain", "set_0_1") => vec![("alpha0[sub=test_Man0]", 1f64)],
        ("long_chain", "set_n_1") => vec![("alpha10[sub=test_Man0]", 1f64)],
        ("mid_chain", "set_0_1") => vec![("alpha0[sub=test_Man0]", 1f64)],
        ("mid_chain", "set_n_1") => vec![("alpha4[sub=test_Man0]", 1f64)],
        _ => panic!("Case name not recognized"),
    };
    let r = repl_state.set_pairs_by_name(connection, &pairs);
    Ok(r)
}

pub fn run_inference_rounds(
    connection: &mut Connection,
    scenario_name: &str,
    test_scenario: &str,
) -> Result<Vec<MarginalTable>, Box<dyn Error>> {
    let model = InferenceModel::new_shared(scenario_name.to_string()).unwrap();
    let fact_memory = EmptyBeliefTable::new_shared(scenario_name)?;
    let target = model.graph.get_target(connection)?;
    let proposition_graph = PropositionGraph::new_shared(connection, &model.graph, target)?;
    proposition_graph.visualize();
    let mut inferencer =
        Inferencer::new_mutable(model.clone(), proposition_graph.clone(), fact_memory)?;
    inferencer.initialize_chart(connection)?;
    let mut repl = ReplState::new(inferencer);
    let mut buffer = vec![];
    buffer.push(repl.inferencer.log_table_to_file()?);
    let evidence_node = setup_test_scenario(connection, scenario_name, test_scenario, &mut repl)?;
    if evidence_node.is_some() {
        for _i in 0..50 {
            repl.inferencer
                .do_fan_out_from_node(connection, &evidence_node.clone().unwrap())?;
            buffer.push(repl.inferencer.log_table_to_file()?);
        }
    } else {
        for _i in 0..50 {
            repl.inferencer
                .do_full_forward_and_backward(connection)?;
            buffer.push(repl.inferencer.log_table_to_file()?);
        }
    }
    Ok(buffer)
}
