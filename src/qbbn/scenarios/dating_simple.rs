use crate::qbbn::common::graph::InferenceGraph;
// use crate::qbbn::common::interface::BeliefTable;
use crate::qbbn::common::proposition_db::RedisBeliefTable;
use crate::qbbn::common::resources::{ResourceContext};
use crate::qbbn::common::train::TrainingPlan;
use crate::qbbn::model::creators::{predicate, relation, variable_argument};
use crate::qbbn::{
    common::interface::ScenarioMaker,
    model::{
        creators::{conjunction, constant, implication, obj, proposition, sub, variable},
        objects::{Domain, Entity, RoleMap},
    },
};
use rand::Rng; // Import Rng trait
use std::{collections::HashMap, error::Error};
#[allow(dead_code)]
fn cointoss() -> f64 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    if rng.r#gen::<f64>() < 0.5 {
        1.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn weighted_cointoss(threshold: f64) -> f64 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    if rng.r#gen::<f64>() < threshold {
        1.0
    } else {
        0.0
    }
}

pub struct SimpleDating {}

impl ScenarioMaker for SimpleDating {
    fn setup_scenario(&self, resources: &ResourceContext) -> Result<(), Box<dyn Error>> {
        let mut connection = resources.connection.lock().unwrap();
        let namespace = "dating_simple".to_string();
        let mut graph = InferenceGraph::new_mutable(namespace.clone())?;
        let proposition_db = RedisBeliefTable::new_mutable(namespace.clone())?;
        let mut plan = TrainingPlan::new(namespace.clone())?;
        // Use the configured number of entities per domain from the resource context
        let total_members_each_class = resources.config.entities_per_domain as usize;
        let entity_domains = [Domain::MAN.to_string(), Domain::WOMAN.to_string()];

        // Retrieve entities in the Man domain
        let jack_domain = Domain::MAN.to_string(); // Convert enum to string and make lowercase
        let jacks: Vec<Entity> = graph.get_entities_in_domain(&mut connection, &jack_domain)?;
        println!("Initial number of jacks: {}", jacks.len());
        graph.register_domain(&mut connection, &jack_domain)?;
        // Retrieve entities in the Woman domain
        let jill_domain = Domain::WOMAN.to_string(); // Convert enum to string and make lowercase
        let jills = graph.get_entities_in_domain(&mut connection, &jill_domain)?;
        println!("Initial number of jills: {}", jills.len());
        graph.register_domain(&mut connection, &jill_domain)?;

        let exciting_jill_relation = relation(
            "exciting".to_string(),
            vec![variable_argument(jill_domain.clone())],
        );
        graph.register_relation(&mut connection, &exciting_jill_relation)?;
        println!("exciting: {}", jills.len());
        let lonely_jack_relation = relation(
            "lonely".to_string(),
            vec![variable_argument(jack_domain.clone())],
        );
        graph.register_relation(&mut connection, &lonely_jack_relation)?;
        println!("lonely jack: {}", jills.len());
        let lonely_jill_relation = relation(
            "lonely".to_string(),
            vec![variable_argument(jill_domain.clone())],
        );
        graph.register_relation(&mut connection, &lonely_jill_relation)?;
        println!("lonely jill: {}", jills.len());
        let jack_like_jill_relation = relation(
            "like".to_string(),
            vec![
                variable_argument(jack_domain.clone()),
                variable_argument(jill_domain.clone()),
            ],
        );
        graph.register_relation(&mut connection, &jack_like_jill_relation)?;
        println!("like jack jill: {}", jills.len());
        let jill_like_jack_relation = relation(
            "like".to_string(),
            vec![
                variable_argument(jill_domain.clone()),
                variable_argument(jack_domain.clone()),
            ],
        );
        graph.register_relation(&mut connection, &jill_like_jack_relation)?;
        println!("jill like jack: {}", jills.len());
        let jack_date_jill_relation = relation(
            "date".to_string(),
            vec![
                variable_argument(jack_domain.clone()),
                variable_argument(jill_domain.clone()),
            ],
        );
        graph.register_relation(&mut connection, &jack_date_jill_relation)?;
        println!("jill date jack: {}", jills.len());

        for i in 0..total_members_each_class {
            println!("i: {}", i);
            let is_test = i == 0;
            let is_training = !is_test;
            let mut domain_entity_map: HashMap<String, Entity> = HashMap::new();
            for domain in entity_domains.iter() {
                println!("domain: {}", domain);
                let prefix = if is_test { "test" } else { "train" };
                let name = format!("{}_{}{}", &prefix, domain, i); // Using Debug formatting for Domain enum
                let entity = Entity {
                    domain: domain.clone(),
                    name: name.clone(),
                };
                graph.store_entity(&mut connection, &entity)?;
                println!("Stored entity: {:?}", &entity);
                domain_entity_map.insert(domain.to_string(), entity);
            }

            let jack_entity = &domain_entity_map[&Domain::MAN.to_string()];
            let jill_entity = &domain_entity_map[&Domain::WOMAN.to_string()];

            let p_jack_lonely = weighted_cointoss(0.3f64);
            let p_jill_exciting: f64 = weighted_cointoss(0.6f64);
            let p_jill_likes_jack: f64 = weighted_cointoss(0.4f64);
            let p_jack_likes_jill = weighted_cointoss(numeric_or(p_jack_lonely, p_jill_exciting));
            let p_jack_dates_jill = numeric_and(p_jack_likes_jill, p_jill_likes_jack);

            {
                println!("Man entity part 2: {:?}", jack_entity);
                let jack = constant(jack_entity.domain.clone(), jack_entity.name.clone());
                let jack_lonely = proposition(lonely_jack_relation.clone(), vec![sub(jack)]);

                println!(
                    "Man Lonely: {:?}, Probability: {}",
                    jack_lonely.predicate.hash_string(),
                    p_jack_lonely
                );
                proposition_db.store_proposition_probability(
                    &mut connection,
                    &jack_lonely,
                    p_jack_lonely,
                )?;
                plan.maybe_add_to_training(&mut connection, is_training, &jack_lonely)?;
                graph.ensure_existence_backlinks_for_proposition(&mut connection, &jack_lonely)?;
            }

            {
                let jill = constant(jill_entity.domain.clone(), jill_entity.name.clone());
                let jill_exciting = proposition(exciting_jill_relation.clone(), vec![sub(jill)]);

                println!(
                    "Woman Exciting: {:?}, Probability: {}",
                    jill_exciting.predicate.hash_string(),
                    p_jill_exciting
                );
                proposition_db.store_proposition_probability(
                    &mut connection,
                    &jill_exciting,
                    p_jill_exciting,
                )?;
                plan.maybe_add_to_training(&mut connection, is_training, &jill_exciting)?;
                graph
                    .ensure_existence_backlinks_for_proposition(&mut connection, &jill_exciting)?;
            }

            {
                let jill = constant(jill_entity.domain.clone(), jill_entity.name.clone());
                let jack = constant(jack_entity.domain.clone(), jack_entity.name.clone());

                // "likes(jill, jack)"
                let jill_likes_jack = proposition(
                    jill_like_jack_relation.clone(),
                    vec![sub(jill.clone()), obj(jack.clone())],
                );
                println!(
                    "Woman likes Man: {:?}, Probability: {}",
                    jill_likes_jack.predicate.hash_string(),
                    p_jill_likes_jack
                ); // Logging
                proposition_db.store_proposition_probability(
                    &mut connection,
                    &jill_likes_jack,
                    p_jill_likes_jack,
                )?;
                plan.maybe_add_to_training(&mut connection, is_training, &jill_likes_jack)?;
                graph.ensure_existence_backlinks_for_proposition(
                    &mut connection,
                    &jill_likes_jack,
                )?;
            }

            {
                let jill = constant(jill_entity.domain.clone(), jill_entity.name.clone());
                let jack = constant(jack_entity.domain.clone(), jack_entity.name.clone());
                let jack_likes_jill = proposition(
                    jack_like_jill_relation.clone(),
                    vec![sub(jack.clone()), obj(jill.clone())],
                );
                println!(
                    "Man likes Woman: {:?}, Probability: {}",
                    jack_likes_jill.predicate.hash_string(),
                    p_jack_likes_jill
                ); // Logging
                if is_training {
                    proposition_db.store_proposition_probability(
                        &mut connection,
                        &jack_likes_jill,
                        p_jack_likes_jill,
                    )?;
                }
                plan.maybe_add_to_training(&mut connection, is_training, &jack_likes_jill)?;
                // graph.ensure_existence_backlinks_for_proposition(&jack_likes_jill)?;
            }
            {
                let jill = constant(jill_entity.domain.clone(), jill_entity.name.clone());
                let jack = constant(jack_entity.domain.clone(), jack_entity.name.clone());

                // "dates(jack, jill)" based on "likes(jack, jill) and likes(jill, jack)"
                let jack_dates_jill =
                    proposition(jack_date_jill_relation.clone(), vec![sub(jack), obj(jill)]);
                println!(
                    "Man dates Woman: {:?}, Probability: {}",
                    jack_dates_jill.predicate.hash_string(),
                    p_jack_dates_jill
                ); // Logging

                if is_training {
                    proposition_db.store_proposition_probability(
                        &mut connection,
                        &jack_dates_jill,
                        p_jack_dates_jill,
                    )?;
                }
                plan.maybe_add_to_training(&mut connection, is_training, &jack_dates_jill)?;
                plan.maybe_add_to_test(&mut connection, is_test, &jack_dates_jill)?;
                // graph.ensure_existence_backlinks_for_proposition(&jack_dates_jill)?;

                if i == 0 {
                    graph.register_target(&mut connection, &jack_dates_jill)?;
                }
            }
        }

        let xjack = variable(Domain::MAN.to_string());
        let xjill = variable(Domain::WOMAN.to_string());

        let implications = vec![
            // if jack is lonely, he will date any jill
            implication(
                conjunction(vec![predicate(
                    lonely_jack_relation,
                    vec![sub(xjack.clone())],
                )]),
                predicate(
                    jack_like_jill_relation.clone(),
                    vec![sub(xjack.clone()), obj(xjill.clone())],
                ),
                vec![RoleMap::new(HashMap::from([(
                    "sub".to_string(),
                    "sub".to_string(),
                )]))],
            ),
            // if jill is exciting, any jack will date her
            implication(
                conjunction(vec![predicate(
                    exciting_jill_relation,
                    vec![sub(xjill.clone())],
                )]),
                predicate(
                    jack_like_jill_relation.clone(),
                    vec![sub(xjack.clone()), obj(xjill.clone())],
                ),
                vec![RoleMap::new(HashMap::from([(
                    "obj".to_string(),
                    "sub".to_string(),
                )]))],
            ),
            // if jill likes jack, then jack dates jill
            implication(
                conjunction(vec![
                    predicate(
                        jill_like_jack_relation.clone(),
                        vec![sub(xjill.clone()), obj(xjack.clone())],
                    ),
                    predicate(
                        jack_like_jill_relation.clone(),
                        vec![sub(xjack.clone()), obj(xjill.clone())],
                    ),
                ]),
                predicate(
                    jack_date_jill_relation.clone(),
                    vec![sub(xjack.clone()), obj(xjill.clone())],
                ),
                vec![
                    RoleMap::new(HashMap::from([
                        ("sub".to_string(), "obj".to_string()),
                        ("obj".to_string(), "sub".to_string()),
                    ])),
                    RoleMap::new(HashMap::from([
                        ("sub".to_string(), "sub".to_string()),
                        ("obj".to_string(), "obj".to_string()),
                    ])),
                ],
            ),
        ];

        for implication in implications.iter() {
            println!("Storing implication: {:?}", implication);
            graph.store_predicate_implication(&mut connection, implication)?;
        }

        // Additional functions
        fn numeric_or(a: f64, b: f64) -> f64 {
            f64::min(a + b, 1.0)
        }

        fn numeric_and(a: f64, b: f64) -> f64 {
            a * b
        }

        Ok(())
    }
}
