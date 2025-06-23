#[cfg(test)]
mod test_and_gate {
    use bayeslog::qbbn::{
        common::{
            resources::ResourceContext,
            interface::ScenarioMaker,
            setup::{CommandLineOptions, StorageType},
        },
        scenarios::and_gate_training::AndGateTraining,
        model::{
            creators::{relation, variable_argument, constant, sub, proposition},
        },
    };
    use bayeslog::GraphDatabase;
    use std::sync::{Arc, Mutex};
    
    #[test]
    fn test_and_gate_behavior() {
        // Create a test environment
        let graph_db = Arc::new(GraphDatabase::new(":memory:").unwrap());
        let connection = Arc::new(Mutex::new(
            bayeslog::qbbn::common::redis::MockConnection::new(
                bayeslog::qbbn::graphdb::GraphDBAdapter::new(graph_db, "test")
            )
        ));
        
        let config = CommandLineOptions {
            entities_per_domain: 10,
            scenario_name: "test_and".to_string(),
            storage_type: StorageType::InMemory,
            db_path: None,
            test_scenario: None,
            print_training_loss: false,
            test_example: None,
            marginal_output_file: None,
        };
        
        let resources = ResourceContext {
            config,
            connection: connection.clone(),
        };
        
        // Set up the AND gate training scenario
        let scenario = AndGateTraining {};
        scenario.setup_scenario(&resources).expect("Failed to setup scenario");
        
        // Train the model (in practice, this would be done through the training pipeline)
        // For now, we'll just verify that the features are extracted correctly
        
        // Create a simple test case: input(A) AND input(B) -> output(A)
        let test_domain = "TestDomain".to_string();
        let input_relation = relation("input".to_string(), vec![variable_argument(test_domain.clone())]);
        let output_relation = relation("output".to_string(), vec![variable_argument(test_domain.clone())]);
        
        let const_a = constant(test_domain.clone(), "A".to_string());
        let const_b = constant(test_domain.clone(), "B".to_string());
        
        // Create propositions
        let input_a = proposition(
            input_relation.clone(),
            vec![sub(const_a.clone())],
        );
        let input_b = proposition(
            input_relation.clone(),
            vec![sub(const_b.clone())],
        );
        let output_a = proposition(
            output_relation.clone(),
            vec![sub(const_a.clone())],
        );
        
        // Test feature extraction
        use bayeslog::qbbn::model::exponential::features_from_factor;
        use bayeslog::qbbn::common::model::FactorContext;
        use bayeslog::qbbn::inference::graph::PropositionFactor;
        use bayeslog::qbbn::model::objects::{ImplicationFactor, PredicateGroup, GroupRoleMap, PropositionGroup};
        
        // Create a mock factor context for AND gate
        let implication = ImplicationFactor {
            premise: PredicateGroup {
                terms: vec![
                    input_a.predicate.clone(),
                    input_b.predicate.clone(),
                ],
            },
            conclusion: output_a.predicate.clone(),
            role_maps: GroupRoleMap { role_maps: vec![] },
        };
        
        let factor = PropositionFactor {
            premise: PropositionGroup::new(vec![input_a.clone(), input_b.clone()]),
            conclusion: output_a.clone(),
            inference: implication,
        };
        
        let context = FactorContext {
            factor: vec![factor.clone()],
            probabilities: vec![1.0, 1.0], // Both inputs are true
        };
        
        // Extract features
        let features = features_from_factor(&context).unwrap();
        
        // Verify AND-specific features are present
        assert!(features[0].contains_key("and_all_true_0"), "Should have and_all_true feature");
        assert!(features[0].contains_key("and_size_0"), "Should have and_size feature");
        assert!(features[0].contains_key("and_num_true_0"), "Should have and_num_true feature");
        
        // Verify feature values for all-true case
        assert_eq!(features[0].get("and_all_true_0"), Some(&1.0));
        assert_eq!(features[0].get("and_any_false_0"), Some(&0.0));
        assert_eq!(features[0].get("and_num_true_0"), Some(&2.0));
        assert_eq!(features[0].get("and_size_0"), Some(&2.0));
        
        // Test with one false input
        let context_one_false = FactorContext {
            factor: vec![factor.clone()],
            probabilities: vec![1.0, 0.0], // Second input is false
        };
        
        let features_one_false = features_from_factor(&context_one_false).unwrap();
        assert_eq!(features_one_false[0].get("and_all_true_0"), Some(&0.0));
        assert_eq!(features_one_false[0].get("and_any_false_0"), Some(&1.0));
        assert_eq!(features_one_false[0].get("and_num_true_0"), Some(&1.0));
    }
    
    #[test]
    fn test_and_gate_soft_behavior() {
        // Test the soft AND features with probabilistic inputs
        use bayeslog::qbbn::model::exponential::features_from_factor;
        use bayeslog::qbbn::common::model::FactorContext;
        use bayeslog::qbbn::inference::graph::PropositionFactor;
        use bayeslog::qbbn::model::objects::{ImplicationFactor, PredicateGroup, GroupRoleMap, PropositionGroup};
        use bayeslog::qbbn::model::creators::{relation, variable_argument, constant, sub, proposition};
        
        // Create a proper 3-input AND gate
        let test_domain = "TestDomain".to_string();
        let input_relation = relation("input".to_string(), vec![variable_argument(test_domain.clone())]);
        let output_relation = relation("output".to_string(), vec![variable_argument(test_domain.clone())]);
        
        let const_a = constant(test_domain.clone(), "A".to_string());
        let const_b = constant(test_domain.clone(), "B".to_string()); 
        let const_c = constant(test_domain.clone(), "C".to_string());
        
        // Create propositions for 3-input AND
        let input_a = proposition(
            input_relation.clone(),
            vec![sub(const_a.clone())],
        );
        let input_b = proposition(
            input_relation.clone(),
            vec![sub(const_b.clone())],
        );
        let input_c = proposition(
            input_relation.clone(),
            vec![sub(const_c.clone())],
        );
        let output_a = proposition(
            output_relation.clone(),
            vec![sub(const_a.clone())],
        );
        
        // Create AND implication with 3 premises
        let implication = ImplicationFactor {
            premise: PredicateGroup {
                terms: vec![
                    input_a.predicate.clone(),
                    input_b.predicate.clone(),
                    input_c.predicate.clone(),
                ],
            },
            conclusion: output_a.predicate.clone(),
            role_maps: GroupRoleMap { role_maps: vec![] },
        };
        
        let factor = PropositionFactor {
            premise: PropositionGroup::new(vec![input_a.clone(), input_b.clone(), input_c.clone()]),
            conclusion: output_a.clone(),
            inference: implication,
        };
        
        // Test with soft probabilities
        let context = FactorContext {
            factor: vec![factor.clone()],
            probabilities: vec![0.8, 0.9, 0.7], // Three probabilistic inputs
        };
        
        let features = features_from_factor(&context).unwrap();
        
        // Check soft AND feature (product of probabilities)
        let expected_soft_and = 0.8 * 0.9 * 0.7; // 0.504
        let actual_soft_and = features[0].get("and_soft_0").unwrap();
        assert!((actual_soft_and - expected_soft_and).abs() < 0.001);
        
        // Check min feature (weakest link)
        assert_eq!(features[0].get("and_min_0"), Some(&0.7));
        
        // Check size feature
        assert_eq!(features[0].get("and_size_0"), Some(&3.0));
        
        // Check num_true feature (none are > 0.5)
        assert_eq!(features[0].get("and_num_true_0"), Some(&3.0)); // All are > 0.5
        
        // Check all_true and any_false based on 0.5 threshold
        assert_eq!(features[0].get("and_all_true_0"), Some(&1.0)); // All > 0.5
        assert_eq!(features[0].get("and_any_false_0"), Some(&0.0)); // None <= 0.5
    }
}