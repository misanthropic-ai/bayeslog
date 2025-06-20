#[cfg(test)]
mod test_weight_manager {
    use bayeslog::qbbn::model::{
        weight_manager::{WeightManager, WeightManagerConfig, ConsolidationTrigger},
        weights::ExponentialWeights,
    };
    use bayeslog::qbbn::common::redis::MockConnection;
    use bayeslog::qbbn::graphdb::GraphDBAdapter;
    use bayeslog::GraphDatabase;
    use std::collections::HashMap;
    use std::sync::Arc;
    use chrono::Duration;

    fn create_test_connection() -> MockConnection {
        let graph_db = Arc::new(GraphDatabase::new(":memory:").unwrap());
        let adapter = GraphDBAdapter::new(graph_db, "test");
        MockConnection::new(adapter)
    }

    #[test]
    fn test_weight_manager_basic() {
        let mut conn = create_test_connection();
        let base_weights = ExponentialWeights::new("test".to_string()).unwrap();
        let mut manager = WeightManager::new(base_weights, "test".to_string());
        
        // Test reading a weight (should be default initialized)
        let weight = manager.read_weight(&mut conn, "test_feature").unwrap();
        assert_eq!(weight, 0.0); // Default weight
        
        // Update a weight
        manager.update_weight("test_feature", 0.5);
        
        // Read again - should include delta
        let weight = manager.read_weight(&mut conn, "test_feature").unwrap();
        assert_eq!(weight, 0.5);
        
        // Update again
        manager.update_weight("test_feature", 0.3);
        let weight = manager.read_weight(&mut conn, "test_feature").unwrap();
        assert_eq!(weight, 0.8); // 0.0 + 0.5 + 0.3
    }
    
    #[test]
    fn test_weight_vector_with_deltas() {
        let mut conn = create_test_connection();
        let base_weights = ExponentialWeights::new("test".to_string()).unwrap();
        let mut manager = WeightManager::new(base_weights, "test".to_string());
        
        // Update some weights
        let mut updates = HashMap::new();
        updates.insert("feature1".to_string(), 0.5);
        updates.insert("feature2".to_string(), -0.2);
        updates.insert("feature3".to_string(), 0.1);
        manager.update_weights(&updates);
        
        // Read weight vector
        let features = vec!["feature1".to_string(), "feature2".to_string(), "feature4".to_string()];
        let weights = manager.read_weight_vector(&mut conn, &features).unwrap();
        
        assert_eq!(weights.get("feature1"), Some(&0.5));
        assert_eq!(weights.get("feature2"), Some(&-0.2));
        assert_eq!(weights.get("feature4"), Some(&0.0)); // Not updated, should be base weight
    }
    
    #[test]
    fn test_consolidation() {
        let mut conn = create_test_connection();
        let base_weights = ExponentialWeights::new("test".to_string()).unwrap();
        let mut config = WeightManagerConfig::default();
        config.hot_feature_threshold = 2; // Low threshold for testing
        
        let mut manager = WeightManager::with_config(base_weights, "test".to_string(), config);
        
        // Create some hot and cold features
        manager.update_weight("hot_feature", 0.1);
        manager.update_weight("hot_feature", 0.1);
        manager.update_weight("hot_feature", 0.1); // 3 updates - hot
        
        manager.update_weight("cold_feature", 0.2); // 1 update - cold
        
        // Check delta stats before consolidation
        let stats = manager.get_delta_stats();
        assert_eq!(stats.num_features, 2);
        
        // Consolidate
        let result = manager.consolidate_with_connection(&mut conn, ConsolidationTrigger::Manual).unwrap();
        assert_eq!(result.features_consolidated, 1); // Only cold feature
        assert_eq!(result.hot_features_kept, 1); // Hot feature kept with decay
        
        // Check delta stats after consolidation
        let stats = manager.get_delta_stats();
        assert_eq!(stats.num_features, 1); // Only hot feature remains
        
        // Check that cold feature was consolidated to base
        let cold_weight = manager.read_weight(&mut conn, "cold_feature").unwrap();
        assert_eq!(cold_weight, 0.2); // Still accessible through base weights
        
        // Hot feature should still have delta (but decayed)
        let hot_weight = manager.read_weight(&mut conn, "hot_feature").unwrap();
        assert!(hot_weight < 0.3); // Less than original due to decay
        assert!(hot_weight > 0.0); // But still positive
    }
    
    #[test]
    fn test_export_weights() {
        let mut conn = create_test_connection();
        let mut base_weights = ExponentialWeights::new("test".to_string()).unwrap();
        
        // Initialize some base weights
        let mut base_values = HashMap::new();
        base_values.insert("base_feature1".to_string(), 1.0);
        base_values.insert("base_feature2".to_string(), 2.0);
        base_weights.save_weight_vector(&mut conn, &base_values).unwrap();
        
        let mut manager = WeightManager::new(base_weights, "test".to_string());
        
        // Add some delta weights
        manager.update_weight("base_feature1", 0.5); // Update existing
        manager.update_weight("delta_only_feature", 3.0); // New feature
        
        // Export all weights
        let exported = manager.export_weights(&mut conn).unwrap();
        
        assert_eq!(exported.weights.get("base_feature1"), Some(&1.5)); // 1.0 + 0.5
        assert_eq!(exported.weights.get("base_feature2"), Some(&2.0)); // Unchanged
        assert_eq!(exported.weights.get("delta_only_feature"), Some(&3.0)); // Delta only
        assert_eq!(exported.namespace, "test");
    }
}