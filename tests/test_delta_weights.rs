#[cfg(test)]
mod test_delta_weights {
    use bayeslog::qbbn::model::delta_weights::DeltaWeights;
    use chrono::Duration;
    use std::thread;
    use std::time::Duration as StdDuration;

    #[test]
    fn test_delta_weights_basic() {
        let mut delta = DeltaWeights::new("test".to_string());
        
        // Test initial state
        assert_eq!(delta.size(), 0);
        assert_eq!(delta.total_updates, 0);
        
        // Add some deltas
        delta.update_weight("feature1", 0.5);
        delta.update_weight("feature2", -0.3);
        delta.update_weight("feature1", 0.2); // Update same feature again
        
        // Check state
        assert_eq!(delta.size(), 2);
        assert_eq!(delta.total_updates, 3);
        assert_eq!(delta.get_delta("feature1"), 0.7); // 0.5 + 0.2
        assert_eq!(delta.get_delta("feature2"), -0.3);
        assert_eq!(delta.get_delta("feature3"), 0.0); // Non-existent feature
        
        // Check update counts
        assert_eq!(*delta.update_counts.get("feature1").unwrap(), 2);
        assert_eq!(*delta.update_counts.get("feature2").unwrap(), 1);
    }
    
    #[test] 
    fn test_hot_features() {
        let mut delta = DeltaWeights::new("test".to_string());
        
        // Create hot and cold features
        for _ in 0..5 {
            delta.update_weight("hot_feature", 0.1);
        }
        delta.update_weight("cold_feature", 0.1);
        
        let hot_features = delta.get_hot_features(3);
        assert_eq!(hot_features.len(), 1);
        assert_eq!(hot_features[0], "hot_feature");
        
        let all_hot = delta.get_hot_features(0);
        assert_eq!(all_hot.len(), 2); // Both features
    }
    
    #[test]
    fn test_clear_features() {
        let mut delta = DeltaWeights::new("test".to_string());
        
        delta.update_weight("feature1", 0.5);
        delta.update_weight("feature2", 0.3);
        delta.update_weight("feature3", 0.1);
        
        assert_eq!(delta.size(), 3);
        
        // Clear specific features
        delta.clear_features(&["feature1".to_string(), "feature3".to_string()]);
        
        assert_eq!(delta.size(), 1);
        assert_eq!(delta.get_delta("feature2"), 0.3);
        assert_eq!(delta.get_delta("feature1"), 0.0);
    }
    
    #[test]
    fn test_stats() {
        let mut delta = DeltaWeights::new("test".to_string());
        
        delta.update_weight("feature1", 0.5);
        delta.update_weight("feature1", 0.5);
        delta.update_weight("feature2", 0.3);
        
        let stats = delta.get_stats();
        assert_eq!(stats.num_features, 2);
        assert_eq!(stats.total_updates, 3);
        assert_eq!(stats.avg_update_count, 1.5);
        assert_eq!(stats.max_update_count, 2);
    }
    
    #[test]
    fn test_save_load() {
        let mut delta = DeltaWeights::new("test".to_string());
        delta.update_weight("feature1", 0.5);
        delta.update_weight("feature2", -0.3);
        
        // Save to file
        let path = "/tmp/test_delta_weights.json";
        delta.save_to_file(path).unwrap();
        
        // Load from file
        let loaded = DeltaWeights::load_from_file(path).unwrap();
        
        assert_eq!(loaded.namespace, "test");
        assert_eq!(loaded.size(), 2);
        assert_eq!(loaded.get_delta("feature1"), 0.5);
        assert_eq!(loaded.get_delta("feature2"), -0.3);
        
        // Clean up
        std::fs::remove_file(path).ok();
    }
}