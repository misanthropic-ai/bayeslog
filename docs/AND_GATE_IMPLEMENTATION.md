# AND Gate Implementation in BayesLog

## Overview

BayesLog now supports probabilistic AND gates using the same learned model approach as OR gates. This document describes the implementation details, feature extraction, and usage patterns for AND gates in the Quantified Boolean Bayesian Network (QBBN).

## Background

Previously, AND gates in BayesLog used a simple heuristic approach:
- If all inputs were true (1.0), output was 1.0
- If any input was false (0.0), output was 0.0

This binary approach didn't capture the nuanced probabilistic reasoning needed for real-world scenarios where inputs have uncertainty.

## Implementation Details

### Unified Model Architecture

Both AND and OR gates now use the `ExponentialModel` class, allowing:
- Learned probabilistic behavior rather than hard-coded heuristics
- Consistent training and inference pipelines
- Support for online learning with delta weights
- GPU acceleration (when using TorchExponentialModel)

### Feature Extraction

The system automatically detects AND gates by examining the factor structure. When a conjunction is detected (multiple premises), it extracts AND-specific features:

```rust
// Detection logic in features_from_factor()
let is_conjunction = factor.factor.iter().any(|f| {
    f.premise.terms.len() > 1
});
```

#### AND-Specific Features

For each AND gate, the following features are extracted:

1. **`and_size_{class}`**: Number of premises in the conjunction
   - Helps the model learn patterns based on gate complexity
   - Example: 2-input AND vs 4-input AND may have different characteristics

2. **`and_num_true_{class}`**: Count of premises with probability > 0.5
   - Captures how many inputs are "mostly true"
   - Useful for learning partial satisfaction patterns

3. **`and_all_true_{class}`**: Binary indicator (1.0 or 0.0)
   - Set to 1.0 if all premises have probability > 0.5
   - Captures the traditional AND gate behavior

4. **`and_any_false_{class}`**: Binary indicator (1.0 or 0.0)
   - Set to 1.0 if any premise has probability ≤ 0.5
   - Inverse of all_true, useful for learning failure modes

5. **`and_soft_{class}`**: Product of all premise probabilities
   - Implements "soft" AND logic: P(A ∧ B) ≈ P(A) × P(B)
   - Useful for probabilistic reasoning with uncertain inputs

6. **`and_min_{class}`**: Minimum probability among all premises
   - Implements "weakest link" logic
   - The AND gate is only as strong as its weakest input

### Training Process

AND gates are trained using the same process as OR gates:

1. Create training scenarios with various input combinations
2. Set expected output probabilities based on domain knowledge
3. Run the standard training pipeline
4. The model learns appropriate weights for AND-specific features

Example training data:
```rust
// All inputs true -> high probability output
(vec![true, true], true, 0.9)

// Any input false -> low probability output  
(vec![true, false], false, 0.1)
(vec![false, true], false, 0.1)

// All inputs false -> very low probability
(vec![false, false], false, 0.05)
```

## Usage Examples

### Creating AND Gate Rules

```rust
use bayeslog::qbbn::model::creators::*;

// Create a 2-input AND gate: 
// If has_fever(Person) AND has_cough(Person), then likely_flu(Person)
let and_rule = implication(
    conjunction(vec![
        predicate(has_fever_relation, vec![sub(person_var.clone())]),
        predicate(has_cough_relation, vec![sub(person_var.clone())]),
    ]),
    predicate(likely_flu_relation, vec![sub(person_var.clone())]),
    vec![], // role mappings
);
```

### Training AND Gates

Use the provided `AndGateTraining` scenario or create custom training data:

```rust
use bayeslog::qbbn::scenarios::and_gate_training::AndGateTraining;

let scenario = AndGateTraining {};
scenario.setup_scenario(&resources)?;
```

### Inference with AND Gates

Inference works identically to OR gates - the system automatically detects and applies appropriate features:

```rust
// Set premise probabilities
proposition_db.store_proposition_probability(&mut conn, &has_fever_prop, 0.8)?;
proposition_db.store_proposition_probability(&mut conn, &has_cough_prop, 0.7)?;

// Run inference - AND features are automatically extracted
let result = engine.infer_proposition(&mut conn, &likely_flu_prop)?;
// Result will reflect learned AND gate behavior
```

## Advantages of Probabilistic AND Gates

1. **Uncertainty Handling**: Can reason with partial evidence
   - Traditional: has_fever=0.8 ∧ has_cough=0.7 → undefined
   - Probabilistic: has_fever=0.8 ∧ has_cough=0.7 → likely_flu≈0.56

2. **Learned Behavior**: Model can learn domain-specific patterns
   - Some domains may require stricter AND logic
   - Others may be more forgiving with partial satisfaction

3. **Soft Failures**: Graceful degradation with uncertain inputs
   - Instead of binary success/failure
   - Probability decreases smoothly as inputs become less certain

4. **Feature Interactions**: Can learn complex patterns
   - Maybe 3-input ANDs behave differently than 2-input ANDs
   - The model can capture these nuances

## Performance Considerations

1. **Feature Extraction**: AND gates generate 6 additional features per class
   - Minimal overhead during inference
   - Features are computed on-demand

2. **Training**: AND gates require appropriate training data
   - Include various input combinations
   - Balance positive and negative examples

3. **Memory**: No additional memory overhead
   - Uses same weight storage as OR gates
   - Delta weights system handles online updates efficiently

## Migration Guide

For existing systems using the heuristic AND gate approach:

1. **Retrain Models**: Existing models need retraining to learn AND gate weights
2. **Update Training Data**: Ensure training scenarios include AND gate examples
3. **Test Behavior**: Verify probabilistic outputs match expected behavior
4. **Adjust Thresholds**: May need to adjust decision thresholds for binary outcomes

## Best Practices

1. **Training Data Quality**
   - Include diverse input combinations
   - Set realistic output probabilities (not always 0.0 or 1.0)
   - Consider domain-specific AND semantics

2. **Feature Selection**
   - The default 6 features work well for most cases
   - Can extend with domain-specific features if needed

3. **Evaluation**
   - Test with both hard (0/1) and soft (probabilistic) inputs
   - Verify behavior matches domain expectations
   - Monitor performance on real-world data

## Technical Reference

### Key Files

- `src/qbbn/model/exponential.rs`: Feature extraction logic
- `src/qbbn/inference/engine.rs`: AND gate inference
- `src/qbbn/scenarios/and_gate_training.rs`: Training scenario
- `tests/test_and_gate.rs`: Comprehensive tests

### Feature Extraction Code

```rust
if is_conjunction {
    // Extract AND-gate specific features
    let num_premises = factor.probabilities.len() as f64;
    result.insert(format!("and_size_{}", class_label), num_premises);
    
    let num_true = factor.probabilities.iter()
        .filter(|&&p| p > 0.5).count() as f64;
    result.insert(format!("and_num_true_{}", class_label), num_true);
    
    let all_true = factor.probabilities.iter().all(|&p| p > 0.5);
    result.insert(format!("and_all_true_{}", class_label), 
                  if all_true { 1.0 } else { 0.0 });
    
    let any_false = factor.probabilities.iter().any(|&p| p <= 0.5);
    result.insert(format!("and_any_false_{}", class_label), 
                  if any_false { 1.0 } else { 0.0 });
    
    let soft_and = factor.probabilities.iter().product::<f64>();
    result.insert(format!("and_soft_{}", class_label), soft_and);
    
    let min_prob = factor.probabilities.iter()
        .fold(1.0f64, |a, &b| a.min(b));
    result.insert(format!("and_min_{}", class_label), min_prob);
}
```

## Future Enhancements

1. **Configurable Thresholds**: Allow customizing the 0.5 threshold for true/false
2. **Additional Features**: 
   - Variance among inputs
   - Specific position features (first input, last input)
   - Correlation features
3. **Specialized AND Types**:
   - Strict AND (all must be very high)
   - Majority AND (most must be true)
   - Weighted AND (some inputs matter more)
4. **Debugging Tools**: Visualize learned AND gate behavior

## Conclusion

The probabilistic AND gate implementation in BayesLog provides a powerful and flexible approach to conjunction reasoning in uncertain environments. By learning from data rather than using fixed heuristics, the system can adapt to domain-specific requirements while maintaining the efficiency and scalability of the overall QBBN framework.