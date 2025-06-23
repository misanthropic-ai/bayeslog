# Benchmark Results: Standard vs Torch ExponentialModel

## Summary

Based on the benchmarks run, here are the performance comparisons:

### Small Entity Counts (High Overhead Impact)
- **1 entity**: Standard model is ~1384% faster (0.003s vs 0.037s)
- **2 entities**: Standard model is ~979% faster (0.008s vs 0.081s) 
- **5 entities**: Standard model is ~43% faster (0.034s vs 0.048s)

### Medium Entity Counts (Approaching Parity)
- **10 entities**: Torch model is ~0.7% faster (0.113s vs 0.112s)
- **25 entities**: Torch model is ~3.8% faster (0.676s vs 0.649s)

### Larger Entity Counts (GPU Advantage Emerging)
- **50 entities**: Torch model is ~10.5% faster (2.280s vs 2.041s)
- **100 entities**: Torch model is ~13.9% faster (10.957s vs 9.436s)
- **200 entities**: Torch model is ~17.3% faster (49.094s vs 40.610s)

## Analysis

1. **Overhead Impact**: For small datasets (< 10 entities), the PyTorch initialization overhead dominates, making the standard model significantly faster.

2. **Break-even Point**: Around 10 entities, both models perform similarly.

3. **GPU Acceleration Benefits**: For larger datasets (50+ entities), the GPU-accelerated model shows increasing performance gains:
   - 10.5% faster at 50 entities
   - 13.9% faster at 100 entities
   - 17.3% faster at 200 entities

4. **Scalability**: The performance gap widens as the dataset grows, suggesting the torch model will provide even greater benefits for production-scale datasets.

## Recommendations

1. Use the standard model for small datasets (< 10 entities)
2. Use the torch model for larger datasets (> 50 entities)
3. Consider implementing dynamic model selection based on dataset size
4. Further optimizations possible:
   - True batch training (currently training one example at a time)
   - Mixed precision training
   - Model parallelism for very large networks