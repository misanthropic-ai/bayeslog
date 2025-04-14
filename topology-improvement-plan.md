# Topology-Aware Adaptive Iteration Implementation Plan

## Current Status
- Basic network topology classification exists in IBP class (Unknown, Acyclic, SparseCyclic, DenseCyclic)
- Simple cycle detection using DFS algorithm to check if networks contain cycles
- Basic density calculation used to distinguish between sparse and dense cyclic networks
- Initial damping factors and iteration limits are selected based on topology
- No specialized iteration strategies for different network types
- No tests specifically for topology detection or adaptive strategies

## Implementation Plan

### 1. Enhanced Network Topology Classification

#### NetworkTopology enum expansion
- Update the NetworkTopology enum to include more specific types:
  - `Tree`: Strictly tree-structured (only one path to any node)
  - `Polytree`: DAG with possible multiple parents but no cycles
  - `SparseCyclic`: Network with few cycles
  - `ModerateCyclic`: Network with moderate cycle density
  - `DenseCyclic`: Heavily interconnected network

#### Topology metrics
- Implement algorithm to count number of cycles in the network
- Calculate cycle lengths and their distribution
- Measure network diameter (longest shortest path)
- Identify articulation points (nodes that, when removed, disconnect the graph)
- Calculate node centrality measures

#### Classification refinement
- Improve `analyze_network_topology` to use the new metrics
- Add detection for bipartite/near-bipartite structures
- Add detection for hub-and-spoke patterns
- Implement efficiency improvements for large networks

### 2. Specialized Scheduling Strategies

#### Tree/Polytree networks
- Implement two-pass strategy:
  - First pass: top-down from root to leaves (π messages)
  - Second pass: bottom-up from leaves to root (λ messages)
  - Skip the iterative convergence check for guaranteed convergence

#### Sparse cyclic networks
- Implement "cycle-breaking" scheduling:
  - Identify feedback arc set (minimum set of edges to remove to make acyclic)
  - Schedule nodes in topological order with feedback arcs delayed
  - Use stronger damping only for nodes in cycles
  - Implement oscillation detection for cycle nodes

#### Dense cyclic networks
- Implement gradual relaxation strategy:
  - Start with very strong damping
  - Use special message aggregation for nodes with many connections
  - Implement adaptive residual-based scheduling
  - Add proactive convergence check to detect oscillation patterns
  - Implement early stopping based on pattern detection

### 3. Adaptive Parameter Selection

#### Damping factor selection
- Implement context-aware damping:
  - Per-node damping based on position in network
  - Stronger damping for nodes in cycles
  - Calculate optimal damping based on eigenvalues of adjacency matrix
  - Dynamic adjustment during iterations based on delta trends

#### Convergence thresholds
- Implement adaptive thresholds:
  - Stricter for trees and DAGs (exact convergence possible)
  - Gradual threshold relaxation for cyclic networks
  - Analyze delta patterns to identify when oscillation is minimal

#### Iteration limits
- Dynamic iteration limits based on:
  - Network size and complexity
  - Convergence behavior in early iterations
  - Predicted oscillation amplitude

### 4. Algorithm Switching Mechanism

#### Selection logic
- Create a mechanism that can switch propagation algorithms mid-run:
  - Start with optimistic algorithm (assumes quick convergence)
  - Monitor convergence patterns
  - Switch to more specialized algorithm if convergence is problematic

#### Fallback strategies
- Implement recovery strategies for difficult cases:
  - Automatic re-initialization with different starting conditions
  - Dynamic nodes freezing for persistent oscillations
  - Progressive evidence application for stability

#### Diagnostics and reporting
- Add detailed metrics collection:
  - Convergence speed per network region
  - Oscillation amplitude and frequency
  - Evidence impact propagation paths
  - Iteration-by-iteration state history for debugging

## Testing Plan

### Unit Tests
- Network classification tests with known graph structures
- Verify correct topology detection for various test cases
- Measure topology metrics accuracy
- Test cycle detection and counting

### Integration Tests
- End-to-end tests for different network types
- Convergence tests with varied network structures
- Performance comparison between strategies
- Tests for dynamic algorithm switching

### Benchmark Tests
- Performance tests for large networks
- Convergence speed across different network types
- Memory usage monitoring
- Oscillation behavior analysis
