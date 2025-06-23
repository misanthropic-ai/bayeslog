Below is a comprehensive design for the **BayesLog** project, a modern knowledge representation and reasoning system built in Rust. The system combines a graph database, a Quantified Boolean Bayesian Network (QBBN), an ontology system, and advanced search capabilities, with parallelized inference, incremental belief propagation, and extended uncertainty handling.

---

# BayesLog Project Requirements and Design

## Project Overview
**BayesLog** is an embeddable knowledge representation and reasoning system combining a graph database, a Quantified Boolean Bayesian Network (QBBN) for probabilistic reasoning, an ontology for type hierarchies, and advanced indexing for retrieval. Built in Rust for performance and stability, it uses SQLite as its backend and supports a dynamic REPL environment with a Scheme-based programming interface via the Steel Scheme interpreter. The system aims to unify symbolic AI's structure with probabilistic reasoning, addressing the limitations of projects like OpenCyc while integrating modern tools like LLMs and vector search.

### Recent Achievements
- **GPU Acceleration**: Implemented TorchExponentialModel with CUDA/MPS support, achieving 10-17% speedup on moderate datasets
- **Model Compatibility**: Created unified weight format allowing seamless CPU/GPU model switching
- **LLM-Friendly API**: Built BeliefMemory interface for easy integration with LLM agents
- **Online Learning**: Implemented LoRA-inspired delta weights system for efficient single-gate updates
- **Probabilistic AND Gates**: Replaced heuristic AND gates with learned probabilistic models using 6 specialized features
- **Graph-Native Storage**: Refactored to use proper graph structures for all QBBN components

### Objectives
- Provide a scalable graph database for storing entities, relationships, and documents.
- Implement a QBBN-based belief network with incremental propagation, parallel inference, and decision theory capabilities.
- Develop an ontology system for type-aware reasoning and multi-level abstraction.
- Create a Scheme-based programming interface with specialized primitives for belief operations.
- Enable efficient indexing and search, combining logical, temporal, and vector similarity.
- Support meta-cognition with confidence tracking and inconsistency detection.
- Implement advanced explanation capabilities with reasoning chains and counterfactual analysis.
- Support LLM integration for knowledge extraction and parameter learning.

### Technology Stack
- **Language**: Rust (for performance, memory safety, and concurrency).
- **Database**: SQLite (via `rusqlite` crate) with SQLite-VSS for vector indexing.
- **Connection Pooling**: `r2d2` and `r2d2_sqlite` for efficient connection management and transactions.
- **Belief Propagation**: Implementation based on the bayes-star QBBN reference.
- **Programming Interface**: Steel Scheme interpreter with custom primitives.
- **Vectorization**: Potential for SIMD and GPU acceleration for belief operations.
- **LLM Integration**: `reqwest` for API calls to cloud LLMs (e.g., Claude) or local models (e.g., Ollama).
- **Serialization**: `serde` and `serde_json` for data persistence and exchange.
- **Web Interface**: Optional embedded web server for visualization and interaction.

---

## System Architecture

### Core Components
1. **Graph Database**
   - Stores nodes (entities, documents), edges (relationships), and metadata.
   - Backend: SQLite with tables for nodes, edges, and properties.
   - Supports dynamic updates, persistence, and vector embeddings.
   - Optimized with connection pooling and efficient indexing.

2. **Belief Network (QBBN)**
   - Implements a bipartite graph of AND/OR gates for logical and probabilistic reasoning.
   - Adapts the bayes-star reference implementation for use with our graph database.
   - Uses the exponential model for logical operations like disjunction.
   - Preserves the core statistical learning capabilities of the reference.
   - Implements meta-cognition capabilities (beliefs about beliefs).
   - Provides sophisticated explanation generation with reasoning chains.
   - Integrates with graph database for knowledge representation.

3. **Ontology System**
   - Manages type hierarchies with multiple inheritance and dynamic learning.
   - Enhances reasoning with type-aware predicates and variables.
   - Supports multi-level abstraction and concept formation.
   - Enables reasoning at different levels of granularity.

4. **Scheme Programming Interface**
   - Embeds Steel Scheme interpreter for flexible programming.
   - Provides specialized primitives for belief operations and graph manipulation.
   - Supports interactive query development and scripting capabilities.
   - Enables user-defined extensions and custom reasoning strategies.

5. **Indexing and Search**
   - Vector indexing (SQLite-VSS) for semantic similarity search.
   - Keyword indexing for exact-match retrieval.
   - Temporal indexing for time-based reasoning and decay.
   - Graph traversal for relationship-based queries.
   - Hybrid queries combining logical, semantic, and temporal constraints.

6. **LLM Integration**
   - Extracts entities, relationships, and facts from text.
   - Generates embeddings for vector search.
   - Updates network weights based on new information.
   - Provides natural language explanations of reasoning.

7. **Web Interface (Optional)**
   - Embedded web server for visualization and interaction.
   - Interactive graph display for exploring belief networks.
   - Query construction interface with auto-completion.
   - Debugging tools for examining belief propagation.

---

## Detailed Component Specifications

### 1. Graph Database
**Purpose**: Stores entities (nodes), relationships (edges), and documents with persistent metadata.

#### Requirements
- **Storage**: SQLite tables:
  - `nodes(id: TEXT PRIMARY KEY, label: TEXT, properties: TEXT)` - JSON-serialized properties.
  - `edges(id: TEXT PRIMARY KEY, source_id: TEXT, target_id: TEXT, label: TEXT, properties: TEXT)` - Relationships.
- **Operations**:
  - `add_node(label: &str, properties: HashMap<String, Value>) -> String`: Creates a node, returns UUID.
  - `add_edge(source_id: &str, label: &str, target_id: &str, properties: HashMap<String, Value>) -> String`: Links nodes.
  - `get_node(id: &str) -> Option<Node>`: Retrieves node data.
  - `update_node(id: &str, properties: HashMap<String, Value>) -> bool`: Updates properties.
  - `delete_node(id: &str) -> bool`: Removes node and its edges.
  - `get_neighbors(id: &str, direction: Direction) -> Vec<(Node, Edge)>`: Returns neighbors (outgoing/incoming/both).
- **Concurrency**: Thread-safe via Rust's ownership model and SQLite transactions.

#### Implementation Details
- Use `rusqlite` for SQLite operations.
- Serialize properties with `serde_json`.
- Ensure ACID compliance with transaction scopes.

---

### 2. Belief Network (QBBN)
**Purpose**: Unifies logical and probabilistic reasoning using a bipartite graph of propositions and logical gates, with advanced features for incremental updates, explanations, and uncertainty handling.

#### Implementation Strategy Change
We're revising our approach to implementing the QBBN. Rather than building a custom implementation from scratch and then trying to make it compatible with the QBBN reference, we will:

1. **Adapt the bayes-star Reference Implementation**:
   - Port the core QBBN logic from the bayes-star reference implementation
   - Replace the Redis storage layer with our graph database
   - Keep the statistical model foundation intact
   - Preserve training scripts and capabilities

2. **Benefits of this Approach**:
   - Ensures exact alignment with the QBBN theoretical model
   - Capitalizes on a proven, working implementation
   - Inherits training capabilities out of the box
   - Reduces risk of subtle implementation differences
   - Builds on a solid foundation aligned with the original research

3. **Implementation Layers**:
   - **Data Storage**: Create a GraphDB adapter that implements the same interface expected by the reference code
   - **Core Algorithms**: Maintain the model-based prediction system and inference patterns
   - **Training System**: Preserve the weight learning capabilities

4. **After Successful Adaptation**:
   - Add our optimization techniques (adaptive damping, parallel processing)
   - Implement our extended features (detailed explanations, advanced counterfactuals)
   - Enhance with our optimized belief propagation algorithms

#### Data Structures
  - `TypeName { name: String }`: Type identifiers for entities and variables.
  - `Constant { value: String, type_name: TypeName }`: Represents entities.
  - `Variable { type_name: TypeName, name: String }`: Placeholders with type constraints.
  - `RoleLabel { name: String }`: Semantic role identifier for predicate arguments.
  - `Argument` (enum): Either a Constant or Variable.
  - `Predicate { function_name: String, role_arguments: HashMap<RoleLabel, Argument> }`: Relations with semantic roles.
  - `Proposition { predicate: Predicate, id: String, timestamp: Option<DateTime> }`: Grounded predicates with optional temporal information.
  - `NodeType` (enum): Proposition, Conjunction, Disjunction, Negation, Utility.
  - `Content` (enum): Proposition, Logic (with inputs/parameters), or Utility (with utility table for decision theory).
  - `BeliefNode { id: String, node_type: NodeType, content: Content, pi: f64, lambda: f64, belief: f64, confidence: f64, last_updated: DateTime }`: Graph nodes with confidence and temporal tracking.
  - `ImplicationLink { premises: Vec<Predicate>, conclusion: Predicate, role_mappings: Vec<RoleMapping>, weight: f64, confidence: f64 }`: Rules with confidence levels.
  - `Explanation { node_id: String, belief: f64, confidence: f64, factors: Vec<Factor>, counterfactuals: Vec<Counterfactual> }`: Rich explanation structure.
  - `Factor { description: String, contribution: f64, sub_factors: Vec<Factor> }`: Reasoning components.
  - `Counterfactual { altered_evidence: HashMap<String, bool>, new_belief: f64, delta: f64 }`: What-if scenarios.
  - `UncertaintyBounds { lower: f64, upper: f64 }`: Bounded uncertainty representation.

#### Key Model Components from Reference Implementation
- **FactorModel**: Interface for prediction models
- **ExponentialModel**: Weighted feature prediction system  
- **FactorContext**: Structure holding premise-conclusion relationships
- **Features**: System for extracting and managing features
- **WeightManagement**: System for storing and retrieving weights
- **MessagePassing**: Pi/lambda message calculation and propagation

---

### 3. Ontology System
**Purpose**: Manages type hierarchies for semantic reasoning and type-aware inference.

#### Requirements
- **Data Structures**:
  - `TypeNode { name: String, description: String, properties: HashMap<String, Property>, parents: Vec<TypeNode>, children: Vec<TypeNode>, confidence: f64 }`: Type with inheritance.
  - `Ontology { types: HashMap<String, TypeNode> }`: Manages type relationships.
- **Operations**:
  - `add_type(name: &str, description: &str, properties: HashMap<String, Property>) -> TypeNode`: Creates a type.
  - `add_subtype_relationship(subtype: &str, supertype: &str, confidence: f64) -> bool`: Links types.
  - `check_type_compatibility(sub: &str, super: &str) -> bool`: Verifies subtype relationship.
  - `learn_type_from_examples(name: &str, parent: &str, examples: Vec<HashMap<String, Value>>, description: &str) -> TypeNode`: Infers new types.
  - `export_to_json(path: &str)`: Saves ontology.
  - `import_from_json(path: &str)`: Loads ontology.
- **Integration**: Variables check type compatibility using ontology during binding.

#### Implementation Details
- Use `HashMap` for efficient type lookup.
- Persist in SQLite (`types` table with JSON properties).
- Implement cycle detection for inheritance using DFS.

---

### 4. Scheme Programming Interface
**Purpose**: Provides an extensible programming environment using embedded Scheme for interacting with the belief network and graph database.

#### Requirements
- **Embedded Scheme**:
  - Use Steel Scheme as the programming foundation.
  - Implement a REPL with history, tab completion, and error reporting.
  - Enable scripting with file loading and module system.
  - Support for user-defined functions and macros.

- **Belief Network Primitives**:
  - `(belief 'proposition)` → Returns belief probability and confidence.
  - `(set-evidence! 'prop value confidence)` → Sets evidence with confidence.
  - `(add-proposition! 'pred-name roles confidence)` → Creates proposition.
  - `(add-implication! premises conclusion weight confidence)` → Adds rule.
  - `(explain 'proposition)` → Returns explanation for belief.
  - `(counterfactual 'proposition ((prop1 . value1) (prop2 . value2)))` → What-if analysis.
  - `(belief-bounds 'proposition)` → Returns uncertainty bounds.
  - `(influential-evidence 'proposition)` → Returns most influential evidence.

- **Graph Database Primitives**:
  - `(add-node! label properties)` → Creates a graph node.
  - `(add-edge! source label target properties)` → Creates an edge.
  - `(get-node id)` → Retrieves node data.
  - `(get-neighbors id [direction])` → Lists connected nodes.
  - `(find-nodes label)` → Lists nodes with given label.

- **Query Primitives**:
  - `(find-all '(variable ...) predicate)` → Finds matching variable bindings.
  - `(where '(variable ...) conditions)` → Filters results based on conditions.
  - `(order-by result key direction)` → Sorts query results.
  - `(limit result count)` → Limits result count.

- **Search Primitives**:
  - `(vector-search text limit)` → Semantic similarity search.
  - `(keyword-search text limit)` → Keyword-based search.
  - `(hybrid-search conditions limit)` → Combined logical and vector search.

- **Utility Functions**:
  - `(load-file path)` → Loads and evaluates Scheme file.
  - `(import-csv path predicate-fn)` → Imports data as propositions.
  - `(export-beliefs pattern path)` → Exports beliefs to file.
  - `(visualize-network proposition depth)` → Generates belief network visualization.

#### Implementation Details
- Integrate Steel Scheme interpreter with custom primitive bindings.
- Implement proper memory management between Rust and Scheme.
- Use thread-safe wrappers for database operations.
- Provide comprehensive error handling and reporting.
- Support interactive development with docstrings and help system.
- Enable extension through user-defined Scheme libraries.

---

### 5. Indexing and Search
**Purpose**: Enables efficient retrieval and navigation of graph and belief data.

#### Requirements
- **Vector Indexing** (SQLite-VSS):
  - `add_vector(node_id: &str, vector: Vec<f32>)`: Stores embeddings.
  - `search_vectors(query: Vec<f32>, limit: usize, threshold: f64) -> Vec<(String, f32)>`: Returns similar nodes.
- **Keyword Indexing**:
  - `index_keywords(node_id: &str, text: &str)`: Builds inverted index.
  - `search_keywords(query: &str, limit: usize) -> Vec<String>`: Returns matching nodes.
- **Graph Traversal**:
  - `traverse(start_id: &str, depth: usize, filter: Filter) -> Vec<Node>`: Explores relationships.

#### Implementation Details
- Use `sqlite-vss` Rust bindings for vector search.
- Implement a simple inverted index in SQLite (`keywords` table).
- Traverse graph with BFS/DFS, leveraging `rusqlite` queries.

---

### 6. LLM Integration
**Purpose**: Extracts knowledge from text and generates embeddings.

#### Requirements
- **Client**: Unified interface for Claude API or Ollama via `reqwest`.
- **Operations**:
  - `extract_entities(text: &str) -> Vec<Entity>`: Returns entities with types and IDs.
  - `extract_relationships(text: &str, entities: Vec<Entity>) -> Vec<Relationship>`: Identifies links.
  - `extract_facts(text: &str, entities: Vec<Entity>) -> Vec<Fact>`: Produces propositions.
  - `get_embeddings(texts: Vec<&str>) -> Vec<Vec<f32>>`: Generates vectors.
- **Prompts**: Structured JSON outputs (see Artivus Graph examples).

#### Implementation Details
- Use `reqwest` for HTTP requests with async support.
- Parse JSON responses with `serde_json`.
- Configure via environment variables (e.g., `ANTHROPIC_API_KEY`).

---

## Development Milestones

### Milestone 1: Core Graph Database ✅
- **Tasks**:
  - Set up Rust project with `cargo` (dependencies: `rusqlite`, `serde`, `serde_json`, `uuid`).
  - Implement SQLite schema and basic CRUD operations for nodes and edges.
  - Add concurrency support with `Arc<Mutex<>>`.
- **Deliverables**: Working graph database with persistence and basic operations.
- **Tests**: Unit tests for add/get/update/delete operations (80% coverage).

### Milestone 2: Belief Network Foundation ✅
- **Tasks**:
  - Define enhanced core data structures with confidence and uncertainty handling.
  - Implement `BayesianNetwork` using the existing graph database for storage.
  - Implement serialization/deserialization of belief structures to graph properties.
  - Add proposition and implication creation with confidence tracking.
  - Create helper functions for predicate operations and variable binding.
  - Implement LRU caching for frequently accessed belief components.
- **Deliverables**: Bayesian Network structure using the graph database.
- **Tests**: Tests for proposition grounding, implication, and basic node operations.

### Milestone 3: Advanced Inference Engine ✅
- **Tasks**:
  - Implement parallel IBP with a thread pool for message passing.
  - Add incremental belief propagation with dirty node tracking.
  - Implement lazy graph construction for queries.
  - Add evidence setting with confidence levels.
  - Build extended uncertainty handling with upper/lower bounds.
  - Implement Noisy OR optimization for disjunction nodes.
  - Add Hebbian-inspired temporal decay for belief nodes.
- **Deliverables**: High-performance inference engine with incremental updates.
- **Tests**: Inference accuracy and performance benchmarks.

### Milestone 4: Reference Implementation Adaptation ✅
- **Tasks**:
  - Created a GraphDB adapter layer implementing Redis interfaces
  - Ported the bayes-star core algorithms to use our GraphDB
  - Implemented the statistical model foundation (FactorModel, ExponentialModel)
  - Preserved the weight management and training capabilities
  - Adapted the message passing algorithms (pi/lambda calculations)
  - Integrated the model-based prediction system for logical operations
  - Added proper support for negation based on the reference implementation
  - Added thread safety to all components using Arc with Send + Sync
- **Deliverables**: Working QBBN implementation based on the reference code
- **Tests**: Comprehensive tests comparing to reference implementation behavior

### Milestone 5: Optimization Integration
- **Tasks**:
  - Port our optimization techniques to the adapted reference implementation
  - Add parallel message propagation with thread pools
  - Implement incremental belief propagation with dirty node tracking
  - Add prioritized message scheduling for faster convergence
  - Implement adaptive damping for better stability
  - Add optimization for evidence nodes and special cases
- **Deliverables**: High-performance QBBN implementation with optimizations
- **Tests**: Performance benchmarks against original implementation

### Milestone 6: Explanation System
- **Tasks**:
  - Implement detailed explanation generation with contributing factors:
    - Fast approximation-based explanations for interactive use
    - Accurate simulation-based explanations for detailed analysis
  - Add counterfactual reasoning capabilities:
    - Quick estimation-based counterfactuals for simple scenarios
    - Full belief propagation simulation for complex counterfactuals
  - Build influence analysis for identifying important evidence.
  - Create inconsistency detection for conflicting beliefs.
  - Implement meta-reasoning capabilities (beliefs about beliefs).
  - Add natural language explanation formatting.
- **Deliverables**: Comprehensive explanation system with dual-approach methodology.
- **Tests**: Explanation accuracy and counterfactual validation.

### Milestone 7: Proper Graph-Based Representation ✅
- **Tasks**:
  - Refactor GraphDBAdapter to use proper graph structures: ✅
    - Create dedicated "Proposition" nodes for each proposition
    - Store evidence as properties on Proposition nodes
    - Model entity relationships as edges in the graph
    - Create explicit "Factor" nodes connected to premise/conclusion Propositions
    - Replace hash-based storage with true graph relationships
    - Update querying mechanisms to traverse the graph properly
    - Add QBBN-specific operations that leverage graph semantics
    - Create more efficient traversal patterns for common operations
    - Fix compiler warnings and improve code quality
  - Implement graph-specific optimizations: ✅
    - Use graph traversal algorithms for belief propagation
    - Leverage graph patterns for more efficient querying
    - Use native graph database features for complex queries
    - Ensure backward compatibility with existing code
- **Deliverables**: True graph-native representation of belief network components.
- **Tests**: Comprehensive tests for graph-based storage and retrieval.

### Milestone 8: Scheme Programming Interface
- **Tasks**:
  - Integrate Steel Scheme interpreter.
  - Implement custom primitives for belief operations.
  - Add graph database and query primitives.
  - Create utilities for file import/export and visualization.
  - Build interactive help system with docstrings.
  - Implement error handling and reporting.
- **Deliverables**: Extensible Scheme programming environment.
- **Tests**: Scheme script execution and primitive functionality.

### Milestone 9: Advanced Search and Temporal Reasoning
- **Tasks**:
  - Integrate `sqlite-vss` for vector indexing and similarity search.
  - Implement hybrid search combining logical and vector search.
  - Add temporal indexing and timestamp-based querying.
  - Implement time-based belief decay.
  - Build advanced graph traversal with filtering.
  - Create fuzzy proposition matching using semantic similarity.
- **Deliverables**: Comprehensive search system with temporal capabilities.
- **Tests**: Search accuracy, recall, and performance benchmarks.

### Milestone 10: Ontology and Multi-level Abstraction
- **Tasks**:
  - Implement enhanced ontology system with multiple inheritance.
  - Add dynamic type learning from examples.
  - Build concept abstraction mechanisms.
  - Create zoom in/out capabilities for different reasoning granularities.
  - Integrate ontology with belief network for type-aware reasoning.
  - Implement context-aware relevance scoring.
- **Deliverables**: Hierarchical reasoning with dynamic abstractions.
- **Tests**: Concept formation, abstraction reasoning, and type inference.

### Milestone 11: LLM Integration and Learning
- **Tasks**:
  - Implement `UnifiedClient` with `reqwest` for Claude/Ollama.
  - Add knowledge extraction for propositions and implications.
  - Create embedding generation for entities and concepts.
  - Build parameter learning system for updating weights.
  - Implement structure learning for discovering new implications.
  - Add natural language explanation generation.
- **Deliverables**: Automated knowledge extraction and learning.
- **Tests**: Extraction accuracy and learning performance.

### Milestone 12: Web Interface and Visualization
- **Tasks**:
  - Create embedded web server for visualization.
  - Implement interactive belief network graph display.
  - Build query construction interface with auto-completion.
  - Add real-time belief update monitoring.
  - Implement explanation visualization.
  - Create debugging tools for examining propagation.
- **Deliverables**: Graphical interface for exploration and analysis.
- **Tests**: Usability tests and visual correctness.

### Milestone 13: GPU-Accelerated ExponentialModel ✅
- **Tasks**:
  - Add PyTorch bindings (tch-rs) to Cargo.toml with CUDA/MPS support ✅
  - Create TorchExponentialModel implementing FactorModel trait ✅
  - Implement automatic device selection (MPS on macOS, CUDA on Linux/Windows, CPU fallback) ✅
  - Convert weight storage to tensor-based operations ✅
  - Implement modern optimizers (Adam, AdamW, SGD with momentum) ✅
  - Add batch training support for efficient GPU utilization ✅
  - Create learning rate scheduling and early stopping (partial - optimizer created, scheduling pending)
  - Implement vectorized inference for batch predictions ✅
  - Add configuration for optimizer selection and hyperparameters ✅
  - Create benchmarks comparing CPU vs GPU performance ✅
- **Deliverables**: GPU-accelerated training and inference with 10-17% speedup on moderate datasets (50-200 entities)
- **Tests**: Performance benchmarks completed, showing:
  - Standard model faster for small datasets (<10 entities) due to PyTorch overhead
  - Break-even at ~10 entities
  - GPU advantage emerges at 50+ entities: 10.5% faster at 50, 13.9% faster at 100, 17.3% faster at 200
  - MPS device support confirmed on macOS
- **Implementation Notes**:
  - TorchExponentialModel successfully integrated with FactorModel trait
  - Thread-safe implementation using Arc<Mutex<>> for optimizer and VarStore
  - Automatic device selection prioritizes MPS on macOS, CUDA on Linux/Windows
  - Tensor-based weight storage with dynamic resizing
  - Adam optimizer with configurable hyperparameters
  - Batch training infrastructure ready but currently processes examples individually
  - Environment variable BAYESLOG_USE_TORCH controls model selection

### Milestone 14: Model Compatibility and LLM-Friendly Interface ✅
- **Tasks**:
  - Implement weight export/import for ExponentialModel ✅
  - Create ModelWeights common format with metadata ✅
  - Add save/load methods to FactorModel trait ✅
  - Create UnifiedExponentialModel for automatic CPU/GPU switching ✅
  - Implement BeliefMemory high-level API for LLM integration ✅
  - Add entity linking and graph integration ✅
  - Create proposition management with prior beliefs ✅
  - Implement belief updates from observations ✅
  - Add entity-based belief queries ✅
  - Set up automatic libtorch path configuration ✅
- **Deliverables**: Seamless model compatibility and LLM-friendly interface
- **Tests**: Created examples demonstrating:
  - Model compatibility between CPU and GPU implementations
  - BeliefMemory API for agent memory scenarios
  - Automatic environment setup for easier development
- **Implementation Notes**:
  - ModelWeights format supports both ExponentialModel and TorchExponentialModel
  - BeliefMemory provides high-level API for LLM agents
  - Entities and propositions are stored in graph database with proper linking
  - Quick belief updates without full training for interactive scenarios
  - Cargo configuration automatically handles libtorch paths

### Milestone 15: Enhanced GPU Optimization
- **Tasks**:
  - Implement true batch training in TorchExponentialModel (currently processes examples individually)
  - Add learning rate scheduling (cosine annealing, step decay, exponential decay)
  - Implement early stopping based on validation loss
  - Add mixed precision training (FP16/BF16) for faster computation
  - Create GPU memory optimization with gradient checkpointing
  - Implement model parallelism for very large networks
  - Add distributed training support for multi-GPU systems
  - Create ONNX export for inference deployment
  - Optimize batch size based on available GPU memory
  - Add warmup steps for optimizer stability
- **Deliverables**: 10-100x speedup for large-scale training scenarios
- **Tests**: Large-scale benchmarks, memory usage profiling, multi-GPU scaling tests

### Milestone 16: Delta Weights & Online Learning Infrastructure ✅
- **Tasks**:
  - Phase 1: Delta Weight Infrastructure ✅
    - Create DeltaWeights structure for sparse weight updates ✅
    - Implement WeightManager to combine base + delta weights ✅
    - Add update tracking (counts, timestamps) per feature ✅
    - Modify ExponentialModel to use WeightManager ✅
  - Phase 2: Consolidation Logic ✅
    - Implement multiple consolidation triggers (size, time, count, memory) ✅
    - Create smart merging strategies for hot features ✅
    - Add audit trail for consolidation history (partial - stats tracking done)
    - Implement rollback capability for delta weights (deferred - not critical)
  - Phase 3: Configuration & Persistence ✅
    - Add configuration for consolidation thresholds ✅
    - Implement save/load for delta weights ✅
    - Create versioning system for compatibility (using ModelWeights versioning)
    - Support partial weight loading by namespace (implicit via namespace filtering)
- **Deliverables**: LoRA-inspired delta weight system for efficient online learning ✅
- **Tests**: Unit tests for delta operations, consolidation benchmarks, memory usage profiling ✅
- **Implementation Notes**:
  - DeltaWeights tracks sparse updates with timestamps and counts
  - WeightManager seamlessly combines base + delta weights
  - Hot features are kept with decay during consolidation
  - ExponentialModel now uses Arc<RwLock<WeightManager>> for thread safety
  - Consolidation can be triggered manually or automatically based on thresholds

### Milestone 17: Probabilistic AND Gate Implementation ✅
- **Tasks**:
  - Unified AND/OR gate handling in ExponentialModel ✅
  - Designed and implemented AND gate features: ✅
    - `and_size`: Number of premises in conjunction
    - `and_num_true`: Count of true premises (> 0.5 threshold)
    - `and_all_true`: Binary indicator for all premises true
    - `and_any_false`: Binary indicator for any premise false
    - `and_soft`: Product of all premise probabilities (soft AND)
    - `and_min`: Minimum probability (weakest link)
  - Implemented soft AND behavior with probabilistic reasoning ✅
  - Created AND gate training scenario with diverse input patterns ✅
  - Updated inference engine to use learned AND gates ✅
  - Removed legacy heuristic AND gate code (1.0/0.0 returns) ✅
  - Integrated with WeightManager for online updates ✅
- **Deliverables**: Learned probabilistic AND gates using same mechanism as OR gates ✅
- **Tests**: Comprehensive AND gate tests for hard and soft inputs ✅
- **Documentation**: Created detailed AND_GATE_IMPLEMENTATION.md guide ✅
- **Implementation Notes**:
  - AND gates now use the same ExponentialModel as OR gates
  - Feature extraction automatically detects conjunctions
  - 6 specialized features capture AND gate semantics
  - Training examples demonstrate various input combinations
  - Probabilistic outputs enable nuanced reasoning beyond boolean logic

### Milestone 18: Online Learning API & Integration
- **Tasks**:
  - Add targeted update methods to BeliefMemory:
    - update_or_gate() for single OR gate updates
    - update_and_gate() for single AND gate updates
    - consolidate_weights() for manual consolidation
  - Implement importance sampling for selective updates
  - Add weight decay for unused implications
  - Create monitoring and visualization tools:
    - Weight change heatmaps
    - Learning curves per gate
    - Consolidation history dashboard
  - Optimize for memory efficiency with configurable caching
  - Add performance metrics and logging
- **Deliverables**: Complete online learning system for continuous belief network updates
- **Tests**: Online learning convergence tests, API integration tests, performance benchmarks

### Milestone 19: Performance Optimization and Integration
- **Tasks**:
  - Implement SIMD optimization for numeric operations.
  - Add optional GPU acceleration using WGPU.
  - Optimize memory usage with smart caching strategies.
  - Connect all components into a cohesive system.
  - Create comprehensive benchmarking suite.
  - Add thorough error handling and logging (`log` crate).
- **Deliverables**: High-performance integrated system.
- **Tests**: End-to-end tests and performance evaluation.

---

## Implementation Strategy for QBBN Adaptation

### Transparent Graph Database Adapter Approach

After examining the reference implementation, we're adopting a transparent adapter approach for the GraphDB integration. Rather than reimagining the entire QBBN architecture, we'll create a GraphDBAdapter that implements the exact same interface expected by the reference, but stores all data in our graph database instead of Redis.

#### Key Advantages of this Approach

1. **Minimal Code Changes**:
   - Preserves the exact behavior of the reference implementation
   - Requires minimal changes to the core QBBN algorithms
   - Allows direct testing and verification against the reference
   - Reduces risk of subtle implementation differences

2. **Training and Inference Preservation**:
   - Maintains the existing scenarios and training scripts
   - Preserves the ExponentialModel and its training capabilities
   - Keeps all message passing and inference algorithms intact
   - Allows using the same evaluation methods for verification

3. **Structured Graph Storage**:
   - Uses a well-defined schema with consistent node and edge types
   - Stores QBBN components in a semantically appropriate graph structure
   - Leverages the natural structure of graphs for relationships
   - Provides a foundation for future graph-specific optimizations

#### Implementation Components

1. **GraphDBAdapter**:
   - Implements Redis-compatible operations (GET, SET, HGET, HSET, SADD, etc.)
   - Stores data using appropriate graph nodes and relationships
   - Uses consistent naming and structure through schema enums
   - Provides a seamless replacement for RedisManager

2. **Schema Definition**:
   - Clearly defined NodeLabel enum for all node types
   - EdgeLabel enum for relationship types
   - FactorType enum for different logical operations
   - Property naming conventions for consistent data access

3. **Storage Patterns**:
   - Proper graph representation of network components:
     - Proposition nodes with belief values and evidence markers
     - Factor nodes connected to premise and conclusion propositions
     - Predicate nodes with connections to arguments
     - Entity nodes with connections to domains
   - Specialized QBBN operations that leverage graph structure:
     - store_proposition - Creates a full proposition with all related nodes
     - create_factor - Creates a factor with premises and conclusion
     - set_evidence - Sets evidence on proposition nodes
     - update_belief - Updates pi/lambda/belief values on nodes
   - Legacy compatibility through Redis-like primitives:
     - Key-Value storage using KeyValue nodes
     - Hash storage using proper node types based on semantic meaning
     - Set storage using specialized edge connections
     - List storage using ordered edges with position properties

4. **Training Support**:
   - Weight storage in the graph database
   - Training queues for examples
   - Support for the ExponentialModel and its operations
   - Preservation of scenario definitions and training procedures

### Core QBBN Algorithms Preservation

1. **Model-Based Prediction**:
   - Keep the `ExponentialModel` implementation intact
   - Preserve the feature extraction system 
   - Maintain the weight-based prediction mechanism
   - Ensure the same mathematical foundations

2. **Message Passing**:
   - Preserve the pi/lambda message calculations
   - Keep the factor context building mechanisms
   - Maintain the special case handling logic

3. **Training Capabilities**:
   - Keep the training scripts and methods
   - Preserve the weight update mechanisms
   - Maintain the model initialization capabilities

### Enhanced Optimization Integration

After successfully adapting the reference implementation, the following optimizations will be added:

1. **Parallel Message Processing**:
   - Add thread pool for concurrent message calculations
   - Implement work-stealing for efficient thread utilization
   - Add synchronization primitives for concurrent updates

2. **Prioritized Message Scheduling**:
   - Implement message impact scoring
   - Create priority queue for message scheduling
   - Add adaptive priority adjustment

3. **Incremental Propagation**:
   - Add dirty node tracking
   - Implement incremental updates based on node changes
   - Create dependency tracking for efficient update propagation

4. **Adaptive Damping**:
   - Add oscillation detection
   - Implement feedback-based damping adjustment
   - Create topology-aware damping strategies

### Future Enhancements

After successful adaptation, further enhancements will include:

1. **Extended Uncertainty Handling**:
   - Add lower/upper probability bounds
   - Implement confidence-weighted belief propagation
   - Create meta-certainty mechanisms

2. **Temporal Reasoning**:
   - Add time-based belief decay
   - Implement temporal conflict resolution
   - Create recency bias mechanisms

3. **Optimization for Large Networks**:
   - Implement graph partitioning for distributed inference
   - Add boundary detection for localized reasoning
   - Create sparse representation for efficient computation

4. **Counterfactual Reasoning**:
   - Enhance simulation-based counterfactuals
   - Add minimum change analysis
   - Implement counterfactual chain discovery

These enhancements will build on the solid foundation of the adapted reference implementation while adding unique capabilities and optimizations.

---

## Additional Considerations
- **Performance Profiling**: Use `cargo flamegraph` and `tracy` for comprehensive profiling and bottleneck identification.
- **Parallelism**: Utilize thread pools, work stealing, and SIMD for maximizing CPU utilization.
- **GPU Acceleration**: Consider WGPU for large matrix operations in belief propagation.
- **Memory Management**: 
  - Implement smart caching strategies with configurable eviction policies.
  - Use arena allocation for short-lived objects during belief propagation.
  - Employ reference counting for shared belief structures.
- **Scalability**: 
  - Support sharding for large graphs in future iterations.
  - Implement partition schemes for distributed reasoning.
  - Add persistence strategies for billion-scale proposition networks.
- **Extensibility**: 
  - Design plugin system for custom inference algorithms.
  - Provide FFI interface for integration with other languages.
  - Create extension points for custom belief propagation strategies.
- **Real-time Capabilities**:
  - Support event-driven belief updates.
  - Implement streaming interfaces for continuous data ingestion.
  - Add priority-based processing for critical belief updates.
- **Security**: 
  - Add permission system for belief network modifications.
  - Implement validation of external inputs.
  - Support sandboxed execution of user-defined Scheme code.
- **Documentation**: 
  - Generate API docs with `cargo doc` and include comprehensive examples.
  - Create interactive tutorials for the Scheme interface.
  - Provide visualization tools for understanding the system architecture.
  - **Specialized Guides**:
    - `docs/AND_GATE_IMPLEMENTATION.md`: Comprehensive guide for probabilistic AND gates
    - Additional guides to be created for other major features
- **Testing and Verification**:
  - Implement property-based testing for belief propagation.
  - Add fuzzing for robustness against unexpected inputs.
  - Create benchmark suite for performance regression detection.

---

## Instructions for Code Agents
- **Setup**: Initialize with `cargo new bayeslog --lib` and add dependencies in `Cargo.toml`.
- **Structure**: Use a modular design:
  - `src/graph`: Graph database implementation.
  - `src/belief`: Belief network with models, network, and inference submodules.
  - `src/ontology`: Ontology system with types and concept formation.
  - `src/scheme`: Steel Scheme integration with custom primitives.
  - `src/search`: Vector, keyword, and temporal search capabilities.
  - `src/llm`: LLM integration for knowledge extraction and learning.
  - `src/web`: Optional web interface for visualization (later milestone).
- **Testing**: 
  - Use `cargo test` with `#[test]` for each milestone.
  - Target 80% code coverage measured with `cargo tarpaulin`.
  - Implement property-based tests with `proptest` for complex components.
- **Error Handling**: 
  - Use `thiserror` for defining custom error types.
  - Leverage `anyhow` for error context and propagation.
  - Provide clear error messages with context information.
- **Concurrency**: 
  - Use `rayon` for CPU-based parallelism in belief propagation.
  - Implement `crossbeam` channels for message passing between components.
  - Use `parking_lot` for more efficient mutexes where needed.
  - Employ `Arc` for shared ownership of thread-safe components.

This design provides a comprehensive blueprint for BayesLog as a modern knowledge representation and reasoning system, combining the strengths of symbolic AI with probabilistic reasoning, uncertainty handling, and advanced temporal and vector search capabilities.