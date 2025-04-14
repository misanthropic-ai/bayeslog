# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test single: `cargo test test_name`
- Lint: `cargo clippy`
- Format: `cargo fmt`

## Code Style Guidelines
- Follow Rust standard naming conventions (snake_case for variables/functions, CamelCase for types)
- Use descriptive variable names that convey meaning
- Prefer Result/Option handling with `?` operator over unwrap/expect where appropriate
- Keep functions focused on a single responsibility
- Use meaningful error messages in custom error types
- Document public API with rustdoc comments
- Use cargo fmt for consistent formatting
- Follow the standard Rust module organization pattern
- Prefer immutable variables (let) over mutable ones (let mut) when possible

## Project Requirements
Project requirements are stored in the PROJECT.md of the base directory. This should be consulted regularly, and at the start of each session to ensure the project goals are met.

## Project Management
Current and future tasks are stored and tracked in TODO.md. This should be initialized with all tasks to be complete for the current milestone and regularly updated with progress, expanded with new tasks, and pruning of any that are found redundant.

## Reference Implementations
The repository contains reference implementations in the `reference/` directory:
- `reference/bayes-star` - A reference implementation of the Quantified Boolean Bayesian Network (QBBN)
- `reference/loopybayesnet` - A reference implementation of Loopy Belief Propagation 

IMPORTANT: These reference implementations are FOR REFERENCE ONLY. You should NOT modify or work on these files. They are provided solely to help understand the concepts and algorithms that will be implemented in our own codebase. Our implementation will be built from scratch using our existing graph database code as foundation.