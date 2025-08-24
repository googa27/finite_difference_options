# ADR-002: Unified Multi-Dimensional PDE Pricing Framework

## Status
Accepted

## Context
The original codebase had separate implementations for 1D and multi-dimensional processes, leading to code duplication and inconsistent interfaces. Different process types required different usage patterns, making the framework difficult to extend and maintain.

## Decision
Implement a unified framework that provides a single interface for all stochastic processes regardless of dimension, organized by domain rather than dimensionality.

## Architecture

### Package Structure
```
src/
├── processes/          # Unified stochastic process implementations
│   ├── base.py        # Abstract interfaces
│   ├── affine.py      # Affine models (GBM, OU, CIR, Heston)
│   └── nonaffine.py   # Non-affine models (CEV, SABR)
├── pricing/           # Financial instruments and pricing engines
├── solvers/           # PDE solvers (ADI, etc.)
└── utils/             # Shared utilities and validation
```

### Key Design Decisions

1. **Unified Interface**: All processes implement `drift(time, state)` and `covariance(time, state)` methods
2. **Covariance Parameterization**: Use covariance matrices instead of diffusion matrices for PDE formulation
3. **Domain Organization**: Organize by mathematical/financial domain rather than dimensionality
4. **Factory Functions**: Provide convenient creation functions for common model configurations
5. **Automatic Solver Selection**: Pricing engine selects appropriate solver based on process dimension

## Consequences

### Positive
- **Maintainability**: Clear domain separation and consistent interfaces
- **Extensibility**: Easy to add new processes and instruments
- **Usability**: Single API for all process types
- **Performance**: Vectorized computations and efficient state handling
- **Type Safety**: Comprehensive dataclasses and NumPy typing

### Negative
- **Migration Effort**: Existing code needs to be updated to use new interfaces
- **Learning Curve**: Users need to understand new package structure

### Neutral
- **Backward Compatibility**: Maintained through deprecation warnings and adapter patterns

## Implementation Notes

### Process Hierarchy
- `StochasticProcess`: Abstract base class
- `AffineProcess`: For processes with affine drift/covariance structure
- `NonAffineProcess`: For processes with non-affine structure

### Validation Strategy
- Parameter validation in `__post_init__` methods
- State validation with dimension and positivity checks
- Covariance matrix validation for positive definiteness

### Performance Optimizations
- Vectorized evaluation of drift and covariance
- Efficient state matrix creation using meshgrid
- Batch processing for multiple grid points

## Alternatives Considered

1. **Dimension-Specific Packages**: Keep separate 1D and multi-D implementations
   - Rejected: Leads to code duplication and inconsistent interfaces

2. **Template-Based Approach**: Use generic programming patterns
   - Rejected: Python's dynamic typing makes this unnecessary and complex

3. **Plugin Architecture**: Separate process implementations as plugins
   - Rejected: Adds unnecessary complexity for this use case

## References
- Original refactoring plan in `.gemini_project/refactoring_plan.md`
- Mathematical foundation in `docs/explanation/black_scholes_fdm.md`
- Implementation details in `docs/explanation/unified_framework.md`
