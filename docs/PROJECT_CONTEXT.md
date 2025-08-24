# Finite Difference Options Pricing - AI Project Context

## Template Metadata
- **Template Version**: 1.0
- **Project Type**: Quantitative Finance Library
- **AI Maintenance**: Enabled
- **Last Updated**: 2025-08-22T14:00:00-04:00

## Project Overview
Unified multi-dimensional PDE pricing framework for financial derivatives using finite difference methods. Supports both 1D and multi-dimensional stochastic processes with automatic solver selection and comprehensive validation.

## Architecture Map
- **Core Package**: src/
- **Domain Packages**: processes/, pricing/, solvers/, utils/
- **External APIs**: FastAPI REST service, Streamlit web app, CLI interface
- **Deployment**: Python package with multiple interfaces

## Current Architecture
- **processes/**: Unified stochastic process implementations (affine/non-affine)
  - Base classes: `StochasticProcess`, `AffineProcess`, `NonAffineProcess`
  - 1D Models: GBM, OU, CIR, CEV
  - Multi-D Models: Heston (2D), SABR (2D)
- **pricing/**: Financial instruments and pricing engines
  - Instruments: `EuropeanOption`, `BasketOption`
  - Engine: `UnifiedPricingEngine` with automatic solver selection
- **solvers/**: PDE solvers with multi-dimensional support
  - 1D: Finite difference with boundary conditions
  - Multi-D: ADI (Alternating Direction Implicit) solver
- **utils/**: Shared utilities and validation

## Key Design Principles
1. **Unified Interface**: Single interface for 1D and multi-dimensional processes `(time, state)`
2. **Covariance Parameterization**: Use covariance matrices for PDE formulation
3. **Domain-Focused Organization**: Organize by financial domain, not dimension
4. **Factory Functions**: Convenient creation of standard models
5. **Comprehensive Validation**: Parameter validation and error handling
6. **Vectorized Computation**: Batch evaluation for performance

## Available Models
- **1D Affine**: GBM, OU (Ornstein-Uhlenbeck), CIR (Cox-Ingersoll-Ross)
- **1D Non-Affine**: CEV (Constant Elasticity of Variance)
- **Multi-D Affine**: Heston (stochastic volatility)
- **Multi-D Non-Affine**: SABR (stochastic alpha-beta-rho)

## Recent Changes
- ✅ Completed unified framework refactoring (2025-08-22)
- ✅ Removed legacy files (unified_*.py) after testing
- ✅ Updated all imports and tests to new package structure
- ✅ Created comprehensive documentation and ADRs
- ✅ Implemented deprecation warnings for backward compatibility

## Next Priorities
- Performance optimization leveraging affine process properties
- Complete ADI solver implementation beyond placeholders
- Additional model implementations (jump-diffusion, regime-switching)
- Production deployment guides and monitoring
- Enhanced Greeks computation and risk management

## AI Maintenance Rules
- Update `reference/api-reference.md` on code changes in src/
- Update `reference/model-catalog.md` when new processes added
- Update `planning/roadmap.md` from tasks.sqlite weekly
- Validate all code examples monthly
- Generate `compliance/validation-report.md` on releases
- Create ADR for any architectural decisions

## Template Variables
- **PROJECT_NAME**: "Finite Difference Options Pricing"
- **MAIN_DOMAIN**: "Quantitative Finance PDE Solving"
- **PRIMARY_MODELS**: ["GBM", "Heston", "SABR", "CIR", "OU", "CEV"]
- **TARGET_USERS**: ["Quantitative Analysts", "AI Agents", "Financial Engineers"]
- **DEPLOYMENT_TYPES**: ["Python Package", "REST API", "Web App", "CLI"]

## Dependencies
- **Core**: numpy, scipy, findiff
- **Web**: fastapi, streamlit, uvicorn
- **CLI**: typer, click
- **Dev**: pytest, mypy, ruff, black
- **Optional**: plotly (enhanced plotting)

## Testing Strategy
- Unit tests for all process implementations
- Integration tests for complete pricing workflows
- Validation against analytical Black-Scholes solutions
- Performance benchmarks for large grids
- Regression tests for API stability
