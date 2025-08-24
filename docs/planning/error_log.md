# Error Log - Refactor Implementation

## Refactor 1: Complete ADI Solver Implementation

### Issues Encountered and Resolutions

#### 1. Placeholder ADI Solver
**Issue**: The ADI solver in `src/solvers/adi.py` was just a placeholder that returned the initial condition without solving the PDE.
**Resolution**: Replaced with a full implementation based on the working code in `src/multidimensional_solver.py`.

#### 2. Unified Pricing Engine Not Using Solver
**Issue**: The unified pricing engine was not actually calling the solver, just returning the initial condition.
**Resolution**: Updated the `price_option` method to call the appropriate solver methods.

#### 3. Affine Process Drift Computation
**Issue**: The drift computation in `src/processes/base.py` was using element-wise multiplication instead of matrix multiplication for the beta coefficient.
**Resolution**: Fixed to use proper matrix multiplication (`@` operator) for affine processes.

#### 4. Grid Utility Functions
**Issue**: The `create_log_grid` function didn't support the `center` parameter expected by tests.
**Resolution**: Enhanced the function to support centered grids.

#### 5. Payoff Computation for Multi-dimensional Grids
**Issue**: The payoff function in `src/pricing/instruments/options.py` was returning a 1D array for multi-dimensional grids.
**Resolution**: Updated to create proper multi-dimensional payoff grids using `meshgrid`.

#### 6. Time Direction Conceptual Mismatch
**Issue**: The tests expect `prices[-1]` to match the terminal payoff, but conceptually in option pricing we solve backward from payoff at maturity to price at time 0.
**Status**: Partially resolved - the solver now works but there's still a conceptual mismatch with test expectations.

#### 7. Missing Boundary Conditions
**Issue**: The current implementation lacks proper boundary condition handling for multi-dimensional processes.
**Status**: Identified for future work in Refactor 3.

### Test Results After Changes

#### Passing Tests:
- `tests/test_unified_pricing_engine.py::TestUnifiedPricingEngine::test_engine_initialization_2d`
- ADI solver standalone tests

#### Still Failing Tests:
- `tests/test_unified_pricing_engine.py::TestUnifiedPricingEngine::test_price_option_2d_heston` - Due to time direction mismatch
- Most other tests in `test_unified_pricing_engine.py` - Due to various issues with the unified framework

### Root Cause Analysis

The main issues stem from the conceptual mismatch between how the tests expect the solver to work and how option pricing PDEs should be solved:
1. Tests expect forward-time solving with terminal condition at `prices[-1]`
2. Option pricing requires backward-time solving from terminal payoff to initial price
3. The current implementation solves forward in time but the test expectations are for a backward-time solution

This suggests that either:
1. The tests need to be updated to reflect proper option pricing semantics
2. The solver needs to be modified to solve backward in time
3. The interface needs to be adjusted to match test expectations while maintaining correct pricing semantics