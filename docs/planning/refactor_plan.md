# Refactor Plan

## Overview
This document outlines the planned refactors to enhance the finite difference options pricing framework based on analysis of the current codebase and documentation.

## Refactor 1: Complete ADI Solver Implementation - COMPLETED

### Problem
The current ADI (Alternating Direction Implicit) solver in `src/solvers/adi.py` only contained placeholder methods without actual implementation. This prevented proper multi-dimensional pricing.

### Solution Implemented
I've replaced the placeholder implementation with a complete ADI solver that includes:
1. Proper 2D and 3D PDE solving capabilities using the Alternating Direction Implicit method
2. Operator splitting techniques for multi-dimensional problems
3. Tridiagonal solvers using the Thomas algorithm
4. Proper handling of drift and covariance terms
5. Boundary condition placeholders (to be enhanced in future work)

### Key Changes Made
1. Replaced the minimal placeholder ADI solver in `src/solvers/adi.py` with a full implementation based on the working code in `src/multidimensional_solver.py`
2. Updated the unified pricing engine in `src/pricing/engines/unified.py` to actually use the solver instead of returning the initial condition
3. Fixed the affine process drift computation in `src/processes/base.py` to use proper matrix multiplication instead of element-wise multiplication
4. Enhanced the grid utility functions in `src/pricing/engines/unified.py` to support centered grids
5. Fixed the payoff computation in `src/pricing/instruments/options.py` to properly handle multi-dimensional grids

### Testing Results
- The ADI solver now properly solves PDEs as demonstrated by test cases
- The unified pricing engine now calls the actual solver
- Several previously failing tests now pass

### Issues Encountered
1. **Time Direction Mismatch**: The tests expect `prices[-1]` to match the terminal payoff, but in option pricing we typically solve backward from payoff at maturity (T) to price at time 0. The solver works forward in time, so this creates a conceptual mismatch.
2. **Boundary Conditions**: The current implementation lacks proper boundary condition handling for multi-dimensional processes, which is critical for accurate pricing.

## Refactor 2: Enhance Greeks Computation - PLANNED

### Problem
The current Greeks computation in `src/pricing/engines/unified.py` is incomplete and doesn't handle all cases properly, particularly for multi-dimensional processes.

### Solution
Implement a complete Greeks computation system with:
1. Proper finite difference methods for all Greeks
2. Support for multi-dimensional processes
3. Better numerical stability
4. Comprehensive error handling

### Implementation Plan
1. Implement proper finite difference computation for all Greeks
2. Add support for multi-dimensional processes
3. Improve numerical stability with adaptive step sizes
4. Add comprehensive tests

## Refactor 3: Proper Boundary Conditions for Multi-Dimensional Processes - PLANNED

### Problem
The current implementation lacks proper boundary condition handling for multi-dimensional processes, which is critical for accurate pricing.

### Solution
Implement a comprehensive boundary condition system with:
1. Proper boundary condition classes for multi-dimensional processes
2. Support for different types of boundary conditions (Dirichlet, Neumann, etc.)
3. Automatic boundary condition selection based on process type
4. Proper integration with the ADI solver

### Implementation Plan
1. Create boundary condition classes for multi-dimensional processes
2. Implement automatic boundary condition selection
3. Integrate with the ADI solver
4. Add comprehensive tests