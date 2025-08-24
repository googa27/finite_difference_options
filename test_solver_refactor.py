#!/usr/bin/env python3
"""Test the solver abstraction refactor."""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_solver_factory():
    """Test solver factory creation."""
    print("Testing solver factory...")
    
    # Import only what we need directly
    from src.processes.base import ProcessDimension
    from src.solvers.base import SolverFactory, FDSolver1D, ADISolverWrapper
    
    # Create mock processes with different dimensions
    class Mock1DProcess:
        @property
        def dimension(self):
            return ProcessDimension(1)
    
    class Mock2DProcess:
        @property
        def dimension(self):
            return ProcessDimension(2)
    
    # Test factory creation
    process_1d = Mock1DProcess()
    process_2d = Mock2DProcess()
    
    solver_1d = SolverFactory.create_solver(process_1d)
    solver_2d = SolverFactory.create_solver(process_2d)
    
    assert isinstance(solver_1d, FDSolver1D), "Factory should create FDSolver1D for 1D process"
    # For 2D, it should create ADISolverWrapper
    assert isinstance(solver_2d, ADISolverWrapper), "Factory should create ADISolverWrapper for multi-D process"
    
    print("✓ Solver factory test passed")


def test_solver_interface():
    """Test solver interface."""
    print("\nTesting solver interface...")
    
    from src.solvers.base import Solver, FDSolver1D
    import numpy as np
    
    # Test that FDSolver1D implements the Solver interface
    solver = FDSolver1D()
    assert isinstance(solver, Solver), "FDSolver1D should implement Solver interface"
    
    print("✓ Solver interface test passed")


def main():
    """Run all tests."""
    print("Testing solver abstraction refactor...")
    
    try:
        test_solver_factory()
        test_solver_interface()
        print("\n✓ All solver abstraction tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()