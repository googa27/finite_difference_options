# 3. Refactor Architecture for Better Extensibility

## Status
Accepted

## Context
The finite difference options pricing framework had several architectural issues that made it difficult to extend and maintain:

1. **Tight coupling** between instrument classes and payoff calculation logic
2. **Inconsistent solver interface** with special handling for different dimensions
3. **Embedded Greeks calculation** within the pricing engine with dimension-specific logic

These issues made it difficult to:
- Add new payoff structures
- Implement new solver types
- Extend Greeks calculation methods
- Maintain consistent APIs across dimensions

## Decision
We will implement three major refactors to improve the architecture:

### 1. Payoff Calculator Abstraction
Create a strategy pattern implementation for payoff calculation:

```python
class PayoffCalculator(ABC):
    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

class EuropeanPayoffCalculator(PayoffCalculator):
    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        # European option payoff logic

class BasketPayoffCalculator(PayoffCalculator):
    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        # Basket option payoff logic

class PayoffCalculatorFactory:
    @staticmethod
    def create_calculator(instrument: UnifiedInstrument) -> PayoffCalculator:
        # Factory method to create appropriate calculator
```

### 2. Solver Abstraction
Create a unified solver interface:

```python
class Solver(ABC):
    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        ...

class FDSolver1D(Solver):
    def solve(self, initial_condition, instrument, *grids, time_grid=None):
        # 1D finite difference solver

class ADISolverWrapper(Solver):
    def solve(self, initial_condition, instrument, *grids, time_grid=None):
        # Multi-dimensional ADI solver wrapper

class SolverFactory:
    @staticmethod
    def create_solver(process: StochasticProcess) -> Solver:
        # Factory method to create appropriate solver
```

### 3. Greeks Calculator Abstraction
Create a strategy pattern implementation for Greeks calculation:

```python
class GreeksCalculator(ABC):
    def calculate(
        self,
        prices: NDArray[np.float64],
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        ...

class FDCalculator1D(GreeksCalculator):
    def calculate(self, prices, *grids, time_grid=None):
        # 1D Greeks calculation

class FDCalculator2D(GreeksCalculator):
    def calculate(self, prices, *grids, time_grid=None):
        # 2D Greeks calculation

class GreeksCalculatorFactory:
    @staticmethod
    def create_calculator(process: StochasticProcess) -> GreeksCalculator:
        # Factory method to create appropriate calculator
```

## Consequences

### Positive
- **Better separation of concerns**: Each component has a single responsibility
- **Easier extensibility**: Adding new payoff types, solvers, or Greeks calculators requires minimal changes
- **Improved testability**: Each component can be tested independently
- **Consistent interfaces**: Uniform APIs across different dimensions
- **Reduced coupling**: Components depend on abstractions rather than concrete implementations

### Negative
- **Increased complexity**: More classes and interfaces to understand
- **Potential performance overhead**: Factory methods and delegation may add minimal overhead
- **Learning curve**: Developers need to understand the new architecture

## Implementation Plan
1. **Payoff Calculator Refactor**: Implement payoff calculators and update instrument classes
2. **Solver Abstraction Refactor**: Create solver interface and update pricing engine
3. **Greeks Calculator Refactor**: Implement Greeks calculators and update pricing engine

## Related Issues
- Issue #123: Need to support exotic option payoffs
- Issue #124: Want to add new solver types
- Issue #125: Need more flexible Greeks calculation

## Tags
architecture, refactoring, extensibility, strategy-pattern, factory-pattern