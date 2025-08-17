# Finite Difference Option Pricing

[![CI](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml/badge.svg)](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml)

Educational project demonstrating how to price European options by solving
the Black--Scholes partial differential equation (PDE) with
[`findiff`](https://github.com/findiff/findiff).

## Features

- Clean object‑oriented design following SOLID principles.
- Modular boundary condition builder for easy extension.
- Unit tests covering calls and puts.
- Continuous integration with linting (`ruff`) and tests (`pytest`).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from src.models import Market, GeometricBrownianMotion
from src.options import EuropeanCall
from src.pde_pricer import BlackScholesPDE

market = Market(rate=0.05)
model = GeometricBrownianMotion(rate=0.05, sigma=0.2)
option = EuropeanCall(strike=1.0)

s = np.linspace(0, 3, 100)
t = np.linspace(0, 1, 100)
pricer = BlackScholesPDE(model=model, market=market)
values = pricer.price(option, s, t)
price_at_S0 = values[-1, np.searchsorted(s, 1.0)]
print(price_at_S0)
```

## Development

Run the linter and tests locally:

```bash
ruff .
pytest
```

## License

MIT
