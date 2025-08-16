# Finite Difference Option Pricing

[![CI](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml/badge.svg)](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml)

A small educational project that demonstrates how to price European
options with finite difference methods using the excellent
[`findiff`](https://github.com/findiff/findiff) package.

## Features

- Black–Scholes model implemented with the
  [`findiff.pde.PDE`](https://findiff.readthedocs.io/en/latest/)
  solver.
- Object oriented design following the SOLID principles.
- Unit tests and continuous integration.

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
pricer = BlackScholesPDE(model, market)
values = pricer.price(option, s, t)
price_at_S0 = values[-1, np.searchsorted(s, 1.0)]
print(price_at_S0)
```

## Running the tests

```bash
pytest
```

## License

MIT
