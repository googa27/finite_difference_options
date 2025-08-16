# Finite Difference Option Pricing

This project demonstrates how to price European vanilla options using
finite difference methods.  The numerical scheme is implemented with
[`findiff`](https://pypi.org/project/findiff/) and its `findiff.pde`
module.  The code has been refactored following object oriented and
SOLID principles.

## Features

- Black--Scholes model for European call and put options
- Crank--Nicolson finite difference scheme
- Unit tests comparing results with the analytical Black--Scholes
  formula
- Continuous integration with GitHub Actions

## Usage

Install the dependencies and run the tests:

```bash
pip install -r requirements.txt pytest
pytest -q
```

The core API is provided by the ``FiniteDifferencePricer`` class:

```python
from src import Market, EuropeanOption, FiniteDifferencePricer

market = Market(r=0.05)
option = EuropeanOption(strike=100, maturity=1.0, is_call=True)
pricer = FiniteDifferencePricer(market, sigma=0.2, s_max=200, ns=200, nt=200)
price = pricer.price(option, s0=100)
print(price)
```

## Development

Run the test-suite with `pytest` before submitting changes.  The CI
workflow defined in `.github/workflows/ci.yml` ensures that all commits
on the `main` branch pass the tests.
