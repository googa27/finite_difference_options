# Finite Difference Option Pricing

[![CI](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml/badge.svg)](https://github.com/PLACEHOLDER/finite_difference_options/actions/workflows/ci.yml)

Educational project demonstrating how to price European options by solving
the Black--Scholes partial differential equation (PDE) with
[`findiff`](https://github.com/findiff/findiff).

## Features

- Clean object‑oriented design following SOLID principles.
- Modular boundary condition builder for easy extension.
- Pluggable time-stepping schemes (Explicit Euler, Crank--Nicolson).
- Unit tests covering calls and puts.
- Finite difference Greeks (Delta, Gamma, Theta).
- Continuous integration with linting (`ruff`), type checking (`mypy`) and tests (`pytest`).

## Installation

```bash
pip install -r requirements.txt
```

For development with linting and test tools:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

```python
import numpy as np
from src.option_pricer import OptionPricer
from src.options import EuropeanCall

pricer = OptionPricer(rate=0.05, sigma=0.2)
option = EuropeanCall(strike=1.0)
s = np.linspace(0, 3, 100)
t = np.linspace(0, 1, 100)
_, _, values, delta, gamma, theta = pricer.compute_grid(
    strike=option.strike,
    maturity=t[-1],
    option_type="Call",
    s_max=s[-1],
    s_steps=len(s),
    t_steps=len(t),
    return_greeks=True,
)
price_at_S0 = values[-1, np.searchsorted(s, 1.0)]
print(price_at_S0)
```

## Development

Install the development tools and activate the pre-commit hooks:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

Run formatting, linting and tests in one go:

```bash
pre-commit run --all-files
```

## Streamlit Demo

Launch an interactive application to explore option prices:

```bash
streamlit run apps/streamlit_app.py
```

## Command-line Interface

Use [Typer](https://typer.tiangolo.com/) commands for quick pricing and plotting:

```bash
python -m cli price --option-type Call --strike 1 --maturity 1 --s0 1 --rate 0.05 --sigma 0.2
python -m cli plot --option-type Call --strike 1 --maturity 1 --output plot.png
```

## FastAPI Service

Start a REST API that exposes pricing and Greek calculations:

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `POST /price` → `{ "price": float }`
- `POST /greeks` → `{ "delta": float, "gamma": float, "theta": float }`

## Next.js Client

A minimal client in `nextjs-client/` demonstrates how to call the API from the browser:

```bash
cd nextjs-client
npm install
npm run dev
```

It expects the FastAPI server to run locally on port 8000.

## License

MIT
