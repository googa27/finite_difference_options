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
- Basic regulatory reporting utilities (CRIF, CUSO, Basel, FRTB placeholders).

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
res = pricer.compute_grid(
    maturity=t[-1],
    s_max=s[-1],
    s_steps=len(s),
    t_steps=len(t),
    strike=option.strike,
    option_type="Call",
    return_greeks=True,
)
# Grid orientation: values.shape == (len(t), len(s))
price_at_S0 = res.values[-1, np.searchsorted(res.s, 1.0)]
print(price_at_S0)
```

### Pricing a callable bond

```python
import numpy as np
from src.option_pricer import OptionPricer
from src.pde_pricer import CallableBondPDEModel
from src.models import Market, GeometricBrownianMotion

market = Market(rate=0.03)
short_rate = GeometricBrownianMotion(rate=0.03, sigma=0.01)
bond_model = CallableBondPDEModel(
    face_value=100.0,
    call_price=105.0,
    market=market,
    model=short_rate,
)
pricer = OptionPricer(pde_model=bond_model)
s, t, values = pricer.compute_grid(
    maturity=1.0,
    s_max=150.0,
    s_steps=100,
    t_steps=100,
)
price_at_par = values[-1, np.searchsorted(s, 100.0)]
print(price_at_par)
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

In the app you can switch plotting backends between Matplotlib and Plotly.
Plotly is optional but recommended for interactivity (requirements already pin
`plotly>=6.3,<7`).

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
- `POST /reports/crif` → `{ "crif": str }`
- `POST /reports/cuso` → `{ "status": str }`
- `POST /reports/basel` → `{ "status": str }`
- `POST /reports/frtb` → `{ "status": str }`

## Next.js Client

A minimal client in `nextjs-client/` demonstrates how to call the API from the browser:

```bash
cd nextjs-client
npm install
npm run dev
```

It expects the FastAPI server to run locally on port 8000.

## Regulatory Documentation

Details on assumptions and limitations are available in [docs/regulatory.md](docs/regulatory.md).

## License

MIT
