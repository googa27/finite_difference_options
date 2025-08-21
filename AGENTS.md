# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core library (PDE models, pricer, Greeks, plotting, risk). Examples: `pde_pricer.py`, `option_pricer.py`, `greeks.py`, `plotting/`, `risk/`.
- `tests/`: Unit tests for pricing, boundary conditions, and options (e.g., `tests/test_pde_pricer.py`).
- `api/`: FastAPI service (run with `uvicorn api.main:app --reload`).
- `cli/`: Typer-based CLI entry points (run with `python -m cli ...`).
- `apps/`: Streamlit demo (`apps/streamlit_app.py`).
- `nextjs-client/`: Next.js example client consuming the API.
- `docs/`: Explanations and regulatory notes.

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt -r requirements-dev.txt`
- Pre-commit: `pre-commit install` then `pre-commit run --all-files`
- Lint: `ruff check`  | Types: `mypy`  | Format: `black .`
- Tests: `pytest -n auto` (parallel) or `pytest -q` for concise output
- Streamlit: `streamlit run apps/streamlit_app.py`
- API: `uvicorn api.main:app --reload`
- CLI: `python -m cli price --option-type Call --strike 1 --maturity 1`
- Next.js client: `cd nextjs-client && npm ci && npm run dev`

## Coding Style & Naming Conventions
- Python: Black formatting, Ruff linting (E, F, B), line length 120 (`pyproject.toml`).
- Types: `mypy` enforced; avoid untyped defs; keep `src/` importable.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Tests: name files `tests/test_*.py`; use descriptive test names and fixtures where helpful.

## Testing Guidelines
- Use `pytest` with small, deterministic unit tests covering the PDE solver, boundary conditions, and Greeks.
- Prefer analytical Black–Scholes formulas for oracles where possible.
- Run locally with `pytest -n auto` and ensure CI parity.

## Version Control & Pull Requests
- Commits: use imperative mood and a short scope. Conventional prefixes are encouraged (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `plotting:`). Examples in history: `docs: explain Black-Scholes finite difference scheme`, `chore: expand pre-commit and CI pipelines`.
- PRs: include a clear description, linked issues, and rationale. Add tests for new behavior, update docs as needed, and include screenshots for UI changes (Streamlit/Next.js).
- Quality gate: all pre-commit hooks pass, `mypy` clean, `ruff` clean, and `pytest` green before requesting review.

