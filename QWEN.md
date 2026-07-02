# Repository Guidelines

## General Guidelines
- Preferably, follow OOP principles (encapsulation, inheritance, polymorphism).
- Preferably, follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion).
- Preferably, use design patterns (Factory, Singleton, Observer, Strategy, Decorator, Adapter, Bridge, Composite, Iterator, Command, Mediator, Memento, State, Visitor).
- Make tests frequently as you code. Make sure tests pass before you continue coding.
- After ending a large chunk of code, test it and review it.
- All failures to pass tests should be registered in '.gemini_project'.
- Use a virtual environment.
- Use type hints.
- Prefer composition over inheritance.
- Before coding or reading anything else, you need to know the contents of 'README.md' and '.agent_workspace/README.md' for the project status, past refactors and future tasks.
- When writing python code, follow PEP8 standards.

## Project Structure & Module Organization
- `src/finite_difference_options/`: Installable core package (PDE models, pricers, solvers, validation, integrations, optional API/CLI/plotting/risk adapters).
- `tests/`: Unit, architecture, packaging, integration, and numerical tests.
- `src/finite_difference_options/api/`: FastAPI service (run with `uvicorn finite_difference_options.api.main:app --reload`).
- `src/finite_difference_options/cli/`: Typer CLI entry point (`fd-options`).
- `nextjs-client/`: Optional Next.js example client consuming the API.
- `docs/`: Explanations, architecture, CI policy, and regulatory notes.

## Build, Test, and Development Commands
- Install: `python -m pip install -e '.[dev]'` (or `python -m pip install .` for runtime only)
- Lock smoke: `python -m pip install -r requirements-dev.lock.txt && python -m pip check`
- Pre-commit: `pre-commit install` then `pre-commit run --all-files`
- Lint: `ruff check . --select E9,F63,F7,F82`  | Types: `mypy --ignore-missing-imports --follow-imports=silent src/finite_difference_options/contracts src/finite_difference_options/validation scripts/check_architecture_contract.py`  | Format: `black .`
- Tests: `pytest -q tests/architecture tests/test_packaging_contract.py --no-cov`, then `pytest -q`
- Build/audit: `python -m build --sdist --wheel && python -m twine check dist/* && python -m pip_audit --progress-spinner=off --skip-editable`
- API: `uvicorn finite_difference_options.api.main:app --reload`
- CLI: `fd-options price --option-type Call --strike 1 --maturity 1`
- Next.js client: `cd nextjs-client && npm ci && npm run dev`

## Coding Style & Naming Conventions
- Python: Black formatting, Ruff linting (E, F, B), line length 120 (`pyproject.toml`).
- Types: `mypy` enforced on contract-critical modules; avoid untyped defs; keep `finite_difference_options` importable as an installed package.
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

## AI Project Management
- **Primary Project State:** For all project-specific context, including requirements, goals, tasks, decisions, and long-term memory, refer to the `.gemini_project/` directory at the project root. This directory is the authoritative source for the project's current state and history.
- **Task Management:** All active tasks are managed via the `project_tasks.sqlite` database within `.gemini_project/`.
- **Long-Term Memory:** Semantic search for project history and code context should utilize the vector store located in `.gemini_project/project_memory/`.
- **User Instructions:** For detailed guidance on project setup and context recovery, consult `.gemini_project/INSTRUCTIONS.md`.
