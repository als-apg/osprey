`CONTRIBUTING.md` now runs its test steps through `uv run` and
`uv run python scripts/check.py quick`, instead of a bare `pytest tests/` that
also collects the CANOPY tests and so fails on a fresh checkout.
