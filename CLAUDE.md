# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Osprey is a production-ready framework for deploying agentic AI in safety-critical control system environments (particle accelerators, fusion experiments, beamlines). It uses Claude Code with MCP servers for orchestration, mandatory human approval for hardware writes, and protocol-agnostic connectors (EPICS, Mock).

## Common Commands

### Environment
```bash
# Install all dependencies (creates .venv automatically)
uv sync --extra dev

# Add a new dependency
uv add <package>

# Run any command in the project venv
uv run <command>

# Build the package
uv build
```

### Testing
```bash
# Unit tests (fast, no API keys required)
uv run pytest tests/ --ignore=tests/e2e -v

# E2E tests (MUST use path, NOT marker)
uv run pytest tests/e2e/ -v

# Single test file
uv run pytest tests/path/to/test_file.py -v

# Single test
uv run pytest tests/path/to/test_file.py::test_function_name -v
```

**Critical**: E2E tests MUST be run with `pytest tests/e2e/` NOT `pytest -m e2e`. The marker-based approach causes registry state leaks and service conflicts.

**Note**: Plain `pytest` also works if the `.venv` is activated after `uv sync`.

### Validation Scripts
```bash
./scripts/quick_check.sh          # Pre-commit: formatting + fast tests (~30s)
./scripts/ci_check.sh             # Pre-push: full CI replication (2-3 min)
./scripts/premerge_check.sh main  # Pre-PR: comprehensive validation (1-2 min)
```

### Linting & Type Checking
```bash
uv run ruff check src/ tests/     # Lint
uv run ruff format src/ tests/    # Format (auto-fix)
uv run mypy src/                  # Type check
```

### Documentation
```bash
cd docs && make html              # Build docs (after uv sync --extra docs)
```

### Pre-commit Hooks
```bash
pre-commit install                # One-time setup
pre-commit run --all-files        # Manual run
```

## Architecture

### Core Components (src/osprey/)

- **cli/**: 13 lazy-loading Click commands (`init`, `deploy`, `config`, `health`, `migrate`, `tasks`, `claude`, `eject`, `channel-finder`, `ariel`, `artifacts`, `web`, `prompts`)
- **connectors/**: Control system adapters (EPICS, Mock) via `ConnectorFactory`
- **mcp_server/**: 5 FastMCP server packages (`control_system`, `python_executor`, `workspace`, `accelpapers`, `matlab`)
- **models/providers/**: LiteLLM-based adapters (Anthropic, OpenAI, Google, CBORG, AMSC, ALS-APG, Argo, Stanford, Ollama, AskSage, vLLM)
- **services/**: Channel Finder (3 pipeline variants), Python Executor (containerized), ARIEL Search, DePlot
- **interfaces/**: Web Terminal (FastAPI + PTY), ARIEL (Flask + MCP), Artifacts Gallery, Channel Finder UI
- **registry/**: `RegistryManager` for component discovery with lazy-loading
- **runtime/**: `write_channel`/`read_channel` API for Claude-generated Python scripts
- **generators/**: Config update utilities for CLI
- **templates/**: App templates, Claude Code hooks/rules/agents/skills, project scaffolding
- **errors.py**: Framework-level exceptions

### Key Patterns

- **Convention Over Configuration**: Reflection-based validation, automatic component discovery
- **Human-in-the-Loop**: Mandatory approval for hardware writes via Claude Code prompts
- **Protocol-Agnostic**: Pluggable connectors selected via configuration
- **Lazy Loading**: CLI commands load dependencies only when invoked

### Data Flow
1. User → Web Terminal → Claude Code → MCP tools
2. MCP tools → Connectors → Control System → Response

## Code Standards

- **Line length**: 100 characters (ruff)
- **Python**: 3.11+
- **Docstrings**: Google-style
- **Type hints**: Gradual typing enforced with mypy
- **Formatting**: Ruff (linting and formatting)

## Bug Fixing Workflow

When a bug is reported, do NOT start by trying to fix it. Instead:
1. **Write a test first** that reproduces the bug
2. **Use subagents** to attempt the fix and prove it works with the passing test

This ensures bugs are properly reproduced before fixes are attempted and that fixes are verified.

## Test Organization

```
tests/
├── cli/, connectors/, mcp_server/, registry/, ...  # Unit tests by component
├── integration/                                   # Integration tests
├── e2e/                                          # End-to-end (requires API keys)
└── conftest.py                                   # Shared fixtures
```

**Markers**: `unit`, `integration`, `e2e`, `e2e_smoke`, `slow`, `requires_api`, `asyncio`

## Git Workflow

- Always use feature branches; do not commit directly to `main`
- Create pull requests for all changes
- Make atomic commits (one logical change per commit)
- Always keep the changelog in sync with every commit
- Keep changelog entries short for simple changes
- Prefer conventional commit style: `type(scope): short description`
  - Examples: `feat(eval): group logs per test`, `fix(cli): handle missing API key`

## Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Test improvements

