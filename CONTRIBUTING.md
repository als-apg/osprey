# Contributing to Osprey Framework

Thank you for your interest in contributing to Osprey! 🎉

This document provides a quick start guide. For comprehensive contribution guidelines, please visit our **[full Contributing Guide in the documentation](https://als-apg.github.io/osprey/contributing/)**.

## Quick Start

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/osprey.git
cd osprey
```

### 2. Set Up Development Environment

```bash
# Install all development dependencies (creates .venv automatically)
uv sync --extra dev --extra docs
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes and Test

```bash
# Run tests
pytest tests/ --ignore=tests/e2e -v

# Run linters
ruff check src/ tests/
ruff format --check src/ tests/
```

### 5. Submit Pull Request

- Push your branch to GitHub
- Open a Pull Request with a clear description
- Address review feedback

## Branch Strategy

Osprey uses **GitHub Flow**: `main` is the single long-lived branch and is always the PR target. Releases are CalVer tags (`vYYYY.M.P`) on `main` — there is no separate `develop`, `release/*`, or `next` branch. Hotfixes follow the same flow: branch from the tag or from `main`, open a PR back to `main`, and tag a follow-up release.

For details (CI gates, branch protection, release cuts) see the [full Contributing Guide](https://als-apg.github.io/osprey/contributing/).

## Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

## Code Standards

- Follow PEP 8 (100 character line length)
- Use Ruff for linting and formatting
- Add tests for new functionality
- Write Google-style docstrings
- Update documentation as needed

## Running Tests

```bash
# Unit tests (fast)
pytest tests/ --ignore=tests/e2e -v

# E2E tests (requires API keys)
pytest tests/e2e/ -v
```

## Building Documentation

```bash
cd docs
make html
```

## Need Help?

- Read the [full Contributing Guide](https://als-apg.github.io/osprey/contributing/)
- Check [existing issues](https://github.com/als-apg/osprey/issues)
- Join [GitHub Discussions](https://github.com/als-apg/osprey/discussions)

## Code of Conduct

Be respectful, welcoming, and inclusive. Focus on what's best for the community.

---

For detailed guidelines please visit our **[complete Contributing documentation](https://als-apg.github.io/osprey/contributing/)**.
