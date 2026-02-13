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
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install in development mode
pip install -e ".[dev,docs]"
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

## Developer Workflows

We provide comprehensive workflow guides optimized for AI-assisted development:

### Browsing Tasks

```bash
# List all available tasks
osprey tasks list

# Show details about a specific task
osprey tasks show pre-merge-cleanup
```

### Essential Workflows

**Before committing:**
- `pre-merge-cleanup` - Scan for issues before commit
- `commit-organization` - Organize atomic commits

**While coding:**
- `docstrings` - Professional docstring writing
- `comments` - Strategic inline comments

**After changes:**
- `update-documentation` - Keep docs in sync

### Using with AI Assistants

For Claude Code, install tasks as skills:

```bash
osprey claude install pre-commit
```

For other AI assistants (Cursor, Copilot), copy tasks to your project:

```bash
osprey tasks copy pre-merge-cleanup
```

Then reference with `@` mentions:

```
@.ai-tasks/pre-merge-cleanup/instructions.md Scan my uncommitted changes
```

Each task includes an "AI Quick Start" section with detailed prompt guidance.

### Automation

Run automated pre-merge checks:

```bash
./scripts/premerge_check.sh main
```

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

For detailed guidelines, workflow files, AI-assisted development tips, and more, please visit our **[complete Contributing documentation](https://als-apg.github.io/osprey/contributing/)**.
