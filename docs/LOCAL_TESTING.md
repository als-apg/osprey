# Local Documentation Testing

Test your documentation builds locally before committing to avoid GitHub Actions failures.

## Quick Start

```bash
# Test locally (replicates GitHub Actions exactly)
cd docs
make test-build
```

That's it! This single command:
- Installs the exact same dependencies as GitHub Actions
- Runs the exact same build process
- Shows the exact same errors/warnings

## Alternative Commands

| Command | Purpose |
|---------|---------|
| `make test-build` | **Recommended**: Full GitHub Actions simulation |
| `make pre-commit-test` | Quick test (assumes dependencies installed) |
| `make test-deps` | Install dependencies only |

## Troubleshooting

- **Import errors**: Check that new modules are either in `pyproject.toml` dependencies or mocked in `conf.py`
- **Missing files**: Ensure all documented modules have proper `__init__.py` files
- **Build fails**: Run `make clean` then try again

The local build shows the same errors as GitHub Actions, so if it passes locally, it should pass in CI.
