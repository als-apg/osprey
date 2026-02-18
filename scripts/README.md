# Osprey Testing Scripts

This directory contains testing and validation scripts for the Osprey Framework development workflow.

## Quick Reference

| Script | Purpose | Duration | When to Use |
|--------|---------|----------|-------------|
| `quick_check.sh` | Fast pre-commit validation | < 30s | Before every commit |
| `ci_check.sh` | Full CI replication | 2-3 min | Before pushing |
| `premerge_check.sh` | Pre-merge validation | 1-2 min | Before creating PR |

## Scripts

### quick_check.sh

**Purpose**: Fast pre-commit validation to catch common issues.

**What it does**:
- Auto-fixes code formatting with ruff
- Runs fast unit tests (stops on first failure)

**Usage**:
```bash
./scripts/quick_check.sh
```

**When to use**: Before every commit. This is your first line of defense.

**Exit codes**:
- `0`: All checks passed
- `1`: Checks failed

---

### ci_check.sh

**Purpose**: Replicate the entire GitHub Actions CI workflow locally.

**What it does**:
1. **Linting**: Runs ruff (linting + formatting check) and mypy
2. **Testing**: Runs pytest with coverage reporting
3. **Documentation**: Builds Sphinx docs and checks links
4. **Package**: Builds Python package and validates with twine

**Usage**:
```bash
./scripts/ci_check.sh
```

**When to use**: Before pushing to GitHub. If this passes, CI will almost certainly pass.

**Requirements**:
- Virtual environment must be present (`venv` or `.venv`)
- All dev dependencies installed: `uv sync --extra dev --extra docs`
- Build tools installed: `uv tool install build twine` or `uv pip install build twine`

**Exit codes**:
- `0`: All checks passed (safe to push)
- `1`: One or more checks failed

**Tips**:
- Run this before every push to save CI minutes
- Uses exact same commands as `.github/workflows/ci.yml`
- Shows detailed output for each check

---

### premerge_check.sh

**Purpose**: Comprehensive validation before creating a pull request.

**What it does**:
- Detects debug code (print, breakpoint, pdb)
- Finds commented-out code
- Checks for hardcoded secrets
- Validates CHANGELOG updates
- Checks type hints
- Validates TODO/FIXME comments have issue links
- Runs code formatters and linters
- Runs test suite

**Usage**:
```bash
# Check against main branch (default)
./scripts/premerge_check.sh main

# Check against develop branch
./scripts/premerge_check.sh develop

# Check against current branch's upstream
./scripts/premerge_check.sh
```

**When to use**: Final validation before creating a PR or merging.

**Exit codes**:
- `0`: All checks passed (ready for PR)
- `1`: Blocking issues found (must fix before PR)

**Severity levels**:
- **BLOCKERS**: Must fix (debug code, secrets, test failures)
- **CRITICAL**: Should fix (missing CHANGELOG, missing type hints)
- **HIGH**: Address before merge (unlinked TODOs)
- **MEDIUM**: Good to fix (formatting issues)

---

## Development Workflow

### Recommended Testing Flow

```bash
# 1. Make changes
vim src/osprey/some_file.py

# 2. Quick check before commit
./scripts/quick_check.sh

# 3. Commit if passed
git add .
git commit -m "feat: add new feature"

# 4. Full CI check before push
./scripts/ci_check.sh

# 5. Push if passed
git push origin feature/my-feature

# 6. Final pre-merge check before PR
./scripts/premerge_check.sh main

# 7. Create PR if passed
gh pr create
```

### Pre-commit Hook Integration

For automatic validation on every commit, install pre-commit hooks:

```bash
# One-time setup
pre-commit install

# Now pre-commit runs automatically on git commit
# Manual trigger:
pre-commit run --all-files
```

## Troubleshooting

### "Permission denied" error

Make scripts executable:
```bash
chmod +x scripts/*.sh
```

### "Virtual environment not found"

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### "Module not found" errors

Install dependencies:
```bash
uv sync --extra dev --extra docs
# or
pip install -e ".[dev,docs]"
```

### Tests pass locally but fail in CI

Common causes:
- Different Python version (test on 3.11 and 3.12)
- Different OS (test on both Ubuntu and macOS if possible)
- Missing dependencies in `pyproject.toml`

### Documentation build fails

```bash
# Clean and rebuild
cd docs
make clean
make html

# Check for missing dependencies
uv sync --extra docs
# or
pip install -e ".[docs]"
```

## CI/CD Integration

These scripts are designed to match the GitHub Actions workflows:

- `.github/workflows/ci.yml`: Main CI pipeline
- `.github/workflows/release.yml`: Release automation
- `.pre-commit-config.yaml`: Pre-commit hooks

See `docs/source/contributing/05_ci-cd-testing.rst` for comprehensive CI/CD testing guide.

## Contributing

If you modify these scripts:

1. Test them thoroughly on both macOS and Linux
2. Update this README
3. Update `docs/source/contributing/05_ci-cd-testing.rst`
4. Ensure exit codes are correct (0 = success, 1 = failure)
5. Add helpful error messages

## See Also

- [Git and GitHub Workflow](../docs/source/contributing/02_git-and-github.rst)
- [CI/CD Testing Guide](../docs/source/contributing/05_ci-cd-testing.rst)
- [Code Standards](../docs/source/contributing/03_code-standards.rst)
