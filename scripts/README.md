# Osprey Testing Scripts

This directory contains testing and validation scripts for the Osprey Framework development workflow.

## Quick Reference

| Script | Purpose | Duration | When to Use |
|--------|---------|----------|-------------|
| `quick_check.sh` | Fast pre-commit validation | < 30s | Before every commit |
| `ci_check.sh` | Full CI replication | 2-3 min | Before pushing |
| `premerge_check.sh` | Pre-merge validation | 1-2 min | Before creating PR |
| `check_config_keys.py` | Config-key resurrection guard | 2-5s | After touching a `config.yml.j2`, a preset, or config-reading code |
| `changelog_fragments.py` | Changelog-fragment gate and release fold | < 1s | After touching `src/` or `packages/`; when cutting a release |

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
- Validates that a changelog fragment was added (runs `changelog_fragments.py check`)
- Checks type hints
- Validates TODO/FIXME comments have issue links
- Runs code formatters and linters
- Runs test suite

**Usage**:
```bash
# Check against main branch (default)
./scripts/premerge_check.sh main

# Check against current branch's upstream
./scripts/premerge_check.sh
```

**When to use**: Final validation before creating a PR or merging.

**Exit codes**:
- `0`: All checks passed (ready for PR)
- `1`: Blocking issues found (must fix before PR)

**Severity levels**:
- **BLOCKERS**: Must fix (debug code, secrets, test failures)
- **CRITICAL**: Should fix (missing changelog fragment, missing type hints)
- **HIGH**: Address before merge (unlinked TODOs)
- **MEDIUM**: Good to fix (formatting issues)

---

### check_config_keys.py

**Purpose**: Stop deleted config keys from coming back, and stop live keys from
quietly losing their reader.

**What it does**: Renders the shipped `config.yml.j2` templates, collects every
dotted key they produce, and checks each one against
`scripts/config_key_manifest.yml` — which records either the code fragment that
reads the key or the structural reason it has none. It also re-checks the keys
that were deliberately deleted (they must not reappear in a rendered template,
a preset `config:` override, or the loader's synthesized defaults), the code
sites that went with them, cross-template parity, and the manifest's own
internal consistency. The script's module docstring is the authoritative list
of failure modes.

**Usage**:
```bash
# Plain run (~2.4s) — works anywhere, including a shallow clone
uv run python scripts/check_config_keys.py

# What CI runs (~4.6s). Same checks, plus the back-test: every orphan-site
# regex must match at the recorded baseline commit AND not match on this
# branch, which is what proves it can actually fail. Extracts the baseline
# tree with `git archive`, so it needs full history — in a shallow clone it
# errors out rather than skipping.
uv run python scripts/check_config_keys.py --back-test

# Same, against a specific baseline instead of the manifest's recorded one
uv run python scripts/check_config_keys.py --back-test <commit>
```

**When to use**: After editing any `config.yml.j2`, any preset under
`src/osprey/profiles/presets/`, or code that reads configuration. Use the plain
form for a quick local check; use `--back-test` to reproduce CI exactly. CI runs
the `--back-test` form as a step of the `lint` job, which checks out with
`fetch-depth: 0`. `tests/scripts/test_config_key_guard.py` exercises the guard
from the unit-test lane as well, back-test included — but those cases skip
themselves when the baseline is unreachable, so that coverage is revocable by a
checkout-depth or marker change. The step's own `!cancelled()` guard is what
pins it down: it runs even when the lint steps before it fail.

**Exit codes**:
- `0`: No findings
- `1`: One or more failure records (each is printed with its key and location),
  or a hard error — notably an unreachable `--back-test` baseline, which raises
  a traceback rather than skipping

---

### changelog_fragments.py

**Purpose**: Make sure a change under `src/` or `packages/` ships a changelog
fragment, and fold the fragments into `CHANGELOG.md` when a release is cut.

**What it does**: `check` validates every fragment in `changelog.d/` (`README.md` is skipped) — the filename
grammar `<name>.<type>.md` and the body rules — and then compares the branch
against its base: a change under `src/` or `packages/` must add a fragment, and
`## [Unreleased]` in `CHANGELOG.md` must not be edited by hand. `apply` inserts
each fragment as a bullet under its `### <Type>` heading in `## [Unreleased]`
and deletes the fragment files.

**Usage**:
```bash
# What the `lint` CI job and premerge_check.sh run
uv run python scripts/changelog_fragments.py check --base origin/main

# The release fold — run once while cutting a release
uv run python scripts/changelog_fragments.py apply
```

**When to use**: `check` after touching `src/` or `packages/`, though CI and
`./scripts/premerge_check.sh` already run it for you. `apply` only when cutting
a release.

**Exit codes**:
- `0`: No findings
- `1`: Something you can fix — a missing fragment, a malformed fragment, a
  hand-written entry in `## [Unreleased]`, or a deleted fragment
- `2`: Environment problem — the base ref was not found, or the clone is too
  shallow to find a common ancestor

See `changelog.d/README.md` for the fragment format and the list of types.

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

See `docs/source/contributing/development-setup.rst` for the CI and testing guide.

## Contributing

If you modify these scripts:

1. Test them thoroughly on both macOS and Linux
2. Update this README
3. Update `docs/source/contributing/development-setup.rst`
4. Ensure exit codes are correct (0 = success, 1 = failure)
5. Add helpful error messages

## See Also

- [Contribution Workflow](../docs/source/contributing/workflow.rst)
- [Development Setup](../docs/source/contributing/development-setup.rst)
