---
name: osprey-release
description: >
  Guides through the complete OSPREY release workflow. Use when the user wants
  to create a release, bump versions, publish to PyPI, create a tag, or needs
  help with pre-release testing and version consistency checks.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Release Workflow

This skill guides you through creating a properly versioned OSPREY release.

## Instructions

Follow the detailed release workflow in [instructions.md](../../../tasks/release-workflow/instructions.md).

## Quick Reference

### Version Files to Update

Update these files **BEFORE** creating the tag:

| File | Location | Example |
|------|----------|---------|
| `pyproject.toml` | Line 7 | `version = "X.Y.Z"` |
| `src/osprey/__init__.py` | Line 15 | `__version__ = "X.Y.Z"` |
| `src/osprey/cli/main.py` | Line 20 | `__version__ = "X.Y.Z"` |
| `RELEASE_NOTES.md` | Line 1 | `# Osprey Framework - Latest Release (vX.Y.Z)` |
| `CHANGELOG.md` | New section | `## [X.Y.Z] - YYYY-MM-DD` |
| `README.md` | Line 12 | `**Latest Release: vX.Y.Z**` |

### Version Consistency Check

```bash
echo "=== VERSION CONSISTENCY CHECK ==="
echo "pyproject.toml:     $(grep 'version = ' pyproject.toml)"
echo "osprey/__init__.py: $(grep '__version__ = ' src/osprey/__init__.py)"
echo "cli/main.py:        $(grep '__version__ = ' src/osprey/cli/main.py)"
echo "RELEASE_NOTES.md:   $(head -1 RELEASE_NOTES.md)"
echo "README.md:          $(grep 'Latest Release:' README.md)"
echo "CHANGELOG.md:       $(grep -m1 '## \[' CHANGELOG.md)"
```

### Pre-Release Testing

```bash
# Create clean environment (catches missing dependencies)
python -m venv .venv-release-test && source .venv-release-test/bin/activate
pip install -e ".[dev]"

# Run unit tests (~1850 tests, ~2 min, free)
pytest tests/ --ignore=tests/e2e -v

# Run e2e tests (~32 tests, ~12 min, ~$1-2)
pytest tests/e2e/ -v

# Cleanup
deactivate && rm -rf .venv-release-test
```

### Create and Push Tag

```bash
git checkout main && git pull origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Workflow Summary

| Step | Action | Key Commands |
|------|--------|--------------|
| **0A** | Review CHANGELOG | Read `## [Unreleased]`, identify theme |
| **0B** | Pre-release testing | `pytest tests/ --ignore=tests/e2e` then `pytest tests/e2e/` |
| **1** | Version updates | Update 6 files, run consistency check |
| **2** | Create tag | `git tag vX.Y.Z && git push origin vX.Y.Z` |
| **3** | Verify release | Check GitHub Actions, PyPI, test install |

## What Happens Automatically

After pushing the tag, GitHub Actions (`.github/workflows/release.yml`):
1. Builds the package (wheel + source)
2. Publishes to PyPI (trusted publishing/OIDC)
3. Creates GitHub Release (extracts notes from CHANGELOG.md)

## Verification

```bash
# Monitor GitHub Actions
gh run list --limit 5

# Test installation after PyPI publishes
pip install --upgrade osprey-framework
python -c "import osprey; print(osprey.__version__)"
```
