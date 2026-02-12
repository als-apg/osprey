---
name: osprey-release
description: >
  Guides through the complete OSPREY release workflow. Use when the user wants
  to create a release, bump versions, publish to PyPI, create a tag, or needs
  help with pre-release testing and version consistency checks.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Release Workflow

Definitive workflow for creating releases that ensure documentation and version numbers stay in sync.

## Instructions

Walk the user through each step and verify completion before moving to the next.

### Step 0A: Review and Sanitize CHANGELOG (CRITICAL - DO THIS FIRST!)

**DO NOT PROCEED** until you clearly understand what this release contains!

1. **Sanitize the CHANGELOG**

   When multiple PRs are merged, the CHANGELOG can develop issues. Check for and fix:
   - **Duplicate entries**: Same change listed multiple times
   - **Entries in wrong sections**: Changes that ended up in previous version sections instead of `[Unreleased]`
   - **Modifications to released versions**: Previous `## [X.Y.Z]` sections should never be modified
   - **Formatting inconsistencies**: Missing blank lines, inconsistent bullet styles
   - **Orphaned entries**: Items outside of any section header

   ```bash
   # Review the CHANGELOG structure
   grep -n "^## \[" CHANGELOG.md
   ```

   If any issues are found, fix them before proceeding.

2. **Read the `## [Unreleased]` section in `CHANGELOG.md`**
   - What are the major features?
   - What's the theme/focus of this release?
   - Are there breaking changes?

4. **Determine Release Theme and Title**
   - Create a descriptive title based on the main features
   - Examples:
     - "Middle Layer Pipeline for Channel Finder"
     - "Developer Experience & CI/CD Improvements"
     - "Performance Optimizations & Bug Fixes"

5. **Verify Completeness**
   - All merged PRs documented?
   - All breaking changes noted?
   - Migration steps included if needed?

6. **Check for Breaking Changes → Migration Document**
   - Does CHANGELOG "Changed" or "Removed" section affect public API?
   - If YES: Follow the migration workflow to create migration document
   - Migration document must be committed before tagging release
   - See: `src/osprey/assist/tasks/migrate/authoring/README.md`

7. **Plan Release Notes**
   - Identify the top 3-5 features to highlight
   - Note any important upgrade instructions
   - Prepare user-facing descriptions

**Confirm the release theme and title with the user before proceeding.**

### Step 0B: Pre-Release Testing (CRITICAL)

**Always run tests before starting the release process.**

1. **Create Clean Virtual Environment (Dependency Verification)**

   Your development venv may have packages installed that aren't in `pyproject.toml`. A clean venv catches missing dependencies before users encounter them.

   ```bash
   # Create a fresh venv (don't delete your main one)
   python -m venv .venv-release-test
   source .venv-release-test/bin/activate

   # Install osprey with dev dependencies from scratch
   pip install -e ".[dev]"
   ```

   **Why this matters**: We've had CI failures where tests passed locally but failed in CI because developers had manually installed packages (like `langchain-openai`, `fastmcp`) that weren't declared as dependencies.

2. **Run Unit & Integration Tests**
   ```bash
   # Run all unit tests (excluding e2e tests)
   # IMPORTANT: Must exclude e2e tests due to registry mocking
   pytest tests/ --ignore=tests/e2e -v
   ```
   - **Any failures = STOP**: Fix issues before proceeding
   - **Expected**: ~1850 tests, completes in ~1-2 minutes
   - **Cost**: Free (no API calls)

3. **Run End-to-End Tutorial Tests**
   ```bash
   # Run e2e tests separately (uses real registry, no mocks)
   # IMPORTANT: Must run separately from unit tests
   pytest tests/e2e/ -v
   ```
   - **Any failures = STOP**: These validate the core user experience
   - **Expected**: ~32 tests, completes in ~10-12 minutes
   - **Cost**: ~$1-2 in API calls

   **Why Separate Test Commands?**
   - Unit tests use mocked registry for fast, isolated testing
   - E2E tests use real registry for integration testing
   - Running together causes registry contamination
   - Always run both test suites before releasing

4. **Verify All Tests Pass**
   - Unit & Integration tests: must pass
   - End-to-End tests: must pass
   - **PROCEED**: Continue to Step 1

5. **Cleanup Test Environment**
   ```bash
   # Return to your main development venv
   deactivate
   rm -rf .venv-release-test
   source venv/bin/activate  # or your main venv
   ```

### Step 1: Pre-Release Version Updates (CRITICAL)

**Update all version numbers BEFORE creating the GitHub release/tag.**

1. **Update Version Numbers** in these files:

   | File | Location | Example |
   |------|----------|---------|
   | `pyproject.toml` | Line 7 | `version = "X.Y.Z"` |
   | `src/osprey/__init__.py` | Line 15 | `__version__ = "X.Y.Z"` |
   | `src/osprey/cli/main.py` | Line 20 | `__version__ = "X.Y.Z"` |
   | `RELEASE_NOTES.md` | Line 1 | `# Osprey Framework - Latest Release (vX.Y.Z)` |
   | `CHANGELOG.md` | New section | `## [X.Y.Z] - YYYY-MM-DD` |
   | `README.md` | Line 12 | `**Latest Release: vX.Y.Z**` |

2. **Run Version Consistency Check**
   ```bash
   echo "=== VERSION CONSISTENCY CHECK ==="
   echo "pyproject.toml:        $(grep 'version = ' pyproject.toml)"
   echo "osprey/__init__.py: $(grep '__version__ = ' src/osprey/__init__.py)"
   echo "cli/main.py:          $(grep '__version__ = ' src/osprey/cli/main.py)"
   echo "RELEASE_NOTES.md:     $(head -1 RELEASE_NOTES.md)"
   echo "README.md:            $(grep 'Latest Release:' README.md)"
   echo "CHANGELOG.md:         $(grep -m1 '## \[' CHANGELOG.md)"
   ```

3. **Commit Version Updates**
   ```bash
   git add pyproject.toml src/osprey/__init__.py src/osprey/cli/main.py RELEASE_NOTES.md CHANGELOG.md README.md
   git commit -m "release: Bump version to X.Y.Z"
   git push origin main
   ```

**Verify all versions match before proceeding.**

### Step 2: Create and Push Tag (Triggers Automated Release)

**GitHub Actions handles build, PyPI publishing, and release creation!**

```bash
# 1. Ensure you're on main and up to date
git checkout main
git pull origin main

# 2. Create and push tag (use your version number)
git tag vX.Y.Z
git push origin vX.Y.Z
```

**What happens automatically:**
1. GitHub Actions triggers (`.github/workflows/release.yml`)
2. Builds package (creates wheel and source distribution)
3. Publishes to PyPI (using trusted publishing/OIDC)
4. Creates GitHub Release (extracts notes from CHANGELOG.md)

### Step 3: Verify Release

**Monitor the GitHub Actions workflow:**

```bash
# Option 1: Use GitHub CLI to monitor
gh run list --limit 5

# Option 2: Check GitHub web interface
# Go to: https://github.com/als-apg/osprey/actions
```

**Verify deployment:**

1. **Check PyPI** - Package should appear at: `https://pypi.org/project/osprey-framework/X.Y.Z/`
2. **Check GitHub Release** - Release should appear at: `https://github.com/als-apg/osprey/releases/tag/vX.Y.Z`
3. **Test installation**:
   ```bash
   pip install --upgrade osprey-framework
   python -c "import osprey; print(osprey.__version__)"  # Should print: X.Y.Z
   ```

### Step 4: Manual PyPI Publishing (Fallback Only)

**ONLY USE IF GITHUB ACTIONS FAILS**

If the automated workflow fails, you can manually publish:

```bash
# 1. Clean previous builds
rm -rf dist/ build/ src/*.egg-info/

# 2. Build the package
python -m build

# 3. Check the built package
twine check dist/*

# 4. Upload to PyPI (requires PyPI credentials)
twine upload dist/*
```

## Workflow Summary

| Step | Action | Verification |
|------|--------|--------------|
| **0A** | Review CHANGELOG | Confirm theme with user |
| **0B** | Pre-release testing | All tests pass |
| **1** | Version updates | Consistency check passes |
| **2** | Create tag | Tag pushed successfully |
| **3** | Verify release | PyPI + GitHub Release exist, install works |

After each major step, confirm success before proceeding to next step.
If any step fails, help troubleshoot before continuing.
