---
workflow: release-workflow
category: release-management
applies_when: [before_release, version_bump, publishing]
estimated_time: 30-45 minutes
ai_ready: true
related: [pre-merge-cleanup, commit-organization]
---

# Release Workflow - Documentation Version Sync Fix

Definitive workflow for creating releases that ensure documentation and version numbers stay in sync.

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant (Cursor/Copilot):**

```
I'm ready to create a new release. Following @src/osprey/assist/tasks/release-workflow/instructions.md,
guide me through the complete release process for version X.X.X.

Walk me through each step and verify completion before moving to the next:

STEP 0A - Review and Sanitize CHANGELOG (CRITICAL - DO THIS FIRST):
1. Sanitize CHANGELOG: check for duplicates, entries in wrong sections, modifications to released versions
2. Read the ## [Unreleased] section in CHANGELOG.md
3. Summarize what this release is about (major features, theme)
4. Verify the changelog accurately reflects all changes
5. Confirm the release theme and title with me before proceeding

STEP 0B - Pre-Release Testing (Clean Environment):
1. Create a fresh venv: python -m venv .venv-release-test && source .venv-release-test/bin/activate
2. Install from scratch: pip install -e ".[dev]"
3. Run unit tests: pytest tests/ --ignore=tests/e2e -v
4. Run e2e tests: pytest tests/e2e/ -v
5. Verify all tests pass before proceeding
6. Cleanup: deactivate && rm -rf .venv-release-test

STEP 1 - Version Updates (BEFORE creating tag):
1. Show me all files that need version updates
2. For each file, show current version and what it should be changed to
3. Generate the version consistency check commands
4. After I update, verify all versions match
5. Stage and commit version bump

STEP 2 - Create and Push Tag (Automated Release):
1. Verify I'm on main branch and pulled latest
2. Generate the git tag command for version X.X.X
3. Generate the git push command for the tag
4. Explain what GitHub Actions will do automatically

STEP 3 - Verify Automated Release:
1. Show me how to monitor GitHub Actions workflow
2. Guide me to verify PyPI publication
3. Guide me to verify GitHub Release creation
4. Help me test the new version installation

After each major step, confirm success before proceeding to next step.
If any step fails, help me troubleshoot before continuing.
```

**Important**: This workflow includes critical testing (Step 0) and proper version sequencing. Don't skip steps!

**Related workflows**: [pre-merge-cleanup.md](pre-merge-cleanup.md), [commit-organization.md](commit-organization.md)

## 🎯 Problem Solved

**Issue**: Documentation shows previous version (e.g., v0.7.1) instead of current version (v0.7.2) because:
1. Documentation builds from current commit using `git describe --tags`
2. Tag creation happens after documentation build
3. Result: Documentation gets the old tag, not the new one

**Solution**: Use GitHub's `GITHUB_REF` environment variable in GitHub Actions to get the exact tag being built.

## 📋 Correct Release Workflow

### **Step 0A: Review and Sanitize CHANGELOG (CRITICAL - DO THIS FIRST!)**

**⚠️ CRITICAL**: Before doing ANYTHING else, understand what you're releasing!

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

**DO NOT PROCEED** until you clearly understand what this release contains!

### **Step 0B: Pre-Release Testing (CRITICAL)**

**⚠️ IMPORTANT**: Always run tests before starting the release process.

1. **Create Clean Virtual Environment (Dependency Verification)**

   **⚠️ CRITICAL**: Your development venv may have packages installed that aren't in `pyproject.toml`. A clean venv catches missing dependencies before users encounter them.

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
   - **What it tests**:
     - Capability unit tests (with mocked registry)
     - Infrastructure components
     - Registry system
     - Configuration management
     - Channel finder databases

3. **Run End-to-End Tutorial Tests**
   ```bash
   # Run e2e tests separately (uses real registry, no mocks)
   # IMPORTANT: Must run separately from unit tests
   pytest tests/e2e/ -v

   # Optional: See detailed execution progress and LLM judge reasoning
   # pytest tests/e2e/ -v -s --e2e-verbose --judge-verbose
   ```
   - **Any failures = STOP**: These validate the core user experience
   - **Expected**: ~32 tests, completes in ~10-12 minutes
   - **Cost**: ~$1-2 in API calls
   - **What it tests**:
     - Complete tutorial workflows (BPM analysis, weather tutorial)
     - Project creation from templates (minimal, hello_world_weather, control_assistant)
     - Multi-capability orchestration
     - Channel finder benchmarks
     - End-user experience

   **⚠️ Why Separate Test Commands?**
   - Unit tests use mocked registry for fast, isolated testing
   - E2E tests use real registry for integration testing
   - Running together causes registry contamination
   - Always run both test suites before releasing

4. **Verify All Tests Pass**
   - Unit & Integration tests: ✅
   - End-to-End tests: ✅
   - **PROCEED**: Continue to Step 1

5. **Fix Any Failing Tests**
   - If tests fail, fix issues first
   - Re-run tests until all pass
   - Commit fixes before proceeding

6. **Cleanup Test Environment**
   ```bash
   # Return to your main development venv
   deactivate
   rm -rf .venv-release-test
   source venv/bin/activate  # or your main venv
   ```

### **Step 1: Pre-Release Version Updates (CRITICAL)**

**⚠️ IMPORTANT**: Update all version numbers BEFORE creating the GitHub release/tag.

1. **Update Version Numbers** (see checklist below)
   - Update `pyproject.toml`, `src/osprey/__init__.py`, `src/osprey/cli/main.py`
   - Update `RELEASE_NOTES.md`, `CHANGELOG.md`, `README.md`
   - Verify documentation files

2. **Run Version Consistency Check**
   ```bash
   # Check all version references
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
   git commit -m "release: Bump version to 0.7.3"
   git push origin main
   ```

### **Step 2: Create and Push Tag (Triggers Automated Release)**

**✅ AUTOMATED**: GitHub Actions handles build, PyPI publishing, and release creation!

```bash
# 1. Ensure you're on main and up to date
git checkout main
git pull origin main

# 2. Create and push tag (use your version number)
git tag v0.9.9
git push origin v0.9.9
```

**What happens automatically:**
1. ✅ **GitHub Actions triggers** (`.github/workflows/release.yml`)
2. ✅ **Builds package** (creates wheel and source distribution)
3. ✅ **Publishes to PyPI** (using trusted publishing/OIDC)
4. ✅ **Creates GitHub Release** (extracts notes from CHANGELOG.md)

### **Step 3: Verify Release**

**Monitor the GitHub Actions workflow:**

```bash
# Option 1: Use GitHub CLI to monitor
gh run list --limit 5

# Option 2: Check GitHub web interface
# Go to: https://github.com/als-apg/osprey/actions
```

**Verify deployment:**

1. **Check PyPI** - Package should appear at: `https://pypi.org/project/osprey-framework/0.9.9/`
2. **Check GitHub Release** - Release should appear at: `https://github.com/als-apg/osprey/releases/tag/v0.9.9`
3. **Test installation**:
   ```bash
   pip install --upgrade osprey-framework
   python -c "import osprey; print(osprey.__version__)"  # Should print: 0.9.9
   ```

### **Step 4: Manual PyPI Publishing (Fallback Only)**

**⚠️ ONLY USE IF GITHUB ACTIONS FAILS**

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

# Optional: Upload to test PyPI first for verification
# twine upload --repository testpypi dist/*
```

## 🔧 Technical Implementation

### **Documentation Version Detection (Fixed)**

The documentation now uses this logic in `docs/source/conf.py`:

```python
def get_version_from_git():
    # 1. In GitHub Actions: Use GITHUB_REF for exact tag
    github_ref = os.environ.get('GITHUB_REF', '')
    if github_ref.startswith('refs/tags/v'):
        return github_ref.replace('refs/tags/v', '')

    # 2. Local builds: Use git describe
    result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], ...)
    return result.stdout.strip().lstrip('v')
```

### **GitHub Actions Workflow**

The workflow triggers on:
- `push` to `main` branch (for development docs)
- `tags: v*` (for release docs with correct version)

## 🧪 Testing the Fix

### **Test Locally**
```bash
# Simulate GitHub Actions environment
export GITHUB_REF="refs/tags/v0.7.3"
cd docs
make clean html

# Check that version appears correctly in build output
grep -r "0.7.3" build/html/
```

## 📚 Version Update Checklist

Before creating a release, ensure these files have the correct version. **All version numbers must be updated BEFORE creating the GitHub release/tag.**

### 🔧 Core Package Files (CRITICAL)
- [ ] **`pyproject.toml`** - Line 7: `version = "0.7.3"`
- [ ] **`src/osprey/__init__.py`** - Line 15: `__version__ = "0.7.3"`
- [ ] **`src/osprey/cli/main.py`** - Line 20: `__version__ = "0.7.3"` (fallback version)

### 📝 Documentation & Release Notes
- [ ] **`RELEASE_NOTES.md`** - Line 1: `# Osprey Framework - Latest Release (v0.7.3)`
- [ ] **`CHANGELOG.md`** - Add new section: `## [0.7.3] - 2025-MM-DD`
- [ ] **`README.md`** - Line 12: `**🎉 Latest Release: v0.7.3**`

### 📚 Documentation Files (Auto-detected via git tags, but verify)
- [ ] **`docs/source/conf.py`** - Version auto-detected from git tags (✅ Fixed)
- [ ] **Documentation RST files** - Update any hardcoded version references:
  - `docs/source/getting-started/migration-guide.rst`
  - `docs/source/getting-started/installation.rst`
  - `docs/source/getting-started/hello-world-tutorial.rst`
  - Any files with "New in v0.x.x" admonitions

## 🧪 Test Architecture

Osprey has two separate test suites that **must be run independently**:

### Unit Tests (Fast, Mocked)
```bash
# Run all unit tests excluding e2e tests
pytest tests/ --ignore=tests/e2e -v
```
- **Purpose**: Fast, isolated component testing
- **Registry**: Mocked (in `tests/capabilities/conftest.py`)
- **Duration**: ~1-2 minutes
- **Count**: ~1850 tests
- **Cost**: Free (no API calls)

### E2E Tests (Slow, Real Integration)
```bash
# Run e2e tests separately
pytest tests/e2e/ -v

# With verbose progress (recommended for releases)
pytest tests/e2e/ -v -s --e2e-verbose
```
- **Purpose**: End-to-end user workflow validation
- **Registry**: Real (no mocks)
- **Duration**: ~10-12 minutes
- **Count**: ~32 tests
- **Cost**: ~$1-2 in API calls

### ⚠️ Critical: Why Separate?

**DO NOT** run `pytest tests/` without `--ignore=tests/e2e`!

The capability unit tests mock the registry globally, which will break e2e tests. The two test suites are incompatible when run together:

- **Unit tests** need mocked registry for fast, isolated testing
- **E2E tests** need real registry for integration testing
- Running together causes registry contamination and failures

**Always run both test commands separately before releasing.**

### Representative E2E Tests

The test suite includes ~32 tests across multiple categories. Here are representative examples:

| Test | What It Validates |
|------|-------------------|
| `test_simple_query_smoke_test` | Basic framework initialization and query processing |
| `test_bpm_timeseries_and_correlation_tutorial` | Control assistant workflow: channel finding → archiver retrieval → Python plotting |
| `test_hello_world_weather_tutorial` | Hello World tutorial: weather capability, registry, mock API integration |
| `test_in_context_pipeline_benchmark` | In-context channel finder pipeline performance |
| `test_hierarchical_pipeline_benchmark` | Hierarchical channel finder pipeline performance |
| `test_osprey_claude_install_*` | Claude Code skill installation and invocation |
| `test_channel_finder_queries` | Channel finder query parsing and execution |
| `test_code_generator_workflows` | Capability and MCP server code generation |

### Why E2E Tests Matter for Releases

- **User Experience Validation**: Tests exactly what users see in tutorials
- **Multi-Component Integration**: Validates orchestration across capabilities
- **Regression Detection**: Catches issues that unit tests miss
- **LLM Judge Evaluation**: Flexible validation that adapts to reasonable variations
- **Template Validation**: Ensures all project templates work correctly

### Cost & Performance

- **Unit tests runtime**: ~1-2 minutes
- **E2E tests runtime**: ~10-12 minutes
- **Total API cost**: ~$1-2 per release
- **Value**: Prevents broken tutorials and user-facing bugs

See `tests/e2e/README.md` for complete e2e test documentation.
