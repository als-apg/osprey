# Osprey Framework - Latest Release (v0.9.8)

🎉 **Developer Experience & Infrastructure Improvements** - CI/CD Pipeline, Documentation Versioning, Workflow Guides, and Code Quality Enhancements

## What's New in v0.9.8

### 🚀 Major New Features

#### CI/CD Infrastructure
- **Comprehensive GitHub Actions Pipeline**: Full CI/CD automation with parallel test jobs
  - Multi-version testing: Python 3.11 & 3.12 on Ubuntu & macOS
  - Automated linting (Ruff), type checking (mypy), and package validation
  - Code coverage reporting with Codecov integration
  - Documentation builds with PR preview artifacts (7-day retention)
- **Release Automation**: `.github/workflows/release.yml` for automated PyPI publishing
  - Trusted publishing using OIDC (no manual credentials needed)
  - Version verification and optional TestPyPI deployment
- **Pre-commit Hooks**: `.pre-commit-config.yaml` with automated quality checks
  - Ruff linting and formatting
  - File quality checks (trailing whitespace, merge conflicts, large files)
  - Optional mypy type checking
- **Dependabot**: Automated weekly dependency updates with intelligent grouping

#### Documentation Version Switcher
- **Multi-Version Documentation**: PyData Sphinx Theme version switcher for GitHub Pages
  - Dynamic `versions.json` generation from git tags
  - Historical versions preserved in separate directories (e.g., `/v0.9.7/`, `/latest/`)
  - Seamless navigation between documentation versions
- **Custom Sphinx Extension**: `workflow_autodoc.py` for auto-documenting workflow files
  - New directives: `.. workflow-summary::` and `.. workflow-list::`
  - Parses YAML frontmatter from markdown workflow files
  - Custom CSS styling for workflow documentation

#### Developer Workflows System
- **10 Comprehensive Workflow Guides**: New `docs/workflows/` directory
  - Pre-merge cleanup, commit organization, release process
  - Testing strategy, AI code review, docstrings, comments
  - Documentation updates and quality standards
  - YAML frontmatter metadata for machine parsing
  - AI assistant integration prompts for automated workflows
- **Professional Contributing Guide**: `CONTRIBUTING.md` with quick start
  - Branch naming conventions and code standards summary
  - Links to comprehensive documentation
  - Learning paths for new contributors

#### Testing & Quality
- **Pre-merge Check Script**: `scripts/premerge_check.sh` automated scanning
  - Debug code, commented code, hardcoded secrets detection
  - Missing CHANGELOG entries and incomplete docstrings
  - Unlinked TODOs and code quality issues
- **Enhanced Test Coverage**: Comprehensive test suites for new features
  - Hello World Weather template: Mock API validation, response formatting, error handling
  - Workflow autodoc extension: Frontmatter parsing, directive rendering, integration tests
  - All changes verified with 976 unit tests + 16 e2e tests
- **Status Badges**: README.md badges for CI, docs, coverage, PyPI, Python, license

### 🔧 Improvements & Changes

#### Code Quality
- **Comprehensive Linting Cleanup**: Fixed issues across 47 files
  - B904 exception chaining (30 instances)
  - E722 bare except clauses (5 instances)
  - B007 unused loop variables (4 instances)
  - Removed B904 from ruff ignore list; added intentional per-file ignores
- **Code Formatting**: Applied automated Ruff formatting across codebase
  - Modernized type hints to Python 3.10+ style (`Optional[T]` → `T | None`)
  - Normalized quotes, cleaned whitespace, removed unused imports
  - No functional changes

#### Documentation
- **Workflow Migration**: Moved workflows from `docs/resources/other/` to `docs/workflows/`
  - Consistent YAML frontmatter for machine parsing
  - Updated references throughout documentation
- **Contributing Guide**: Restructured to 400+ line comprehensive guide
  - 6 dedicated sections: Getting Started, Git & GitHub, Code Standards
  - Developer Workflows, AI-Assisted Development, Community Guidelines
  - Sphinx-design cards and grids for better organization
- **Citation Update**: Updated paper citation to "Osprey: Production-Ready Agentic AI for Safety-Critical Control Systems"
- **Framework Name Cleanup**: Replaced all remaining "Alpha Berkeley Framework" references with "Osprey Framework"

#### Hello World Weather Template
- **LLM-based Location Extraction**: Intelligent parsing of natural language queries
  - Handles nicknames, abbreviations, and defaults to "local"
  - Replaces simple string matching with structured output parser
- **Mock API Simplification**: Accepts any location string with random weather data
  - Removed hardcoded city list for flexible tutorial demonstrations
  - Improved error handling and response formatting
- **Enhanced E2E Testing**: Exercises both weather AND Python capabilities
  - Multi-step query validation
  - Configuration defaults and context passing
  - Code generation and execution workflows

### 🐛 Bug Fixes

#### Configuration & Templates
- **Python Code Generation Defaults**: Added missing code generator configuration
  - Fixed "Unknown provider: None" errors in minimal configurations
  - Now includes `code_generator: "basic"` in `ConfigBuilder._get_execution_defaults()`
- **Hello World Weather Template**: Fixed conditional to include execution infrastructure
  - Ensures Python code generation works out-of-the-box
  - Excludes only EPICS-specific settings

#### Channel Finder
- **Multiple Direct Signal Selection**: Fixed leaf node detection
  - Properly handles selections like "status and heartbeat" at optional levels
- **Optional Levels LLM Awareness**: Enhanced database descriptions and prompts
  - Better distinction between direct signals and subdevice-specific signals
- **Separator Overrides**: Respect `_separator` metadata from tree nodes
  - New `_collect_separator_overrides()` method
  - Proper navigation through expanded instance names
- **Navigation Through Expanded Instances**: Fixed at optional levels
  - Base containers with `_expansion` no longer appear as selectable options
  - Correct handling in `_navigate_to_node()` and `_extract_tree_options()`

#### Testing
- **Channel Finder Test Path**: Fixed incorrect database path in `test_multiple_direct_signals_fix.py`
- **CI Workflow Autodoc**: Fixed `ModuleNotFoundError: No module named 'sphinx'`
  - Added `pytest.importorskip` for graceful skipping when Sphinx unavailable
  - Sphinx only required for documentation builds, not `[dev]` dependencies

### 🗑️ Removed
- **Documentation Local Server**: Removed `docs/launch_docs.py` script
  - Use standard Sphinx commands: `make html` and `python -m http.server`

## Migration Guide

### For Users

**No breaking changes.** All updates are backward compatible:
1. **CI/CD**: New automation is optional (existing workflows continue to work)
2. **Documentation**: Version switcher enhances navigation but doesn't affect existing docs
3. **Templates**: Existing projects continue to work unchanged
4. **Configuration**: New defaults improve out-of-box experience without breaking existing configs

### For Developers

**Workflow Improvements**: Take advantage of new developer tools:
1. **Pre-commit hooks**: Run `.pre-commit install` to enable automated quality checks
2. **Pre-merge script**: Use `scripts/premerge_check.sh` before creating PRs
3. **Workflow guides**: Consult `docs/workflows/` for best practices and AI integration
4. **CI pipeline**: All PRs automatically tested across Python 3.11 & 3.12, Ubuntu & macOS

## Performance & Quality

- **Test Coverage**: 976 unit tests + 16 e2e tests, all passing
- **Unit Test Runtime**: ~45 seconds
- **E2E Test Runtime**: ~7 minutes
- **Code Quality**: Comprehensive linting cleanup across 47 files
- **CI/CD**: Parallel testing across 4 environments (2 Python versions × 2 OS)

## Installation

```bash
pip install osprey-framework==0.9.8
```

## What's Next

Stay tuned for upcoming features:
- Additional control system connectors
- Enhanced plotting capabilities
- Production deployment guides
- Multi-agent orchestration patterns

---

**Full Changelog**: https://github.com/als-apg/osprey/compare/v0.9.7...v0.9.8
