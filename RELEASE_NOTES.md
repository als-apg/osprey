# Osprey Framework - Latest Release (v0.9.6)

🎉 **Control Assistant Enhancements & Channel Finder Improvements** - Custom Task Extraction, Enhanced Database Preview, and Flexible Hierarchy Configuration

## What's New in v0.9.6

### 🚀 Major New Features

#### Control Assistant Template - Custom Task Extraction Prompt
- **Domain-Specific Task Extraction**: Control-system-specific task extraction prompt builder replaces framework defaults
  - 14 control system examples covering channel references, temporal context, write operations, and visualization requests
  - Unit test suite verifying custom prompt usage without LLM invocation
  - Documentation in Part 4 tutorial explaining single-point-of-failure importance
  - Ensures accurate task decomposition for accelerator control operations

#### Channel Finder - Enhanced Database Preview Tool
- **Flexible Display Options**: Better hierarchy visibility and exploration
  - `--depth N` parameter to control tree depth display (default: 3, -1 for unlimited)
  - `--max-items N` parameter to limit items shown per level (default: 10, -1 for unlimited)
  - `--sections` parameter with modular output: tree, stats, breakdown, samples, all
  - `--path PATH` parameter to preview any database file directly
  - `--focus PATH` parameter to zoom into specific hierarchy branches
  - New `stats` section showing unique value counts at each hierarchy level
  - New `breakdown` section showing channel count breakdown by path
  - New `samples` section showing random sample channel names
  - Backwards compatible `--full` flag support
  - Comprehensive unit tests covering all preview features and edge cases

#### Hierarchical Channel Finder - Advanced Configuration Features
- **Custom Separator Overrides**: Per-node control of channel name separators
  - New `_separator` metadata field overrides default separators
  - Solves EPICS naming with mixed delimiters (`:` for subdevices, `_` for suffixes, `.` for legacy)
  - Backward compatible: nodes without `_separator` use pattern defaults
- **Automatic Leaf Detection**: Eliminates verbose `_is_leaf` markers
  - Nodes without children automatically detected as leaves
  - `_is_leaf` now only required for nodes with children that are also complete channels
  - Reduces verbosity in database definitions
  - Backward compatible: explicit markers still work
- **Flexible Naming Configuration**: Navigation-only levels and decoupled naming
  - Naming pattern can reference subset of hierarchy levels
  - New `_channel_part` field decouples tree keys from naming components
  - Enables semantic tree organization with PV names at leaf (JLab CEBAF pattern)
  - Enables friendly navigation with technical naming ("Magnets" → "MAG")

#### Channel Finder - Pluggable Pipeline and Database System
- **Registration Pattern**: Custom implementations without modifying framework
  - `register_pipeline()` and `register_database()` methods
  - Discovery API: `list_available_pipelines()` and `list_available_databases()`
  - Config-driven selection
  - Examples for RAG pipeline and PostgreSQL database implementations

### 🧪 Comprehensive Test Suite

#### New Test Coverage
- **Channel Finder Parameterized Tests**: Automated testing for all example databases
  - 80 tests covering all 6 example databases
  - Parameterized tests automatically run on new databases
  - Core functionality: loading, navigation, channel generation, validation
  - Expected channel count validation (total: 30,908 channels)
  - Database-specific feature tests
- **Custom Task Extraction Tests**: Unit tests for control assistant prompt builder
- **Preview Tool Tests**: Comprehensive coverage of all preview features
- **Hierarchical Channel Finder Tests**: 18 new tests for flexible naming functionality

### 🔧 Infrastructure Improvements

#### Channel Finder
- **Default Preview Depth**: Increased from 2 to 3 levels for better visibility
- **Comprehensive Example Coverage**: All example databases now tested
  - `hierarchical_legacy.json` and `optional_levels.json` previously untested

### 🐛 Bug Fixes

#### MCP Server Template
- **Dynamic Timestamps**: Fixed MCP server generation to use current UTC timestamps instead of hardcoded November 15, 2025 dates
  - Prevents e2e test failures due to stale mock data
  - Ensures demo servers return realistic "current" weather data

#### Channel Finder Tests
- **Unit Test Compatibility**: Updated test files for hierarchical database changes (optional levels, custom separators)

#### Registry & Test Infrastructure
- **Mock Cleanup**: Fixed 7 registry isolation test failures
  - Session-level registry mock pollution from capability tests resolved
  - Renamed conflicting test fixtures to prevent pytest naming collisions

#### Python Executor
- **Context File Creation**: Fixed timing issue where `context.json` was not created until execution
  - Caused warnings and test failures when approval was required
  - Context now saved immediately when creating pre-approval notebooks

#### Code Quality
- **Pre-merge Cleanup**: Removed unused imports and applied formatting standards (black + isort)

#### Documentation
- **RST Docstring Formatting**: Corrected docstring syntax in `BaseInfrastructureNode.get_current_task()`
  - Eliminates Sphinx warnings

### 📚 Documentation

#### Control Assistant Tutorial
- **Part 4 Customization**: New section on custom task extraction prompts
  - Explains single-point-of-failure concept
  - Provides examples for domain-specific task extraction

#### Channel Finder Documentation
- **Advanced Hierarchy Patterns**: New "Custom Separators" tab
- **Preview Tool Examples**: Comprehensive examples for all preview options
- **Pluggable System Guide**: Documentation for custom pipeline/database implementations

## Migration Guide

### For Control Assistant Users

No breaking changes. The custom task extraction prompt is automatically used in new control assistant projects. Existing projects continue to work with framework defaults.

### For Channel Finder Database Authors

New features are backward compatible:
1. **Custom Separators**: Optional `_separator` field (uses pattern defaults if omitted)
2. **Automatic Leaf Detection**: No action needed (childless nodes automatically detected)
3. **Flexible Naming**: Optional `_channel_part` field (uses tree keys if omitted)

Existing databases work unchanged.

## Performance & Quality

- **Test Coverage**: 546 unit tests + 9 e2e tests, all passing
- **Example Database Coverage**: 100% (6/6 databases tested)
- **Channel Count Validation**: 30,908 channels across all examples

## Installation

```bash
pip install osprey-framework==0.9.6
```

## What's Next

Stay tuned for upcoming features:
- Additional control system connectors
- Enhanced plotting capabilities
- Production deployment guides
- Multi-agent orchestration patterns

---

**Full Changelog**: https://github.com/als-apg/osprey/compare/v0.9.5...v0.9.6
