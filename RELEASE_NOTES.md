# Osprey Framework - Latest Release (v0.9.7)

🎉 **Configuration Management & Channel Finder Robustness** - Model Configuration CLI, Enhanced Channel Finder Validation, and Bug Fixes

## What's New in v0.9.7

### 🚀 Major New Features

#### CLI - Model Configuration Command
- **Unified Model Configuration**: New `osprey config set-models` command to update all model configurations at once
  - Interactive mode: Guided prompts for selecting AI provider and specific models
  - Direct mode: Pass provider and models as command-line arguments for automation
  - Updates all relevant model fields: `model`, `channel_finder`, `python_generator`, and `mcp_server_generator`
  - Comprehensive unit tests covering interactive and direct configuration scenarios

#### Channel Finder - Enhanced Reliability
- **API Call Context Tracking**: Added context tracking to channel finder pipeline for better debugging and logging
  - Improved visibility into LLM API calls during channel finding operations
  - Better error messages and troubleshooting capabilities
- **Improved Configuration Validation**: Clearer error messages when channel_finder model is not configured
  - Prevents silent failures or confusing error messages
  - Guides users to proper configuration steps

### 📚 Documentation Improvements

#### Python Version Consistency
- **Unified Python Requirements**: Updated all documentation and templates to consistently specify "Python 3.11+"
  - Matches the pyproject.toml requirement of `>=3.11`
  - Eliminates confusion about supported Python versions
  - Updated files: installation guides, README templates, and documentation

### 🔧 Infrastructure & Code Quality

#### Control Assistant Template Cleanup
- **Removed Duplicate Code**: Removed duplicate `completion.py` implementation from channel finder service
  - Now uses `osprey.models.completion` for consistency and maintainability
  - Reduces code duplication and maintenance burden
  - Ensures consistent LLM completion behavior across all capabilities

#### Pre-Merge Code Quality
- **Code Cleanup**: Removed unused imports and applied formatting standards
  - Applied black formatting to 13 files
  - Documented DEBUG and CONFIG_FILE environment variables in `env.example`
  - Improved overall code quality and consistency

### 🐛 Bug Fixes

#### Channel Finder - Optional Levels Navigation
- **Fixed Hierarchy Navigation Bug**: Resolved issue where direct signals incorrectly appeared as subdevice options in optional hierarchy levels
  - System now correctly distinguishes between container nodes (current optional level) and leaf/terminal nodes (next level)
  - Fixed `build_channels_from_selections()` to handle missing optional levels
  - Automatic separator cleanup (removes `::` and trailing separators)
  - Comprehensive test coverage: 18 new tests in `test_hierarchical_optional_levels_regression.py`

#### Template Fixes
- **Hello World Weather Template**: Added service configuration to prevent template generation errors
  - Fixed `'services/docker-compose.yml.j2' not found` error when following installation guide
  - Template now includes proper container runtime and deployed services configuration

#### Capability Fixes
- **Channel Write Capability**: Fixed initialization bug in approval workflow
  - Removed `verification_levels` field from approval `analysis_details`
  - Field incorrectly called `_get_verification_config()` method before connector initialization
  - Added integration test (`test_channel_write_approval_integration.py`) to catch capability-approval interaction bugs

#### Testing Infrastructure
- **Channel Finder Registration Tests**: Updated test mocks to include `channel_finder` model configuration
  - Fixed tests broken by stricter validation introduced in commit 5834de3
  - Ensures proper test coverage of channel finder initialization
- **E2E Workflow Test**: Updated `test_hello_world_template_generates_correctly`
  - Now expects services directory and deployment configuration
  - Matches current template structure
- **E2E Benchmark Tests**: Fixed registry initialization in `test_channel_finder_benchmarks.py`
  - Added `initialize_registry()` call before creating `BenchmarkRunner`
  - Prevents "Registry not initialized" errors

## Migration Guide

### For Users

**No breaking changes.** All updates are backward compatible:
1. **Model Configuration**: New `osprey config set-models` command is optional (existing configuration methods still work)
2. **Channel Finder**: Stricter validation provides better error messages but doesn't change API
3. **Templates**: Existing projects continue to work unchanged

### For Developers

**Test Infrastructure Update**: If you have custom tests that use the channel finder:
- Ensure mocked `configurable` dicts include `channel_finder` model configuration
- See updated tests in `tests/services/channel_finder/test_registration.py` for examples

## Performance & Quality

- **Test Coverage**: 558 unit tests + 12 e2e tests, all passing
- **Unit Test Runtime**: ~3-5 seconds
- **E2E Test Runtime**: ~7 minutes
- **Code Quality**: Consistent formatting and reduced code duplication

## Installation

```bash
pip install osprey-framework==0.9.7
```

## What's Next

Stay tuned for upcoming features:
- Additional control system connectors
- Enhanced plotting capabilities
- Production deployment guides
- Multi-agent orchestration patterns

---

**Full Changelog**: https://github.com/als-apg/osprey/compare/v0.9.6...v0.9.7
