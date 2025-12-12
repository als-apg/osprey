# Testing Guide - Version 0.9.7 Release

**Branch**: `release/0.9.7`
**Requirements**: Python 3.11+

## ⚠️ CRITICAL: How to Run Tests

```bash
# ✅ CORRECT: Run unit tests
pytest tests/ --ignore=tests/e2e -v

# ✅ CORRECT: Run e2e tests
pytest tests/e2e/ -v

# ❌ WRONG: Do NOT use -m e2e (causes test collection issues)
pytest -m e2e  # DON'T DO THIS!
```

**Why?** Using `-m e2e` causes pytest to collect tests in the wrong order, leading to registry initialization failures. Always use `pytest tests/e2e/` directly. See [tests/e2e/README.md](tests/e2e/README.md) for full details.

---

## Overview

This release branch contains improvements for version 0.9.7, including:

1. **CLI Model Configuration Command**: Unified command for managing AI models across all roles
2. **Channel Finder Configuration**: Improved validation and error handling

## Quick Setup

```bash
# 1. Checkout the branch
git checkout release/0.9.7

# 2. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (if needed)
pip install -e "."

# 4. Run the test suite
pytest tests/ --ignore=tests/e2e -v
```

## What Changed

### 1. CLI Model Configuration Command

**Files Modified**:
- `src/osprey/cli/config_cmd.py` - New `set-models` command
- `src/osprey/cli/interactive_menu.py` - Interactive menu handler
- `src/osprey/generators/config_updater.py` - Utility functions
- `tests/generators/test_config_updater.py` - Test coverage
- `docs/source/developer-guides/02_quick-start-patterns/00_cli-reference.rst` - Documentation

**Features**:
- Updates all model configurations simultaneously
- Interactive and non-interactive modes
- Preserves max_tokens settings
- Shows preview before applying changes
- Supports: anthropic, openai, google, cborg, ollama

**Testing**:
```bash
# Run config updater tests (includes 12 new tests)
pytest tests/generators/test_config_updater.py -v

# Test the functions
pytest tests/generators/test_config_updater.py::test_update_all_models_basic -v
pytest tests/generators/test_config_updater.py::test_update_all_models_preserves_max_tokens -v
```

**Manual Testing**:
```bash
# Create test project
osprey init test-project --template control_assistant
cd test-project

# Test interactive mode
osprey config set-models

# Test non-interactive mode
osprey config set-models --provider openai --model gpt-4

# Verify changes
osprey config show
```

### 2. Channel Finder Configuration Improvements

**Files Modified**:
- `src/osprey/templates/apps/control_assistant/config.yml.j2` - Added channel_finder model
- `src/osprey/templates/apps/control_assistant/services/channel_finder/service.py` - Validation

**Features**:
- Added dedicated `channel_finder` model configuration to template
- Enhanced validation with clear error messages
- Uses standard `get_model_config()` utility
- Validates required fields (provider, model_id)

**Testing**:
```bash
# Create new project with template
osprey init test-cf --template control_assistant
cd test-cf

# Verify channel_finder model in config
grep -A 3 "channel_finder:" config.yml

# Test channel finder (should work with proper config)
osprey chat
# Then try: "Find all magnet channels"
```

## Running Full Test Suite

```bash
# Run all relevant tests
pytest tests/generators/test_config_updater.py -v

# Expected: All tests pass including 12 new model config tests
```

## Verification Checklist

### Code Quality
- [ ] All unit tests pass
- [ ] No linter errors
- [ ] Type hints correct

### Functionality
- [ ] `osprey config set-models` works in interactive mode
- [ ] `osprey config set-models` works with flags
- [ ] max_tokens settings preserved
- [ ] Preview shows before applying changes
- [ ] Channel finder validation works correctly

### Documentation
- [ ] CLI reference updated
- [ ] CHANGELOG entries complete
- [ ] Module docstrings updated
- [ ] Examples work as documented

## Known Issues

None - all tests passing ✅

## Publishing Checklist

- [x] Commits created
- [x] CHANGELOG updated
- [x] Documentation complete
- [x] Tests pass
- [ ] Ready for additional features
- [ ] Ready to merge to main (after testing)

## Reporting Issues

Found a bug or have suggestions? https://github.com/als-apg/osprey/issues

## Additional Resources

- **CLI Reference**: `docs/source/developer-guides/02_quick-start-patterns/00_cli-reference.rst`
- **Config Updater**: `src/osprey/generators/config_updater.py`
- **Channel Finder Service**: `src/osprey/templates/apps/control_assistant/services/channel_finder/service.py`
