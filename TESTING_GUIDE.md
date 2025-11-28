# Testing Guide - Flexible Hierarchical Database Feature

**Branch**: `feat/hierarchical-flexibility`  
**Requirements**: Python 3.11+

## Overview

This branch introduces a flexible hierarchical database system that supports arbitrary mixing of tree navigation and instance expansion at any level, plus fixes CLI documentation to use direct script execution instead of module imports.

## Quick Setup

```bash
# 1. Checkout the branch
git checkout feat/hierarchical-flexibility

# 2. Create virtual environment (if not already created)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Osprey in development mode
pip install -e "."

# 4. Run the test suite
pytest tests/services/channel_finder/ -v
```

## What Changed

### Core Implementation
- **Flexible `hierarchy_config`**: Explicit level-by-level configuration with `type`, `structure`, and `allow_branching` properties
- **Arbitrary level mixing**: Support any combination of tree (semantic categories) and instance (numbered expansions) levels
- **Consecutive instances**: Enable multiple instance levels in sequence (e.g., SECTOR→DEVICE, FLOOR→ROOM)
- **Comprehensive validation**: Actionable error messages with troubleshooting guidance
- **Dynamic branching**: Pipeline logic adapts based on level configuration
- **Backward compatibility**: Legacy databases automatically inferred from structure

### Example Databases
- **`mixed_hierarchy.json`** - Building management (1,720 channels)
- **`instance_first.json`** - Manufacturing line (85 channels)
- **`consecutive_instances.json`** - Accelerator naming (4,996 channels)
- **`hierarchical_legacy.json`** - Legacy format reference

### CLI Fix
- Converted `cli.py` → `cli.py.j2` with `sys.path.insert()` pattern
- Updated documentation from `python -m module` to `python src/path/to/cli.py`
- Enables standalone script execution without package installation

## Testing Focus Areas

### 1. Automated Tests

```bash
# Run all hierarchical database tests (41 tests total)
pytest tests/services/channel_finder/ -v

# Run unit tests only (34 tests)
pytest tests/services/channel_finder/test_hierarchical_flexible.py -v

# Run integration tests only (7 tests)
pytest tests/services/channel_finder/test_example_databases.py -v

# Run all tests to verify no regressions
pytest tests/ -v
```

### 2. Test Coverage

#### Unit Tests (`test_hierarchical_flexible.py`)
- ✅ Backward compatibility with legacy format
- ✅ Mixed instance/tree patterns
- ✅ Consecutive instance levels (proper nesting validation)
- ✅ Instance-first level (at root position)
- ✅ Cartesian product channel generation
- ✅ Tree navigation skips instance levels correctly
- ✅ Range and list expansion types
- ✅ Comprehensive validation with error messages
- ✅ Edge cases: all-instance, all-tree, single-level, deep hierarchies (8 levels)

#### Integration Tests (`test_example_databases.py`)
- ✅ All example databases load correctly
- ✅ Navigation through each pattern works
- ✅ Channel generation produces correct counts
- ✅ Backward compatibility with existing `hierarchical.json`
- ✅ Statistics gathering works with flexible system

### 3. Manual Testing - New Database Format

Create a test database with the new format:

```json
{
  "hierarchy_definition": ["line", "station", "parameter"],
  "naming_pattern": "LINE{line}:{station}:{parameter}",
  "hierarchy_config": {
    "levels": {
      "line": {
        "type": "instance",
        "structure": "expand_here",
        "allow_branching": false
      },
      "station": {
        "type": "category",
        "structure": "tree",
        "allow_branching": true
      },
      "parameter": {
        "type": "category",
        "structure": "tree",
        "allow_branching": true
      }
    }
  },
  "tree": {
    "LINE": {
      "_expansion": {
        "_type": "range",
        "_pattern": "{}",
        "_range": [1, 5]
      },
      "ASSEMBLY": {
        "SPEED": {"_description": "Line speed"},
        "STATUS": {"_description": "Status"}
      }
    }
  }
}
```

Test with validation tool:
```bash
cd my-control-assistant
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/hierarchical.json
```

### 4. CLI Testing

Test the fixed CLI invocation pattern:

```bash
# After osprey init my-control-assistant --template control_assistant
cd my-control-assistant

# Test CLI works immediately (no package install needed)
python src/my_control_assistant/services/channel_finder/cli.py

# Test benchmarks work
python src/my_control_assistant/services/channel_finder/benchmarks/cli.py

# Verify the old pattern fails (as expected)
python -m my_control_assistant.services.channel_finder.cli
# Should fail with ModuleNotFoundError (because project isn't installed)
```

### 5. Example Databases

Test all example patterns:

```bash
cd my-control-assistant

# Test instance-first pattern (simplest)
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/examples/instance_first.json

# Test consecutive instances pattern
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/examples/consecutive_instances.json

# Test mixed hierarchy pattern
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/examples/mixed_hierarchy.json

# Test legacy format still works
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/examples/hierarchical_legacy.json
```

### 6. Backward Compatibility

Verify existing databases continue to work without modification:

```bash
# The main hierarchical.json has been updated to new format
# But test that old-style databases still load via auto-inference

cd my-control-assistant

# Use the legacy example
cp src/my_control_assistant/data/channel_databases/examples/hierarchical_legacy.json \
   src/my_control_assistant/data/channel_databases/test_legacy.json

# Update config.yml to point to test_legacy.json
# Run channel finder - should work without errors
python src/my_control_assistant/services/channel_finder/cli.py
```

### 7. Documentation

Verify documentation builds correctly:

```bash
cd docs
pip install -r requirements.txt
python launch_docs.py
# Visit: http://localhost:8082
```

Check that:
- Channel Finder guide shows new format examples
- Dropdown explanations for tree vs instance levels
- CLI commands use `python src/.../cli.py` pattern (not `python -m`)
- Example databases are referenced correctly
- Production troubleshooting guide updated

## Expected Behavior

### Database Loading
- **New format**: Validates `hierarchy_config` with helpful error messages
- **Legacy format**: Automatically infers configuration from structure
- **Validation**: Catches common mistakes with actionable guidance

### Level Types
- **`structure: "tree"`**: Navigate through named semantic choices
- **`structure: "expand_here"`**: Expand across numbered/named instances
- **`structure: "container"`**: Legacy mode (auto-inferred only)

### Consecutive Instances
- Multiple instance levels must be properly nested
- Example: `FLOOR` container contains `ROOM` container (not siblings)
- Validation catches incorrect nesting with clear error messages

### CLI Invocation
- **New pattern**: `python src/my_control_assistant/services/channel_finder/cli.py`
- **Old pattern**: `python -m my_control_assistant.services.channel_finder.cli` (should fail)
- **Works immediately**: No `pip install -e .` required

## Common Test Scenarios

### Scenario 1: Create New Database with Instance-First Pattern

```bash
# Copy example template
cp src/my_control_assistant/data/channel_databases/examples/instance_first.json \
   src/my_control_assistant/data/channel_databases/my_database.json

# Edit to match your system
# Validate
python src/my_control_assistant/data/tools/validate_database.py \
  --database src/my_control_assistant/data/channel_databases/my_database.json

# Update config.yml
# Test with CLI
python src/my_control_assistant/services/channel_finder/cli.py
```

### Scenario 2: Migrate Legacy Database to New Format

```bash
# Start with legacy database
# Add hierarchy_config section following examples
# Rename containers (devices → DEVICE with _expansion)
# Flatten nested structure (fields/subfields → direct children)
# Validate
python src/my_control_assistant/data/tools/validate_database.py
```

### Scenario 3: Test Consecutive Instance Levels

Use `consecutive_instances.json` as reference for proper nesting:
- SECTOR and DEVICE are both instance levels
- DEVICE container is nested inside SECTOR container (not siblings)
- Validation will catch incorrect nesting

## Known Issues

None - all tests passing ✅

## Reporting Issues

Found a bug or have suggestions? https://github.com/als-apg/osprey/issues

## Additional Resources

- **Example README**: `src/osprey/templates/apps/control_assistant/data/channel_databases/examples/README.md`
- **Channel Finder Guide**: `docs/source/getting-started/control-assistant-part2-channel-finder.rst`
- **Pull Request**: https://github.com/als-apg/osprey/pull/35
