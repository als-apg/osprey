# Osprey Framework - Latest Release (v0.9.10)

🎉 **Modular Prompts & Production Hardening** - Channel Finder Customization & Massive Test Coverage Expansion

## What's New in v0.9.10

### 🚀 Channel Finder Enhancements

#### Modular Prompt Structure
- **Simplified Customization**: Split monolithic `system.py` into focused modules
  - `facility_description.py` (REQUIRED) - Your facility-specific content
  - `matching_rules.py` (OPTIONAL) - Custom matching logic
  - `system.py` auto-combines modules - no manual editing needed
- **Query Splitter Enhancement**: Now accepts `facility_name` parameter for better context
- **All Pipelines Updated**: Hierarchical, in-context, and middle layer all use modular structure

#### Explicit Detection
- **New Detection Module**: `explicit_detection.py` identifies explicit channel/PV/IOC names in queries
  - Catches direct references before semantic search
  - Works across all pipeline implementations
  - `build_result()` helper method in BasePipeline for consistent result construction

#### Query Splitting Control
- **New Parameter**: `query_splitting` for hierarchical and middle_layer pipelines
  - Disable for facility-specific terminology that shouldn't be split
  - Enabled by default for backward compatibility

### 🧪 Production Hardening (~500+ New Tests)

#### Test Coverage Expansion
Major test coverage improvements across the codebase:

| Module | Before | After |
|--------|--------|-------|
| Ollama provider | 24.2% | 96.0% |
| Memory provider | 32.2% | 94.9% |
| Error node | 33.6% | 91.8% |
| CLI main | 28.6% | 95.2% |
| YAML loader | 0% | 86.6% |
| Preview styles | 0% | 88.1% |
| Health cmd | 0% | 69.6% |
| Memory capability | 37.7% | 62.4% |

#### New Test Suites
- **CLI Commands**: chat, config, deploy, generate, remove, registry, health
- **Infrastructure**: error_node, respond_node, task_extraction_node, orchestration_node, classification_node
- **Models**: generators, memory_storage, completion, logging
- **Providers**: Anthropic, Ollama, ARGO

### 🐛 Bug Fixes

- **Channel Finder**: Fixed `AttributeError` - `query_splitting` attribute initialization in HierarchicalPipeline
- **CLI**: Fixed broken imports in `config_cmd.py` (incorrect function names)

### 🔧 Quality of Life

- **Control Assistant Template**: Write access now enabled by default for mock connector
  - Simplifies tutorial experience
  - Production deployments should review before enabling
- **License**: Added explicit "BSD 3-Clause License" header
- **Benchmark Dataset**: Renamed `in_context_main.json` → `in_context_benchmark.json` for consistency

### 📚 Documentation

- Updated Hello World tutorial for current weather capability implementation
- Fixed version picker 404 errors in documentation
- Fixed image path typos for channel finder CLI screenshots
- Added "Viewing Exported Workflows" section to AI-assisted development guide
- Removed obsolete v0.9.2+ migration guide (no longer needed)
- Added academic reference (Hellert et al. 2025, arXiv:2512.18779)

---

## Installation

```bash
pip install --upgrade osprey-framework
```

Or install with all optional dependencies:

```bash
pip install --upgrade "osprey-framework[all]"
```

## Upgrading from v0.9.9

### Prompt Structure Migration

If you've customized channel finder prompts:

1. **Check your `system.py` files** - They now auto-combine modules
2. **Move facility content** to `facility_description.py`
3. **Move matching rules** to `matching_rules.py` (optional)

The new structure makes future upgrades easier - framework updates won't overwrite your customizations.

### Query Splitting

If you have facility-specific terms being incorrectly split:

```python
# In your pipeline configuration
pipeline = HierarchicalPipeline(
    query_splitting=False  # Disable splitting for your facility
)
```

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- Channel Finder prompt customization guide
- AI-assisted development workflows
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
