# Osprey Framework - Latest Release (v0.10.2)

🎉 **Unified Artifact System & TUI Enhancements** - Streamlined artifact management and interactive browsing

## What's New in v0.10.2

### 🎨 Unified Artifact System

A single source of truth for all artifacts produced during execution:

- **New `register_artifact()` API** - Single method to register any artifact type
- **`ArtifactType` enum** - IMAGE, NOTEBOOK, COMMAND, HTML, FILE
- **Backward compatible** - Legacy methods (`register_figure`, etc.) still work
- **Clean accumulation** - No more duplicate registration issues

```python
from osprey.state import register_artifact, ArtifactType

# New unified API
register_artifact(state, ArtifactType.IMAGE, path="/path/to/plot.png", label="Analysis Plot")

# Legacy methods still work
register_figure(state, path, label)  # Delegates to register_artifact
```

### 🖼️ TUI Artifact Gallery

Interactive artifact browsing in the terminal UI (`osprey chat --tui`):

- **Artifact Gallery** - Browse all generated artifacts with keyboard navigation
- **Keyboard shortcuts**: `Ctrl+a` focus, `j/k` navigate, `Enter` view, `o` open external
- **Native image rendering** - Sixel (iTerm2/WezTerm), Kitty Graphics Protocol
- **New/seen tracking** - `[NEW]` badges for artifacts from current turn
- **Type-specific viewers** - Details and actions for each artifact type

### 🔧 Tooling Consolidation

Simplified developer tooling:

- **Ruff only** - Consolidated formatting and linting (removed Black and Isort)
- **Faster checks** - Single tool for both linting and formatting
- **Updated templates** - All project templates use Ruff

### 🐛 Bug Fixes

- **Gateway**: `/chat` without arguments now correctly displays capabilities without triggering execution
- **Orchestrator**: Fixed time range context key collision (similar date ranges no longer reuse wrong context)
- **Approval**: Fixed KeyError when optional approval config keys are omitted
- **Templates**: Deployment infrastructure config now included for all templates
- **CLI**: Fixed `python-dotenv` warnings when users have shell config `~/.env` files (#95)

---

## Installation

```bash
pip install --upgrade osprey-framework
```

Or install with all optional dependencies:

```bash
pip install --upgrade "osprey-framework[all]"
```

## Upgrading from v0.10.1

### Artifact Registration

The new unified API is optional - existing code continues to work:

```python
# Old way (still works)
register_figure(state, path, label)
register_notebook(state, path, label)

# New way (recommended)
register_artifact(state, ArtifactType.IMAGE, path, label)
register_artifact(state, ArtifactType.NOTEBOOK, path, label)
```

### Tooling

If you have local pre-commit hooks using Black or Isort, update to use Ruff:

```bash
# Old
black src/ tests/
isort src/ tests/

# New
ruff format src/ tests/
ruff check src/ tests/
```

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- TUI mode guide
- Artifact system API reference
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
