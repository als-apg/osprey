# Multi-Project Support in Osprey PyQt GUI

The Osprey PyQt GUI now supports discovering and combining multiple osprey projects into a unified interface.

## Overview

This feature allows you to:
- Discover all osprey projects in subdirectories
- Generate unified configuration and registry files
- Access capabilities from multiple projects in a single GUI session

## How It Works

### 1. Project Discovery

The GUI searches for osprey projects by looking for `config.yml` files in immediate subdirectories (non-recursive, 1 level deep).

**Requirements:**
- Each project must have a `config.yml` file
- Each project must have a `registry_path` defined in its config
- Projects can have any directory name (no naming requirements)

**Example directory structure:**
```
current-directory/
├── weather-agent/
│   ├── config.yml
│   └── src/weather_agent/registry.py
├── mps-agent/
│   ├── config.yml
│   └── src/mps_agent/registry.py
└── my-custom-project/
    ├── config.yml
    └── src/my_custom_project/registry.py
```

### 2. Unified Configuration Generation

When you generate unified files, the GUI creates:

**`unified_config.yml`** - Located in the project root directory
- Merges configuration from all discovered projects
- Uses first project's config as base
- Combines models, API settings, execution settings, etc.
- Points to `unified_registry.py`

**`unified_registry.py`** - Located in the project root directory
- Combines all project registries into one
- Imports all capabilities, context classes, data sources, etc.
- Uses `extend_framework_registry()` to merge with framework defaults

### 3. Loading Unified Configuration

Once generated, you can load the unified configuration to:
- Access all capabilities from all projects
- Use a single GUI session for multiple projects
- Route questions to the appropriate project capabilities

## Usage

### Step 1: Discover Projects

1. Open the Osprey GUI
2. Go to **Multi-Project → Discover Projects**
3. Review the list of discovered projects

### Step 2: Generate Unified Files

1. Go to **Multi-Project → Generate Unified Config**
2. Review the list of projects to be combined
3. Click **Yes** to generate the files

This creates:
- `unified_config.yml` (in the project root directory)
- `unified_registry.py` (in the project root directory)

### Step 3: Load Unified Configuration

1. Go to **Multi-Project → Load Unified Config**
2. Click **Yes** to reinitialize with unified configuration
3. All project capabilities are now available!

## File Locations

The unified files are stored in the project root directory (where you run `osprey-gui`):
```
project-root/
├── unified_config.yml      # Auto-generated, should be git-ignored
├── unified_registry.py     # Auto-generated, should be git-ignored
├── mps-agent/              # Example project 1
│   └── config.yml
├── weather-agent/          # Example project 2
│   └── config.yml
└── src/osprey/interfaces/pyqt/
    ├── project_discovery.py    # Discovery and generation logic
    └── gui.py                  # Main GUI application
```

**Note:** Add `unified_config.yml` and `unified_registry.py` to your `.gitignore` file in the project root.

## Important Notes

### Git Ignore

The unified files are automatically git-ignored because they are:
- Auto-generated
- Environment-specific
- Easily regenerated

### Regeneration

You can regenerate the unified files at any time:
1. Make changes to individual project configs/registries
2. Use **Multi-Project → Generate Unified Config** again
3. Use **Multi-Project → Load Unified Config** to reload

### Naming Consistency

Both unified files use underscores for consistency:
- `unified_config.yml` (not `unified-config.yml`)
- `unified_registry.py` (not `unified-registry.py`)

## Comparison with osprey-aps

The osprey-aps project has similar functionality but with key differences:

| Feature | osprey-aps | osprey framework |
|---------|-----------|------------------|
| Discovery method | Requires `-agent` suffix | Any directory with `config.yml` |
| File location | Project root | Project root (same) |
| Naming | Mixed (hyphen + underscore) | Consistent (underscores) |
| Integration | Standalone GUI | Integrated into framework GUI |

## Troubleshooting

### "No Projects Found"

**Cause:** No subdirectories contain `config.yml` files

**Solution:**
- Ensure projects are in subdirectories of the current directory
- Each project must have a `config.yml` file
- Check that directories aren't in the ignore list (node_modules, venv, etc.)

### "No projects have registry_path defined"

**Cause:** Projects' `config.yml` files don't specify `registry_path`

**Solution:**
- Add `registry_path: ./src/my_project/registry.py` to each project's config.yml
- Ensure the registry file exists at that path

### Import Errors After Loading

**Cause:** Project src directories not on Python path

**Solution:**
- The unified registry automatically adds project src directories to sys.path
- Ensure each project has the expected structure (src/project_name/registry.py)
- Check that module names in registry.py match directory names

## Example Workflow

```bash
# 1. Navigate to directory containing multiple projects
cd /path/to/projects/

# 2. Launch GUI
osprey-gui

# 3. In GUI:
#    - Multi-Project → Discover Projects (see what's available)
#    - Multi-Project → Generate Unified Config (create unified files)
#    - Multi-Project → Load Unified Config (activate multi-project mode)

# 4. Now you can use capabilities from all projects!
```

## Advanced: Manual Unified Files

You can also create unified files manually if needed:

```python
from pathlib import Path
from osprey.interfaces.pyqt.project_discovery import (
    discover_projects,
    create_unified_config,
    create_unified_registry
)

# Discover projects
projects = discover_projects(Path.cwd())

# Generate files
config_path = create_unified_config(projects)
registry_path = create_unified_registry(projects)

print(f"Created: {config_path}")
print(f"Created: {registry_path}")
```

## See Also

- [PyQt GUI Installation Guide](INSTALLATION_GUIDE.md)
- [PyQt GUI README](README.md)
- CLI's `discover_nearby_projects()` in `src/osprey/cli/interactive_menu.py`