# Channel Database Tools

Tools for building, validating, and inspecting channel databases for the in-context pipeline.

## Available Tools

### 1. build_channel_database.py

**Purpose:** Build template-based channel database from CSV file (Workflow A)

**Features:**
- Reads simple CSV format with optional device family metadata
- Intelligent template detection and grouping
- LLM-based descriptive channel name generation for standalone channels
- Efficient template-based storage for device families

**Usage:**

```bash
# From project root (your-control-assistant directory)
python src/your_assistant_name/data/tools/build_channel_database.py

# With LLM naming (recommended)
python src/your_assistant_name/data/tools/build_channel_database.py --use-llm --config config.yml

# With custom paths
python src/your_assistant_name/data/tools/build_channel_database.py \
    --csv path/to/address_list.csv \
    --output path/to/output.json \
    --config config.yml
```

**What it does:**

1. Loads CSV from configured `data_source.path` or `--csv` argument
2. Groups channels by `family_name` column (for templating)
3. Creates template entries for device families
4. Generates descriptive channel names using LLM for standalone channels (with `--use-llm`)
5. Outputs template database JSON to configured output path

**Configuration:**
- Uses `config.yml` from project root for all settings
- CSV input path: `facility.data_source.path` in config
- Output path: configurable via `--output` or defaults to `data/channel_databases/in_context.json`

**Required CSV Format** (see `../data/raw/CSV_EXAMPLE.csv` for a comprehensive example):

```csv
address,description,family_name,instances,sub_channel
TerminalVoltage,Terminal voltage readback,,,
BPM{instance:02d}{sub_channel},Beam position monitor,BPM,10,XPosition
BPM{instance:02d}{sub_channel},Beam position monitor,BPM,10,YPosition
```

**Columns:**
- `address` - Channel address/PV name (required) - can include patterns like `{instance:02d}`
- `description` - Human-readable description (required)
- `family_name` - Device family name (optional, for templating)
- `instances` - Number of instances to generate (optional, assumes starts at 01)
- `sub_channel` - Sub-channel name (optional, for templating)

**Template Detection:**
- Rows with `family_name` filled → grouped into template entries
- Rows with empty `family_name` → standalone channel entries

**Example:**

Input CSV:
```csv
address,description,family_name,instances,sub_channel
BeamCurrent,Total beam current,,,
Valve{instance:02d}SetPoint,Valve setpoint,Valve,5,SetPoint
Valve{instance:02d}ReadBack,Valve readback,Valve,5,ReadBack
```

Output JSON (conceptual):
```json
{
  "channels": [
    {
      "template": false,
      "channel": "BeamCurrent",
      "address": "BeamCurrent"
    },
    {
      "template": true,
      "base_name": "Valve",
      "instances": [1, 5],
      "sub_channels": ["SetPoint", "ReadBack"]
    }
  ]
}
```

### 2. validate_database.py

**Purpose:** Validate channel database JSON files (Workflow B helper)

**Features:**
- JSON structure and schema validation
- Template entry validation (instances, sub_channels, patterns)
- Standalone entry validation (channel, address, description)
- Database loading test through actual database class
- Statistics and diagnostics

**Usage:**

```bash
# Validate configured database
python tools/validate_database.py

# Validate specific file
python tools/validate_database.py --database my_database.json

# Verbose output with detailed stats
python tools/validate_database.py --verbose
```

**What it checks:**

✅ **Structure:**
- Valid JSON syntax
- Correct top-level format (dict with 'channels' key)
- Version and metadata presence

✅ **Template entries:**
- Required fields: `base_name`, `instances`, `description`
- Valid instance range [start, end]
- Sub-channels format (list)
- Address pattern presence
- Channel descriptions mapping

✅ **Standalone entries:**
- Required fields: `channel`, `address`, `description`
- Non-empty values

✅ **Database loading:**
- Can be loaded by TemplateChannelDatabase class
- All channels accessible
- Statistics generation

**Example output:**

```
============================================================
DATABASE VALIDATION REPORT
============================================================
✅ VALID - Database passed all checks

⚠️  WARNINGS (2):
  • Template 3: missing 'channel_descriptions'. Will use generic descriptions.
  • Template 5: 'address_pattern' not specified. Will use default pattern.

📊 STATISTICS:
  • Format: template
  • Total channels: 251
  • Template entries: 5
  • Standalone entries: 53
============================================================
```

### 3. preview_database.py

**Purpose:** Flexible database preview with customizable depth, display options, and modular sections

**Features:**
- Auto-detects database type (hierarchical or in-context)
- Configurable tree depth and item limits
- Modular output sections (tree, stats, breakdown, samples)
- Focus on specific branches
- Direct path support for previewing any database file
- Rich console output with color-coded hierarchy

**Usage:**

```bash
# Quick overview (default: 3 levels, 10 items per level)
python src/your_assistant_name/data/tools/preview_database.py

# Show 4 levels with statistics (addresses colleague request)
python src/your_assistant_name/data/tools/preview_database.py --depth 4 --sections tree,stats

# Preview a specific database file directly
python src/your_assistant_name/data/tools/preview_database.py --path data/channel_databases/examples/optional_levels.json

# Complete view with all sections
python src/your_assistant_name/data/tools/preview_database.py --depth -1 --max-items -1 --sections all

# Focus on specific system/branch
python src/your_assistant_name/data/tools/preview_database.py --focus MAG:DIPOLE --depth 4

# Just statistics, no tree
python src/your_assistant_name/data/tools/preview_database.py --sections stats

# Backwards compatible (legacy)
python src/your_assistant_name/data/tools/preview_database.py --full
```

**Parameters:**

- `--depth N` - Tree depth to display (default: 3, -1 for unlimited)
- `--max-items N` - Maximum items per level (default: 10, -1 for unlimited)
- `--sections SECTIONS` - Comma-separated sections: tree, stats, breakdown, samples, all (default: tree)
- `--path PATH` - Direct path to database file (overrides config, auto-detects type)
- `--focus PATH` - Focus on specific branch (e.g., "M:QB" for QB family in M system)
- `--full` - Show complete hierarchy (shorthand for --depth -1 --max-items -1)

**Output Sections:**

1. **tree** (default) - Hierarchy tree visualization showing structure and channel counts
2. **stats** - Per-level statistics table showing unique value counts at each hierarchy level
3. **breakdown** - Channel count breakdown by path (top 20 paths)
4. **samples** - Random sample channel names from the database

**Example output:**

```
╭─────────────────────────────────────────────────────────────────╮
│                                                                 │
│  Hierarchical Database Preview                                  │
│  Shows the tree structure of the hierarchical channel database  │
│                                                                 │
╰─────────────────────────────────────────────────────────────────╯

╭───────────────────────── Configuration ──────────────────────────╮
│                                                                  │
│  Database Path      data/channel_databases/hierarchical.json     │
│  Hierarchy Levels   system → family → sector → device → property│
│  Display Depth      4                                            │
│  Max Items/Level    10                                           │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯

✓ Successfully loaded 4996 channels

╭───────────────────────── Hierarchy Tree ─────────────────────────╮
│                                                                  │
│  Channel Database Hierarchy                                      │
│  ┣━━ M (3916 channels)                                          │
│  ┃   ┣━━ QB (2376 channels)                                     │
│  ┃   ┃   ┗━━ SECTOR (0 channels)                                │
│  ┃   ┃       ┣━━ 0L (396 channels)                              │
│  ┃   ┃       ┣━━ 1A (396 channels)                              │
│  ┃   ┃       └━━ ... 4 more sectors                             │
│  ┃   └━━ ... 2 more families                                    │
│  └━━ ... 2 more systems                                         │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯

╭────────────── Hierarchy Level Statistics ────────────────────────╮
│                                                                  │
│  ╭───────┬──────────┬───────────────╮                            │
│  │ Level │ Name     │ Unique Values │                            │
│  ├───────┼──────────┼───────────────┤                            │
│  │ 1     │ system   │ 3             │                            │
│  │ 2     │ family   │ 5             │                            │
│  │ 3     │ sector   │ 6             │                            │
│  │ 4     │ device   │ 99            │                            │
│  │ 5     │ property │ 10            │                            │
│  ╰───────┴──────────┴───────────────╯                            │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────╮
│ ✓ Preview complete! 4996 total channels │
╰─────────────────────────────────────────╯
```

### 4. llm_channel_namer.py

**Purpose:** Library for LLM-based channel name generation

**Note:** This is a library module used by `build_channel_database.py`, not a standalone tool.

**Features:**
- Batch processing for efficiency
- Configurable LLM providers (Claude, GPT, Gemini, etc.)
- Structured output using Pydantic models
- Validation and quality checks
- Generates descriptive, human-friendly channel names

**API Example:**

```python
from llm_channel_namer import create_namer_from_config

# Create namer from config
namer = create_namer_from_config()

# Generate names for channels
channels = [
    {'short_name': 'TEMP_01', 'description': 'Temperature sensor in room 1'},
    {'short_name': 'PRESS_01', 'description': 'Pressure gauge at inlet'}
]

# Returns: ['RoomOneTemperatureSensor', 'InletPressureGauge']
names = namer.generate_names(channels)
```

**Configuration:**

In `facility_config.yml`:
```yaml
channel_finder:
  channel_name_generation:
    llm_batch_size: 10          # Process channels in batches
    llm_model:
      provider: cborg
      model_id: anthropic/claude-haiku
      max_tokens: 2000
```

## Workflow Examples

### Workflow A: Build from CSV

1. **Prepare your CSV:**
   ```bash
   # Edit or create your channels CSV in src/your_assistant_name/data/raw/
   vim src/your_assistant_name/data/raw/address_list.csv
   ```

2. **Build the database:**
   ```bash
   # From your-control-assistant directory
   python src/your_assistant_name/data/tools/build_channel_database.py --use-llm --config config.yml
   ```

3. **Validate the database:**
   ```bash
   python src/your_assistant_name/data/tools/validate_database.py
   ```

4. **Preview the result:**
   ```bash
   python src/your_assistant_name/data/tools/preview_database.py
   ```

5. **Test the system:**
   ```bash
   python scripts/channel_finder_cli.py
   ```

### Workflow B: Create JSON Directly

1. **Create your JSON:**
   ```bash
   # Create your database manually
   vim src/your_assistant_name/data/channel_databases/my_database.json
   ```

2. **Validate it:**
   ```bash
   # From your-control-assistant directory
   python src/your_assistant_name/data/tools/validate_database.py --database src/your_assistant_name/data/channel_databases/my_database.json
   ```

3. **Update config:**
   ```yaml
   # In config.yml
   channel_finder:
     pipelines:
       in_context:
         database:
           path: src/your_assistant_name/data/channel_databases/my_database.json
   ```

4. **Preview it:**
   ```bash
   python src/your_assistant_name/data/tools/preview_database.py
   ```

5. **Test the system:**
   ```bash
   python scripts/channel_finder_cli.py
   ```

## CSV Format Details

### Simple Format (Minimal)

For facilities without device families:

```csv
address,description,family_name,instances,sub_channel
BEAM:CURRENT,Total beam current in milliamps,,,
VACUUM:PRESSURE,Main beamline vacuum pressure,,,
RF:FREQUENCY,RF cavity frequency in MHz,,,
```

All `family_name`, `instances`, and `sub_channel` columns are empty.

### Template Format (Device Families)

For facilities with device families:

```csv
address,description,family_name,instances,sub_channel
BEAM:CURRENT,Total beam current,,,
BPM{instance:02d}X,BPM horizontal position,BPM,10,XPosition
BPM{instance:02d}Y,BPM vertical position,BPM,10,YPosition
Valve{instance:02d}:SP,Valve setpoint,Valve,5,SetPoint
Valve{instance:02d}:RB,Valve readback,Valve,5,ReadBack
```

**Pattern Syntax:**
- `{instance:02d}` - Replaced with 01, 02, 03, ... (zero-padded 2 digits)
- `{sub_channel}` - Replaced with sub_channel value from the row

**Grouping:**
- Rows with same `family_name` → grouped into single template
- `instances` value determines how many to generate (assumes 01 to N)

### Multi-Axis Devices

For devices with X/Y axes (like steering coils):

```csv
address,description,family_name,instances,sub_channel
Steering{instance:02d}X:SP,Steering X setpoint,Steering,5,XSetPoint
Steering{instance:02d}X:RB,Steering X readback,Steering,5,XReadBack
Steering{instance:02d}Y:SP,Steering Y setpoint,Steering,5,YSetPoint
Steering{instance:02d}Y:RB,Steering Y readback,Steering,5,YReadBack
```

The builder will detect the X/Y pattern and create a template with axes.

## JSON Format Details

See `data/processed/TEMPLATE_EXAMPLE.json` for comprehensive examples.

### Standalone Entry

```json
{
  "template": false,
  "channel": "BeamCurrent",
  "address": "BEAM:CURRENT",
  "description": "Total electron beam current measured in milliamps"
}
```

### Template Entry (Simple)

```json
{
  "template": true,
  "base_name": "BPM",
  "instances": [1, 10],
  "sub_channels": ["XPosition", "YPosition"],
  "description": "Beam Position Monitors measure beam location",
  "address_pattern": "BPM{instance:02d}{suffix}",
  "channel_descriptions": {
    "XPosition": "Horizontal position from BPM {instance:02d} in mm",
    "YPosition": "Vertical position from BPM {instance:02d} in mm"
  }
}
```

Expands to: `BPM01XPosition`, `BPM01YPosition`, ..., `BPM10XPosition`, `BPM10YPosition`

### Template Entry (With Axes)

```json
{
  "template": true,
  "base_name": "Corrector",
  "instances": [1, 5],
  "axes": ["X", "Y"],
  "sub_channels": ["SetPoint", "ReadBack"],
  "description": "Corrector magnets adjust beam trajectory",
  "address_pattern": "Corrector{instance:02d}{axis}{suffix}",
  "channel_descriptions": {
    "XSetPoint": "Horizontal corrector {instance:02d} setpoint",
    "XReadBack": "Horizontal corrector {instance:02d} readback",
    "YSetPoint": "Vertical corrector {instance:02d} setpoint",
    "YReadBack": "Vertical corrector {instance:02d} readback"
  }
}
```

Expands to: `Corrector01XSetPoint`, `Corrector01XReadBack`, `Corrector01YSetPoint`, etc.

## Configuration

All tools use settings from `facility_config.yml`:

```yaml
facility:
  path: examples/in_context

  # Data source for build_channel_database.py (Workflow A)
  data_source:
    type: csv
    path: data/raw/address_list.csv
    field_mapping:
      address: "address"
      description: "description"
    encoding: utf-8
    skip_empty_rows: true

channel_finder:
  # Runtime database (used by all tools)
  pipelines:
    in_context:
  database:
        type: template
        path: examples/in_context/data/processed/channel_database.json
        presentation_mode: template

  # LLM configuration for name generation (build tool)
  channel_name_generation:
    llm_batch_size: 10
    llm_model:
      provider: cborg
      model_id: anthropic/claude-haiku
      max_tokens: 2000
```

## Troubleshooting

### Build Tool Issues

**Problem:** "No database path provided"
   ```bash
# Solution: Make sure you're in the right directory
cd examples/in_context
   python tools/build_channel_database.py
   ```

**Problem:** CSV columns not found
```bash
# Solution: Check your CSV has the required columns
# Required: address, description
# Optional: family_name, instances, sub_channel
```

**Problem:** LLM errors during name generation
```bash
# Solution: Check your API keys and model configuration
# Edit facility_config.yml to configure the LLM provider
```

### Validation Tool Issues

**Problem:** Module import errors
```bash
# Solution: Activate venv and make sure channel_finder is installed
source venv/bin/activate
pip install -e .
```

**Problem:** Template validation errors
   ```bash
# Solution: Check the error messages and fix JSON structure
# Common issues:
# - instances should be [start, end] not a single number
# - sub_channels should be a list ["Set", "Read"] not a string
# - channel_descriptions should map sub_channel names to descriptions
```

### Preview Tool Issues

**Problem:** Can't find database
   ```bash
# Solution: Make sure database path in config is correct
# Check facility_config.yml: channel_finder.pipelines.in_context.database.path
   ```

## Architecture

These tools work with the in-context pipeline:
- **Input**: Simple CSV format (address, description, optional family metadata)
- **Output**: Template-based JSON database
- **Runtime**: TemplateChannelDatabase class expands templates on-the-fly
- **Location**: All tools should be run from your project root directory (your-control-assistant)

The hierarchical pipeline uses a different approach with pre-built hierarchical databases.

## References

- **Main README**: `../README.md` - Overall example documentation
- **CSV example**: `../data/raw/CSV_EXAMPLE.csv` - CSV format reference (Workflow A)
- **JSON example**: `../data/processed/TEMPLATE_EXAMPLE.json` - JSON format reference (Workflow B)
- **Cleanup summary**: `../CLEANUP_SUMMARY.md` - Recent changes and cleanup details
