# Hierarchical Database Examples

This directory contains example hierarchical database configurations demonstrating the flexible hierarchy system. Each example showcases different patterns for organizing control system channels.

## Quick Reference

|            Example            |  Depth   |   Pattern Type    |  Tests  |
|-------------------------------|----------|-------------------|---------|
| `instance_first.json`         | 3 levels | Instance-driven   | ✅ AUTO |
| `consecutive_instances.json`  | 5 levels | Compact naming    | ✅ AUTO |
| `mixed_hierarchy.json`        | 5 levels | Variable subtrees | ✅ AUTO |
| `optional_levels.json`        | 6 levels | Optional levels   | ✅ AUTO |
| `hierarchical_jlab_style.json`| 5 levels | Navigation-only   | ✅ AUTO |
| `hierarchical_legacy.json`    | 5 levels | Legacy format     | ✅ AUTO |

> **Note:** All example databases are automatically tested by `tests/services/channel_finder/test_all_example_databases.py`. New databases added to this directory are automatically discovered and tested.

---

## 0. Legacy Format Reference
**File:** `hierarchical_legacy.json`

### Overview
Accelerator control system database in the **legacy format**. This file demonstrates the old container-based structure using `devices`, `fields`, and `subfields` containers. Kept as a reference for understanding the migration from legacy to new format.

### Legacy Format Structure
```json
{
  "hierarchy_definition": ["system", "family", "device", "field", "subfield"],
  "tree": {
    "FAMILY": {
      "devices": {
        "_type": "range",
        "_pattern": "B{:02d}",
        "_range": [1, 24]
      },
      "fields": {
        "CURRENT": {
          "subfields": {
            "SP": {...},
            "RB": {...}
          }
        }
      }
    }
  }
}
```

### New Format Equivalent
The same structure in new format (see `../hierarchical.json`):
```json
{
  "hierarchy_config": {
    "levels": {
      "device": {"structure": "expand_here", ...}
    }
  },
  "tree": {
    "FAMILY": {
      "DEVICE": {
        "_expansion": {
          "_type": "range",
          "_pattern": "B{:02d}",
          "_range": [1, 24]
        },
        "CURRENT": {
          "SP": {...},
          "RB": {...}
        }
      }
    }
  }
}
```

### Key Differences
**Legacy Format:**
- ❌ Implicit hierarchy configuration (inferred from structure)
- ❌ Uses container keys: `devices`, `fields`, `subfields`
- ❌ Nested `subfields` within each field
- ✓ Backward compatible (still supported)

**New Format:**
- ✓ Explicit `hierarchy_config` section
- ✓ Consistent structure: `DEVICE` with `_expansion`
- ✓ Flat field/subfield structure (no nesting)
- ✓ Clearer semantics and easier validation

### Migration Notes
The legacy format is **still supported** through automatic inference in `hierarchical.py`. However, new databases should use the explicit configuration format for better clarity and validation.

---

## 1. Instance First Pattern
**File:** `instance_first.json`

### Overview
Manufacturing production line with **numbered lines** sharing the same station structure. Perfect first example of instance expansion.

### Pattern Visualization
```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1: LINE        [Instance]  → Expands to: 1, 2, 3, 4, 5│
│   └─ LEVEL 2: STATION    [Tree]  → ASSEMBLY, INSPECTION, .. │
│       └─ LEVEL 3: PARAMETER [Tree] → SPEED, STATUS, ...     │
└─────────────────────────────────────────────────────────────┘
```

### Example Expansion
**Query:** `"LINE{1-5}:ASSEMBLY:SPEED"`

**Expands to 5 channels:**
```
LINE1:ASSEMBLY:SPEED
LINE2:ASSEMBLY:SPEED
LINE3:ASSEMBLY:SPEED
LINE4:ASSEMBLY:SPEED
LINE5:ASSEMBLY:SPEED
```

### Use Case
✓ Numbered/lettered primary divisions (lines, sectors, buildings)
✓ Each division has identical subsystems
✓ Simple facilities where "copy this structure N times" is the pattern

---

## 2. Consecutive Instances Pattern
**File:** `consecutive_instances.json`

### Overview
Accelerator magnet naming following **CEBAF convention**: compact names where multiple instance indices appear consecutively (sector AND device number).

### Pattern Visualization
```
┌────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: SYSTEM  [Tree] → M (Magnet), V (Vacuum), D (Diagnostics) │
│   └─ LEVEL 2: FAMILY [Tree] → QB (Quadrupole), DP (Dipole), ...   │
│       └─ LEVEL 3: SECTOR    [Instance] → 0L, 1A, 1B, 2A, 2B, 3A   │
│           └─ LEVEL 4: DEVICE    [Instance] → 01, 02, 03, ..., 99  │
│               └─ LEVEL 5: PROPERTY [Tree] → .S, .M, .BDL, .X, ... │
└────────────────────────────────────────────────────────────────────┘
```

### Example Expansion
**Query:** `"MQB{0L,1A}0{1-3}.S"`
(Magnet Quadrupole in sectors 0L or 1A, devices 01-03, Setpoint)

**Expands to 6 channels:**
```
MQB0L01.S    (Sector 0L, Device 01)
MQB0L02.S    (Sector 0L, Device 02)
MQB0L03.S    (Sector 0L, Device 03)
MQB1A01.S    (Sector 1A, Device 01)
MQB1A02.S    (Sector 1A, Device 02)
MQB1A03.S    (Sector 1A, Device 03)
```

### Use Case
✓ **Compact naming conventions** (no delimiters between instance parts)
✓ Multiple instance dimensions (sector × device, row × column, etc.)
✓ Accelerator magnets, detector arrays, sensor grids

**Key Innovation:** Two consecutive instance levels without tree navigation between them.

---

## 3. Mixed Hierarchy Pattern
**File:** `mixed_hierarchy.json`

### Overview
Building management system where **different buildings have different structures**. Demonstrates that tree branches can have different subtree shapes.

### Pattern Visualization
```
┌───────────────────────────────────────────────────────────────┐
│ LEVEL 1: SECTOR   [Instance] → 01, 02, 03, 04               │
│   └─ LEVEL 2: BUILDING  [Tree]                              │
│       ├─ MAIN_BUILDING → 5 floors × 20 rooms × 3 equip types│
│       ├─ ANNEX         → 3 floors × 15 rooms × 2 equip types│
│       └─ LAB           → 2 floors × named rooms × 4 equip   │
│           └─ LEVEL 3: FLOOR    [Instance]                   │
│               └─ LEVEL 4: ROOM      [Instance]              │
│                   └─ LEVEL 5: EQUIPMENT [Tree]              │
└───────────────────────────────────────────────────────────────┘
```

### Example Expansion
**Query 1:** `"S01:MAIN_BUILDING:F{1-2}:R{101,102}:HVAC"`

**Expands to 4 channels:**
```
S01:MAIN_BUILDING:F1:R101:HVAC
S01:MAIN_BUILDING:F1:R102:HVAC
S01:MAIN_BUILDING:F2:R101:HVAC
S01:MAIN_BUILDING:F2:R102:HVAC
```

**Query 2:** `"S04:LAB:F2:R{LAB_A,CLEAN_ROOM}:PRESSURE"`

**Expands to 2 channels:**
```
S04:LAB:F2:RLAB_A:PRESSURE
S04:LAB:F2:RCLEAN_ROOM:PRESSURE
```

### Use Case
✓ **Heterogeneous facilities** (different areas have different structures)
✓ Building/campus management with variable floor/room counts
✓ Complex systems where not all branches are identical

**Key Innovation:** Tree branches (buildings) define different subtree structures for the same instance levels (floors, rooms).

---

## 4. Optional Levels Pattern
**File:** `optional_levels.json`

### Overview
Demonstrates **optional hierarchy levels** where some channels skip intermediate levels while others include them. This feature supports real-world naming conventions where levels like subdevice or suffix are conditionally present.

### Pattern Visualization
```
┌─────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: SYSTEM     [Tree]                                          │
│   └─ LEVEL 2: SUBSYSTEM [Tree]                                      │
│       └─ LEVEL 3: DEVICE    [Instance] → DEV-01 to DEV-24           │
│           ├─ DIRECT PATH (skips subdevice)                          │
│           │   └─ LEVEL 5: SIGNAL [Tree] → SIGNAL-X, SIGNAL-Y        │
│           │       └─ LEVEL 6: SUFFIX [Tree, Optional] → RB, SP      │
│           └─ SUBDEVICE PATH                                          │
│               └─ LEVEL 4: SUBDEVICE [Tree, Optional] → SUBDEVX, ... │
│                   └─ LEVEL 5: SIGNAL [Tree] → SIGNAL-XX, ...        │
│                       └─ LEVEL 6: SUFFIX [Tree, Optional] → RB, SP  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

**1. Optional Subdevice Level**
- Some signals go directly from device to signal: `SYSTEM-SSYS-01:DEV-01:SIGNAL-X`
- Others include subdevice: `SYSTEM-SSYS-01:DEV-01:SUBDEVX:SIGNAL-XX`
- System automatically cleans up separator artifacts (`::`  → `:`)

**2. Optional Suffix Level**
- Base signals without suffix: `SIGNAL-Y` → `SYSTEM-SSYS-01:DEV-01:SIGNAL-Y`
- Suffixed variants: `SIGNAL-Y_RB`, `SIGNAL-Y_SP`
- Trailing separators automatically removed (`_` at end)

**3. Automatic Leaf Detection**
- Nodes without children are automatically detected as leaves (no `_is_leaf` needed)
- Explicit `_is_leaf: true` ONLY needed for nodes that have children but are also complete channels
- Example: `SIGNAL-Y` needs `_is_leaf` (generates base channel AND has children `RB`, `SP`)
- Example: `RB` and `SP` don't need `_is_leaf` (no children = automatic leaf)

### Example Expansion

**Query 1:** Direct signal (no subdevice, no suffix)
```
Path: SYSTEM → SSYS → DEV-01 → SIGNAL-X
Channel: SYSTEM-SSYS-01:DEV-01:SIGNAL-X
```

**Query 2:** Direct signal with suffix (no subdevice)
```
Path: SYSTEM → SSYS → DEV-01 → SIGNAL-Y → RB
Channel: SYSTEM-SSYS-01:DEV-01:SIGNAL-Y_RB
```

**Query 3:** Subdevice signal with suffix
```
Path: SYSTEM → SSYS → DEV-01 → SUBDEVX → SIGNAL-XY → SP
Channel: SYSTEM-SSYS-01:DEV-01:SUBDEVX:SIGNAL-XY_SP
```

**Query 4:** Instanced subdevice
```
Path: SYSTEM → SSYS → DEV-01 → SUBDEVY-03 → SIGNAL-YY
Channel: SYSTEM-SSYS-01:DEV-01:SUBDEVY-03:SIGNAL-YY
```

### Configuration Details

**Marking Optional Levels:**
```json
{
  "hierarchy": {
    "levels": [
      {"name": "subdevice", "type": "tree", "optional": true},
      {"name": "suffix", "type": "tree", "optional": true}
    ]
  }
}
```

**Marking Leaf Nodes:**
```json
{
  "SIGNAL-X": {
    "_description": "Complete channel (automatic leaf - no children)"
  },
  "SIGNAL-Y": {
    "_is_leaf": true,
    "_description": "Complete channel, but also has children (explicit _is_leaf needed)",
    "RB": {
      "_description": "Readback (automatic leaf - no children, no _is_leaf needed)"
    }
  }
}
```

### Total Channels
With 24 devices, this configuration generates **1,440 channels**:
- 24 × SIGNAL-X (direct, no suffix)
- 24 × SIGNAL-Y base (direct, no suffix)
- 24 × SIGNAL-Y_RB, SIGNAL-Y_SP (direct with suffix)
- 24 × SUBDEVX channels (4 signals × 1 instance)
- 24 × SUBDEVY channels (4 signals × 8 instances)

### Use Case
✓ **Variable-depth naming conventions** (some paths longer than others)
✓ Real accelerator/control naming (JLab, ALS, etc.)
✓ Devices with optional sub-channels
✓ Signals with optional suffixes (RB/SP, raw/calibrated, etc.)

**Real-World Example:**
```
LIN-VAC-SW:Version              ← Short path
LIN-MAG-01-QF-02:Ch-03:Current_RB  ← Full path with all levels
```

---

## Choosing the Right Example

### Learning Path
1. **Start with `instance_first.json`** - Understand basic instance expansion (3 levels, 85 channels)
2. **Move to `consecutive_instances.json`** - See real-world compact naming (5 levels, 4,996 channels)
3. **Explore `optional_levels.json`** - Learn optional levels for variable-depth hierarchies (6 levels, 1,440 channels)
4. **Advanced: `mixed_hierarchy.json`** - Variable subtree patterns (5 levels, 1,720 channels)

### Pattern Selection Guide

**Your facility has...**

| Characteristic | Use Example |
|----------------|-------------|
| Numbered/lettered divisions (lines, sectors, zones) | `instance_first.json` |
| Compact naming (MQB1A03, R2C4, etc.) | `consecutive_instances.json` |
| Variable-depth paths (some short, some long) | `optional_levels.json` |
| Optional intermediate levels (subdevices, suffixes) | `optional_levels.json` |
| Different branches with different structures | `mixed_hierarchy.json` |
| Simple hierarchy to learn the system | `instance_first.json` |

---

## Key Concepts Illustrated

### Instance vs Tree Levels
- **Tree level**: Navigate choices (ASSEMBLY vs INSPECTION vs PACKAGING)
- **Instance level**: Expand across numbered/named copies (1, 2, 3 or 0L, 1A, 2B)

### Consecutive Instances
Multiple instance levels in a row (like SECTOR then DEVICE) create **combinatorial expansion** without requiring tree navigation between them.

### Variable Subtrees
Tree branches can define different child structures - not all paths through the hierarchy need to be identical.

### Optional Levels (NEW!)
Hierarchy levels marked `"optional": true` can be skipped in some paths:
- **Automatic leaf detection**: Nodes without children are automatically detected as leaves (no `_is_leaf` needed)
- **Explicit leaf markers**: `"_is_leaf": true` ONLY needed for nodes that have children but are also complete channels
- **Leaf-with-children**: Same node can be both a leaf AND have children (requires explicit `_is_leaf`)
- **Automatic cleanup**: Separator artifacts (`::`, trailing `_`) removed automatically
- **Variable depth**: Different channels can terminate at different hierarchy levels

This enables real-world naming conventions where intermediate levels (like subdevice or suffix) are conditionally present.
