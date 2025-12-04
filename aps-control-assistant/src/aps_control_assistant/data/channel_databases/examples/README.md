# Hierarchical Database Examples

This directory contains example hierarchical database configurations demonstrating the flexible hierarchy system. Each example showcases different patterns for organizing control system channels.

## Quick Reference

|            Example           |  Depth   |   Pattern Type    |
|------------------------------|----------|-------------------|
| `instance_first.json`        | 3 levels | Instance-driven   |
| `consecutive_instances.json` | 5 levels | Compact naming    |
| `mixed_hierarchy.json`       | 5 levels | Variable subtrees |
| `hierarchical_legacy.json`   | 5 levels | Legacy format     |

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

## Choosing the Right Example

### Learning Path
1. **Start with `instance_first.json`** - Understand basic instance expansion (3 levels, 85 channels)
2. **Move to `consecutive_instances.json`** - See real-world compact naming (5 levels, 4,996 channels)
3. **Explore `mixed_hierarchy.json`** - Learn advanced variable subtree patterns (5 levels, 1,720 channels)

### Pattern Selection Guide

**Your facility has...**

| Characteristic | Use Example |
|----------------|-------------|
| Numbered/lettered divisions (lines, sectors, zones) | `instance_first.json` |
| Compact naming (MQB1A03, R2C4, etc.) | `consecutive_instances.json` |
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
