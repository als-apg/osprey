# Channel Limits: EPICS DRVH/DRVL Integration

## Overview

APS EPICS records store drive limits in `.DRVH` (drive high) and `.DRVL` (drive low) fields on setpoint PVs. These are the authoritative IOC-level safety limits. The `channel_limits.json` file is used by OSPREY to validate channel writes at runtime.

The script `scripts/update_channel_limits.py` reads `.DRVH`/`.DRVL` from live EPICS PVs and merges them into `channel_limits.json`, while preserving manual overrides like `max_step`, `verification`, and `writable`.

## Usage

```bash
cd /home/oxygen8/SHANG/next_osprey/aps-control-assistant

# Preview what would change (no writes)
python scripts/update_channel_limits.py --dry-run

# Update channel_limits.json from live EPICS
python scripts/update_channel_limits.py

# Custom timeout for slow network
python scripts/update_channel_limits.py --timeout 5
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Show what would change without writing |
| `--pv-database` | `data/channel_databases/its_pv_in_context.json` | Path to PV database |
| `--output` | `data/channel_limits.json` | Path to channel limits file |
| `--timeout` | `3.0` | EPICS caget timeout in seconds |

## Merge Logic (Safety-First)

The merge uses the **tighter** bound when both EPICS and manual values exist:

```
final min_value = max(EPICS_DRVL, manual_min)   # tighter lower bound
final max_value = min(EPICS_DRVH, manual_max)   # tighter upper bound
```

### Priority Rules

- **EPICS `.DRVH`/`.DRVL`** provides `min_value`/`max_value`
- **Manual overrides preserved:** `max_step`, `verification`, `writable`, `_comment`
- **Tighter wins:** If manual entry has tighter bounds than EPICS, the tighter (safer) value is kept
- **New channels:** PVs found in EPICS but not in the limits file are added with EPICS limits and default verification
- **Missing from EPICS:** Channels in the limits file but not reachable via EPICS are kept as-is
- **Unset limits skipped:** PVs where `DRVH == DRVL == 0` are skipped (limits not configured in IOC)

### EPICS Source Tracking

After merging, each entry is tagged with the original EPICS values for traceability:

```json
"LTS:H1:CurrentAO": {
    "min_value": -5.0,
    "max_value": 5.0,
    "max_step": 0.2,
    "_epics_drvl": -5.0,
    "_epics_drvh": 5.0,
    "verification": {
        "level": "readback",
        "tolerance_percent": 0.2
    }
}
```

## Backup

The script creates a `.bak` backup of `channel_limits.json` before overwriting. The backup is saved as `data/channel_limits.json.bak`.

## Current ITS Channel Limits

22 writable ITS channels are configured with readback verification at 0.2% tolerance:

| Category | Channels | Address Pattern | Min | Max | Step |
|----------|----------|----------------|-----|-----|------|
| H Correctors | H1-H4 | `LTS:H{N}:CurrentAO` | -5 A | +5 A | 0.2 A |
| H Corrector | H5 | `LTS:H5:CurrentAO` | -4 A | +4 A | 0.2 A |
| V Correctors | V1-V4 | `LTS:V{N}:CurrentAO` | -5 A | +5 A | 0.2 A |
| V Corrector | V5 | `LTS:V5:CurrentAO` | -4 A | +4 A | 0.2 A |
| Quadrupoles | Q1-Q5 | `LTS:Q{N}:CurrentAO` | -5 A | +5 A | 0.2 A |
| Quadrupoles | Q6-Q9 | `LTS:Q{N}:CurrentAO` | -4 A | +4 A | 0.2 A |
| Bending Magnets | BM1-BM2 | `LTS:BM{N}:CurrentAO` | 0 A | +33 A | 0.2 A |
| Raw PS | RAW | `LTS:RAW:CurrentAO` | -8 A | +8 A | 0.2 A |
| Alpha Magnet | Alpha | `LTS:Alpha:CurrentAO` | -15 A | +15 A | 0.5 A |
