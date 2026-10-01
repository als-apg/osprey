---
type: Reference
title: EPICS Channel Access Reference
description: Channel naming conventions, access modes, and common PV types used at the Example Research Facility.
tags: [epics, channel-access, pv, naming, convention, reference]
resource: https://epics.anl.gov/EpicsDocumentation/AppDevManuals/ChannelAccess/cadoc_4.htm
---

# EPICS Channel Access Reference

The Example Research Facility uses EPICS (Experimental Physics and Industrial
Control System) Channel Access as its primary control-system protocol.  This
document summarises ERF-specific naming conventions and common record types.

## Naming Convention

ERF channel names follow the pattern:

```
{System}:{Subsystem}:{Device}:{Field}
```

Where `{Field}` encodes the access mode:

| Field suffix | Mode | Description |
|-------------|------|-------------|
| (none / bare) | Read | Readback / process variable |
| `:SP` | Write | Setpoint — triggers a hardware action |
| `:Cmd` | Write | Command — typically a single-shot action |
| `:Enable` | Read/Write | Enable gate |
| `:Status` | Read | Derived status string |
| `:Fault` | Read | Fault flag (0=OK, 1=fault) |

## Common Record Types

| EPICS record | ERF usage | Example |
|-------------|-----------|---------|
| `ai` / `ao` | Analog readback / setpoint | `PS:DCCT:Current`, `PS:RF:CAV01:Power:SP` |
| `bi` / `bo` | Binary readback / output | `ES3:SHUTTER:State`, `ES3:SHUTTER:Cmd` |
| `stringin` | String status | `ES{N}:Status`, `TD:BPM:H01:Quality` |
| `motor` | Motor record | `ES{N}:MOT:{Axis}:Position` |
| `waveform` | Array data | (used for BPM turn-by-turn data) |

## Access Control

- **Read access** is unrestricted for diagnostic channels.
- **Write access** to setpoints and commands requires the OSPREY approval
  workflow (see [Safety Rules](/references/safety-rules.md)).
- **Interlocked channels** (PSS, vacuum gate valves) have hardware-enforced
  write guards that the software cannot override.

## Units Convention

| Quantity | Unit |
|----------|------|
| Beam current | mA |
| Magnet current | A |
| RF power | kW |
| RF phase | degrees |
| Pressure | Torr |
| Position (BPM, motor) | mm |
| Rotation (motor) | degrees |
| Voltage | V |
| Temperature | °C |

## Cross-links

- [Safety Rules](/references/safety-rules.md) — write-access policy
- [Vacuum System](/subsystems/vacuum.md) — pressure channels
- [Transport and Delivery](/subsystems/transport-delivery.md) — magnet and BPM channels

# Citations

[1] [EPICS Channel Access Reference Manual](https://epics.anl.gov/EpicsDocumentation/AppDevManuals/ChannelAccess/cadoc_4.htm)
