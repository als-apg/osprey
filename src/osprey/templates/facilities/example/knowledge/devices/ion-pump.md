---
type: Device
title: Ion Pump
description: Sputter-ion vacuum pump maintaining UHV in beamline sections; current readback indicates pressure indirectly.
tags: [vacuum, uhv, pump, ion-pump, pressure]
---

# Ion Pump

Ion pumps (IPs) are the primary vacuum-maintaining devices in the Example
Research Facility beamline.  They operate continuously to maintain UHV
conditions without moving parts or oil contamination.

## Channel Pattern

Ion pumps follow the naming convention `{SYS}:VAC:IP{N}:{param}`:

```
TD:VAC:IP01:Current       # Pump current readback (mA) — proxy for pressure
TD:VAC:IP01:Voltage       # High-voltage supply voltage (V)
TD:VAC:IP01:Enable        # Pump enabled: 1=on, 0=off
TD:VAC:IP01:Fault         # Fault status: 0=OK, 1=fault
```

## Current-to-Pressure Relationship

Ion pump current is a rough proxy for pressure at UHV:

| Current (mA) | Approximate pressure (Torr) |
|-------------|---------------------------|
| `< 0.001` | `< 1e-9` (nominal UHV) |
| `0.01–0.1` | `1e-8 – 1e-7` (marginal) |
| `> 0.1` | `> 1e-7` (interlock range) |

Use a calibrated ion gauge (`{SYS}:VAC:{Section}:Pressure`) for accurate
pressure readings; see [Vacuum System](/subsystems/vacuum.md).

## Fault Conditions

Common ion pump faults and their meaning:

| Fault | Likely cause | Action |
|-------|-------------|--------|
| `Fault=1, Current=0` | HV supply trip | Check HV supply, reset if safe |
| `Fault=0, Current > 1 mA` | High gas load / leak | Investigate upstream leak |
| `Enable=0` | Deliberately disabled for venting | Re-enable after pumpdown |

## Operational Notes

- Ion pumps must remain on at all times during beam operation.
- Turning off a pump while others are running triggers a gate-valve closure
  in the affected section (pressure excursion interlock).
- Pumps must be conditioned at reduced voltage after a vent; see
  [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md).

## Cross-links

- [Vacuum System](/subsystems/vacuum.md) — overall vacuum architecture
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — post-vent pumpdown
