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

## Channels

Each pump is a device record of the facility file.  Pump
[SR/ION-PUMP01](facility:SR/ION-PUMP01), for example, carries:

- [SR:VAC:ION-PUMP:01:CURRENT:RB](facility:SR:VAC:ION-PUMP:01:CURRENT:RB) — pump current, a proxy for pressure
- [SR:VAC:ION-PUMP:01:PRESSURE:RB](facility:SR:VAC:ION-PUMP:01:PRESSURE:RB) — the pressure calculated from the pump current
- [SR:VAC:ION-PUMP:01:VOLTAGE:SP](facility:SR:VAC:ION-PUMP:01:VOLTAGE:SP) — high-voltage supply setpoint
- [SR:VAC:ION-PUMP:01:VOLTAGE:RB](facility:SR:VAC:ION-PUMP:01:VOLTAGE:RB) — high-voltage supply readback
- [SR:VAC:ION-PUMP:01:STATUS:ON](facility:SR:VAC:ION-PUMP:01:STATUS:ON) — the pump is on
- [SR:VAC:ION-PUMP:01:STATUS:FAULT](facility:SR:VAC:ION-PUMP:01:STATUS:FAULT) — the pump reports a fault

Ask the channel finder for the other pumps rather than composing addresses by
hand.

## Current-to-Pressure Relationship

Ion pump current is a rough proxy for pressure at UHV:

| Current | Approximate pressure (Torr) |
|---------|---------------------------|
| `< 1 µA` | `< 1e-9` (nominal UHV) |
| `10–100 µA` | `1e-8 – 1e-7` (marginal) |
| `> 100 µA` | `> 1e-7` (interlock range) |

Use a calibrated vacuum gauge, such as
[SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB), for
accurate pressure readings; see [Vacuum System](/subsystems/vacuum.md).

## Fault Conditions

Common ion pump faults and their meaning:

| Fault | Likely cause | Action |
|-------|-------------|--------|
| Fault true, current near zero | HV supply trip | Check HV supply, reset if safe |
| No fault, current above 1 mA | High gas load / leak | Investigate upstream leak |
| Pump off, no fault | Deliberately switched off for venting | Switch on after pumpdown |

## Operational Notes

- Ion pumps must remain on at all times during beam operation.
- Turning off a pump while others are running triggers a gate-valve closure
  in the affected section (pressure excursion interlock).
- Pumps must be conditioned at reduced voltage after a vent; see
  [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md).

## Cross-links

- [Vacuum System](/subsystems/vacuum.md) — overall vacuum architecture
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — post-vent pumpdown
