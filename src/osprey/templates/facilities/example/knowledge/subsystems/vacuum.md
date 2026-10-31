---
type: Subsystem
title: Vacuum System (VAC)
description: Ultra-high vacuum across all beamline sections, maintained by ion pumps, turbomolecular pumps, and gate valves.
tags: [vacuum, uhv, ion-pump, gate-valve, pressure]
---

# Vacuum System (VAC)

The Vacuum System maintains ultra-high vacuum (UHV) conditions throughout the
Example Research Facility beamline.  Nominal operating pressure is
`< 1e-8 Torr` in the photon beam path and `< 1e-9 Torr` in SR.

## Pressure Channels

Vacuum gauges read the pressure of their section, for example gauge
[SR/GAUGESR01](facility:SR/GAUGESR01):

- [SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB) — measured pressure
- [SR:VAC:GAUGE:SR01:STATUS:VALID](facility:SR:VAC:GAUGE:SR01:STATUS:VALID) — the reading is valid

Ion pumps also report a pressure calculated from their current, for example
[SR:VAC:ION-PUMP:01:PRESSURE:RB](facility:SR:VAC:ION-PUMP:01:PRESSURE:RB).
Ask the channel finder for the gauges of a section rather than composing
addresses by hand.

## Gate Valves

Gate valves isolate sections when a pressure excursion is detected, for
example valve [SR/VALVE01](facility:SR/VALVE01):

- [SR:VAC:VALVE:01:POSITION:OPEN](facility:SR:VAC:VALVE:01:POSITION:OPEN) — the valve is fully open
- [SR:VAC:VALVE:01:POSITION:CLOSED](facility:SR:VAC:VALVE:01:POSITION:CLOSED) — the valve is fully closed
- [SR:VAC:VALVE:01:STATUS:READY](facility:SR:VAC:VALVE:01:STATUS:READY) — the valve is ready to move

Gate valves are **interlock-controlled**: they close automatically when
neighbouring section pressure exceeds `1e-6 Torr`.  Manual open commands
require both adjacent pressures to be `< 1e-7 Torr`.

## Pump Types

| Device | Typical location | Example record |
|--------|-----------------|----------------|
| [Ion Pump](/devices/ion-pump.md) | All sections | [SR/ION-PUMP01](facility:SR/ION-PUMP01) |
| Turbomolecular Pump | Injection section | none: a cart connected by hand |
| Roughing Pump | Fore-vacuum stages | none: a cart connected by hand |

## Venting and Pumpdown

Venting any section requires shutting the adjacent gate valves first.  A
complete pumpdown from atmosphere to operating pressure takes 24–48 hours.

See [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) for the
full pumpdown sequence.

## Cross-links

- [Ion Pump](/devices/ion-pump.md) — primary UHV pump device
- [Primary Source](/subsystems/primary-source.md) — injection vacuum requirements
- [Personnel Safety System](/subsystems/pss.md) — vacuum interlock authority
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — recovery from a vacuum event
