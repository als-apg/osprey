---
type: Subsystem
title: Primary Source (PS)
description: Main particle/photon source system including power supplies, vacuum, RF, and diagnostics subsystems.
tags: [source, rf, vacuum, diagnostics, power-supply]
---

# Primary Source (PS)

The Primary Source (PS) is the injector of the Example Research Facility: the
electron gun and the gun-to-linac line (GTL), and the linac (LN) that takes the
beam to 150 MeV.  It generates and conditions the beam before the
[Transport and Delivery system](/subsystems/transport-delivery.md) carries it
on.

## Components

| Component | Role | Example records |
|-----------|------|-----------------|
| Electron gun | Produces the beam | [GTL/EGUN01](facility:GTL/EGUN01), [GTL:GUN:EGUN:01:VOLTAGE:RB](facility:GTL:GUN:EGUN:01:VOLTAGE:RB) |
| Bunchers | Compress the gun pulse into bunches | [GTL:RF:SHB:01:VOLTAGE:SP](facility:GTL:RF:SHB:01:VOLTAGE:SP), [GTL:RF:SHB:01:PHASE:SP](facility:GTL:RF:SHB:01:PHASE:SP) |
| Power supplies | Regulate solenoid and magnet currents | [GTL:MAG:SOL:01:CURRENT:SP](facility:GTL:MAG:SOL:01:CURRENT:SP), [LN:MAG:QF:01:CURRENT:SP](facility:LN:MAG:QF:01:CURRENT:SP) |
| Linac RF | Accelerates the beam | [LN/MASTER01](facility:LN/MASTER01), [LN:RF:ACC:02:PHASE:SP](facility:LN:RF:ACC:02:PHASE:SP) |
| Cooling water | Holds the accelerating sections at temperature | [LN:RF:WATER:01:TEMPERATURE:RB](facility:LN:RF:WATER:01:TEMPERATURE:RB) |
| Diagnostics | Monitor charge and position | [GTL:DIAG:ICT:01:CHARGE:RB](facility:GTL:DIAG:ICT:01:CHARGE:RB), [LN:DIAG:BPM:01:POSITION:X](facility:LN:DIAG:BPM:01:POSITION:X) |

Ask the channel finder for the other devices of the injector rather than
composing addresses by hand.

## Beam Charge

The injector is single-pass, so it measures the charge per pulse rather than a
circulating current: [GTL:DIAG:ICT:01:CHARGE:RB](facility:GTL:DIAG:ICT:01:CHARGE:RB)
after the gun and [LN:DIAG:ICT:01:CHARGE:RB](facility:LN:DIAG:ICT:01:CHARGE:RB)
after the linac.  The stored current is read in SR, at
[SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB); typical
operating range: 10–500 mA depending on mode.

## RF System

The master RF ([LN:RF:MASTER:01:PHASE:SP](facility:LN:RF:MASTER:01:PHASE:SP))
phases the linac's buncher cavity and accelerating sections; SR's RF runs at
500 MHz ([SR:RF:CAVITY:01:FREQUENCY:RB](facility:SR:RF:CAVITY:01:FREQUENCY:RB)).

## Operational Notes

- The injector requires a 30-minute warmup after a cold start before beam
  injection.
- RF trips above 5% power excursion trigger automatic cavity detuning.
- Vacuum interlocks close gate valves automatically when pressure exceeds
  `1e-7 Torr`; see the [Vacuum System](/subsystems/vacuum.md) for the
  full pressure hierarchy.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — downstream recipient of PS beam
- [Vacuum System](/subsystems/vacuum.md) — shared vacuum infrastructure
- [PS Startup Procedure](/procedures/ps-startup.md) — step-by-step startup
- [Beam Current Calibration](/physics/beam-current-calibration.md) — DCCT calibration physics
