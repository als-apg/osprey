---
type: Subsystem
title: Primary Source (PS)
description: Main particle/photon source system including power supplies, vacuum, RF, and diagnostics subsystems.
tags: [source, rf, vacuum, diagnostics, power-supply]
---

# Primary Source (PS)

The Primary Source (PS) is the main particle/photon source for the Example
Research Facility.  It generates and conditions the beam before injection into
the [Transport and Delivery system](/subsystems/transport-delivery.md).

## Components

| Component | Role | Key channels |
|-----------|------|-------------|
| Power Supplies | Regulate magnet and solenoid currents | `PS:MAG:*:CURRENT` |
| RF System | Accelerate and bunch the beam | `PS:RF:*:Power`, `PS:RF:*:Phase` |
| Vacuum | Maintain beam-quality vacuum | `PS:VAC:*:Pressure` |
| Diagnostics | Monitor beam current, position, size | `PS:DCCT:Current`, `PS:BPM:*:X` |

## Beam Current

The primary beam current is read from the DC current transformer (DCCT):

```
PS:DCCT:Current   # Readback — primary beam current (mA)
```

Typical operating range: 10–500 mA depending on mode.

## RF System

The RF system operates at 500 MHz.  Key setpoints:

```
PS:RF:CAV01:Power:SP    # Cavity power setpoint (kW)
PS:RF:CAV01:Phase:SP    # Cavity phase setpoint (degrees)
```

## Operational Notes

- The PS requires a 30-minute warmup after a cold start before beam injection.
- RF trips above 5% power excursion trigger automatic cavity detuning.
- Vacuum interlocks open gate valves automatically when pressure exceeds
  `1e-7 Torr`; see the [Vacuum System](/subsystems/vacuum.md) for the
  full pressure hierarchy.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — downstream recipient of PS beam
- [Vacuum System](/subsystems/vacuum.md) — shared vacuum infrastructure
- [PS Startup Procedure](/procedures/ps-startup.md) — step-by-step startup
- [Beam Current Calibration](/physics/beam-current-calibration.md) — DCCT calibration physics
