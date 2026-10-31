---
type: PhysicsNote
title: Beam Current Calibration
description: Notes on DCCT calibration, systematic offsets, and how to interpret beam current readbacks accurately.
tags: [diagnostics, beam-current, dcct, calibration, measurement]
---

# Beam Current Calibration

The circulating beam current is measured by DC Current Transformers (DCCTs):
[SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB) in SR and
[BR:DIAG:DCCT:01:CURRENT:RB](facility:BR:DIAG:DCCT:01:CURRENT:RB) in the
booster.  Each DCCT also reports whether its measurement is valid, for example
[SR:DIAG:DCCT:01:STATUS:VALID](facility:SR:DIAG:DCCT:01:STATUS:VALID).  This
note describes the calibration basis and known systematic effects.

The single-pass lines measure charge per pulse instead, with integrating
current transformers: [GTL:DIAG:ICT:01:CHARGE:RB](facility:GTL:DIAG:ICT:01:CHARGE:RB)
after the gun, [LTB:DIAG:ICT:01:CHARGE:RB](facility:LTB:DIAG:ICT:01:CHARGE:RB)
before the booster and [BTS:DIAG:ICT:01:CHARGE:RB](facility:BTS:DIAG:ICT:01:CHARGE:RB)
before SR.

## DCCT Operating Principle

A DCCT measures the magnetic field produced by the circulating beam current.
The readback has a linear relationship to true current:

```
I_true = (I_readback - I_offset) / G_cal
```

Where:
- `I_offset` — residual offset (typically ±0.05 mA; calibrated monthly)
- `G_cal` — calibration gain (nominally 1.000; verified quarterly)

## Calibration Status

Calibration is performed by injecting a known DC reference current through
a calibration winding.  The example facility file holds no records for the
calibration constants; the offset, the gain and the date of the last
calibration are kept in the calibration log.  Check the date of the last
calibration if current readings seem anomalous.

## Known Systematic Effects

| Effect | Magnitude | Condition |
|--------|-----------|-----------|
| Thermal drift | ±0.1 mA | During first 30 min warmup |
| RF interference | ±0.2 mA | High-power RF conditioning |
| Ground loop | ±0.05 mA | If shielding is compromised |

## Interpretation Guidelines

- At currents `< 5 mA`, the DCCT readback is unreliable; use the injection
  charge monitors instead.
- For absolute accuracy requirements `< 1%`, use a dedicated Faraday cup
  measurement (available in the injector during machine studies).
- A sudden step change in the DCCT readback with no beam action is likely
  an electronics fault, not a real current change.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — the injector and its charge monitors
- [Primary Source Startup](/procedures/ps-startup.md) — warmup before trusting readback
