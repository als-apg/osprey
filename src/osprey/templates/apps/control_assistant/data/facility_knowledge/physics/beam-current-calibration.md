---
type: PhysicsNote
title: Beam Current Calibration
description: Notes on DCCT calibration, systematic offsets, and how to interpret beam current readbacks accurately.
tags: [diagnostics, beam-current, dcct, calibration, measurement]
---

# Beam Current Calibration

The beam current is measured by the DC Current Transformer (DCCT) at
`PS:DCCT:Current`.  This note describes the calibration basis and known
systematic effects.

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
a calibration winding.  The result is stored in:

```
PS:DCCT:Cal:Offset    # Current offset correction (mA)
PS:DCCT:Cal:Gain      # Gain correction factor
PS:DCCT:Cal:Date      # Date of last calibration (ISO 8601)
```

Check `PS:DCCT:Cal:Date` if current readings seem anomalous.

## Known Systematic Effects

| Effect | Magnitude | Condition |
|--------|-----------|-----------|
| Thermal drift | ±0.1 mA | During first 30 min warmup |
| RF interference | ±0.2 mA | High-power RF conditioning |
| Ground loop | ±0.05 mA | If shielding is compromised |

## Interpretation Guidelines

- At currents `< 5 mA`, the DCCT readback is unreliable; use injection
  diagnostics instead.
- For absolute accuracy requirements `< 1%`, use a dedicated Faraday cup
  measurement (available in the PS injection section during machine studies).
- A sudden step change in `PS:DCCT:Current` with no beam action is likely
  an electronics fault, not a real current change.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — DCCT location
- [Primary Source Startup](/procedures/ps-startup.md) — warmup before trusting readback
