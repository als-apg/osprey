---
type: PhysicsNote
title: Quadrupole Scan
description: Beam optics measurement technique using quadrupole current variation and BPM response to determine beta functions and emittance.
tags: [optics, quadrupole, emittance, beta-function, bpm, measurement]
---

# Quadrupole Scan

The quadrupole scan is the primary technique for measuring beam optics parameters
(beta function, emittance) in the transport line.  It varies a quadrupole strength
and records beam size or orbit response at downstream BPMs.

## Physical Basis

For a thin-lens quadrupole with focal length `f`, the beam size at a
downstream observation point varies as:

```
σ²(f) = A/f² + B/f + C
```

where A, B, C are constants determined by the upstream Twiss parameters.
Fitting this parabola to measured σ² vs. 1/f gives the emittance and beta
function.

## Measurement Protocol

### Prerequisites

- Stable beam: `PS:DCCT:Current > 50 mA` (minimum for BPM signal quality)
- All BPMs: `Quality = OK`
- No orbit feedback running during the scan

### Channels Varied

Scan over the quadrupole current setpoint in small steps:

```
TD:QF:QF01:CURRENT:SP   # Focusing quadrupole (varied)
```

Typical scan range: ±5% around nominal current, in 1% steps.

### Channels Read

At each step, record:

```
TD:BPM:H02:X   # Horizontal position at downstream BPM
TD:BPM:H02:Y   # Vertical position at downstream BPM
TD:BPM:H02:Sum # Sum signal (proportional to beam intensity at BPM)
```

### Fitting

Fit `σ²` vs. `1/f` (quadrupole current) to the parabolic model.  The
resulting Twiss parameters allow computation of:

- Horizontal emittance εₓ (normalised or geometric)
- Beta function βₓ at the measurement point
- Alpha function αₓ

## Typical Values at ERF

| Parameter | Typical value |
|-----------|--------------|
| εₓ (geometric) | 10–50 nm·rad |
| βₓ at TD BPMs | 5–15 m |
| αₓ | -1 to +1 |

## Notes

- Perform the scan at low beam current (50–100 mA) to minimise space-charge
  effects that would perturb the optics.
- After the scan, restore the quadrupole to its nominal setpoint and verify
  orbit with [Orbit Correction Procedure](/procedures/orbit-correction.md).
- This measurement is typically performed during machine studies only.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — quadrupole locations
- [Beam Position Monitor (BPM)](/devices/bpm.md) — measurement devices
- [Orbit Correction Procedure](/procedures/orbit-correction.md) — post-scan cleanup
