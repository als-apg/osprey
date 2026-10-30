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

- Stable beam: the beam current
  ([SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB)) above
  50 mA, the minimum for BPM signal quality
- All BPMs used: valid and connected (see [BPM status flags](/devices/bpm.md))
- No orbit feedback running during the scan

### Channels Varied

Scan over the quadrupole current setpoint in small steps, for example the
first booster-to-storage focusing quadrupole
[BTS/QF01](facility:BTS/QF01):

- [BTS:MAG:QF:01:CURRENT:SP](facility:BTS:MAG:QF:01:CURRENT:SP) — current setpoint (varied)
- [BTS:MAG:QF:01:CURRENT:RB](facility:BTS:MAG:QF:01:CURRENT:RB) — current readback

Typical scan range: ±5% around nominal current, in 1% steps.

### Channels Read

At each step, record a monitor downstream of the quadrupole, for example
[BTS/BPM02](facility:BTS/BPM02):

- [BTS:DIAG:BPM:02:POSITION:X](facility:BTS:DIAG:BPM:02:POSITION:X) — horizontal position
- [BTS:DIAG:BPM:02:POSITION:Y](facility:BTS:DIAG:BPM:02:POSITION:Y) — vertical position
- [BTS:DIAG:BPM:02:SUM:RB](facility:BTS:DIAG:BPM:02:SUM:RB) — sum signal, proportional to beam intensity at the monitor

The same scan works in SR with a quadrupole such as
[SR:MAG:QF:01:CURRENT:SP](facility:SR:MAG:QF:01:CURRENT:SP) and its readback
[SR:MAG:QF:01:CURRENT:RB](facility:SR:MAG:QF:01:CURRENT:RB).

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
| βₓ at transfer-line BPMs | 5–15 m |
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
