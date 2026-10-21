---
type: Device
title: Beam Position Monitor (BPM)
description: Electromagnetic pickup measuring beam centroid position in horizontal and vertical planes with sub-millimetre resolution.
tags: [diagnostics, beam-position, bpm, orbit, pickup]
---

# Beam Position Monitor (BPM)

Beam Position Monitors (BPMs) measure the transverse position of the beam
centroid using electromagnetic button pickups.  BPMs are distributed along the
entire beamline and are the primary diagnostic for orbit control.

## Channel Pattern

BPMs follow the naming convention `{SYS}:BPM:{ID}:{param}`:

```
TD:BPM:H01:X        # Horizontal position at H01 (mm)
TD:BPM:H01:Y        # Vertical position at H01 (mm)
TD:BPM:H01:Sum      # Total signal sum (a.u.) — proxy for beam current at this point
TD:BPM:H01:Quality  # Data quality flag: OK / NOBEAM / FAULT
```

## Resolution and Range

| Parameter | Value |
|-----------|-------|
| Position range | ±10 mm |
| Nominal resolution | 10 µm rms |
| Update rate | 10 Hz (fast acquisition), 1 Hz (slow orbit) |

## Orbit Correction

The golden orbit is stored in the control system.  Deviations from golden
orbit trigger corrector magnet adjustments via the orbit feedback loop.
See [Orbit Correction Procedure](/procedures/orbit-correction.md).

## Quality Flags

- **`OK`** — Valid position data.
- **`NOBEAM`** — No beam signal detected; position readback is unreliable.
- **`FAULT`** — Electronics fault; data should not be used for orbit control.

Always check `Quality = OK` before trusting position readbacks for
diagnostics or feedback.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — primary BPM location
- [Experimental Stations](/subsystems/experimental-stations.md) — end-station BPMs
- [Orbit Correction Procedure](/procedures/orbit-correction.md) — using BPMs to correct orbit
- [Quadrupole Scan](/physics/quadrupole-scan.md) — BPM data used in optics measurement
