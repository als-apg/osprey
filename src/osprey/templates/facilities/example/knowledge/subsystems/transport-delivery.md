---
type: Subsystem
title: Transport and Delivery (TD)
description: Beam transport, switching, and delivery to experimental stations including magnets, steering, and interlock systems.
tags: [transport, magnets, steering, interlock, beamline]
---

# Transport and Delivery (TD)

The Transport and Delivery (TD) subsystem accepts beam from the
[Primary Source](/subsystems/primary-source.md) and routes it to the
[Experimental Stations](/subsystems/experimental-stations.md).  It comprises
bending and focusing magnets, beam-steering correctors, and the injection
switchyard.

## Components

| Component | Role | Key channels |
|-----------|------|-------------|
| Bending Magnets | Steer beam along transfer line | `TD:BM:*:CURRENT` |
| Quadrupoles | Focus beam transversely | `TD:QF:*:CURRENT`, `TD:QD:*:CURRENT` |
| Horizontal Correctors | Fine steering in x | `TD:HCM:*:CURRENT` |
| Vertical Correctors | Fine steering in y | `TD:VCM:*:CURRENT` |
| Switchyard | Route beam to selected station | `TD:SW:*:Position` |
| BPMs | Measure beam position | `TD:BPM:*:X`, `TD:BPM:*:Y` |

## Beam Position Monitors

TD BPMs report beam centroid position in mm:

```
TD:BPM:H01:X   # Horizontal position at BPM H01 (mm)
TD:BPM:H01:Y   # Vertical position at BPM H01 (mm)
```

Nominal orbit tolerance: ±0.5 mm from golden orbit.

## Switchyard

The switchyard selects which experimental station receives beam:

```
TD:SW:01:Position      # Readback: current routing position (1–6)
TD:SW:01:Position:SP   # Setpoint: requested routing position
```

Position 0 means beam is blocked (dump mode).

## Interlock System

Hard interlocks cut magnet current when:
- BPM orbit exceeds ±5 mm from design
- Personnel Safety System asserts a beam-inhibit

See [Personnel Safety System](/subsystems/pss.md) for the interlock hierarchy.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — beam origin
- [Experimental Stations](/subsystems/experimental-stations.md) — beam destinations
- [Personnel Safety System](/subsystems/pss.md) — interlock authority
- [Orbit Correction Procedure](/procedures/orbit-correction.md) — restore golden orbit
- [Quadrupole Scan](/physics/quadrupole-scan.md) — beam optics measurement
