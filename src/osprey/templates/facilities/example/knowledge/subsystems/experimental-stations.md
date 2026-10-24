---
type: Subsystem
title: Experimental Stations (ES)
description: Six end-stations where users conduct experiments, each with independent motion control, detectors, and sample environments.
tags: [end-station, motion, detector, sample-environment, user]
---

# Experimental Stations (ES)

The Example Research Facility has six experimental stations (ES1–ES6).  Each
station operates independently and may be in beam, setup, or maintenance mode
while other stations continue running.

The example facility file holds no records for the experimental stations:
their status, motors, detectors and shutters are run from each station's own
controls, not through the control tools.  What the stations need from the
machine is stored beam, for example the current
[SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB).

## Per-Station Controls

| Control | Description |
|---------|-------------|
| Station status | Station readiness: `READY`, `SETUP`, `MAINT` |
| Motor position and setpoint | Motor axis position (mm or degrees) |
| Motor status | `MOVING`, `DONE`, `FAULT` |
| Detector counts | Detector count rate (counts/s) |
| Detector enable | Detector enable gate (off / on) |
| Photon shutter state and command | `OPEN` / `CLOSED` |

## Motion Control

Each station has a set of motorized axes:

- **X / Y / Z** — Sample stage translations (range ±50 mm, resolution 1 µm)
- **Theta / Chi / Phi** — Sample rotations (range ±180°, resolution 0.001°)
- **Det_Arm** — Detector arm angle (range 0–180°, resolution 0.01°)

Motors are driven via the EPICS motor record.

## Shutters

Each station has an independent photon shutter.  Opening the shutter without
a `READY` status triggers a Personnel Safety System fault.

See [Personnel Safety System](/subsystems/pss.md) for the interlock hierarchy.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — beam source for stations
- [Personnel Safety System](/subsystems/pss.md) — controls shutter interlocks
- [Sample Scan Procedure](/procedures/sample-scan.md) — automated sample scanning
- [Ion Pump](/devices/ion-pump.md) — vacuum devices at each station
