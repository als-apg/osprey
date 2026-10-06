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

## Per-Station Channels

Channels follow the pattern `ES{N}:<subsystem>:<parameter>`:

| Channel pattern | Description |
|----------------|-------------|
| `ES{N}:Status` | Station readiness: `READY`, `SETUP`, `MAINT` |
| `ES{N}:MOT:{Axis}:Position` | Motor axis position (mm or degrees) |
| `ES{N}:MOT:{Axis}:Position:SP` | Motor position setpoint |
| `ES{N}:MOT:{Axis}:Status` | Motor status: `MOVING`, `DONE`, `FAULT` |
| `ES{N}:DET:Counts` | Detector count rate (counts/s) |
| `ES{N}:DET:Enable` | Detector enable gate (`0`=off, `1`=on) |
| `ES{N}:SHUTTER:State` | Photon shutter state (`OPEN`/`CLOSED`) |
| `ES{N}:SHUTTER:Cmd` | Shutter command (`OPEN`/`CLOSE`) |

## Motion Control

Each station has a set of motorized axes:

- **X / Y / Z** — Sample stage translations (range ±50 mm, resolution 1 µm)
- **Theta / Chi / Phi** — Sample rotations (range ±180°, resolution 0.001°)
- **Det_Arm** — Detector arm angle (range 0–180°, resolution 0.01°)

Motors are driven via the EPICS motor record.  The OSPREY agent must use
`write_channel` for all motor moves — never direct EPICS PV access.

## Shutters

Each station has an independent photon shutter.  Opening the shutter without
a `READY` status triggers a Personnel Safety System fault.

```
ES3:SHUTTER:State     # Current shutter state
ES3:SHUTTER:Cmd       # Issue OPEN or CLOSE command
```

See [Personnel Safety System](/subsystems/pss.md) for the interlock hierarchy.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — beam source for stations
- [Personnel Safety System](/subsystems/pss.md) — controls shutter interlocks
- [Sample Scan Procedure](/procedures/sample-scan.md) — automated sample scanning
- [Ion Pump](/devices/ion-pump.md) — vacuum devices at each station
