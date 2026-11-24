---
# Bundled with OSPREY: the example facility. Delete this line to make the file yours.
type: Subsystem
title: Experimental Stations (ES)
description: Six end-stations where users conduct experiments, each with independent motion control, detectors, and sample environments.
tags: [end-station, motion, detector, sample-environment, user]
---

# Experimental Stations (ES)

The Example Research Facility has six experimental stations (ES1–ES6).  Each
station operates independently and may be in beam, setup, or maintenance mode
while other stations continue running.

The example facility file holds one sample stage and one detector at ES1.
The stage sets its target on
[ES1:MOT:STAGE:01:POSITION:SP](facility:ES1:MOT:STAGE:01:POSITION:SP), reads
its position on
[ES1:MOT:STAGE:01:POSITION:RB](facility:ES1:MOT:STAGE:01:POSITION:RB), starts
a move with [ES1:MOT:STAGE:01:MOVE:SP](facility:ES1:MOT:STAGE:01:MOVE:SP),
reports it on [ES1:MOT:STAGE:01:STATUS:MOVE](facility:ES1:MOT:STAGE:01:STATUS:MOVE)
and stops with [ES1:MOT:STAGE:01:STOP:SP](facility:ES1:MOT:STAGE:01:STOP:SP).
The detector starts an acquisition with
[ES1:DET:MCA:01:ACQUIRE:SP](facility:ES1:DET:MCA:01:ACQUIRE:SP), reports it on
[ES1:DET:MCA:01:STATUS:ACQUIRE](facility:ES1:DET:MCA:01:STATUS:ACQUIRE), and
reads its counts on [ES1:DET:MCA:01:COUNTS:RB](facility:ES1:DET:MCA:01:COUNTS:RB),
its spectrum on
[ES1:DET:MCA:01:SPECTRUM:WAVEFORM](facility:ES1:DET:MCA:01:SPECTRUM:WAVEFORM)
and its temperature on
[ES1:DET:MCA:01:TEMPERATURE:RB](facility:ES1:DET:MCA:01:TEMPERATURE:RB). In
the simulation the spectrum is a fixed array and an acquisition changes no
reading.  Every other station's status, motors, detectors and shutters are run
from the station's own controls, not through the control tools.  What the
stations need from the machine is stored beam, for example the current
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
