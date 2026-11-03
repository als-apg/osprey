---
type: Device
title: Beam Position Monitor (BPM)
description: Electromagnetic pickup measuring beam centroid position in horizontal and vertical planes with sub-millimetre resolution.
tags: [diagnostics, beam-position, bpm, orbit, pickup]
---

# Beam Position Monitor (BPM)

Beam Position Monitors (BPMs) measure the transverse position of the beam
centroid using electromagnetic button pickups.  BPMs sit along every section of
the machine, from the gun-to-linac line to SR, and are the primary diagnostic
for orbit control.

## Channels

Each monitor is a device record of the facility file, and each of its readings
is a channel record.  Monitor [SR/BPM01](facility:SR/BPM01), for example,
carries:

- [SR:DIAG:BPM:01:POSITION:X](facility:SR:DIAG:BPM:01:POSITION:X) — horizontal position
- [SR:DIAG:BPM:01:POSITION:Y](facility:SR:DIAG:BPM:01:POSITION:Y) — vertical position
- [SR:DIAG:BPM:01:STATUS:VALID](facility:SR:DIAG:BPM:01:STATUS:VALID) — the reading is valid
- [SR:DIAG:BPM:01:STATUS:CONNECTED](facility:SR:DIAG:BPM:01:STATUS:CONNECTED) — the monitor electronics are responding
- [SR:DIAG:BPM:01:GOLDEN:X](facility:SR:DIAG:BPM:01:GOLDEN:X) — the golden-orbit position
- [SR:DIAG:BPM:01:OFFSET:X](facility:SR:DIAG:BPM:01:OFFSET:X) — the position offset from the golden orbit

Transfer-line monitors carry fewer channels: the first booster-to-storage
monitor reads [BTS:DIAG:BPM:01:POSITION:X](facility:BTS:DIAG:BPM:01:POSITION:X)
and a sum signal, [BTS:DIAG:BPM:01:SUM:RB](facility:BTS:DIAG:BPM:01:SUM:RB).
Ask the channel finder for the monitors of a section rather than composing
addresses by hand.

## Resolution and Range

| Parameter | Value |
|-----------|-------|
| Position range | ±10 mm |
| Nominal resolution | 10 µm rms |
| Update rate | 10 Hz (fast acquisition), 1 Hz (slow orbit) |

## Orbit Correction

The golden orbit is stored in the control system, one golden-orbit channel per
monitor and plane.  Deviations from golden orbit trigger corrector magnet
adjustments via the orbit feedback loop.
See [Orbit Correction Procedure](/procedures/orbit-correction.md).

## Status Flags

Two status channels say whether a monitor's position reading can be trusted:

- **Valid** ([SR:DIAG:BPM:01:STATUS:VALID](facility:SR:DIAG:BPM:01:STATUS:VALID))
  — true when the monitor reports a valid position reading.
- **Connected** ([SR:DIAG:BPM:01:STATUS:CONNECTED](facility:SR:DIAG:BPM:01:STATUS:CONNECTED))
  — true when the monitor electronics are responding.

Both must be true before a position readback is trusted for diagnostics or
feedback; a monitor with either flag false must not be used for orbit
correction.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — transfer-line BPM locations
- [Experimental Stations](/subsystems/experimental-stations.md) — where the beam is delivered
- [Orbit Correction Procedure](/procedures/orbit-correction.md) — using BPMs to correct orbit
- [Quadrupole Scan](/physics/quadrupole-scan.md) — BPM data used in optics measurement
