---
type: Procedure
title: Sample Scan
description: Procedure to perform an automated position scan of a sample at an experimental station using motor control and detector readout.
tags: [sample, scan, motor, detector, experimental-station, procedure]
---

# Sample Scan

This procedure performs an automated position scan of a sample at an
[Experimental Station](/subsystems/experimental-stations.md), stepping a motor
axis and recording detector counts at each position.

## Prerequisites

- Station status `READY`: `ES{N}:Status = READY`
- Sample mounted and aligned (manual step)
- Photon shutter closed: `ES{N}:SHUTTER:State = CLOSED`
- Beam present: `PS:DCCT:Current > 10 mA`
- PSS permit granted: `PSS:ES{N}:BeamPermit = GRANTED`

## Steps

### 1. Configure the scan

1. Choose the scan axis (e.g., X for a horizontal sample scan).
2. Set the scan range and step size:
   - Start position: `ES{N}:MOT:X:Position:SP → {start}`
   - (Steps will be applied by the scan script)
3. Set detector integration time and confirm detector enabled:
   `ES{N}:DET:Enable → 1`

### 2. Move to start position

1. Move motor to start: `ES{N}:MOT:X:Position:SP → {start}`
2. Wait for: `ES{N}:MOT:X:Status = DONE`
3. Open shutter: `ES{N}:SHUTTER:Cmd → OPEN`
4. Confirm: `ES{N}:SHUTTER:State = OPEN`

### 3. Step through scan

For each step from start to end:

1. Record current position: `ES{N}:MOT:X:Position`
2. Record detector counts: `ES{N}:DET:Counts`
3. Move to next position: `ES{N}:MOT:X:Position:SP → {next_pos}`
4. Wait for: `ES{N}:MOT:X:Status = DONE`

### 4. Close shutter and park

1. Close shutter: `ES{N}:SHUTTER:Cmd → CLOSE`
2. Confirm: `ES{N}:SHUTTER:State = CLOSED`
3. Return motor to home position if required.

### 5. Save data

Record position vs. counts data in the facility logbook with scan parameters.

## Notes

- Never leave the shutter open with the detector disabled — stray photons
  can damage the detector.
- If the motor faults (`Status = FAULT`), close the shutter immediately and
  investigate before resuming.
- Typical scan speeds: 1 mm/s for fine scans, 10 mm/s for coarse surveys.

## Cross-links

- [Experimental Stations](/subsystems/experimental-stations.md) — station capabilities
- [Beam Position Monitor (BPM)](/devices/bpm.md) — verify beam is on target before scanning
