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

The example facility file holds no records for the experimental stations:
their status, motors, detectors and shutters are run from the station's own
controls, not through the control tools.

## Prerequisites

- Station status ready
- Sample mounted and aligned (manual step)
- Photon shutter closed
- Beam present: [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB) above 10 mA
- PSS permit granted for the station

## Steps

### 1. Configure the scan

1. Choose the scan axis (e.g., X for a horizontal sample scan).
2. Set the scan range and step size: the start position, the end position and
   the step (the scan script applies the steps).
3. Set the detector integration time and confirm the detector is enabled.

### 2. Move to start position

1. Move the motor to the start position.
2. Wait for the motor to report done.
3. Open the shutter.
4. Confirm the shutter reports open.

### 3. Step through scan

For each step from start to end:

1. Record the current motor position.
2. Record the detector counts.
3. Move to the next position.
4. Wait for the motor to report done.

### 4. Close shutter and park

1. Close the shutter.
2. Confirm the shutter reports closed.
3. Return the motor to its home position if required.

### 5. Save data

Record position vs. counts data in the facility logbook with scan parameters.

## Notes

- Never leave the shutter open with the detector disabled — stray photons
  can damage the detector.
- If the motor faults, close the shutter immediately and investigate before
  resuming.
- Typical scan speeds: 1 mm/s for fine scans, 10 mm/s for coarse surveys.

## Cross-links

- [Experimental Stations](/subsystems/experimental-stations.md) — station capabilities
- [Beam Position Monitor (BPM)](/devices/bpm.md) — verify beam is on target before scanning
