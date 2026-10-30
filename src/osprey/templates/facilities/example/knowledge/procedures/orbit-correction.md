---
type: Procedure
title: Orbit Correction
description: Procedure to restore the beam orbit to the golden orbit using BPM readings and corrector magnets.
tags: [orbit, correction, bpm, corrector, steering, procedure]
---

# Orbit Correction

This procedure restores the beam orbit to the stored golden orbit using
[BPM](/devices/bpm.md) position readbacks and corrector magnets.  The steps
are written for SR; the transfer lines of
[Transport and Delivery](/subsystems/transport-delivery.md) are corrected the
same way with their own monitors and correctors.

## When to Use

- After a magnet quench or power supply trip
- After an interlock reset that changed magnet currents
- When BPM orbit RMS exceeds 0.5 mm from golden orbit
- At the start of each shift as a routine check

## Prerequisites

- Beam present: [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB) above 10 mA
- Every BPM used is valid and connected, for example
  [SR:DIAG:BPM:01:STATUS:VALID](facility:SR:DIAG:BPM:01:STATUS:VALID) and
  [SR:DIAG:BPM:01:STATUS:CONNECTED](facility:SR:DIAG:BPM:01:STATUS:CONNECTED)
- Corrector magnets powered and responding, for example
  [SR:MAG:HCM:01:STATUS:ON](facility:SR:MAG:HCM:01:STATUS:ON)

## Steps

### 1. Assess orbit distortion

1. Read every BPM position, for example
   [SR:DIAG:BPM:01:POSITION:X](facility:SR:DIAG:BPM:01:POSITION:X) and
   [SR:DIAG:BPM:01:POSITION:Y](facility:SR:DIAG:BPM:01:POSITION:Y); ask the
   channel finder for the full set of monitors.
2. Compute RMS deviation from the golden orbit values, for example
   [SR:DIAG:BPM:01:GOLDEN:X](facility:SR:DIAG:BPM:01:GOLDEN:X).
3. If RMS `< 0.2 mm` in both planes, orbit is acceptable — no action needed.

### 2. Identify dominant kicks

1. Locate the BPM with the largest deviation.
2. The kick is typically one or two BPMs upstream of the largest reading.

### 3. Apply corrector adjustments

1. Adjust horizontal correctors to minimise X RMS, for example
   [SR:MAG:HCM:01:CURRENT:SP](facility:SR:MAG:HCM:01:CURRENT:SP), checking
   its readback [SR:MAG:HCM:01:CURRENT:RB](facility:SR:MAG:HCM:01:CURRENT:RB).
2. Adjust vertical correctors to minimise Y RMS, for example
   [SR:MAG:VCM:01:CURRENT:SP](facility:SR:MAG:VCM:01:CURRENT:SP).
3. Apply corrections in small steps (≤ 10% of total correction per iteration)
   to avoid over-shooting.

### 4. Verify

1. Recheck all BPM positions.
2. Confirm orbit RMS `< 0.3 mm` in both planes.
3. Log correction in the facility logbook.

## Success Criteria

- Horizontal orbit RMS `< 0.3 mm`
- Vertical orbit RMS `< 0.3 mm`

## Notes

- Large distortions (> 5 mm) may indicate a hardware fault rather than
  a routine orbit drift.  Escalate to the on-call accelerator physicist.
- Do not adjust bending magnets during orbit correction — they are tuned
  during machine studies only.

## Cross-links

- [Beam Position Monitor (BPM)](/devices/bpm.md) — measurement device
- [Transport and Delivery](/subsystems/transport-delivery.md) — corrector locations
- [Quadrupole Scan](/physics/quadrupole-scan.md) — optics measurement that may follow
