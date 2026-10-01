---
type: Procedure
title: Orbit Correction
description: Procedure to restore the beam orbit to the golden orbit using BPM readings and corrector magnets.
tags: [orbit, correction, bpm, corrector, steering, procedure]
---

# Orbit Correction

This procedure restores the beam orbit to the stored golden orbit using
[BPM](/devices/bpm.md) position readbacks and corrector magnets in the
[Transport and Delivery](/subsystems/transport-delivery.md) system.

## When to Use

- After a magnet quench or power supply trip
- After an interlock reset that changed magnet currents
- When BPM orbit RMS exceeds 0.5 mm from golden orbit
- At the start of each shift as a routine check

## Prerequisites

- Beam present: `PS:DCCT:Current > 10 mA`
- All BPMs reporting `Quality = OK`
- Corrector magnets powered and responding

## Steps

### 1. Assess orbit distortion

1. Read all BPM positions:
   ```
   TD:BPM:H01:X, TD:BPM:H01:Y
   TD:BPM:H02:X, TD:BPM:H02:Y
   … (all H01–H12)
   ```
2. Compute RMS deviation from golden orbit values.
3. If RMS `< 0.2 mm` in both planes, orbit is acceptable — no action needed.

### 2. Identify dominant kicks

1. Locate the BPM with the largest deviation.
2. The kick is typically one or two BPMs upstream of the largest reading.

### 3. Apply corrector adjustments

1. Adjust horizontal correctors (`TD:HCM:*:CURRENT`) to minimise X RMS.
2. Adjust vertical correctors (`TD:VCM:*:CURRENT`) to minimise Y RMS.
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
