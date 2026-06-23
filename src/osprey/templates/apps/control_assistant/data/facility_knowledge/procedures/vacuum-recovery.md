---
type: Procedure
title: Vacuum Recovery
description: Procedure to recover from a vacuum event (vent or pressure excursion) back to UHV operating conditions.
tags: [vacuum, uhv, recovery, pumpdown, ion-pump, procedure]
---

# Vacuum Recovery

This procedure restores the [Vacuum System](/subsystems/vacuum.md) to operating
conditions after a vacuum event — either an intentional vent or an uncontrolled
pressure excursion.

## Types of Vacuum Events

| Event | Cause | Recovery time |
|-------|-------|---------------|
| Controlled vent | Planned maintenance | 24–48 h |
| Pressure excursion | Leak or pump fault | 2–8 h (if no structural issue) |
| Catastrophic vent | Gate valve failure | 48+ h |

## Prerequisites

- Identify and fix the root cause of the vacuum event before pumping down.
- Beam is **off** (`PSS:Global:BeamInhibit = 1`).
- All gate valves to affected section are **closed**.

## Steps

### 1. Roughing pumpdown (0–4 h)

1. Connect the roughing pump to the affected section.
2. Enable roughing pump: `PS:VAC:RP01:Enable → 1`.
3. Monitor pressure until `< 1e-3 Torr`.
4. Enable turbo pump (if available): `PS:VAC:TMP01:Enable → 1`.
5. Pump down to `< 1e-5 Torr` before proceeding.

### 2. Ion pump conditioning (4–12 h)

1. Start ion pumps at reduced voltage (typically 3 kV):
   - Set `{SYS}:VAC:IP01:Voltage → 3000` V
   - Enable: `{SYS}:VAC:IP01:Enable → 1`
2. Allow pumps to run until pump current drops below 1 mA.
3. Ramp to full voltage (5 kV) once stable: `{SYS}:VAC:IP01:Voltage → 5000`
4. Continue until pressure `< 1e-8 Torr`.

### 3. Gate valve restoration

1. Open gate valves one section at a time, verifying pressure does not spike:
   - `{SYS}:VAC:GV01:Cmd → OPEN`
   - Confirm `{SYS}:VAC:GV01:State = OPEN`
2. Proceed from the most-pumped section toward the vented section.

### 4. Final check

1. Verify all section pressures `< 1e-8 Torr`.
2. Confirm all ion pump currents `< 0.01 mA`.
3. Clear beam inhibit: `PSS:Global:BeamInhibit → 0` (requires shift supervisor).

## Cross-links

- [Vacuum System](/subsystems/vacuum.md) — system architecture
- [Ion Pump](/devices/ion-pump.md) — pump conditioning details
- [PSS Reset Procedure](/procedures/pss-reset.md) — re-establishing beam permits
