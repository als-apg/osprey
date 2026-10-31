---
type: Procedure
title: Primary Source Startup
description: Step-by-step startup sequence for the Primary Source system from cold-start to beam-ready state.
tags: [startup, primary-source, rf, beam, procedure]
---

# Primary Source Startup

This procedure brings the [Primary Source (PS)](/subsystems/primary-source.md)
— the electron gun, the gun-to-linac line and the linac — from a cold start
(all systems off) to beam-ready state, and injects beam through the booster
into SR.  Allow 45–60 minutes for a complete startup.

## Prerequisites

- Vacuum in all sections: `< 1e-7 Torr`.  The example facility file holds no
  vacuum records for the injector; check the SR gauges, for example
  [SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB).
- [Personnel Safety System](/subsystems/pss.md) permits granted for the
  injector area
- Linac cooling water at temperature, with no water interlock:
  [LN:RF:WATER:01:TEMPERATURE:RB](facility:LN:RF:WATER:01:TEMPERATURE:RB) and
  [LN:RF:WATER:01:STATUS:INTERLOCK](facility:LN:RF:WATER:01:STATUS:INTERLOCK)

## Steps

### 1. Power supplies on (0–10 min)

1. Switch on the gun heater and wait for it to settle:
   [GTL:GUN:EGUN:01:HEATER_ON:SP](facility:GTL:GUN:EGUN:01:HEATER_ON:SP), then
   set [GTL:GUN:EGUN:01:HEATER:SP](facility:GTL:GUN:EGUN:01:HEATER:SP).
2. Ramp the gun-to-linac and linac solenoids to standby current, for example
   [GTL:MAG:SOL:01:CURRENT:SP](facility:GTL:MAG:SOL:01:CURRENT:SP) and
   [LN:MAG:SOL:01:CURRENT:SP](facility:LN:MAG:SOL:01:CURRENT:SP).
3. Verify all supply readbacks within 1% of setpoints, for example
   [GTL:MAG:SOL:01:CURRENT:RB](facility:GTL:MAG:SOL:01:CURRENT:RB).

### 2. Vacuum verification (10–15 min)

1. Check pressures `< 1e-7 Torr`.
2. If pressure is marginal, wait for ion pumps to recover; see
   [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md).

### 3. RF conditioning (15–40 min)

1. Switch on the master RF: confirm
   [LN:RF:MASTER:01:STATUS:ON](facility:LN:RF:MASTER:01:STATUS:ON).
2. Switch on the sub-harmonic bunchers and raise their voltage slowly, for
   example [GTL:RF:SHB:01:HV_ON:SP](facility:GTL:RF:SHB:01:HV_ON:SP), then
   [GTL:RF:SHB:01:VOLTAGE:SP](facility:GTL:RF:SHB:01:VOLTAGE:SP) in steps,
   holding 5 min at each.
3. Switch on the gun high voltage,
   [GTL:GUN:EGUN:01:HV_ON:SP](facility:GTL:GUN:EGUN:01:HV_ON:SP), and raise
   [GTL:GUN:EGUN:01:VOLTAGE:SP](facility:GTL:GUN:EGUN:01:VOLTAGE:SP) to the
   operating level.
4. Adjust the accelerating-section phase,
   [LN:RF:ACC:02:PHASE:SP](facility:LN:RF:ACC:02:PHASE:SP), until the charge
   through the linac
   ([LN:DIAG:ICT:01:CHARGE:RB](facility:LN:DIAG:ICT:01:CHARGE:RB)) is
   optimised.

### 4. Beam injection (40–45 min)

1. Confirm the booster injection kicker is on:
   [BR:MAG:KICKER:01:STATUS:ON](facility:BR:MAG:KICKER:01:STATUS:ON).
2. Confirm the SR injection septum is on:
   [SR:MAG:SEPTUM:01:STATUS:ON](facility:SR:MAG:SEPTUM:01:STATUS:ON).
3. Monitor the stored current,
   [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB) —
   nominal injection current: 100–200 mA.
4. Stop injecting once target current is reached.

## Success Criteria

- Stored current above 100 mA (or facility-specified operating current)
- All power supply readbacks within 1% of setpoints
- Pressures `< 1e-7 Torr`

## Failure Modes

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| RF trips | Power ramp too fast | Reduce ramp rate, re-condition |
| No beam after injection | Gate valve closed | Check the SR valves, for example [SR:VAC:VALVE:01:POSITION:OPEN](facility:SR:VAC:VALVE:01:POSITION:OPEN) |
| Current decays rapidly | Vacuum leak | Check all pressure channels |

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — system description
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — if pressure is high
- [PSS Reset Procedure](/procedures/pss-reset.md) — if beam permit is denied
- [Beam Current Calibration](/physics/beam-current-calibration.md) — DCCT accuracy
