---
type: Procedure
title: Primary Source Startup
description: Step-by-step startup sequence for the Primary Source system from cold-start to beam-ready state.
tags: [startup, primary-source, rf, beam, procedure]
---

# Primary Source Startup

This procedure brings the [Primary Source (PS)](/subsystems/primary-source.md)
from a cold-start (all systems off) to beam-ready state.  Allow 45–60 minutes
for a complete startup.

## Prerequisites

- Vacuum in all sections: `< 1e-7 Torr` (check `PS:VAC:INJ:Pressure`)
- [Personnel Safety System](/subsystems/pss.md) permits granted for PS area
- Timing system in MULTI mode (`TIM:MODE:Current = MULTI`)

## Steps

### 1. Power supplies on (0–10 min)

1. Enable main power bus: confirm `PS:PWR:Main:Status = ON`.
2. Ramp magnet power supplies to standby current:
   - `PS:MAG:SOL01:CURRENT:SP → 50 A`
3. Verify all supply readbacks within 1% of setpoints.

### 2. Vacuum verification (10–15 min)

1. Check `PS:VAC:INJ:Pressure < 1e-7 Torr`.
2. If pressure is marginal, wait for ion pumps to recover; see
   [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md).

### 3. RF conditioning (15–40 min)

1. Enable RF low-level: `PS:RF:LLRF:Enable → 1`.
2. Ramp cavity power slowly:
   - `PS:RF:CAV01:Power:SP → 1 kW` (hold 5 min)
   - `PS:RF:CAV01:Power:SP → 5 kW` (hold 5 min)
   - `PS:RF:CAV01:Power:SP → 20 kW` (operating level)
3. Adjust phase: `PS:RF:CAV01:Phase:SP` until beam current optimised.

### 4. Beam injection (40–45 min)

1. Open injection gate: `PS:GATE:INJ:Cmd → OPEN`.
2. Confirm injection event enabled: `TIM:EVT:Injection:Ena = 1`.
3. Monitor `PS:DCCT:Current` — nominal injection current: 100–200 mA.
4. Close injection gate once target current is reached.

## Success Criteria

- `PS:DCCT:Current > 100 mA` (or facility-specified operating current)
- All power supply readbacks within 1% of setpoints
- `PS:VAC:INJ:Pressure < 1e-7 Torr`

## Failure Modes

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| RF cavity trips | Power ramp too fast | Reduce ramp rate, re-condition |
| No beam after injection | Gate valve closed | Check `PS:VAC:GV01:State` |
| Current decays rapidly | Vacuum leak | Check all pressure channels |

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — system description
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — if pressure is high
- [PSS Reset Procedure](/procedures/pss-reset.md) — if beam permit is denied
- [Beam Current Calibration](/physics/beam-current-calibration.md) — DCCT accuracy
