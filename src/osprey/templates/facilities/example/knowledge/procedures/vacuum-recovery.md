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
- Beam is **off** (the global beam inhibit is asserted at the PSS panel).
- All gate valves to affected section are **closed**, for example
  [SR:VAC:VALVE:01:POSITION:CLOSED](facility:SR:VAC:VALVE:01:POSITION:CLOSED).

## Steps

### 1. Roughing pumpdown (0–4 h)

The roughing and turbomolecular pumps are carts connected by hand; the example
facility file holds no records for them.

1. Connect the roughing pump to the affected section.
2. Switch on the roughing pump at its cart.
3. Monitor pressure until `< 1e-3 Torr`.
4. Switch on the turbo pump (if available).
5. Pump down to `< 1e-5 Torr` before proceeding.

### 2. Ion pump conditioning (4–12 h)

1. Start ion pumps at reduced voltage (typically 3 kV), for example pump
   [SR/ION-PUMP01](facility:SR/ION-PUMP01):
   - Set [SR:VAC:ION-PUMP:01:VOLTAGE:SP](facility:SR:VAC:ION-PUMP:01:VOLTAGE:SP) to 3000 V
   - Confirm the pump is on: [SR:VAC:ION-PUMP:01:STATUS:ON](facility:SR:VAC:ION-PUMP:01:STATUS:ON)
2. Allow pumps to run until pump current
   ([SR:VAC:ION-PUMP:01:CURRENT:RB](facility:SR:VAC:ION-PUMP:01:CURRENT:RB))
   drops below 1 mA.
3. Ramp to full voltage (5 kV) once stable:
   [SR:VAC:ION-PUMP:01:VOLTAGE:SP](facility:SR:VAC:ION-PUMP:01:VOLTAGE:SP) to 5000 V.
4. Continue until pressure `< 1e-8 Torr`, read at a gauge such as
   [SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB).

### 3. Gate valve restoration

1. Open gate valves one section at a time, verifying pressure does not spike:
   - Open the valve from the vacuum control panel, for example
     [SR/VALVE01](facility:SR/VALVE01)
   - Confirm it is open:
     [SR:VAC:VALVE:01:POSITION:OPEN](facility:SR:VAC:VALVE:01:POSITION:OPEN)
2. Proceed from the most-pumped section toward the vented section.

### 4. Final check

1. Verify all section pressures `< 1e-8 Torr`.
2. Confirm all ion pump currents `< 0.01 mA`.
3. Clear the beam inhibit (requires shift supervisor).

## Cross-links

- [Vacuum System](/subsystems/vacuum.md) — system architecture
- [Ion Pump](/devices/ion-pump.md) — pump conditioning details
- [PSS Reset Procedure](/procedures/pss-reset.md) — re-establishing beam permits
