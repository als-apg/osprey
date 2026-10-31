---
type: Subsystem
title: Timing System (TIM)
description: Facility-wide synchronization and triggering system providing event-based timing to all subsystems.
tags: [timing, synchronization, trigger, event, clock]
---

# Timing System (TIM)

The Timing System (TIM) provides facility-wide synchronization and event-based
triggering.  All subsystems lock their local clocks to the master oscillator.

## Master Clock

The master clock runs at 500 MHz, the RF frequency of SR
([SR:RF:CAVITY:01:FREQUENCY:RB](facility:SR:RF:CAVITY:01:FREQUENCY:RB)).  The
linac's master RF, [LN/MASTER01](facility:LN/MASTER01), is phased to it:

- [LN:RF:MASTER:01:PHASE:SP](facility:LN:RF:MASTER:01:PHASE:SP) — master RF phase setpoint
- [LN:RF:MASTER:01:PHASE:RB](facility:LN:RF:MASTER:01:PHASE:RB) — master RF phase readback
- [LN:RF:MASTER:01:STATUS:ON](facility:LN:RF:MASTER:01:STATUS:ON) — the master RF drive is on

## Events

Derived event clocks include:

| Event | Frequency | Description |
|-------|-----------|-------------|
| Injection | 1 Hz | Injection trigger for the injector |
| Data acquisition | 10 Hz | Detector data acquisition trigger |
| Sync | 1 kHz | General subsystem synchronization |

The example facility file holds no records for the event system: event
enables, delays and the operating mode are set at the timing console.  The
injection kicker it fires reports its state at
[BR:MAG:KICKER:01:STATUS:ON](facility:BR:MAG:KICKER:01:STATUS:ON).

## Operating Modes

- **MULTI** — Normal multi-bunch operation (default).
- **SINGLE** — Single-bunch mode for pump-probe experiments.
- **STUDY** — Machine studies; non-standard timing patterns may be active.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — RF frequency reference
- [Experimental Stations](/subsystems/experimental-stations.md) — detector trigger recipients
