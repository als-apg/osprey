---
type: Subsystem
title: Timing System (TIM)
description: Facility-wide synchronization and triggering system providing event-based timing to all subsystems.
tags: [timing, synchronization, trigger, event, clock]
---

# Timing System (TIM)

The Timing System (TIM) provides facility-wide synchronization and event-based
triggering.  All subsystems lock their local clocks to the TIM master oscillator.

## Master Clock

The master clock runs at 500 MHz (the RF frequency).  Derived event clocks
include:

| Event | Frequency | Description |
|-------|-----------|-------------|
| `TIM:EVT:Injection` | 1 Hz | Injection trigger for PS |
| `TIM:EVT:DataAcq` | 10 Hz | Detector data acquisition trigger |
| `TIM:EVT:Sync` | 1 kHz | General subsystem synchronization |

## Key Channels

```
TIM:CLOCK:Freq        # Master clock frequency (MHz, readback)
TIM:EVT:Injection:Ena # Enable injection event (1=enabled, 0=inhibited)
TIM:EVT:DataAcq:Delay # Data acquisition delay relative to injection (µs)
TIM:MODE:Current      # Operating mode: SINGLE, MULTI, STUDY
```

## Operating Modes

- **MULTI** — Normal multi-bunch operation (default).
- **SINGLE** — Single-bunch mode for pump-probe experiments.
- **STUDY** — Machine studies; non-standard timing patterns may be active.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — RF frequency reference
- [Experimental Stations](/subsystems/experimental-stations.md) — detector trigger recipients
