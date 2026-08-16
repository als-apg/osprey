---
type: subsystem
title: Low-Level RF Control
description: Amplitude, phase, and plunger regulation ahead of the power amplifier.
tags: [rf, controls, feedback]
---

# Low-Level RF Control

The low-level chain generates the drive signal, compares the field probe against
the requested amplitude and phase, and closes three regulation loops before the
signal ever reaches the klystron.

The amplitude loop holds the accelerating voltage flat against beam loading. The
phase loop holds the synchronous phase, which is what actually determines where in
the bucket the bunch sits. The third loop is mechanical and slow: it drives the
plunger motor to keep the resonator on resonance so the amplitude loop does not
have to spend forward power fighting a detuned structure.

All three run on a single digital board with a common clock, so the loops cannot
drift relative to one another. Loop gains and setpoints are exposed as channels
ending in `:SP`, and each loop publishes an in-regulation flag that the machine
protection chain consumes.

When a loop is opened for diagnostics, the board falls back to holding the last
commanded drive rather than to zero, which avoids dropping the stored beam during
routine measurements.
