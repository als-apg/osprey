---
type: device
title: Beam Loss Monitors
description: Scintillator paddles that report where electrons are being lost.
tags: [diagnostics, losses, radiation]
---

# Beam Loss Monitors

Loss monitors are scintillator-and-photomultiplier paddles mounted outside the
vacuum chamber at the places electrons are most likely to strike it: downstream of
the injection septum, at each narrow-gap chamber, and at the scraper. Each paddle
reports a count rate proportional to the local shower intensity.

Operators use the monitors in two ways. During a fill they show which aperture is
limiting injection efficiency. During user operation a sustained rise flags a
degrading vacuum sector or a drifting orbit before anyone notices it elsewhere.

A single paddle crossing its high threshold raises an alarm. Several paddles
crossing together within the same turn is treated as an uncontrolled loss and
issues a beam abort request to the interlock chain, which drops the stored beam
deliberately rather than letting it paint the chamber wall.

Count rates are archived at 10 Hz and are logged next to the shift's
lifetime measurement in the end-of-shift report.
