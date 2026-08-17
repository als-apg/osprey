---
type: device
title: Stored Current Monitor
description: Non-destructive readout of how many electrons are circulating.
tags: [diagnostics, current]
---

# Stored Current Monitor

A parametric current transformer surrounds the vacuum chamber in a field-free
straight and reports the total circulating charge without intercepting the beam.
It is the facility's headline number: the value on every status display, the
trigger for a top-off shot, and the quantity users normalize their data against.

The transformer is zeroed against a calibration winding whenever the ring is
empty, which happens at the start of every maintenance period. Drift between
calibrations is under a tenth of a milliamp.

Because the reading is continuous and non-destructive, the decay of the stored
current between injections is the cheapest available diagnostic of how long
electrons survive in the ring. Operators fit an exponential to a few minutes of
decay and quote the resulting time constant in hours; a sudden shortening points
at a vacuum problem or a mis-set aperture long before anything else complains.
