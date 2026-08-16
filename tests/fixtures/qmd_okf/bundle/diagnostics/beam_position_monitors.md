---
type: device
title: Beam Position Monitors
description: Button pickups that report the transverse position of the stored beam.
tags: [diagnostics, orbit]
---

# Beam Position Monitors

A position monitor is a set of four button electrodes flush with the chamber wall.
The beam's image charge induces a signal on each button; the normalized difference
between opposing buttons gives the horizontal and vertical displacement of the
beam centroid from the electrical centre of the assembly.

The ring carries one assembly per girder, giving enough sampling to reconstruct
the closed orbit turn by turn. Readings are published at 10 Hz for archiving and
at several kilohertz on a dedicated link for the fast loop.

Position monitors are the input to orbit correction: the slow loop compares the
measured orbit to the golden orbit and drives the steering magnets until the
residual is minimized. A monitor whose electronics have drifted will therefore
pull the real orbit away from centre, which is why the offsets are re-measured by
beam-based alignment during every maintenance period.
