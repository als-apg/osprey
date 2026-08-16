---
type: device
title: Insertion Devices
description: Undulators and wigglers installed in the storage ring straight sections.
tags: [beamlines, photon, undulator]
---

# Insertion Devices

Most straight sections of the storage ring hold an insertion device: a periodic
magnetic structure that wiggles the electron beam and radiates a bright, narrow
photon spectrum into the downstream beamline. Undulators produce sharp harmonics;
wigglers produce a broad, hotter spectrum.

The photon spectrum an experiment sees is tuned by moving the two magnet jaws
closer together or further apart. Closing the jaws raises the on-axis field and
shifts the harmonics down in energy; opening them shifts the harmonics up. Some
devices instead shift one jaw longitudinally to change the polarization.

Each gap axis is a motion controller with its own PV name of the form
`<device>:GapMtr:...`, plus a limit-switch summary and a calibration table that
maps requested photon energy to a jaw separation.

Insertion-device motion is coordinated with [Orbit Feedback](/operations/orbit_feedback.md)
because a changing field perturbs the closed orbit for every other user.
