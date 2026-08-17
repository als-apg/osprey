---
type: subsystem
title: Machine Protection System
description: The interlock chain that protects equipment from the stored beam.
tags: [safety, interlock, equipment]
---

# Machine Protection System

The protection chain exists to keep hardware intact. It is distinct from the
personnel chain: this one may drop the stored beam to save a chamber, but it makes
no radiation-safety guarantees and is not credited in the safety analysis.

Inputs come from every subsystem that can be damaged. Each vacuum sector supplies
a pressure readback and a gauge-fault flag; the magnet supplies supply
over-temperature and ground-fault flags; the radio-frequency plant supplies
reflected power and arc-detector status; the loss paddles supply their high
threshold.

Any input outside its window opens the chain, which inhibits injection and fires
the dump kicker within one turn. The chain latches, so the fault survives for
diagnosis and an operator must acknowledge it before beam can be re-established.

A first-fault recorder timestamps every input at microsecond resolution, which is
usually the only way to tell which subsystem actually failed and which merely
reacted to losing the beam.
