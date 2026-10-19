---
type: Subsystem
title: Transport and Delivery (TD)
description: Beam transport, switching, and delivery to experimental stations including magnets, steering, and interlock systems.
tags: [transport, magnets, steering, interlock, beamline]
---

# Transport and Delivery (TD)

The Transport and Delivery (TD) subsystem accepts beam from the
[Primary Source](/subsystems/primary-source.md) and delivers it to SR, whose
light serves the [Experimental Stations](/subsystems/experimental-stations.md).
It comprises two transfer lines: the linac-to-booster line (LTB) and the
booster-to-storage line (BTS), with the booster (BR) between them, and the
kickers and septa that inject and extract the beam.

## Components

| Component | Role | Example records |
|-----------|------|-----------------|
| Bending magnets | Steer beam along the transfer line | [BTS:MAG:DIPOLE:01:CURRENT:SP](facility:BTS:MAG:DIPOLE:01:CURRENT:SP), [LTB:MAG:DIPOLE:01:CURRENT:SP](facility:LTB:MAG:DIPOLE:01:CURRENT:SP) |
| Quadrupoles | Focus beam transversely | [BTS:MAG:QF:01:CURRENT:SP](facility:BTS:MAG:QF:01:CURRENT:SP), [LTB:MAG:QF:01:CURRENT:SP](facility:LTB:MAG:QF:01:CURRENT:SP) |
| Horizontal correctors | Fine steering in x | [BTS:MAG:HCM:01:CURRENT:SP](facility:BTS:MAG:HCM:01:CURRENT:SP) |
| Vertical correctors | Fine steering in y | [BTS:MAG:VCM:01:CURRENT:SP](facility:BTS:MAG:VCM:01:CURRENT:SP) |
| Kickers and septa | Inject and extract the beam | [BR/KICKER01](facility:BR/KICKER01), [BR/SEPTUM01](facility:BR/SEPTUM01), [SR/SEPTUM01](facility:SR/SEPTUM01) |
| BPMs | Measure beam position | [BTS:DIAG:BPM:01:POSITION:X](facility:BTS:DIAG:BPM:01:POSITION:X), [LTB:DIAG:BPM:01:POSITION:X](facility:LTB:DIAG:BPM:01:POSITION:X) |
| Screens | Show the beam spot | [BTS:DIAG:TV:01:PADDLE:SP](facility:BTS:DIAG:TV:01:PADDLE:SP) |
| Charge monitors | Measure the transmitted charge | [LTB:DIAG:ICT:01:CHARGE:RB](facility:LTB:DIAG:ICT:01:CHARGE:RB), [BTS:DIAG:ICT:01:CHARGE:RB](facility:BTS:DIAG:ICT:01:CHARGE:RB) |

Ask the channel finder for the other devices of a line rather than composing
addresses by hand.

## Beam Position Monitors

Transfer-line BPMs report beam centroid position, for example
[BTS:DIAG:BPM:01:POSITION:X](facility:BTS:DIAG:BPM:01:POSITION:X) and
[BTS:DIAG:BPM:01:POSITION:Y](facility:BTS:DIAG:BPM:01:POSITION:Y), with a sum
signal ([BTS:DIAG:BPM:01:SUM:RB](facility:BTS:DIAG:BPM:01:SUM:RB)) and a
valid flag ([BTS:DIAG:BPM:01:STATUS:VALID](facility:BTS:DIAG:BPM:01:STATUS:VALID)).

Nominal orbit tolerance: ±0.5 mm from golden orbit.

## Screens

Each line has screens that are moved into the beam to show the spot; a screen
in the beam blocks it.  The paddle summary
[BTS:DIAG:PADDLES:01:COUNT:INSERTED](facility:BTS:DIAG:PADDLES:01:COUNT:INSERTED)
counts the booster-to-storage screens in the beam: it must read zero before
beam can reach SR.

## Interlock System

Hard interlocks cut magnet current when:
- BPM orbit exceeds ±5 mm from design
- Personnel Safety System asserts a beam-inhibit

See [Personnel Safety System](/subsystems/pss.md) for the interlock hierarchy.

## Cross-links

- [Primary Source](/subsystems/primary-source.md) — beam origin
- [Experimental Stations](/subsystems/experimental-stations.md) — beam destinations
- [Personnel Safety System](/subsystems/pss.md) — interlock authority
- [Orbit Correction Procedure](/procedures/orbit-correction.md) — restore golden orbit
- [Quadrupole Scan](/physics/quadrupole-scan.md) — beam optics measurement
