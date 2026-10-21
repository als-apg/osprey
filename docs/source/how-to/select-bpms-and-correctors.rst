.. _select-bpms-and-correctors:

============================
Select BPMs and Correctors
============================

The trajectory panel's **Setup BPMs** and **Setup CMs** tabs choose which BPMs
and correctors take part in a shot and a correction. Every device starts in
use; you take devices out and put them back. The devices, their labels and
their places come from your facility description, so the panel shows them the
way your facility names them.

The map
=======

Each tab shows a map of the machine: one row per place, rows in ring order,
one square per device. The header counts the devices in use; Setup CMs shows a
horizontal and a vertical strip per row and counts each plane. Hover a square
to see the device's label, place and position.

- **Click a square** to take that device out, or put it back.
- **Click a place name** to take every device in that place out, or put them
  all back.

Find
====

Type a device label, name or place into **Find** and press Enter; it
suggests as you type. A device toggles that device, a place toggles every
device in it. Separate several with commas. A word that names no device and
no place is refused by name, and so are position lists such as ``1 2 8`` or
``3:5``.

Start from
==========

When your facility defines several sets of BPMs or correctors, **Start from**
picks one: every device outside the set is taken out. Adjust from there with
the map or Find.

Commit
======

Nothing takes effect until you press **Setup BPMs** or **Setup CMs**. Until
then each change shows as pending, and **Revert** drops them. A selection that
leaves a plane with no BPM or no corrector is refused.

Once committed, every shot leaves the excluded BPMs out, and the correction
step and Apply use the same BPMs and correctors. The plots show excluded BPMs
as gaps. You can also click a BPM on the Trajectories or Difference plot, or a
corrector bar in the CM step plot, to mark it pending.

The selection is kept by device, so it survives a restart. A rebuild that
drops a device drops its exclusion, with a notice.

See :doc:`thread-the-first-turn` for the correction loop and
:ref:`reference-trajectory-service` for the device catalog the panel reads.
