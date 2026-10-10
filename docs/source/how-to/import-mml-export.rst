.. _how-to-import-mml-export:

==============================
Import a Middle Layer export
==============================

``osprey facility import mml`` reads a MATLAB Middle Layer (MML) export and
writes it as sources of your facility description, under
``data/facility/imported/mml/``. ``osprey build`` merges those sources with
everything else under ``data/facility/``; the import itself builds nothing.

Export from MATLAB
==================

The exporter ships with OSPREY. Print it, copy it to the MATLAB host and run it
once per sub-machine:

.. code-block:: bash

   osprey facility import mml --print-exporter > mml_export.m

Each run writes six files that share a stem. The import is given the
``<stem>.ao.json`` and reads the other five from beside it.
``osprey facility import mml --help`` lists the six files and what a run needs
of the Middle Layer.

What the files contain
----------------------

The lattice is saved first, before the export samples anything. Reading a
nominal reaches its answer through the lattice's own closed orbit and turns
radiation off or the cavity on to get one, leaving the lattice in whichever
state it needed, so a lattice saved afterwards would no longer be the one the
rest of the export describes. A model measurement of the response matrix steps
its correctors on a copy, so what it leaves behind is what its own readings
needed rather than the columns it built.

Each file opens with an ``_export`` block recording the exporter version, the
MATLAB version, the machine, the sub-machine and when the export ran. After it
come the AO families (or the AD fields) as the Middle Layer holds them, with a
few values rewritten so JSON can carry them faithfully:

- function handles become ``{"$fn": "<name>", "file": "<path>"}``;
- padded char matrices become one trimmed string per row;
- ``Inf``, ``-Inf`` and ``NaN`` are written as the strings ``"Inf"``,
  ``"-Inf"`` and ``"NaN"``, never as ``null``, which would lose the value;
- logicals become ``0``/``1``, and the ``Handles`` field (plot handles) is left
  out.

Everything else is written as is. The script only reads from the Middle Layer;
it sets nothing on the machine.

``va.json`` is the part the Middle Layer cannot state as stored data. It opens
with a fingerprint of the lattice the file belongs to: element count, a digest
of the family names in lattice order, the model energy, and where any parameter
element sits. The importer recomputes it from the lattice file: a lattice that
disagrees on the elements is refused, and an energy that differs alone is
reported. Then one block per family: the Setpoint's and the Monitor's own
hardware-to-physics calibrations, and the Monitor's physics-to-hardware
inverse, each sampled through the facility's own conversion functions rather
than read out of their stored parameters; whether the Setpoint's conversion
carries the beam's rigidity; the per-device nominal the Middle Layer's model
read gives; what the family's own readings are corrected by, where it states
it; and, for a family the beam energy is read from, the energy table its
current maps to. The exporter's own header lists every key.

The corrections (a gain, an offset, a roll and a crunch, one number per device)
belong to whichever family states them, not to the beam monitors alone. A
facility that calibrates its magnets and correctors from an orbit measurement
keeps their gains and rolls under the same four names, so an export of a whole
lattice usually carries a block of them on a dozen or more families. They are
four numbers per device and nothing is resampled for them, so what that costs
the file is negligible. The offset is in the family's own **hardware** units
(what its readback answers in, millimetres on most beam monitors) and not the
physics units of the conversion beside it. A number the facility states nowhere
is left out rather than defaulted, and one written in a shape that is not one
per device is refused by name and costs that family only that number.

The numbers are looked up in the three places the Middle Layer looks: the
family's ``Monitor`` field, the family itself, then the facility's physics data
file, which a facility fills from an orbit fit and copies into the Accelerator
Objects at operating-mode set-up. Run the export from a session where that
set-up has run. An answer that came from the physics data can hold ``"NaN"``
for single devices, because that file is stored against its own device list
and a device it does not cover comes back as no number, named on the console as
it happens. Such an entry means the facility states nothing for that one
device.

A family whose conversions refuse a sample keeps the facts already in hand and
records the MATLAB message under ``refused``; it costs the export nothing else.
The same holds for the response matrix. A file the Accelerator Data names but
the Middle Layer cannot find ends in its own measurement of the model, and so
does an Accelerator Data that names no file at all: naming nothing is the one
state the Middle Layer answers with a file-chooser dialog, and an export runs
unattended, so the measurement is asked for directly instead. Each block's
``origin`` says whether the matrix was measured on the machine or computed from
a model, which is not the same question as whether a file answered, since a
facility may keep a computed matrix in a file like any other. A matrix computed
from a model says nothing about the machine: ``osprey facility validate`` then
compares a deck against a deck.

The ``<stem>.model.json`` file is not imported: it is what a check of OSPREY's
model against the Middle Layer's reads.

Run the import
==============

.. code-block:: bash

   osprey facility import mml mymachine.storagering.ao.json mymachine.booster.ao.json

The first run finds no mapping, writes a draft to
``data/facility/imported/mml/mapping.yaml`` and stops with
``import mml: mapping-draft``. Nothing else is written. Review the draft, then
run the same command again.

The import takes every export the mapping names: a run given a subset is
refused by the mapping check, which names each system no given export carries.

Each deck is held to the lattice its export was sampled over. A
``<stem>.lattice.mat`` with another element count, other family names or other
parameter elements than the export states stops the import with
``import mml: export-invalid`` and names the fact that differs: import the
lattice the export was sampled from, or export again over this one. An energy
that differs alone is reported and the import goes on.

Review the mapping
==================

The mapping holds every decision the export does not state. The draft fills in
what the export does say and leaves ``null`` in each slot only you can decide.
Every ``null`` must be replaced: the import stops with
``import mml: mapping-undecided`` while one remains.

``families.<raw>.devices`` says which device each slot of a family is. It takes
one of five forms:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``names``
     - The export's ``CommonNames``, by position. This is the default when the
       key is absent.
   * - ``address``
     - The device segment of each slot's address.
   * - a list of local names
     - One name per device, in the family's order.
   * - ``{coordinates: <stem>}``
     - One device per ``DeviceList`` row, ``<stem>_<sector>_<device>``.
       Families that list the same devices use the same stem.
   * - ``{same_as: <family>}``
     - The named family's devices: matched by ``[sector, device]`` where both
       families state a DeviceList, so the named family may carry more; slot by
       slot otherwise.

.. code-block:: yaml

   families:
     HCM:
       devices: names
     VCM:
       devices: {same_as: HCM}
     BPMx:
       devices: {coordinates: BPM}
     BPMy:
       devices: {same_as: BPMx}
     IonGauge:
       devices: address
     RF:
       devices: [cavity1, cavity2]

The import stops with ``import mml: mapping-invalid`` for a list of the wrong
length, for ``names`` on a family the export leaves nameless, for
``coordinates`` on a family with no DeviceList, for a ``same_as`` target that is
itself a ``same_as``, and for a ``[sector, device]`` the ``same_as`` target
lacks.

**Families that share addresses.** One physical device listed under several
families must resolve to one id. The draft proposes this, giving the largest
family ``{coordinates: <stem>}`` and the others ``{same_as: <it>}``, and says
so in a comment under each answer. An import under which a wired channel would
be two devices stops with ``import mml: mapping-invalid``, naming the channel,
both families, both ids and the fix.

``models.<name>.wiring`` lists the families the model drives or reads, each
with the field, engine attribute and calibration that bind it to the deck.

The ``facility:`` block seeds ``data/facility/identity.yaml`` on the first
import and is then removed from the mapping.

A mapping that disagrees with the exports prints one ``<key>: <message>`` line
per problem and writes nothing. Fix each line and run the import again.

What the import writes
======================

A clean run prints one ``wrote <path>`` line per file. The files under
``data/facility/imported/mml/`` are rewritten by every import. The authored
files beside them --- ``limits.yaml``, ``seeds.yaml``, ``identity.yaml``,
``classes.yaml`` and ``measurement/<model>.yaml`` --- are created only when
absent and never rewritten, so your edits to them survive a re-import.

A setpoint takes its ``tolerance`` from its write field's
``Setpoint.Tolerance``, one number per device or one for the family, written
``{absolute: <x>}`` in the field's ``HWUnits``. A setpoint gets none when the
field states no unit, or its tolerance is not a finite number above ``1e-12``
(an ``Inf``, or a machine-epsilon placeholder), and the import prints one line
counting the setpoint devices that export no usable ``Setpoint.Tolerance``.
A later importer that reads EPICS records instead takes the default from the
record: ``MDEL`` when it is above 0, else ``10^-PREC``, and never ``HOPR`` or
``LOPR``.

The import refuses to run while ``data/facility/`` holds a record source of
your own that would merge against the imported records. It prints
``import mml: authored-present: <n> files`` and one ``rm <path>`` line per
file. Corrections belong in ``fixes.yaml``, which is never in the way.

Scenario files never stop the import. After a clean run it prints
``these scenario files name channels that no longer exist:`` and one
``rm <path>`` line for each scenario file that names a channel or model the
imported facility does not have, followed by an ``rm -r <folder>/`` line when
the scenario keeps a folder of attached files beside it. The import deletes
nothing, and
``osprey build`` stops while one is left: remove each listed file, or point it
at the imported channels.

A deployment that began as a preset states
``control_system.connector.virtual_accelerator.probe_channel`` for the demo's
channels. ``osprey build`` and ``osprey facility validate`` stop with
``profile-invalid`` while it names a channel the imported facility does not
hold, and name one it does. Set it to that channel:

.. code-block:: bash

   osprey set config.control_system.connector.virtual_accelerator.probe_channel=<channel>

Check the result
================

.. code-block:: bash

   osprey facility validate

``validate`` runs every check ``osprey build`` makes. For each model whose
export carried a response matrix it also compares that matrix with the one the
imported model computes, and prints one ``response check <model>: …`` line. A
failing check exits 1. When the check left rows out of the comparison, a second
line for that model follows on stderr, giving the number of rows left out and
the count per reason (unwired, no width, unsolved, table calibration); it does
not change the verdict or the exit code:

``response check <model>: left out <n> rows (<k> unwired, <j> no width, <u> unsolved, <t> table calibration)``

The check converts a monitor reading to position with one slope, taken at the
centred beam. That is exact for a straight-line (gain and offset) calibration;
a table-calibrated monitor is left out of the check and counted on the
left-out line. If you need a more elaborate BPM calibration in the check, open
an issue.

.. seealso::

   :ref:`cli-osprey-facility`
      Every exit and message of ``osprey facility validate`` and
      ``osprey facility import mml``.
