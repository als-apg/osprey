.. _how-to-describe-your-facility:

======================
Describe your facility
======================

A deployment describes its facility once, as source files under
``data/facility/``. ``osprey build`` combines them into one facility file,
``build/facility.json``, and writes every file a service reads from it: the
channel finder's index, the limits database, the simulator's channel list, the
knowledge graph. Nothing else in the deployment lists a channel.

This page walks the tree a fresh ``osprey init --preset control-assistant``
writes, file by file, then the three ways records get into it: by hand, from
an importer, and through a fix. What the build writes from the result is in
:doc:`/architecture/facility-file`.

The tree
========

.. code-block:: text

   data/facility/
     identity.yaml             who the facility is
     records/
       places.yaml             the facility's own tree of places
       devices.yaml            the devices
       channels.yaml           the control-system channels
       groups.yaml             named sets of devices
     classes.yaml              device classes the facility adds
     models.yaml               the simulation models and their wiring
     decks/                    the decks those models name
     limits.yaml               the write limits
     seeds.yaml                how a channel no model wires behaves when simulated
     measurement/<model>.yaml  the measurements a model allows
     scenarios/<name>.yaml     ways the simulated machine departs from its baseline
     fixes.yaml                corrections to records another source states
     imported/<layer>/         records an importer wrote
     knowledge/                the facility knowledge pages

Every file is optional. A deployment with no ``data/facility/`` at all builds
a facility with no channel, and each file added states more. The
control-assistant preset ships all of them except ``classes.yaml``,
``fixes.yaml`` and ``imported/``; the hello-world preset ships
``identity.yaml``, ``records/channels.yaml``, ``limits.yaml`` and
``seeds.yaml``.

Check the tree at any time, without writing anything:

.. code-block:: bash

   osprey facility validate
   osprey facility show
   osprey facility show SR:MAG:HCM:01:CURRENT:SP

``validate`` builds the tree and renders the main profile's views in memory.
It prints the errors of the first check that fails, one line each,
``facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>``; a clean
tree prints its warnings the same way. ``show`` prints the identity, the
record counts, the models and the views; given an id it prints that record
with the layer and file each of its fields came from, the fields the build
filled and the fixes applied. Both are described in full under
:ref:`cli-osprey-facility`.

Identity
========

``identity.yaml`` holds the facility's ``code`` and, optionally, its ``name``
and ``description``:

.. code-block:: yaml

   code: ca
   name: Example Research Facility

The ``code`` names the facility in the knowledge graph and in each served
model's status channel, ``<code>:SIM:<model>:STATUS``. Without the file the
code is the project name, folded to letters, digits and underscores. The ``name`` is what agents and panels
call the facility; a facility that states none is called by the project name.

Records
=======

The four files under ``records/`` each hold a list of records of one kind.
Every record has an ``id``, and a record states only what its author knows:
a field left out is either filled by the build or stays absent.

**Places** are the facility's own tree. A place's id is its path,
``SR/SECT1`` under ``SR``; ``level`` is the facility's word for that depth.
A place with a ``span`` covers a stretch of one model's deck, from the
``from_marker`` element up to the ``to_marker`` element, or to the deck's end
without one. A device the deck puts inside it belongs to the deepest such
place, unless the device states its own ``place``.

.. code-block:: yaml

   - id: SR/SECT1
     level: sector
     span:
       model: SR
       from_marker: SECT1
       to_marker: SECT2

**Devices** name a ``class``, either one of the shared vocabulary's classes
(``Quadrupole``, ``BeamPositionMonitor``, ``Gauge``) or one the facility adds
in ``classes.yaml``. A device may state its ``place``, a ``label``, other
``names`` people use for it, and a position: ``s`` in metres along the deck of
its ``model``. A device a model wires takes its position from the deck.

**Channels** are the control-system addresses. A channel's id is its full
address, written exactly as the control system serves it; no part of the
address means anything to OSPREY. What a channel is comes from its fields:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``role``
     - ``setpoint``, ``readback`` or ``none``. Absent means readback.
   * - ``pair``
     - On a setpoint, the address of its readback. Absent, the build pairs the
       setpoint with the readback that shares its ``on``, or its
       ``endpoint_of``, or neither, and whose signal names the same quantity
       (``gap_setpoint`` with ``gap_readback``). It does so only when exactly
       one setpoint and one readback of that quantity share it, no stated
       ``pair`` names the readback, and the two agree in ``value_type``,
       ``options`` and ``shape``; else it pairs the setpoint with itself.
   * - ``tolerance``
     - On a float setpoint, how close its readback must come before a move
       counts as done: ``{absolute: <x>}`` in the channel's unit or
       ``{relative: <fraction>}`` of its seed's ``nominal``. The setpoint and
       its readback must both state a ``unit``.
   * - ``on``
     - The one device or place the channel belongs to, ``{device: <id>}`` or
       ``{place: <id>}``. Absent, the channel belongs to nothing.
   * - ``endpoint_of``
     - The devices that share the channel, for a supply feeding several.
   * - ``signal``
     - A signal role of the shared vocabulary, such as ``current_setpoint`` or
       ``gap_readback``. A facility cannot add one.
   * - ``unit``, ``description``
     - Free text. A channel that states no description and names a signal is
       given one composed from its owner, its signal and its unit.
   * - ``label``, ``names``
     - The one name a person reads for the channel, and other names in use.
   * - ``tags``
     - Free words. ``in_context`` narrows the in-context channel-finder index
       to the tagged channels.
   * - ``value_type``
     - ``float``, ``int``, ``bool``, ``enum``, ``string`` or ``waveform``.
       Absent means float.
   * - ``options``
     - The labels of a ``bool`` or ``enum`` channel.
   * - ``shape``
     - The dimensions of a ``waveform`` channel.
   * - ``precision``
     - Optional, on float channels only: an integer from 0 to 17, the number
       of decimals a display shows. The simulator serves it as the channel's
       display precision on both Channel Access and PVAccess.
   * - ``former_addresses``, ``attributes``
     - Addresses the channel was known by, and a free map carried verbatim.

.. code-block:: yaml

   - id: SR:MAG:QF:01:CURRENT:SP
     role: setpoint
     pair: SR:MAG:QF:01:CURRENT:RB
     tolerance:
       absolute: 0.01
     unit: A
     label: focusing quadrupole current setpoint
     description: Focusing quadrupole current setpoint

**Groups** are named sets of devices: an id, a ``description`` in the
facility's words and the ``members``. A group says which devices belong
together; what a channel means stays on the channel.

``classes.yaml`` adds device classes the shared vocabulary lacks, each as the
child of an existing class:

.. code-block:: yaml

   - class: Undulator
     parent: Magnet
     aliases: [insertion device, ID]

Layers
======

Records reach the build from several sources at once, and each source is a
*layer*. ``records/``, ``models.yaml``, ``seeds.yaml`` and ``measurement/``
are the layer ``authored``; each directory under ``imported/`` is the layer of
that name. A layer states only
the fields its source states.

The build merges the layers field by field: per record and per field it takes
the one value any layer states. Two layers stating different values for one
field stop the build with ``layer-conflict``, naming the record. A ``set`` fix
settles it (see `Fixes`_).

The importers are ``osprey facility import mml`` and
``osprey facility import list``.

A Middle Layer export
---------------------

``osprey facility import mml`` reads a MATLAB Middle Layer export into
``data/facility/imported/mml/``. From the export to a served model it is five
steps; each is described in :doc:`/how-to/import-mml-export`.

1. **Export.** ``osprey facility import mml --print-exporter > mml_export.m``
   prints the exporter. Run it in MATLAB once per sub-machine.

2. **Draft the mapping.** The first run of
   ``osprey facility import mml <stem>.ao.json`` writes
   ``imported/mml/mapping.yaml`` and stops with ``import mml: mapping-draft``.
   The mapping holds every decision the export does not state.

3. **Review the mapping.** Replace every ``null``. The top-level blocks are
   ``models`` (what each exported system is called as a model),
   ``section_order``, ``families`` (each family's ``class``, ``devices``,
   ``description`` and ``fields``) and ``directions`` (whether each field is
   ``read`` or ``write``). Where the export leaves a family's shape open, a
   ``judgments:`` block carries one slot per question: a channel row beyond
   the family's devices (``drop``, ``device`` or ``{field: <name>}``), a
   device no channel binds (``drop`` or ``keep``), and a channel shared
   across devices (``keep_all`` or an owning device).

4. **Wire the model.** Each model's ``wiring`` block in the mapping lists the
   families the model drives or reads. Per family it names the
   ``element_field``, the family field the model wires; ``engine``, the deck
   attribute, index or axis the field acts on; and ``calibration``,
   ``linear`` or ``table``. From it the import writes one wiring record per
   address into ``imported/mml/models.yaml``, and the deck it serves, its
   wired elements renamed, into ``imported/mml/decks/<model>.json``;
   ``osprey build`` copies both into the simulator view the simulator serves.

5. **Accept.** Run ``osprey facility validate``. For each model whose export
   carried a response matrix it compares that matrix with the one the
   imported model computes and prints one ``response check <model>: …`` line.
   A failing check exits 1. A passing one is the acceptance: the calibrations,
   nominals and element bindings of the imported model agree with the machine
   the export was sampled on.

The record files under ``imported/mml/`` belong to the import and are
rewritten by every run; ``mapping.yaml`` is yours and is only drafted when
absent. The import also creates ``limits.yaml``, ``seeds.yaml``,
``identity.yaml``, ``classes.yaml``, ``measurement/<model>.yaml`` and
``scenarios/readout.yaml`` when they do not exist, and never rewrites them
afterwards. It refuses to run while
authored record sources are present: every file of ``records/`` and
``decks/``, ``models.yaml``, and a ``limits.yaml``, ``seeds.yaml``,
``identity.yaml`` or measurement file it did not create itself.

A channel list
--------------

``osprey facility import list FILE`` reads a CSV of channel addresses into
``data/facility/imported/list/``. A header row names ``address`` and any of
``role``, ``pair``, ``device``, ``place``, ``unit``, ``description``,
``tags``, ``s`` and ``model``; a file with no ``address`` column holds one
address per line.

.. code-block:: text

   address,role,device,unit,description
   SR:MAG:QF:01:CURRENT:SP,setpoint,QF01,A,Focusing quadrupole current
   SR:MAG:QF:01:CURRENT:RB,,QF01,A,Focusing quadrupole current readback

The files under ``imported/list/`` belong to the import and are replaced by
each run, so they are never edited by hand: a correction goes into the CSV or
into ``fixes.yaml``. A run whose list states no ``s`` leaves no
``devices.yaml``. The list import runs beside authored records and merges with
them field by field, where the mml import refuses while authored record
sources are present.

Fixes
=====

``fixes.yaml`` corrects a record another source states, without editing that
source. Each fix names a record and gives the reason:

.. code-block:: yaml

   schema: osprey.facility.fixes/1
   fixes:
     - op: set
       kind: channel
       id: SR:QF:SP
       fields: {role: setpoint}
       was: {role: {mml: readback}}
       why: The export lists the setpoint as a monitor.
     - op: add
       kind: device
       id: SR/QF9
       record: {class: Quadrupole}
       why: Missing from the export.

``set`` replaces the fields it names, each whole; ``add`` adds a record;
``drop`` removes one. ``kind`` is ``place``, ``device``, ``channel``,
``wiring``, ``group`` or ``model``. A ``set`` states under ``was`` the value
each layer held when the fix was written. When ``was`` is missing, or a later
import changes that value, the build stops with ``fix-stale`` and prints the
corrected fix to paste in its place, so a fix never silently outlives the fact it corrected. The result
does not depend on the order of the fixes.

``osprey facility show <id>`` lists the fixes applied to a record.

Limits
======

``limits.yaml`` holds the write limits, one record per channel:

.. code-block:: yaml

   records:
   # Symmetric range: the corrector current stays within [-12, 12].
   - address: SR:MAG:HCM:01:CURRENT:SP
     min_value: -12.0
     max_value: 12.0
   # Range plus largest single step: the cavity frequency in MHz.
   - address: SR:RF:CAVITY:01:FREQUENCY:SP
     min_value: 500.0
     max_value: 500.8
     max_step: 0.01
   # Blocked outright: no write reaches this setpoint.
   - address: SR:VAC:ION-PUMP:01:VOLTAGE:SP
     writable: false

A setpoint record with both ``min_value`` and ``max_value`` is writable within
them; ``max_step`` bounds a single change; ``writable: false`` blocks every
write; ``confirm: false`` turns off the re-read after a write.

``osprey build`` renders the records into ``build/data/channel_limits.json``,
the file the connector's reference monitor and the runtime's limits check
enforce on every write. That file is an output: a profile that ships its own stops the build. The file
holds the records of ``limits.yaml`` and nothing else. What happens to a
channel with no record is the deployment's choice,
``control_system.limits_checking.mode``: under ``exclusive`` the limits file
is the complete list of writable channels and a channel without a record is
refused; under ``optional`` a channel without a record is written with no
limits. The hello-world and control-assistant presets ship ``optional``.

Seeds
=====

``seeds.yaml`` says how a channel behaves in the simulator when no model
wires it. It is keyed by address:

.. code-block:: yaml

   SR:BEAM:CURRENT:
     nominal: 250.0
     noise:
       absolute: 0.1

``nominal`` is the value at start; ``noise`` is Gaussian, ``{absolute:
<sigma>}`` in the channel's unit or ``{relative: <fraction>}``; ``drift`` is
``{amplitude, period_s}``; ``clamp`` is ``[low, high]``; ``linear`` makes the
value a weighted sum of other channels. A wired channel starts from its deck
and needs no seed. The build stops with ``seed-invalid`` when a nominal, or a
wired channel's operating point, lies outside the ``min_value`` and
``max_value`` of the channel's limits record.

``pulse`` gives a waveform channel its trace. It names two other channels:
``driver``, whose reading scales the trace, and ``enable``, which switches it
on. The trace's shape is fixed in code, not in the seed; with ``enable`` off
the trace is flat zero.

.. code-block:: yaml

   INJ:SEPTUM:PULSE:
     pulse:
       driver: INJ:SEPTUM:AMPL
       enable: INJ:SEPTUM:ENABLE

``gate`` says what a channel does while an interlock-like input blocks it.
``inputs`` maps each input's address to its **blocking label**; while any input
reads its blocking label the channel takes ``gated``, and otherwise it behaves
as its other seed keys say. ``gated`` is typed by the gated channel: a factor
greater than zero on a float channel (the value is scaled by it), a label on a
bool or enum channel, an integer on an int channel.

.. code-block:: yaml

   SR:BEAM:CURRENT:
     nominal: 250.0
     gate:
       inputs:
         SR:RF:PERMIT: blocked
       gated: 0.01

The build checks both keys and stops with ``value-invalid`` when:

- ``pulse`` sits on a channel that is not a waveform, or its ``driver`` is not
  an unwired float channel, or its ``enable`` is not an unwired bool channel;
- ``gate`` sits on a channel that is not float, int, bool or enum; an input is
  wired, is not a bool or enum channel, or names a label that channel does not
  have; or ``gated`` does not fit the channel's type as above;
- ``gate`` inputs form a cycle (``a`` gated by ``b`` gated by ``a``).

A ``pulse`` or ``gate`` input that names no channel stops the build with
``reference-missing``.

At run time the ``gate`` acts on the channel's held value, after a ``linear``
sum; ``drift`` and ``noise`` are added on top and ``clamp`` applied. A gated
float reads its value
times the factor, so it keeps a small residual that still moves; a gated bool
or enum reads the label ``gated``; a gated int reads the integer ``gated``.
Each input is read as its own current value, its own gate included, so gates
chain: an input that is itself gated blocks whenever its gate holds it at the
blocking label. An input the simulator does not serve never
blocks. Live reads, change notifications and the archive's seeded history all
see the same gated value.

Models and decks
================

``models.yaml`` lists the simulation models. A model has a ``name``, the
``engine`` that runs it, the ``deck`` it runs over, the engine's ``settings``
and a ``wiring`` list that ties channel addresses to elements of the deck:

.. code-block:: yaml

   - name: LINE
     engine: pyat
     deck: decks/LINE.json
     settings:
       pyat:
         solve: single_pass
         twiss_in:
           beta: [7.448848, 4.786082]
           alpha: [-1.333207, 0.885765]
     wiring:
     - address: LINE:DIAG:BPM:01:POSITION:X
       element: BPM01
       engine:
         axis: x
       calibration:
         curve:
           linear: {gain: 1.0, offset: 0.0}
         energy_scaling: none

A wiring record's ``calibration`` converts between the channel's hardware
value and the engine's physics value, as a ``linear`` gain and offset or a
``table``. Channels no served model wires are served by the texture model from
their seeds. Which models a deployment serves, and what a failed one looks like, is
in :ref:`architecture-simulation-models`.

Measurement
===========

``measurement/<model>.yaml`` states which measurements a model allows and
what they use:

.. code-block:: yaml

   kinds: [orm, dispersion, trm, crm, chromaticity_monitor]
   groups:
     bpm: SR/BPM
     hcor: SR/HCM
     vcor: SR/VCM
     quad: [SR/QF, SR/QD]
     sext: [SR/SF, SR/SD]
   instruments:
     tune: SR:DIAG:TUNE:X
     rf: SR:RF:CAVITY:01:FREQUENCY:SP
   n_step: 5
   n_avg_meas: 1
   fit_order: 2
   sleep_between_step: 0.0
   sleep_between_meas: 0.0
   corrector_delta: 1.0e-05
   quad_delta: 0.001
   sextu_delta: 0.01
   frequency_delta: 100.0

``kinds`` names the allowed measurements: ``orm``, ``dispersion``, ``trm``,
``crm`` and ``chromaticity_monitor``. ``groups`` names the group, or the list
of groups taken together, that plays each part (``bpm``, ``hcor``, ``vcor``,
``quad``, ``sext``) and
``instruments`` the channel each instrument is read or stepped on (``tune``,
``rf``). Each kind needs its own members and step keys, and the build stops
when the file allows a kind without them, names a member its model does
not wire, or names a ``tune`` reading of one plane that no single reading of
the other plane pairs with:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Kind
     - Groups and instruments
     - Keys
   * - ``orm``
     - ``bpm``, ``hcor``, ``vcor``
     - ``corrector_delta``
   * - ``dispersion``
     - ``bpm``, ``hcor``, ``vcor``, ``rf``
     - ``frequency_delta``
   * - ``trm``
     - ``quad``, ``tune``
     - ``quad_delta``
   * - ``crm``
     - ``sext``, ``tune``, ``rf``
     - ``sextu_delta``
   * - ``chromaticity_monitor``
     - ``tune``, ``rf``
     - ``n_step``, ``n_avg_meas``, ``fit_order``, ``sleep_between_step``,
       ``sleep_between_meas``

A model solved in a single pass allows ``orm`` alone. A group plays one role;
only ``hcor`` and ``vcor`` may name the same group. A role naming several
groups is one pyAML array named after the role. The tools
that run these measurements are in :doc:`measure-with-pyaml`.

Scenarios
=========

Each file under ``scenarios/`` is one way the simulated machine departs from
its baseline: value overrides, model and channel faults, its own motion
(drivers, couplings, noise), archive history and logbook entries.
Writing and applying them is in :doc:`/how-to/run-scenarios`.

Knowledge
=========

``knowledge/`` is the facility knowledge bundle: the pages agents read about
subsystems, devices and procedures. A page links itself to a device with the
``device_id`` key of its frontmatter; the build names a page whose id is no
device of the facility and goes on. The bundle's format is in
:doc:`/how-to/facility-knowledge/okf-bundle`.

Build
=====

``osprey build`` runs the checks of ``osprey facility validate`` but the
response check, stops at the first that fails, and then writes
``build/facility.json`` and the views into the deployment's render and each
persona and image render.

.. seealso::

   :doc:`/architecture/facility-file`
      The facility file, each view the build writes from it and what reads
      it.

   :ref:`cli-osprey-facility`
      Every exit and message of ``osprey facility validate``, ``show`` and
      the importers.
