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

``validate`` runs every check the build makes and prints each problem as one
line, ``facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>``.
``show`` prints the record counts, the models and the views; given an id it
prints that record with the file each of its fields came from. Both are
described in full under :ref:`cli-osprey-facility`.

Identity
========

``identity.yaml`` holds the facility's ``code`` and, optionally, its ``name``
and ``description``:

.. code-block:: yaml

   code: ca
   name: Example Research Facility

The ``code`` names the facility in the knowledge graph and in each served
model's status channel, ``<code>:SIM:<model>:STATUS``. Without the file the
code is taken from the project name. The ``name`` is what agents and panels
call the facility; a facility that states none is called by the project name.

Records
=======

The four files under ``records/`` each hold a list of records of one kind.
Every record has an ``id``, and a record states only what its author knows:
a field left out is either filled by the build or stays absent.

**Places** are the facility's own tree. A place's id is its path,
``SR/SECT1`` under ``SR``; ``level`` is the facility's word for that depth.
A place with a ``span`` covers a stretch of one model's deck, from one marker
element to the next, and every device the deck puts inside it belongs to that
place without saying so.

.. code-block:: yaml

   - id: SR/SECT1
     level: sector
     span:
       model: SR
       from_marker: SECT1
       to_marker: SECT2

**Devices** carry a ``class``, either one of the shared vocabulary's classes
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
       setpoint with the one readback on the same device whose signal names the
       same quantity, else with itself.
   * - ``tolerance``
     - On a float setpoint, how close its readback must come before a move
       counts as done: ``{absolute: <x>}`` in the channel's unit or
       ``{relative: <fraction>}``.
   * - ``on``
     - The one device or place the channel belongs to, ``{device: <id>}`` or
       ``{place: <id>}``. Absent, the channel belongs to nothing.
   * - ``endpoint_of``
     - The devices that share the channel, for a supply feeding several.
   * - ``signal``
     - A signal role of the shared vocabulary, such as ``current_setpoint`` or
       ``position_x_readback``.
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

   - class: Kicker
     parent: Magnet
     aliases: [injection kicker]

Layers
======

Records reach the build from several sources at once, and each source is a
*layer*. ``records/`` and ``models.yaml`` are the layer ``authored``; each
directory under ``imported/`` is the layer of that name. A layer states only
the fields its source states.

The build merges the layers field by field: per record and per field it takes
the one value any layer states. Two layers stating different values for one
field stop the build with ``layer-conflict``, naming the record. A fix settles
it (see `Fixes`_).

The importers are ``osprey facility import mml`` and
``osprey facility import list``. A third layer, ``pyaml``, joins them for a
facility described in pyAML.

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
   address into ``imported/mml/models.yaml``, beside the deck the export
   saved, and ``osprey build`` copies both into the simulator view the
   simulator serves.

5. **Accept.** Run ``osprey facility validate``. For each model whose export
   carried a response matrix it compares that matrix with the one the
   imported model computes and prints one ``response check <model>: …`` line.
   A failing check exits 1. A passing one is the acceptance: the calibrations,
   nominals and element bindings of the imported model agree with the machine
   the export was sampled on.

The files under ``imported/mml/`` belong to the import and are rewritten by
every run. The import also creates ``limits.yaml``, ``seeds.yaml``,
``identity.yaml``, ``classes.yaml`` and ``measurement/<model>.yaml`` when they
do not exist, and never rewrites them afterwards. It refuses to run while
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
each layer held when the fix was written. When a later import changes that
value, the build stops with ``fix-stale`` and prints the block to paste in its
place, so a fix never silently outlives the fact it corrected. The result
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
limits. Both presets ship ``optional``.

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
and needs no seed. The build stops with ``seed-invalid`` when a nominal lies
outside the channel's limits.

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
``table``. Channels no model wires are served by the texture model from their
seeds. Which models a deployment serves, and what a failed one looks like, is
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
     quad: SR/QF
     sext: SR/SF
   instruments:
     tune: SR:DIAG:TUNE:X
     chromaticity: SR:DIAG:CHROM:X
     rf: SR:RF:CAVITY:01:FREQUENCY:SP
   n_step: 5
   n_avg_meas: 1
   corrector_delta: 1.0e-05

``kinds`` names the allowed measurements, ``groups`` the group that plays
each part and ``instruments`` the channel each scalar is read on. The step
and settle keys (``n_step``, ``n_avg_meas``, ``corrector_delta`` and the
rest) are carried as written.

Scenarios
=========

Each file under ``scenarios/`` is one way the simulated machine departs from
its baseline: overrides, faults, archive history and logbook entries.
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

.. code-block:: bash

   osprey build

The build runs the same checks as ``osprey facility validate``, in fixed
stages, and stops at the first that fails: each file parses, the layers merge
and the fixes apply, the result fits the schema, every id a record names
exists, and the pair, value, limits and seed rules hold. Then it computes
what no source states (each device's place and position from the decks, each
wiring record's unit and range) and writes ``build/facility.json`` and the
views.

.. seealso::

   :doc:`/architecture/facility-file`
      The facility file, each view the build writes from it and what reads
      it.

   :ref:`cli-osprey-facility`
      Every exit and message of ``osprey facility validate``, ``show`` and
      the importers.
