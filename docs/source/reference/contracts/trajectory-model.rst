.. _reference-trajectory-model:

=========================
Trajectory Model Contract
=========================

A trajectory model is a served pyAT model whose measurement file lists the
``trajectory`` kind: it holds turn-by-turn buffers per beam position monitor,
takes an arm command and an injection request, and fills the buffers with one
tracked particle per shot, losing it where the lattice and its apertures stop
it. :ref:`The Virtual Accelerator <architecture-virtual-accelerator>` places it
among the served models, and :ref:`architecture-simulation-models` says where
an engine and its settings come from. This page is the exact behaviour: the
state a model reports, which records are commands and what a write to one
does, what a buffer holds after a shot, where shot state lives, the files and
tool answer of the first-turn response, the model-wide scenario faults, which
channels the archive records, and the sum levels a reader of the buffers
compares against.

The facility side of the kind, its groups, instruments and keys, is described
in :doc:`/how-to/describe-your-facility`.

Model state
===========

A served physics model is in one of three states, and the text its status
channel ``<code>:SIM:<model>:STATUS`` serves tells them apart. The same text is
what :func:`~osprey_connectors.simulation.status.model_status` returns in
process and what ``osprey sim status`` prints (:doc:`/reference/cli`).

The model state is independent of the serving health state of the virtual
accelerator container, which uses the word *degraded* too: a health record
reads ``degraded`` after failed publishing passes, whatever state each model is
in, and a model reads degraded while every pass succeeds.

.. list-table:: Model state (independent of serving health)
   :header-rows: 1
   :widths: 14 26 60

   * - State
     - Status text
     - What is served
   * - ok
     - ``ok``
     - The model is built and reports no condition. Every channel it wires
       serves its value with no condition.
   * - degraded
     - one of the condition texts the model's engine lists; for the pyAT
       engine, ``no closed orbit``
     - The model is built and serving, and writes commit. Each channel the
       engine names under the condition serves its value, NaN where the value
       has no meaning, with the ``udf`` condition. Every other channel serves
       as in *ok*.
   * - failed
     - any other text: the engine's error text
     - No model is built. Every channel the model wires carries the ``udf``
       condition.

A reader tells degraded from failed by the engine's listed texts, never by the
status not being ``ok``. Nothing about the state is cached: the status and the
conditions are derived from the model each time they are read, so a write, a
refused batch, a reset or a scenario change shows at the next read. A status
text is at most 1023 bytes of UTF-8. The archive seeds a model that is built,
ok or degraded, and refuses only a failed one; a degraded model's listed
channels are seeded as NaN.

The pyAT engine reports ``no closed orbit`` for a ``periodic`` model whose
settings state ``injection`` after a write leaves the lattice with no closed
orbit; that write is kept rather than rolled back. Its listed channels are
every orbit reading, every readback of the tunes or the chromaticity, and the
optics arrays (tunes, chromaticity, beta and orbit at the monitors). A
setpoint, a readback of a setpoint, a fault, a turn series, an armed flag and a
command are never listed. A ``periodic`` model without ``injection`` refuses a
write that loses the closed orbit, and a ``single_pass`` model has no closed
orbit to lose.

Records of a trajectory model
=============================

A trajectory model wires three record classes beyond its orbit readings. None
names an element except the series, which name their monitor's.

.. list-table:: Turn-by-turn records
   :header-rows: 1
   :widths: 22 30 48

   * - Record
     - Engine block
     - Served as
   * - turn series
     - ``{axis: x | y | sum, series: turn}`` on a monitor element
     - A waveform of the record's ``shape``, described as ``turn_series``
       with its plane (``sum`` has none). Every series of one model has the
       same length, at least the pre-trigger samples plus one.
   * - armed flag
     - ``{attribute: armed}``
     - An int reading, described as ``turn_series``: 1 while armed, else 0.
   * - command
     - ``{attribute: arm}`` or ``{attribute: inject}``
     - An int setpoint carrying the command mark. A write acts and holds no
       setting.

The ``injection`` block of ``settings.pyat`` names the deck ``marker`` the beam
is launched from, the launch vector ``x``, ``px``, ``y``, ``py``, ``delta``,
``ct`` (each 0 when left out) and ``pre_trigger_samples`` (10 when left out).
A ``periodic`` model carrying the kind requires the block; a ``single_pass``
model refuses it and launches from ``twiss_in`` with the default pre-trigger.

.. _trajectory-command-slots:

Command slots
=============

A command slot is a channel the measurement writes a command to: every channel
carrying the signal ``bpm_arm_command`` on a device of the model, and the
channel ``instruments.injection_request`` names. The facility build holds each
one to these rules:

- It is a setpoint of value type ``int``.
- It is no corrector of the kind: no member of the measurement's ``hcor`` or
  ``vcor`` group.
- A record in ``limits.yaml`` naming it states both bounds and
  ``confirm: false``, and no ``max_step``.
- The model wires it as a command, and no command record sits on a channel
  without the matching signal.

A write batch holding a command together with a channel another model or the
texture owns is refused whole, before any value is written, naming the first
command and the first foreign channel in address order.

Shots
=====

The arm and inject commands act on one armed flag, one shot counter and the
last request value the model has seen, and through them on every turn series
of the model.

- **Arm.** A write of exactly 1 to any arm command sets the armed flag. Any
  other value leaves the flag as it is. Every arm command of a model sets the
  same flag.
- **Inject.** A write to an inject command that is above the last value the
  model saw for it, and not the request's default, is one shot. Every write to
  a request, shot or not, becomes the last value seen. A write of the default
  is never a shot.
- **A shot.** The counter advances by one. When the flag is set, one particle
  is tracked, every series of the model is refilled and the flag is cleared.
  When it is not, the shot is counted, the flag stays clear and every series
  keeps what it held.
- **Order.** The particle is tracked after the batch's setpoints are applied
  and solved, so it sees the lattice the batch left. The series, the flag, the
  counter and the last request seen change only when the batch commits; a
  refused batch leaves all four as they were.
- **Rest.** A model as built, after ``reset()``, and after a rebuild holds every
  series at zero, the flag at 0, the counter at 0 and every request at its
  default. ``reset()`` is never a shot.

A ``periodic`` model tracks the lattice rotated to its injection marker, one
sample per turn, for the series length less the pre-trigger samples. A
``single_pass`` model tracks its line once, unrotated, for one sample.

What a series holds
-------------------

For each series and each sample, after a tracked shot:

.. list-table:: Turn series samples
   :header-rows: 1
   :widths: 30 23 23 24

   * - Sample
     - ``x``
     - ``y``
     - ``sum``
   * - before the launch (the pre-trigger samples)
     - 0
     - 0
     - 0
   * - tracked, beam passes the monitor
     - position, through the record's calibration
     - position, through the record's calibration
     - 1, through the record's calibration
   * - tracked, beam already lost
     - NaN
     - NaN
     - 0
   * - after the last tracked sample
     - 0
     - 0
     - 0

A tracked sample is lost when the tracked ``x`` at the monitor is NaN, so every
monitor past the loss reads NaN in ``x`` and ``y`` and 0 in ``sum``, on that
turn and on every later one. A lost sample is never passed through a
calibration. A tracked position takes the readout faults its monitor's element
carries, the ones its orbit reading takes, at the same physical size; ``sum``
takes none. Noise is drawn when the shot is tracked, keyed on the series
address, the shot counter and the sample, so a series reads the same until the
next shot refills it and a fault written afterwards reaches the next shot
only.

Shot state and the connector
============================

Shot state lives in the composite of the connector that wrote the commands. The
in-process virtual accelerator connector never appends a command write to its
writes journal and refuses a journal that names a command channel. A command
write it took counts as held state, so every journal replay starts from a reset
composite: every series zero and the armed flag 0. A second connector, or one
whose composite was rebuilt, never repeats a command. A shot and its read
therefore belong together: they are valid only with no write from another
connector to the model's setpoints in between.

Slots
=====

:func:`~osprey.facility.views.pyaml.trajectory_slots` answers the channels a
model's ``trajectory`` measurement arms, fires and watches, and the view facts
carry the same answer per model under ``trajectory_slots``.

.. list-table:: Trajectory slots
   :header-rows: 1
   :widths: 24 20 56

   * - Slot
     - Value
     - Channels
   * - ``arm``
     - list
     - The channel carrying ``bpm_arm_command`` on each member device of the
       ``bpm`` group, in ``(s, id)`` order, one per device.
   * - ``armed``
     - list
     - The channel carrying ``bpm_armed_status`` on each of those devices, in
       the same order.
   * - ``injection_request``
     - address
     - The channel the measurement file names; always present.
   * - ``injection_busy``
     - address
     - The channel the measurement file names; always present.
   * - ``bunch_clean_enable``, ``beam_current``, ``rf_status``
     - address
     - Present only when the measurement file names it.

``arm`` and ``armed`` are a set per device, not aligned with the rows of any
response matrix. A model whose measurement file does not list the kind raises
``KeyError``.

First-turn response
===================

For every model carrying the kind, the pyAML view build writes two files to
``data/pyaml/<model>/``. A refusal by the engine, or an engine with no
``response_matrix()``, stops the build.

.. list-table:: First-turn response files
   :header-rows: 1
   :widths: 30 70

   * - File
     - Content
   * - ``trajectory_rm.json``
     - The ``pyaml.tuning_tools.orbit_response_matrix_data`` document:
       ``matrix``, ``observable_names`` (the turn-by-turn ``x`` series
       addresses, then the ``y`` ones, in lattice order from the injection),
       ``variable_names`` (the ``hcor`` group's corrector setpoints, then the
       ``vcor`` group's), ``observable_planes`` and ``variable_planes``
       (``H`` or ``V`` each). It carries no ``rf_response``.
   * - ``trajectory_rm.index.json``
     - The lattice index of every row and column: ``bpm_index`` (one entry per
       row), ``hcor_index`` then ``vcor_index`` (together one entry per
       column).

Each corrector steps by the measurement file's ``corrector_delta`` (radians),
converted to its hardware unit through its wiring calibration. An entry is a
series reading change, through its record's calibration, per corrector
setpoint change, each in its hardware unit; ``sum`` has no row. Every index is
counted from the injection marker along the lattice rotated to it, so the
marker is index 0; a ``single_pass`` model's indices are its line's own. Values
are written to nine significant digits and every one is finite.

``model_response_matrix`` with ``kind`` ``trajectory`` answers the same
content as JSON for the correctors it is asked for, and writes nothing
(:ref:`executor-measurement-tools`).

Model-wide scenario faults
==========================

A scenario states physics faults of a pyAT model under ``faults.<model>`` by
name, with no channel address. A model is built at them and holds them for
its whole lifetime; none is a variable of the built model.

.. list-table:: Model-wide faults of the pyAT engine
   :header-rows: 1
   :widths: 40 22 38

   * - Name
     - Range
     - Meaning
   * - ``misalignment_seed``
     - whole number, 0 to 2\ :sup:`53`
     - Seeds every misalignment draw; 0 when the scenario states none.
   * - ``<magnet>_<field>_sigma`` for ``<magnet>`` in ``quadrupole``,
       ``sextupole``, ``dipole`` and ``<field>`` in ``dx``, ``dy``, ``roll``
     - 0 or more
     - The standard deviation of one misalignment field of one magnet class:
       ``dx`` and ``dy`` in metres, ``roll`` in radians about the beam axis.
   * - ``aperture_<shape>_<axis>`` for ``<shape>`` in ``elliptical``,
       ``rectangular`` and ``<axis>`` in ``x``, ``y``
     - 1e-6 or more
     - One half-axis, in metres, of a vacuum chamber written on every element
       with a length. A shape is stated with both half-axes or not at all.

Each misaligned magnet takes its sigma times one standard normal draw keyed on
the seed, the magnet's deck name and the field, so a value depends on no other
element and on no order, and magnets sharing a name share one misalignment.
The facility build therefore stops a scenario that states a non-zero sigma for
a magnet class while the deck holds a magnet of that class under a repeated
name; the stop names the fault and up to three repeated names with their
counts.

Archived channels
=================

The archived channel set of a simulator view is every channel of the view but
the ones a served model's wiring describes as ``turn_series``: the turn series
and the armed flags. Those hold the samples of one shot and have no value at an
instant between shots. Command channels are archived like any other setpoint.
Archive seeding and the live archiver recorder both take their channels from
this one set, and the mock archiver answers a request for a channel outside it
with ``<address> has no archived history``.

Sum levels
==========

A ``sum`` sample reads 1 where the beam passes the monitor and 0 where it does
not, through the sum record's calibration. A reader that separates beam from no
beam compares against a level between the two. The example facility documents
two such levels, both 0.5 in the sum record's unit: ``shot_sum_threshold``,
below which a shot's mean first-turn sum counts the shot as failed, and
``loss_sum_cutoff``, the rise a monitor's sum has to exceed to count as seeing
the beam.

.. seealso::

   :doc:`/architecture/virtual-accelerator`
       How the virtual accelerator serves its models, and its serving health
       record.

   :doc:`/how-to/describe-your-facility`
       The measurement file of a model, its kinds, groups and instruments.

   :doc:`/how-to/run-scenarios`
       Applying the scenarios that state model-wide faults.

   :doc:`python-executor`
       The ``model_response_matrix`` tool and its ``kind``.
