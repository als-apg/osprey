.. _reference-trajectory-model:

=========================
Trajectory Model Contract
=========================

A trajectory model is a served pyAT model whose measurement file lists the
``trajectory`` kind: it holds turn-by-turn buffers per beam position monitor,
takes an arm command and an injection request, and fills the buffers with one
tracked particle, or a bunch, per shot, losing it where the lattice and its
apertures stop it. :ref:`The Virtual Accelerator <architecture-virtual-accelerator>` places it
among the served models, and :ref:`architecture-simulation-models` says where
an engine and its settings come from. This page is the exact behaviour: the
state a model reports, which records are commands and what a write to one
does, the settings and readings around a shot, what a buffer holds after a
shot, where shot state lives, the files and tool answer of the first-turn
response, the model-wide scenario faults, and which channels the archive
records.

The facility side of the kind, its groups, instruments, keys and
``trajectory`` block, is described in :doc:`/how-to/describe-your-facility`.

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

A trajectory model wires these records beyond its orbit readings. None names
an element except the series, which name their monitor's, the readout
settings, which name a monitor element, and the RF switch, which names the
cavities.

.. list-table:: Records of a trajectory model
   :header-rows: 1
   :widths: 22 30 48

   * - Record
     - Engine block
     - Served as
   * - turn series
     - ``{axis: x | y | sum, series: turn}`` on a monitor element
     - A waveform of the record's ``shape``, described as ``turn_series``
       with its plane (``sum`` has none). Every series of one model has the
       same length, the trigger sample plus at least one turn.
   * - armed flag
     - ``{attribute: armed}``
     - An int reading, described as ``turn_series``: 1 while armed, else 0.
   * - command
     - ``{attribute: arm}``, ``{attribute: inject}`` or
       ``{attribute: clear_stored_charge}``
     - An int setpoint carrying the command mark. A write acts and holds no
       setting.
   * - turns tracked
     - ``{attribute: turns_tracked}``
     - An int setting: the turns after the trigger sample every series
       carries beam on, from 1 to the series length less the trigger sample.
       Its default is the record's ``default``, else 100 or the most the
       series hold.
   * - RF switch
     - ``{attribute: Voltage}`` on the cavities, with two labels
     - A discrete setting, on by default: on writes each cavity's deck
       voltage, off writes 0. A scenario never sets it.
   * - stored current
     - ``{attribute: stored_current, surviving_fraction, charge_floor}``
     - A reading of the stored charge times the revolution frequency,
       through its calibration.
   * - readout setting
     - ``{attribute: readout_noise_scale | readout_tone_amplitude |
       readout_gain | readout_sum_scale}`` on a monitor element, optionally
       ``discrete: true``
     - A setting of that monitor's readout. A discrete one maps its label
       index through a calibration table; it starts at its ``default``, else
       the first label.

The ``injection`` block of ``settings.pyat`` names the deck ``marker`` the beam
is launched from. The build writes the measurement file's
``trajectory.trigger_sample`` into it (10 when the file states none). A
``periodic`` model carrying the kind requires the block; a ``single_pass``
model refuses it and launches from ``twiss_in``. Where the beam is launched
is a scenario's ``launch_*`` faults (see `Model-wide scenario faults`_).

.. _trajectory-command-slots:

Command slots
=============

A command slot is a channel the measurement writes a command to: every channel
carrying the signal ``bpm_arm_command`` on a device of the model, the channel
``instruments.injection_request`` names and the channel
``instruments.bunch_clean_enable`` names. The facility build holds each
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
- **A shot.** The counter advances by one. When the flag is set, the particle
  or the bunch is tracked, every series of the model is refilled and the flag
  is cleared. When it is not, the shot is counted, the flag stays clear and
  every series keeps what it held.
- **Order.** The beam is tracked after the batch's setpoints are applied and
  solved, so it sees the lattice the batch left. The series, the flag, the
  counter, the last request seen and the stored charge change only when the
  batch commits; a refused batch leaves them as they were.
- **Rest.** A model as built, after ``reset()``, and after a rebuild holds every
  series at zero, the flag at 0, the counter at 0, every request at its
  default and the stored charge at the scenario's. ``reset()`` is never a
  shot.

A ``periodic`` model tracks the lattice rotated to its injection marker, one
sample per turn, for the turns tracked. A ``single_pass`` model tracks its
line once, unrotated, for one sample.

A bunch
-------

A scenario that states ``bunch_particles`` declares a **bunch**; without it a
shot tracks one particle. A bunch needs a ``periodic`` model with an
``injection`` marker. Its shape is fixed at build, matched to the lattice at
the marker; the one random part is a centroid jitter per shot, keyed on the
model and the shot counter, so a restart or a reset replays the same shots.
At each sample ``sum`` reads the share of the bunch alive and ``x`` and ``y``
the survivors' centroid.

Accumulation
------------

Under a bunch, every shot, armed or not, adds its survivors at the end of the
tracked turns, times ``bunch_charge``, to the **stored charge**, which the
stored-current record serves. A write of the cleaning command empties it,
before the batch's shots; it is not a shot.

- **RF on, closed orbit.** The stored beam is in every sample of every series,
  the trigger samples included: ``sum`` adds the stored charge over
  ``bunch_charge``, and ``x`` and ``y`` read the charge-weighted centroid of
  the survivors and the stored beam on the closed orbit.
- **RF off.** There is no closed orbit; the model reports ``no closed orbit``,
  and orbit, tune and chromaticity read NaN. The stored charge keeps
  ``surviving_fraction`` of itself per shot, and is gone at or below
  ``charge_floor``. It shows only on the stored-current record, never in a
  series.
- **No closed orbit with RF on.** The stored charge is lost at once.

What a series holds
-------------------

A series reads one value for each sample, beam or not; after a shot it holds
no NaN and its ``sum`` no exact zero.

.. list-table:: Turn series samples
   :header-rows: 1
   :widths: 30 35 35

   * - Sample
     - ``x``, ``y``
     - ``sum``
   * - beam passes the monitor
     - position plus readout noise, through the record's calibration
     - the share alive (1 for a particle), through the readout and the
       record's calibration
   * - no injected beam: before the trigger sample, after the turns tracked,
       or past the loss
     - 0 plus readout noise, through the record's calibration
     - the floor 0.03, through the readout and the record's calibration

Stored beam with RF on adds to both columns as `Accumulation`_ says.

The readout of a monitor is its settings:

- **noise.** ``x`` and ``y`` take a Gaussian noise of
  ``readout_noise_scale / max(sum, 0.03)``: noise over sum.
- **pilot tone.** ``readout_tone_amplitude`` adds to ``sum``, and puts a spike
  of ``2e-3`` times it times a normal draw on every tenth sample of ``x`` and
  ``y``.
- **gain.** ``sum`` is ``(max(physics sum, 0.03) + tone) × readout_gain ×
  readout_sum_scale``.

A tracked position also takes the readout faults its monitor's element
carries, the ones its orbit reading takes, at the same physical size; ``sum``
takes none. Every draw is keyed on the series address, the shot counter and
the sample, so a series reads the same until the next shot refills it and a
fault or setting written afterwards reaches the next shot only.

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

:func:`~osprey.facility.views.pyaml.trajectory_slots` answers a model's
trajectory declaration: its measurement file's ``trajectory`` block read
against the channel facts. The view facts carry the same answer per model
under ``trajectory_slots``.

.. list-table:: Trajectory slots
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Value
   * - ``devices``
     - Per member device of the ``bpm`` group, in ``(s, id)`` order: its
       ``x_row``, ``y_row``, ``sum_row``, ``arm`` and ``armed`` channels, the
       ``units`` of the three rows, and every signal it carries as ``tags``.
   * - ``trigger_sample``
     - The sample the injected shot arrives at.
   * - ``injection_request``, ``injection_busy``
     - The instruments the measurement file names.
   * - ``machine``
     - ``rf_on`` and ``stored_current``: the reading of the ``rf_status`` and
       ``beam_current`` instruments, a setpoint's ``pair`` when it states one.
   * - ``settings``
     - Per declared setting: ``id``, ``label``, ``description``, ``scope``,
       and its written ``setting``, ``readback`` and ``tolerance`` (per device
       for a ``device`` setting) and ``timeout_s``.
   * - ``steps``
     - Per declared step, in chain order: ``id``, ``label``,
       ``description``, ``default``, ``position``, and a command step's
       ``command``, ``timeout_s`` and ``done``.

A model whose measurement file does not list the kind raises ``KeyError``.

First-turn response
===================

For every model carrying the kind, the pyAML view build writes two files to
``data/pyaml/<model>/``. A refusal by the engine, an answer in which no
corrector solves, or an engine with no ``response_matrix()``, stops the build.
So does an answer whose ``x`` and ``y`` rows do not read the same monitors in
the same order, one whose rows of a plane do not sit at strictly increasing
indices, and one reading a monitor the model wires no sum series on. A
corrector whose step loses the launched particle has no solve: it keeps its
column, as a column of zeros, and the index file names it.

.. list-table:: First-turn response files
   :header-rows: 1
   :widths: 30 70

   * - File
     - Content
   * - ``trajectory_rm.json``
     - The ``pyaml.tuning_tools.orbit_response_matrix_data`` document:
       ``matrix``, ``observable_names`` (the turn-by-turn ``x`` series
       addresses, then the ``y`` ones of the same monitors in the same
       order, in lattice order from the injection), ``variable_names`` (the
       ``hcor`` group's corrector setpoints, then the ``vcor`` group's, each
       plane in lattice order from the injection, correctors at one index in
       their group's order), ``observable_planes`` and ``variable_planes``
       (``H`` or ``V`` each). It carries no ``rf_response``.
   * - ``trajectory_rm.index.json``
     - ``bpm_index``, the lattice index of every row; ``sum_names``, the
       address of the sum series the model wires on every row's monitor (one
       entry per row, so the ``x`` and the ``y`` row of a monitor name the
       same series); ``hcorr_index`` then ``vcorr_index``, the lattice index of
       every column (together one entry per column, in the columns' order);
       and ``unsolved``, each corrector whose step has no solve mapped to the
       engine's text.

Each corrector steps by the measurement file's ``corrector_delta`` (radians),
converted to its hardware unit through its wiring calibration. An entry is a
series reading change, through its record's calibration, per corrector
setpoint change, each in its hardware unit; ``sum`` has no row. Every index is
counted from the injection marker along the lattice rotated to it, so the
marker is index 0; a ``single_pass`` model's indices are its line's own. Values
are written to nine significant digits and every one is finite.

``model_response_matrix`` with ``kind`` ``trajectory`` answers the same
entries as JSON for the correctors it is asked for, each row and column
carrying its plane and index rather than taking the files' order, and writes
nothing
(:ref:`executor-measurement-tools`). There, as for every kind, an unsolved
corrector has no column and is listed under ``unsolved``, and the call is
refused only when no corrector solves.

Model-wide scenario faults
==========================

A scenario states physics faults of a pyAT model under ``faults.<model>`` by
name, with no channel address. A model is built at them and holds them for
its whole lifetime; none is a variable of the built model. The launch and
bunch faults need a ``periodic`` model with an ``injection`` marker.

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
   * - ``launch_x``, ``launch_px``, ``launch_y``, ``launch_py``,
       ``launch_delta``, ``launch_ct``
     - any number
     - The launch at the injection marker, in pyAT's coordinates; 0 when
       unstated.
   * - ``bunch_particles``
     - whole number, 1 to 2\ :sup:`53`
     - Declares a bunch of that many particles.
   * - ``bunch_emittance_x``, ``bunch_emittance_y``, ``bunch_sigma_delta``,
       ``bunch_sigma_z``
     - 0 or more
     - The bunch's emittances (m·rad), relative energy spread and length (m).
   * - ``centroid_jitter_x``, ``centroid_jitter_y``
     - 0 or more
     - The standard deviation of the per-shot centroid jitter, in metres.
   * - ``bunch_charge``
     - above 0, default 1
     - The charge one bunch carries.
   * - ``stored_charge``
     - 0 or more, default 0
     - The stored charge at build and after ``reset()``.

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
the ones a physics model's wiring describes as ``turn_series``, whether the
render serves the model or not: the turn series and the armed flags. Those hold
the samples of one shot and have no value at an instant between shots. Command
channels are archived like any other setpoint. Archive seeding and the live
archiver recorder both take their channels from this one set. The mock archiver
answers a request for a channel of the facility outside it with
``ChannelNotArchivedError`` (``<address> has no archived history: it is a shot
record, read it live after a shot``), and an address outside the facility with
``ChannelNotFoundError``.

Sum levels
==========

``sum`` reads the share of the injected charge alive at the monitor, plus
stored beam, on a floor of 0.03, scaled by the readout and the record's
calibration. The calibration is the monitor's charge-to-sum factor ``k``,
declared in the machine as ``gain: 1/k``: the record reads ``k`` times the
physics sum, and a reader multiplies a raw sum by the gain to compare monitors.
The example's factors are log-normal with ``σ = 0.2`` and span about 0.6 to
1.9, so equal charge reads up to three times apart between monitors. A reader that separates beam from no beam compares a rise
against a fraction of the shot's largest rise, never against an absolute
level.

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
