.. _how-to-run-scenarios:

=======================
Run and write scenarios
=======================

A scenario is one way the simulated machine departs from its baseline. It can
set values in the machine's models, write faults into a physics model, shape
the history the archive shows, and add logbook entries that narrate it.

You write each scenario as one YAML file under ``data/facility/scenarios/``.
``osprey build`` serves the scenarios from the simulator view under
``build/data/simulator/``. The ``osprey sim`` commands list, apply and report
on them; :ref:`cli-osprey-sim` has their options.

The scenarios the demo ships
============================

The demo preset ships seven scenarios in
``data/facility/scenarios/``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Scenario
     - What it shows
   * - ``nominal``
     - All systems nominal. It carries the background logbook narrative of
       the weeks before the demo and is always in the active set.
   * - ``rf-thermal``
     - Three thermal excursions on cavity 1 in the week before an
       investigation. Each is a temperature rise, a reflected-power spike and
       a forward-power trip; the last one is a beam dump. The archive shows
       the excursions. The logbook narrates the dump, the investigation and
       the cooling repair. Cavity 1 reads 26.5 degC live, and cavity 2 shows
       one minor excursion for contrast.
   * - ``rf-thermal-live``
     - Live thermal wander on cavity 1, for correlation plots. One slow
       shared driver moves its temperature, tuner, reflected and forward
       power together, each with its own noise. It adds no archived history.
       It does not compose with ``rf-thermal``, which writes the same
       channels.
   * - ``vacuum-burst``
     - A vacuum burst in one sector with a correlated beam-current drop at
       14:32:08 local time, recurring daily in the archive.
   * - ``bpm-polarity``
     - BPM 17 reports an inverted reading in both planes. No single channel
       is out of range; the fault shows only as an orbit correction that does
       not converge.
   * - ``orm-dual-fault``
     - The BPM 17 inversion and corrector HCM01 at half its nominal gain at
       once, both visible only through an orbit-response measurement.
   * - ``demo-history``
     - Three past agent sessions (a sector-7 beam loss, an approved corrector
       nudge, an orbit review), with the archive events and logbook entries
       behind them.

The preset starts a fresh deployment in ``rf-thermal`` and ``demo-history``,
as the profile key ``simulation.default_scenarios`` names (see
:doc:`/reference/configuration/config`).

A scenario's sessions are laid down only by a deployment's first deploy, from
its default set. ``osprey sim apply`` never adds or removes sessions: a set
without ``demo-history`` drops its archive events and logbook entries only.

Apply a scenario
================

Applying a scenario makes it part of the *active set*: the scenarios the
simulated machine shows now. ``nominal`` is always in the active set.

.. warning::

   ``osprey sim apply`` deletes every entry in the ARIEL logbook and reseeds
   the logbook from the active scenarios. It also rewrites windows of the
   stored archive. On a virtual-accelerator deployment, those windows can
   hold samples recorded from the running machine. The command asks before
   each of these, unless you pass ``--yes``. To leave both stores untouched,
   pass ``--no-seed``.

To switch the simulated machine to the ``vacuum-burst`` scenario, run these
commands in order:

.. code-block:: bash

   osprey sim list
   osprey sim apply vacuum-burst
   osprey sim status

#. List the scenarios with ``osprey sim list``. The list marks the active set
   with ``*`` and says which scenarios carry a logbook narrative.
#. Apply the scenario with ``osprey sim apply vacuum-burst``. The command
   replaces the active set with the names you give it, plus ``nominal``. It
   then purges and reseeds the logbook and rewrites the affected windows of
   the archive. Live values, history and narrative now tell one story.
#. Check the simulator with ``osprey sim status``. It shows each served
   physics model's status and the log it writes.

Several scenarios apply together only when no two of them write the same
target, a channel or a model's fault. A set that breaks this rule is refused
before anything is written, and the refusal names the target. Archive events
are not a target: events of several scenarios on one channel combine, so
``[vacuum-burst, rf-thermal]`` composes.

Hold readings still
-------------------

Every facility has a scenario named ``still`` that stills every float
reading: the reading serves without drift, couplings or noise. Run
``osprey sim apply still``. For a deployment that should start still, set
``simulation.default_scenarios: [still]``.

A scenario file may define its own ``still``, which replaces the built-in one.
``still`` composes with overrides and faults. It is refused beside a scenario
that sets the same reading's ``noise`` or ``couple``.

``still`` holds in both *venues* of the simulator. A venue is where the
simulator runs: in its own container, or inside the process that asks (see
:ref:`va-two-venues`).

Select readings to still
------------------------

Any scenario can still a family of readings by selecting it from the facility
file's own records. Each selection names a model or a place, and the build
resolves it to plain addresses:

.. code-block:: yaml

   still:
     - {model: <model>, engine_role: monitor}
     - {place: <place>, signal: <signal>}
     - <address>

- ``model``: channels a wiring record of that model names, or on a device of
  that model.
- ``place``: channels on that place or on a place under it.
- ``group``: channels on a device of that group.
- ``signal``: channels stating that ``signal``.
- ``engine_role``: channels whose wiring record has that engine role
  (``setpoint``, ``readback``, ``monitor`` or ``output``).
- ``plane``: channels whose wiring record has that plane.

A channel must match every key of a selection. The build stops on a
selection that names neither a model nor a place. It also stops on a
selection that matches no float channel. A matched channel that is not a
float is left out.

Write a scenario
================

To add a scenario to your facility:

#. Create ``data/facility/scenarios/<name>.yaml``. The file name is the
   scenario name.
#. Add the keys the scenario needs, from the table below. Every key is
   optional.
#. If a logbook entry has attachments, put the files in the folder
   ``data/facility/scenarios/<name>/`` beside the scenario file.
#. Check the sources with ``osprey facility validate``, which writes nothing.
   An address, model or key the facility does not have stops the check, and
   the error names the file.
#. Run ``osprey build``. The new scenario appears in ``osprey sim list``.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Meaning
   * - ``description``
     - What the scenario shows. ``osprey sim list`` prints it.
   * - ``overrides``
     - ``{<address>: <value>}``, the value each channel reads live while the
       scenario is active. On an injection chain's setpoint it is the value
       the chain starts at, and an override of the request starts no shot.
   * - ``faults``
     - ``{<model>: {<address or engine variable>: <value>}}``, faults written
       into a physics model. A value is a scalar or a map
       ``{<fault field>: <value>}`` such as ``{polarity: -1}``.
   * - ``channel_faults``
     - ``{<address>: stuck | frozen | disconnected}``, faults of the channel
       itself. They work on any simulated channel, wired to a model or not.
       `Channel faults`_ says what each does.
   * - ``archiver``
     - ``[{channel, events}]``, the history each channel's archive shows.
       `Archive events`_ describes the events.
   * - ``logbook``
     - The entries the scenario narrates, each with ``entry_id``, ``when``
       (``{days_ago, time}``), ``author``, ``title`` and ``text``, and
       optionally ``tags``, ``categories``, ``loto_tag``, ``extra`` and
       ``attachments``. ``attachments`` lists the entry's pictures, each
       ``{path: <picture file>}`` or ``{plot: <plot spec .json>}``, relative
       to the folder ``data/facility/scenarios/<name>/``.
   * - ``drivers``, ``couple``
     - Slow shared signals, and the channels that follow them.
   * - ``noise``
     - Per-channel noise while the scenario is active, replacing the seed's.
       Write ``{<address>: {absolute: <sigma>}}`` for a sigma in the
       channel's unit. Write ``{<address>: {relative: <fraction>}}`` for a
       sigma that is a fraction of the reading.
   * - ``still``
     - ``all``, or a list of addresses and selections (see
       `Select readings to still`_). While the scenario is active, those
       readings serve without drift, couplings or noise; readout faults
       still apply.
   * - ``sessions``
     - Past agent sessions, ``[{slug, title, when, parts}]``. Each part has
       a ``title`` and ``turns``; a ``/clear`` starts the next part. A turn
       has ``prompt`` and ``steps`` (``{tool: {server, name}, input, result,
       approved?}`` or ``{answer}``), and optionally ``at_offset_s``,
       ``channels``, ``logbook_ids`` and ``artifacts``. ``channels`` must be
       facility channels, and ``logbook_ids`` entries of ``nominal`` or the
       scenario itself, or the build stops. The scenario must also state the
       events its sessions tell of, under ``archiver`` and ``logbook``. A
       resumed session that reads the archive then sees the same story.

Archive events
--------------

Each entry under ``archiver`` names a channel and its ``events``. Each event
has a shape and one position.

The ``shape`` is one of these:

- ``step``, with ``to``.
- ``ramp``, with ``to``.
- ``spike``, with ``amplitude`` and ``width``, a Gaussian sigma in seconds.
- ``burst``, with ``amplitude``, reached at the event and then decaying
  exponentially over ``decay_s``. An optional slower tail ``tail_decay_s``
  carries ``tail_share`` of it.
- ``loss``, with ``amount``, dropped at the event. Only the channel's top-off
  refills it.

The position is one of these:

- ``at_offset``: seconds from the apply-time anchor. A ramp ends at
  ``until_offset``.
- ``at_time``: a daily ``HH:MM:SS`` in the facility time zone.
- ``at_when``: ``{days_ago, time}``. It places the event at the instant a
  logbook entry's ``when`` names.

``at_time`` and ``at_when`` take every shape but ``ramp``.

A ``step`` on a float channel may end with ``until``, ``until_offset`` or
``until_when``. The end must match the step's position and come later than it.
After the end, the channel returns to its baseline. A daily ``at_time`` step
does not end.

Channel faults
--------------

.. list-table::
   :header-rows: 1
   :widths: 14 14 18 18 22 14

   * - Fault
     - Names
     - Setpoint read
     - Setpoint write
     - Readback or reading read
     - Severity
   * - ``stuck``
     - a setpoint
     - the value last written, its start value before any write
     - accepted, held, forwarded to nothing
     - wired: the model where it was; otherwise its held value, with no echo
     - none
   * - ``frozen``
     - a reading (a readback or a ``none`` channel)
     - n/a
     - n/a
     - the value served when the fault became active; no motion, no model
       update
     - none
   * - ``disconnected``
     - any channel
     - the value last written
     - accepted, held, forwarded to nothing
     - float: not-a-number; any other type: the value served when the fault
       became active
     - ``udf`` on the disconnected reading

A disconnected setpoint's readback is the disconnected reading. A setpoint
that is its own readback reads its demand and reports ``udf`` itself.

A channel fault holds in both venues of the simulator. ``offline``, a read
that never answers, is not one of these faults.

On an injection chain (:ref:`va-injection-chain`) a ``stuck`` or
``disconnected`` request starts no shot, and a ``frozen`` reading raises no
monitor event while the fault is active. Applying a scenario mid-shot
abandons the shot: the chain restarts at the new scenario's start state.
