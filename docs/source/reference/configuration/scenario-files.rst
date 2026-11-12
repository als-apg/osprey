.. _reference-scenario-files:

==============
Scenario files
==============

A scenario file is one YAML file under ``data/facility/scenarios/``; the file
name is the scenario name. This page lists what the file may hold. Applying
and writing scenarios is in :ref:`how-to-run-scenarios`.

Keys
====

Every key is optional.

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

Select readings to still
========================

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

Archive events
==============

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

.. _scenario-channel-faults:

Channel faults
==============

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

A channel fault holds in both venues of the simulator
(:ref:`va-two-venues`). ``offline``, a read that never answers, is not one of
these faults.

On an injection chain (:ref:`va-injection-chain`) a ``stuck`` or
``disconnected`` request starts no shot, and a ``frozen`` reading raises no
monitor event while the fault is active. Applying a scenario mid-shot
abandons the shot: the chain restarts at the new scenario's start state.
