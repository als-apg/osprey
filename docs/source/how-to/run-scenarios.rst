.. _how-to-run-scenarios:

=======================
Run and write scenarios
=======================

A scenario is one way the simulated machine departs from its baseline: values
written into its models, faults in a physics model, the history its archive
shows, and the logbook entries that narrate it. Scenarios are authored under
``data/facility/scenarios/``, one YAML file each, and ``osprey build`` serves
them from the simulator view under ``build/data/simulator/``. The
``osprey sim`` commands list, apply and report on them; their options are in
:ref:`cli-osprey-sim`.

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
       investigation, each a temperature rise, a reflected-power spike and a
       forward-power trip, the last one a beam dump. The archive shows the
       excursions, the logbook narrates the dump, the investigation and the
       cooling repair, and cavity 1 reads 26.5 degC live. Cavity 2 shows one
       minor excursion for contrast.
   * - ``rf-thermal-live``
     - Live thermal wander on cavity 1: one slow shared driver moves its
       temperature, tuner, reflected and forward power together, each with
       its own noise, for correlation plots. It adds no archived history and
       does not compose with ``rf-thermal``, which writes the same channels.
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

.. code-block:: bash

   osprey sim list
   osprey sim apply vacuum-burst
   osprey sim status

``osprey sim list`` marks the active set with ``*`` and says which scenarios
carry a logbook narrative. ``osprey sim apply`` replaces the active set with
the names it is given, plus ``nominal``, then purges and reseeds the ARIEL
logbook and rewrites the affected windows of the stored archive, so live
values, history and narrative tell one story. It asks before the purge and
the rewrite; ``--yes`` skips the prompts and ``--no-seed`` leaves both stores
untouched.

Several scenarios apply together only when no two of them write one target,
a channel or a model's fault. A set that breaks the rule is refused before
anything is written, and the refusal names the target. Archive events are not
a target: events of several scenarios on one channel combine, so
``[vacuum-burst, rf-thermal]`` composes.

Every facility has a scenario named ``still`` that stills every float reading:
``osprey sim apply still``, or ``simulation.default_scenarios: [still]`` for a
deployment that should start still. A file may define its own ``still``,
which replaces the built-in one. ``still`` composes with overrides and faults,
and is refused beside a scenario that sets the same reading's ``noise`` or
``couple``. It holds in both venues of the simulator.

Any scenario can still a family of readings by selecting it over the facility
file's own records. Each selection names a model or a place, and the build
resolves it to plain addresses:

.. code-block:: yaml

   still:
     - {model: <model>, role: monitor}
     - {place: <place>, signal: <signal>}
     - <address>

- ``model``: channels a wiring record of that model names, or on a device of
  that model.
- ``place``: channels on that place or on a place under it.
- ``group``: channels on a device of that group.
- ``signal``: channels stating that ``signal``.
- ``role``: channels whose wiring record has that role (``setpoint``,
  ``readback``, ``monitor`` or ``output``), not the channel's own ``role``.
- ``plane``: channels whose wiring record has that plane.

The keys of one selection all apply. The build stops on a selection that names
neither a model nor a place, and on one that matches no float channel; a
matched channel that is not a float is left out.

Write a scenario
================

Add ``data/facility/scenarios/<name>.yaml``; the file name is the scenario
name. Every key is optional:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Meaning
   * - ``description``
     - What the scenario shows. ``osprey sim list`` prints it.
   * - ``overrides``
     - ``{<address>: <value>}``, the value each channel reads live while the
       scenario is active.
   * - ``faults``
     - ``{<model>: {<address or engine variable>: <value>}}``, faults written
       into a physics model; a value is a scalar or a map
       ``{<fault field>: <value>}`` such as ``{polarity: -1}``.
   * - ``channel_faults``
     - ``{<address>: stuck | frozen | disconnected}``, faults of the channel
       itself, on any simulated channel, wired to a model or not; the table
       below says what each does.
   * - ``archiver``
     - ``[{channel, events}]``, the history each channel's archive shows.
       Each event has a ``shape``, ``step`` (``to``), ``ramp`` (``to``) or
       ``spike`` (``amplitude`` and ``width``, a Gaussian sigma in seconds),
       and one position: ``at_offset`` (seconds from the apply-time anchor; a
       ramp ends at ``until_offset``), ``at_time`` (a daily ``HH:MM:SS`` in the
       facility time zone) or ``at_when`` (``{days_ago, time}``), which places
       an event at the instant a logbook entry's ``when`` names. ``at_time``
       and ``at_when`` take ``step`` and ``spike`` only. A ``step`` on a float
       channel may end with ``until``, ``until_offset`` or ``until_when``,
       matching its position and later than it; after the end the channel
       returns to its baseline. A daily ``at_time`` step does not end.
   * - ``logbook``
     - The entries the scenario narrates, each with ``entry_id``, ``when``
       (``{days_ago, time}``), ``author``, ``title`` and ``text``, and
       optionally ``tags``, ``categories``, ``loto_tag``, ``extra`` and
       ``attachments``.
   * - ``drivers``, ``couple``
     - Slow shared signals, and the channels that follow them.
   * - ``noise``
     - Per-channel noise while the scenario is active, replacing the seed's:
       ``{<address>: {absolute: <sigma>}}`` with the sigma in the channel's
       unit, or ``{<address>: {relative: <fraction>}}`` with the sigma a
       fraction of the reading.
   * - ``still``
     - ``all``, or a list of addresses and selections (see `Apply a scenario`_):
       while the scenario is active, those readings serve without drift,
       couplings or noise; readout faults still apply.
   * - ``sessions``
     - Past agent sessions, ``[{slug, title, when, parts}]``; each part (a
       ``/clear`` starts the next) has a ``title`` and ``turns``. A turn has
       ``prompt`` and ``steps`` (``{tool: {server, name}, input, result,
       approved?}`` or ``{answer}``), and optionally ``at_offset_s``,
       ``channels``, ``logbook_ids`` and ``artifacts``. ``channels`` must be facility channels
       and ``logbook_ids`` entries of ``nominal`` or the scenario itself, or
       the build stops. The scenario must also state, under ``archiver`` and
       ``logbook``, the events its sessions tell of, so a resumed session that
       reads the archive sees the same story.

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

A disconnected setpoint's readback is the disconnected reading; a setpoint
that is its own readback reads its demand and reports ``udf`` itself. A
channel fault holds in both venues of the simulator; ``offline``, a read that
never answers, is not one of them.

An entry's ``attachments`` lists its pictures, each ``{path: <picture file>}``
or ``{plot: <plot spec .json>}``, relative to the folder
``data/facility/scenarios/<name>/`` beside the file. Run ``osprey build``, or
``osprey facility validate`` to check the sources without writing anything;
an address, model or key the facility does not have stops the build and names
the file. The new scenario then appears in ``osprey sim list``.
