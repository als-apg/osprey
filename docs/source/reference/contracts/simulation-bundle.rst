.. _reference-simulation-bundle:

==========================
Simulation Bundle Contract
==========================

A simulated machine is a set of JSON files a facility writes: one
``machine.json`` that defines every simulated channel, and one directory per
scenario that says how the machine departs from its baseline. This page is the
shape of those files, how scenarios compose, and every refusal the loader and
``osprey sim apply`` raise.

The config key ``control_system.connector.<type>.simulation_file`` locates the
bundle, where ``<type>`` is ``control_system.type`` (``mock`` when unset). A
type with no key of its own falls back to
``control_system.connector.mock.simulation_file``. A relative path resolves
against the project root; the shipped presets set both keys to
``data/simulation/machine.json``. Three readers load the bundle: the mock
control-system and archiver connectors, the simulation engine inside the Virtual
Accelerator container, and the ``osprey sim`` commands.

.. _simulation-bundle-layout:

Layout
======

.. code-block:: text

   data/simulation/
     machine.json               channels (required)
     scenarios/
       <name>/
         scenario.json          overrides, archiver events, physics (required)
         logbook.json           the scenario's logbook narrative (optional)
         <any subpath>          pictures logbook entries attach (optional)

- A scenario is an immediate *subdirectory* of ``scenarios/``, and the
  directory name is the scenario name. Files directly inside ``scenarios/`` are
  ignored.
- A subdirectory without a ``scenario.json`` is refused.
- When no ``nominal/`` directory exists, a ``nominal`` scenario is supplied with
  the description "All systems nominal." and no overrides.
- When ``scenarios/`` exists, a ``scenarios`` block inside ``machine.json`` is
  **ignored**. The inline form is read only when the directory is absent, and
  an inline scenario carries no logbook.

``lattice.json``, ``va_bindings.json`` and the ``standin_bpm_errors`` key of
``machine.json`` sit beside the bundle but are Virtual Accelerator inputs,
outside this page; see :doc:`/architecture/virtual-accelerator`.

The directory is build-owned: the build renders it from the profile's ``data/``
tree and checksums it. The one file that changes at run time, the active
scenario set, therefore lives under the agent-data root instead
(:ref:`simulation-bundle-composition`).

.. _simulation-bundle-machine:

``machine.json``
================

The top-level keys the loader reads are ``channels`` (required, a mapping of
channel name to channel entry), ``name`` and ``description`` (optional
strings), and ``default_scenarios`` (optional, a list of scenario names the
bundle defines; see :ref:`simulation-bundle-composition`). Other top-level keys
are not read by the loader.

.. list-table:: Channel entry
   :header-rows: 1
   :widths: 18 22 60

   * - Key
     - Type, default
     - Rule
   * - ``value``
     - number or string
     - The baseline. Exactly one of ``value`` and ``expr`` is required. A
       string value makes a string-valued channel.
   * - ``expr``
     - string
     - A derived channel, computed from other channels
       (:ref:`simulation-bundle-expressions`).
   * - ``units``
     - string, ``""``
     - Reported with every read.
   * - ``description``
     - string, ``""``
     - Reported with every read.
   * - ``noise``
     - number ≥ 0, ``0``
     - Relative sigma: a read is multiplied by ``1 + N(0, noise)``. Being
       multiplicative, it has no effect on a ``0.0`` baseline.
   * - ``noise_abs``
     - number ≥ 0, ``0``
     - Additive sigma, in the channel's units.
   * - ``min``, ``max``
     - number, none
     - Physical bounds. When both are given, ``min`` must be less than
       ``max``.
   * - ``texture``
     - mapping, none
     - Slow baseline motion. Exactly the keys ``kind``, ``amplitude`` and
       ``period_s``: ``kind`` is ``wander``; ``amplitude`` (in the channel's
       units, the bound of the motion) and ``period_s`` (the slowest period, in
       seconds) are numbers greater than 0.

``noise_abs``, ``min``, ``max`` and ``texture`` are refused on a string-valued
channel. ``texture`` is the only closed block: an unknown key inside it is
refused, while any other unrecognised key of a channel entry is ignored. A
channel whose baseline is ``0.0`` with a relative ``noise`` and neither
``noise_abs`` nor ``texture`` loads, but reads as a flat constant; the loader
names such channels once in its log. That is a warning, not a refusal.

A live read of a numeric channel starts from its effective value: a value
written during the session, else the active scenario's override, else the
baseline. It then adds the texture, applies the relative noise, adds the
absolute noise, and clamps into ``min``/``max``. The clamp applies to what is
read out, never to what is stored: overrides and writes are kept as given.
Synthesized archiver history builds on the baseline and the active scenario's
archiver events rather than on overrides, so a scenario that should show in
history as well as in live reads declares an archiver event for the channel.

.. _simulation-bundle-expressions:

Expressions
-----------

An ``expr`` is an arithmetic expression over other channels. The grammar is:

- int and float literals;
- the binary operators ``+``, ``-``, ``*``, ``/`` and ``**``, unary ``+`` and
  ``-``, and parentheses;
- the functions ``abs``, ``min``, ``max``, ``sqrt`` and ``exp``, each with at
  least one argument and no keyword arguments;
- ``ch('NAME')``, the value of channel ``NAME``, with exactly one string
  literal.

No other names, keywords or syntax are accepted. At load, a reference to a
channel the file does not define is refused, and so is any reference cycle,
reported as ``Expression reference cycle detected: A -> B -> A``.

Evaluation errors are raised when the channel is *read*, not at load: a
division by zero, a domain error such as ``sqrt`` of a negative number, and a
reference to a string-valued channel. A file whose ``expr`` references a string
channel ``S`` loads, and the read raises ``Channel 'D': Channel 'S' holds a
string value and cannot be used in an expression``.

A live read resolves each reference at that channel's effective value, clamped
to its bounds, without its texture or noise; the derived channel's own texture,
noise and bounds then apply. Synthesized history of a derived channel is
computed from its references' synthesized series, so it carries their texture
and noise.

.. _simulation-bundle-scenario:

``scenario.json``
=================

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Rule
   * - ``description``
     - String, ``""`` when absent. Shown by ``osprey sim list``.
   * - ``overrides``
     - Mapping of channel name to a number or string. Replaces the channel's
       baseline in live reads while the scenario is active.
   * - ``archiver``
     - List of ``{"channel": <name>, "events": [...]}`` entries; ``events``
       defaults to an empty list (:ref:`simulation-bundle-events`).
   * - ``physics``
     - Deploy-time lattice faults for the Virtual Accelerator
       (:ref:`simulation-bundle-physics`).

Every channel named under ``overrides`` or ``archiver`` must be defined in
``machine.json``. Unrecognised keys are ignored, so a note to the reader can sit
in a key of its own such as ``_comment``. The type of an override is not checked
against the channel's: a number overriding a string-valued channel loads, and
keeping the two consistent is the author's responsibility.

.. _simulation-bundle-events:

Event scripts
=============

Each archiver event is a mapping with a ``shape``, the value keys that shape
needs, and one position.

.. list-table:: Shapes
   :header-rows: 1
   :widths: 14 22 64

   * - ``shape``
     - Value keys
     - Effect
   * - ``step``
     - ``to``
     - Every sample from the event's position on takes the value ``to``.
   * - ``ramp``
     - ``to``, plus the until-key of its position style
     - Runs linearly from the series value at its position to ``to``, then
       holds ``to``. An until at or before the position behaves as a step.
   * - ``spike``
     - ``amplitude``, ``width``
     - Adds a Gaussian bump of height ``amplitude`` and sigma ``width``
       (greater than 0), centred on the position.

``to`` and ``amplitude`` are numbers. The events of one channel apply in list
order, and the channel's texture and noise ride on the series the events leave.
On a string-valued channel only ``step`` is accepted, and its ``to`` becomes the
string.

.. list-table:: Positions (exactly one per event)
   :header-rows: 1
   :widths: 16 84

   * - Key
     - Meaning
   * - ``at_offset``
     - Seconds from the apply-time anchor T0, any sign (negative is the past).
       A ramp pairs it with ``until_offset``; a spike's ``width`` is in
       seconds.
   * - ``at_when``
     - ``{"days_ago": <int ≥ 0>, "time": "HH:MM:SS"}``, the logbook's own
       ``when``: the calendar day ``days_ago`` days before T0's date, at
       ``time``, in the facility time zone. An event and the entry narrating it
       resolve to one instant whatever time of day T0 falls at. ``step`` and
       ``spike`` only; a spike's ``width`` is in seconds.
   * - ``at_time``
     - ``"HH:MM:SS"`` with no zone offset, recurring daily: the event fires at
       that time of day on every calendar date inside the window read, in the
       facility time zone (``system.timezone``, UTC when unset). ``step`` and
       ``spike`` only; a spike's ``width`` is in seconds.
   * - ``at``
     - A fraction in ``[0, 1]`` of whatever window a reader asks for. A ramp
       pairs it with ``until``, also a fraction; a spike's ``width`` is a
       window fraction. A ramp that mixes the fraction and offset flavours is
       refused.

T0 is written as an ``anchor=<ISO8601>`` line into the state file by ``osprey
sim apply``, which takes the current time unless ``--now`` or the
``OSPREY_SIM_NOW`` environment variable freezes it (a value with no zone offset
takes the facility time zone). The same T0 places telemetry events, the
logbook's ``when`` values and the archive rewrite. When the state file holds no
anchor line, the engine takes the state file's modification time, and with no
state file the current time.

.. important::

   ``osprey sim apply`` refuses an ``at`` event in any active scenario unless
   the archive rewrite is skipped with ``--no-seed-archiver`` or ``--no-seed``.
   The refusal holds on every simulation-backed project, whether or not it has
   a stored archive: a fraction names a position in a reader's window, not an
   instant that can be written. The mock archiver can still draw an ``at``
   event at read time, but a bundle meant to be applied uses ``at_offset``,
   ``at_when`` or ``at_time``, and the shipped bundles are held to that.

.. _simulation-bundle-physics:

``physics``
===========

A ``physics`` block seeds lattice faults into the Virtual Accelerator. The mock
connectors do not read it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Rule
   * - ``bpm_errors``
     - Mapping of device id to an error spec, every field optional:
       ``offset`` (m, default 0), ``gain`` (factor, default 1), ``polarity``
       (``1`` or ``-1``, default 1), ``roll`` (rad, default 0) and ``noise``
       (m, ≥ 0, default 0).
   * - ``corrector_gain``
     - Mapping of device id to a calibration factor; 1.0 is nominal.

Device ids are lattice device ids such as ``BPM17`` or ``HCM01``, not channel
names, and the loader does not check them against anything. Each ``bpm_errors``
field except ``roll`` applies to **both** transverse planes: ``offset`` becomes
the horizontal and the vertical offset, and so on for ``gain``, ``polarity``
and ``noise``.

A physics fault is deploy-time only. ``osprey sim apply`` writes it into the
deployment's ``.env`` as ``VA_BPM_ERRORS`` and ``VA_CORR_GAIN``, and the
Virtual Accelerator reads those when its container is created, so the fault
takes effect at the next ``osprey up``, not at the next poll of the state file;
see :doc:`/how-to/control-systems/use-virtual-accelerator`.

The container, not the loader, refuses a device its served bindings cannot
resolve, a fault on a deployment that serves no lattice, and a value outside
its bounds: a BPM gain from 0.1 to 10, a roll within ±0.1 rad, and a corrector
factor of magnitude at most 5.

.. _simulation-bundle-logbook:

``logbook.json``
================

A JSON array of entries, each a mapping:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Rule
   * - ``entry_id``
     - Non-empty string, required.
   * - ``when``
     - ``{"days_ago": <int ≥ 0>, "time": "HH:MM:SS"}``, required; ``time``
       carries no zone offset.
   * - ``author``, ``title``, ``text``
     - Strings, required.
   * - ``tags``, ``categories``
     - Lists of strings, default empty.
   * - ``loto_tag``
     - String or ``null``, default ``null``.
   * - ``extra``
     - Mapping, default empty. Merged into the stored entry's metadata last,
       so it can overwrite the ``title``, ``tags``, ``categories`` and
       ``loto_tag`` stored there.
   * - ``attachments``
     - List, default empty. Each item is ``{"path": "<relative path>"}`` with
       no other key, naming a ``.png``, ``.jpg``, ``.jpeg``, ``.gif`` or
       ``.webp`` file inside the scenario directory whose bytes are that
       format.

``when`` resolves to the calendar day ``days_ago`` days before T0, at ``time``,
in T0's time zone. A ``days_ago: 0`` entry whose ``time`` is later than T0 lands
in the future.

Applying scenarios purges the logbook and reseeds it from every active
scenario's entries in activation order, ``nominal`` first. Entries are upserted
by ``entry_id``, so when two active scenarios reuse an id only the later one's
entry is kept. Seeding needs an ``ariel:`` block in the project config; without
one it is skipped, and ``--no-seed-logbook`` or ``--no-seed`` skips it too.

Each picture an entry attaches is stored in ARIEL's own attachment store and
linked on the entry, with the rendition ``attachment_view`` returns. Seeding runs
no enhancement module, so a picture's caption, and with it the entry's
searchability by what the picture shows, arrives only once ``osprey ariel
enhance`` or the ingestion poller runs with a vision model configured.

.. _simulation-bundle-composition:

Composition
===========

``nominal`` is always active and always first. The other names keep the order
given, with duplicates dropped.

A scenario *touches* every channel it names under ``overrides`` and every
channel it names in an ``archiver`` entry; an entry whose ``events`` list is
empty **still touches its channel**. No two active scenarios may touch one
channel. Because ``nominal`` is always active, a channel ``nominal`` touches is
off limits to every other scenario.

``physics`` has a rule of its own: no two active scenarios may name one device
under the same field.

``osprey sim apply`` records the active set in ``simulation/active_scenarios``
under the agent-data root (``agent_data.base_dir``, ``var/agent_data`` when
unset, so ``var/agent_data/simulation/active_scenarios``). It writes the file
whole, by rename: an optional ``anchor=`` line, then one scenario name per line.
Blank lines and lines starting with ``#`` are skipped. Every reader re-reads the
file when it changes. On the mock connectors, writing it clears the values
written during the session.

A deployment with no state file has never chosen a set. ``osprey up`` then
activates the machine's ``default_scenarios`` the way ``osprey sim apply`` would,
before it seeds the archive and the logbook, so both carry that set's history
and narrative. A machine with no ``default_scenarios`` runs ``nominal`` alone.
The names are resolved then, as ``osprey sim apply`` resolves its arguments: a
default the bundle does not define, or a set that does not compose, leaves the
deployment on ``nominal`` with a warning.
Once the file exists, ``osprey up`` leaves it alone, and ``osprey sim apply``
always means exactly the set it names (``osprey sim apply nominal`` included).
On a deployment whose logbook already holds entries, the activated set's
entries are not added; the deploy warns and names the ``osprey sim apply``
command that reseeds them.

.. _simulation-bundle-refusals:

Refusals
========

Every message is quoted up to its first variable part; ``<name>``, ``<ch>``,
``<id>`` and ``<key>`` stand for the scenario, channel, device and key the
message names.

.. list-table:: Load: every reader, on engine construction and in ``osprey sim list``, ``status`` and ``apply``
   :header-rows: 1
   :widths: 34 66

   * - Condition
     - Message begins
   * - ``machine.json`` is not valid JSON
     - ``Machine file <path> is not valid JSON:``
   * - ``machine.json`` has no ``channels`` mapping
     - ``Machine file <path> must define a 'channels' mapping``
   * - A channel entry is not a mapping
     - ``Channel '<ch>': entry must be a mapping``
   * - Neither or both of ``value`` and ``expr``
     - ``Channel '<ch>': exactly one of 'value' or 'expr' is required``
   * - ``value`` is not a number or string
     - ``Channel '<ch>': 'value' must be a number or string``
   * - ``expr`` is not a string
     - ``Channel '<ch>': 'expr' must be a string``
   * - ``expr`` outside the grammar
     - ``Channel '<ch>':`` followed by the construct refused, for example
       ``Function 'log' is not allowed in``
   * - Bad ``noise``
     - ``Channel '<ch>': 'noise' must be a non-negative number``
   * - Bad ``noise_abs``
     - ``Channel '<ch>': 'noise_abs' must be a non-negative number``
   * - ``min`` or ``max`` not a number
     - ``Channel '<ch>': 'min' must be a number`` (or ``'max'``)
   * - ``min`` not less than ``max``
     - ``Channel '<ch>': 'min' (``
   * - ``noise_abs``, ``min``, ``max`` or ``texture`` on a string channel
     - ``Channel '<ch>': '<key>' is not supported on string-valued channels``
   * - ``texture`` not a mapping
     - ``Channel '<ch>': 'texture' must be a mapping``
   * - ``texture`` with an unknown key
     - ``Channel '<ch>': 'texture' has unknown keys``
   * - ``texture`` missing a key
     - ``Channel '<ch>': 'texture' missing keys``
   * - ``texture`` ``kind`` not ``wander``
     - ``Channel '<ch>': 'texture' kind must be one of ['wander']``
   * - ``texture`` ``amplitude`` or ``period_s`` not greater than 0
     - ``Channel '<ch>': 'texture' <key> must be a number > 0``
   * - ``expr`` references an undefined channel
     - ``Channel '<ch>': expression references unknown channel``
   * - Reference cycle
     - ``Expression reference cycle detected:``
   * - Inline ``scenarios`` block not a mapping
     - ``'scenarios' must be a mapping of scenario name to definition``
   * - Scenario directory without ``scenario.json``
     - ``Scenario bundle '<name>' is missing scenario.json``
   * - ``scenario.json`` or ``logbook.json`` is not valid JSON
     - ``Scenario bundle '<name>': invalid scenario.json:`` (or
       ``invalid logbook.json:``)
   * - ``logbook.json`` is not an array
     - ``Scenario bundle '<name>': logbook.json must be a JSON array``
   * - Scenario not a mapping
     - ``Scenario '<name>': definition must be a mapping``
   * - Override for an undefined channel
     - ``Scenario '<name>': override for unknown channel``
   * - Override not a number or string
     - ``Scenario '<name>': override for '<ch>' must be a number or string``
   * - Archiver entry not a mapping
     - ``Scenario '<name>': archiver entries must be mappings``
   * - Archiver entry for an undefined channel
     - ``Scenario '<name>': archiver events for unknown channel``
   * - ``physics`` not a mapping
     - ``Scenario '<name>': 'physics' must be a mapping``
   * - ``bpm_errors`` or ``corrector_gain`` not a mapping
     - ``Scenario '<name>' physics: 'bpm_errors' must be a mapping`` (or
       ``'corrector_gain'``)
   * - A device id that is not a non-empty string
     - ``Scenario '<name>' physics: 'bpm_errors' keys must be non-empty device
       id strings`` (or ``'corrector_gain'``)
   * - A ``bpm_errors`` spec not a mapping
     - ``Scenario '<name>' physics: bpm_errors['<id>'] must be a mapping``
   * - ``polarity`` not ``1`` or ``-1``
     - ``Scenario '<name>' physics bpm_errors['<id>']: 'polarity' must be 1 or
       -1``
   * - Another ``bpm_errors`` field not a number, or ``noise`` negative
     - ``Scenario '<name>' physics bpm_errors['<id>']: '<key>' must be``
   * - A ``corrector_gain`` factor not a number
     - ``Scenario '<name>' physics: corrector_gain['<id>'] must be a number``
   * - Logbook entry not a mapping
     - ``Scenario '<name>' logbook: each entry must be a mapping``
   * - Bad ``entry_id``
     - ``Scenario '<name>' logbook: 'entry_id' must be a non-empty string``
   * - ``when`` not a mapping
     - ``Scenario '<name>' logbook entry '<id>': 'when' must be a mapping``
   * - Bad ``days_ago``
     - ``Scenario '<name>' logbook entry '<id>': 'days_ago' must be a
       non-negative integer``
   * - Bad ``time``
     - ``Scenario '<name>' logbook entry '<id>': 'when.time'``
   * - ``author``, ``title`` or ``text`` not a string
     - ``Scenario '<name>' logbook entry '<id>': '<key>' must be a string``
   * - ``tags`` or ``categories`` not a list of strings
     - ``Scenario '<name>' logbook entry '<id>': '<key>' must be a list of
       strings``
   * - Bad ``loto_tag``
     - ``Scenario '<name>' logbook entry '<id>': 'loto_tag' must be a string or
       null``
   * - ``extra`` not a mapping
     - ``Scenario '<name>' logbook entry '<id>': 'extra' must be a mapping``
   * - ``default_scenarios`` not a list of names
     - ``'default_scenarios' must be a list of scenario names``
   * - ``attachments`` not a list, or an item not a mapping
     - ``Scenario '<name>' logbook entry '<id>': 'attachments' must be a list``
       (or ``each attachment must be a mapping``)
   * - An attachment with a key other than ``path``
     - ``Scenario '<name>' logbook entry '<id>': attachment has unknown keys``
   * - Attachment ``path`` empty or not a string
     - ``Scenario '<name>' logbook entry '<id>': attachment 'path' must be a
       non-empty string``
   * - Attachment ``path`` absolute or leaving the scenario directory
     - ``Scenario '<name>' logbook entry '<id>': attachment path '<path>' must
       be relative to the scenario directory``
   * - Attachment suffix not a picture format
     - ``Scenario '<name>' logbook entry '<id>': attachment '<path>' is not a
       picture``
   * - Attachment file missing
     - ``Scenario '<name>' logbook entry '<id>': attachment file '<path>' not
       found``
   * - Attachment bytes not the suffix's format
     - ``Scenario '<name>' logbook entry '<id>': attachment '<path>' does not
       hold``
   * - Event not a mapping
     - ``Scenario '<name>', channel '<ch>': event must be a mapping``
   * - Unknown ``shape``
     - ``Scenario '<name>', channel '<ch>': event shape must be one of``
   * - A value key missing, or a ramp's until-key missing
     - ``Scenario '<name>', channel '<ch>': '<shape>' event missing keys``
   * - Not exactly one position key
     - ``Scenario '<name>', channel '<ch>': event requires exactly one of 'at'``
   * - Ramp positioned by ``at_time`` or ``at_when``
     - ``Scenario '<name>', channel '<ch>': 'ramp' events do not support
       'at_time'`` (or ``'at_when'``)
   * - Ramp mixing fraction and offset keys
     - ``Scenario '<name>', channel '<ch>': 'ramp' event must not mix fraction
       and offset position keys``
   * - ``at_time`` not a string, not a valid time of day, or with a zone offset
     - ``Scenario '<name>', channel '<ch>': event key 'at_time'``
   * - ``at_when`` not a mapping
     - ``Scenario '<name>', channel '<ch>': 'at_when' must be a mapping``
   * - ``at_when`` with a bad ``days_ago``
     - ``Scenario '<name>', channel '<ch>': 'days_ago' must be a non-negative
       integer``
   * - ``at_when`` with a bad ``time``
     - ``Scenario '<name>', channel '<ch>': 'at_when.time'``
   * - A shape other than ``step`` on a string channel
     - ``Scenario '<name>', channel '<ch>': '<shape>' events are not supported
       on string-valued channels``
   * - A position, until, ``to``, ``amplitude`` or ``width`` value not a number
     - ``Scenario '<name>', channel '<ch>': event key '<key>' must be a
       number``
   * - ``at`` or ``until`` outside ``[0, 1]``
     - ``Scenario '<name>', channel '<ch>': event key '<key>' must be between
       0 and 1``
   * - ``width`` not greater than 0
     - ``Scenario '<name>', channel '<ch>': event key 'width' must be a number
       > 0``

``osprey sim apply`` checks the requested set in this order and refuses before
anything is written: the logbook, the archive, the state file and ``.env`` are
all left as they were.

.. list-table:: ``osprey sim apply``, before anything is written
   :header-rows: 1
   :widths: 34 66

   * - Condition
     - Message begins
   * - A requested scenario is not defined
     - ``Unknown scenario '<name>'. Available:``
   * - Two active scenarios touch one channel
     - ``Channel '<ch>' is touched by both``
   * - Two active scenarios name one physics device under the same field
     - ``physics.<field>['<id>'] is declared by both``
   * - An active scenario has an ``at`` event and the archive rewrite is not
       skipped
     - ``Archiver event <event> is positioned by window fraction ('at')``
   * - The project configures no simulation file
     - ``Project <dir> has no mock 'simulation_file' configured`` (or ``has no
       simulation_file configured for control_system.type``)

.. list-table:: Virtual Accelerator container boot
   :header-rows: 1
   :widths: 34 66

   * - Condition
     - Message begins
   * - A physics device the served bindings cannot resolve
     - ``FATAL: VA_BPM_ERRORS:`` (or ``FATAL: VA_CORR_GAIN:``)
   * - A physics fault on a deployment serving no lattice
     - ``FATAL: VA_BPM_ERRORS/VA_CORR_GAIN are lattice-physics faults``
   * - A value outside its bounds
     - ``FATAL: VA_BPM_ERRORS entry`` (or ``FATAL: VA_CORR_GAIN entry``)

Some conditions are not refused. A name in a hand-edited state file that no
scenario defines is logged and ignored; a set that does not compose is logged,
and the engine falls back to ``nominal`` alone; a malformed ``anchor=`` line is
logged and ignored. The one supported way to change the active set is ``osprey
sim apply``.

.. _simulation-bundle-example:

Worked example
==============

A machine with one vacuum gauge, a derived margin and a valve, and one
scenario that raises the pressure, closes the valve in history and records
why.

.. code-block:: json
   :caption: data/simulation/machine.json

   {
     "name": "Example ring",
     "description": "One vacuum sector of a storage ring.",
     "channels": {
       "RING:VAC:SECTOR01:PRESSURE": {
         "value": 1.2,
         "units": "nTorr",
         "description": "Sector 1 ion gauge",
         "noise_abs": 0.02,
         "min": 0.0,
         "texture": {"kind": "wander", "amplitude": 0.05, "period_s": 3600}
       },
       "RING:VAC:SECTOR01:MARGIN": {
         "expr": "max(0.0, 10.0 - ch('RING:VAC:SECTOR01:PRESSURE'))",
         "units": "nTorr",
         "description": "Headroom below the 10 nTorr interlock"
       },
       "RING:VAC:SECTOR01:VALVE:STATE": {
         "value": "OPEN",
         "description": "Sector 1 gate valve"
       }
     }
   }

.. code-block:: json
   :caption: data/simulation/scenarios/vacuum-leak/scenario.json

   {
     "description": "A small leak in sector 1: pressure is up and the gate valve is closed.",
     "overrides": {
       "RING:VAC:SECTOR01:PRESSURE": 4.8
     },
     "archiver": [
       {
         "channel": "RING:VAC:SECTOR01:PRESSURE",
         "events": [
           {"shape": "spike", "at_offset": -7200, "amplitude": 3.5, "width": 900}
         ]
       },
       {
         "channel": "RING:VAC:SECTOR01:VALVE:STATE",
         "events": [
           {"shape": "step", "at_offset": -7200, "to": "CLOSED"}
         ]
       }
     ]
   }

.. code-block:: json
   :caption: data/simulation/scenarios/vacuum-leak/logbook.json

   [
     {
       "entry_id": "VAC-001",
       "when": {"days_ago": 1, "time": "22:15:00"},
       "author": "alice",
       "title": "Sector 1 pressure rise",
       "text": "Sector 1 gauge reading high; gate valve closed until the leak is found.",
       "tags": ["vacuum"]
     }
   ]
