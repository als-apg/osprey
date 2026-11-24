.. _how-to-describe-your-facility:

======================
Describe your facility
======================

A deployment describes its facility once, as source files under
``data/facility/``. ``osprey build`` combines them into one facility file,
``build/facility.json``, and writes every file a service reads from it: the
channel finder's index, the limits database, the simulator's channel list, the
knowledge graph. Nothing else in the deployment lists a channel. The machine
facts can instead live in a
:doc:`machine repository </how-to/control-systems/machine-repository>`.

This page walks the tree a fresh ``osprey init --preset demo``
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
       relations.yaml          claims that two records are related
     classes.yaml              device classes the facility adds
     signals.yaml              signal roles the facility adds
     models.yaml               the simulation models and their wiring
     decks/                    the decks those models name
     limits.yaml               the write limits
     seeds.yaml                how a channel no model wires behaves when simulated
     measurement/<model>.yaml  the measurements a model allows
     scenarios/<name>.yaml     ways the simulated machine departs from its baseline
     documents.yaml            the source documents records and claims cite
     policy.yaml               how the build ranks competing claims
     fixes.yaml                corrections to records another source states
     imported/<layer>/         records an importer wrote
     knowledge/                the facility knowledge pages
     ariel/vocabulary.yml      the logbook shorthand ARIEL expands
     benchmarks/               the channel-finder benchmark queries

Every file is optional. A deployment with no ``data/facility/`` at all builds
a facility with no channel, and each file added states more. The
demo preset ships all of them except ``signals.yaml``, ``fixes.yaml`` and
``imported/``; the hello-world preset ships
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

The files under ``records/`` each hold a list of records of one kind.
Every record has an ``id``, and a record states only what its author knows:
a field left out is either filled by the build or stays absent.

**Places** are the facility's own tree. A place's id is its path,
``SR/SECT1`` under ``SR``; ``level`` is the facility's word for that depth.
A place with a ``span`` covers a stretch of one model's deck, from the
``from_marker`` element up to the ``to_marker`` element, or to the deck's end
without one. A device the deck puts inside it belongs to the deepest such
place, unless the device states its own ``place`` in an authored record or a
fix. A place only an importing layer states (``place_from: imported``) yields
to a span that is that place or lies inside it, and stands under a span that
holds it or where none holds the device; any other span holding the device
contradicts it, and the build stops ``place-conflict``. A device still without
a place takes the one its ``located_in`` claims name (``place_from:
relation``) when no claim on its key is ``contested`` and the accepted claims
that are not coarse all name one place; the claim's unit is not copied.

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
address means anything to OSPREY. A setpoint may be a command: a channel
written to start something, such as a reset, rather than to hold a value. The
build marks it ``command: true`` when its ``signal`` is a command word or a
model's engine marks it (:doc:`/reference/contracts/facility-kinds`). Its
readback is its status, the channel that reports when what the command started,
its act, is over; the fields ``idle`` and ``timeout_s`` say what the status reads
while nothing runs and how long an act may take, so a write of the command can
wait until the act is over. What a channel is comes from its fields:

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
       ``options`` and ``shape`` (a command and its status need not); else it
       pairs the setpoint with itself. A setpoint and readback whose signals
       name one quantity must agree in those three, stated or derived; a
       command and the status it reads back on, or a pair whose quantity no
       signal states, may differ. Several setpoints may name one readback (a
       set and a reset command read back on one channel). The simulation
       drives a readback from its setpoint only where exactly one setpoint
       names it, the two agree in type and the readback states no ``idle``;
       any other readback it drives from no setpoint. The readback's seed ``follows`` changes that for one pair
       (see Seeds).
   * - ``tolerance``
     - On a float setpoint, how close its readback must come before a move
       counts as done: ``{absolute: <x>}`` in the channel's unit or
       ``{relative: <fraction>}`` of its seed's ``nominal``. The setpoint and
       its readback must both state a ``unit``. An address under a wiring
       record's ``also`` settles within the tolerance of the record's
       ``address`` and states none of its own.
   * - ``idle``
     - On a readback a command pairs with, its status: what it reads while no
       act runs, typed by the channel: a label on a bool or enum channel, an
       integer on an int, a number on a float, text on a string; a waveform
       channel is never a status. Several commands may share one status. A status that states ``idle``
       is never followed: ``follows`` naming a setpoint on it stops the build
       with ``command-invalid``.
   * - ``timeout_s``
     - On a command whose status states ``idle``: the longest one act may
       take, in seconds. A command and its status state the two together.
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
       display precision on both Channel Access and PVAccess, and serves a
       float channel that states none with 4.
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
together; what a channel means stays on the channel. Its ``kind`` is
``family`` (the default: devices of one sort, described by the group) or
``section`` (things that belong together, such as a vacuum section). Only a
family is a channel-finder Family and gives its members their purpose text.

**Relations** are claims, one per source, that two records are related: a
``type``, a ``subject`` and an ``object``, each naming one device, place or
group. The build writes each claim's ``standing``: ``accepted``, ``contested``
when claims state different objects and none ranks first, or ``superseded``;
and ``contests``, the ids of the claims it disagrees with, when there are any.
A claim stops the build with ``relation-invalid``, naming the claim, on any of
seven conditions:

* an end names no device, place or group;
* an end names more than one of them;
* an end gives ``unit`` but names no place;
* an end names a kind the type does not allow there;
* ``basis: inferred`` comes without its ``rule``;
* a required attribute is missing;
* the subject and object name the same end, port included.

``policy.yaml`` ranks competing claims by the kind of document they cite, as
declared in ``documents.yaml``: ``relations.precedence`` lists kinds per
relation type or as ``default``. Between claims whose documents rank the same,
a stated claim breaks the tie over an inferred one unless
``stated_over_inferred: false``. Without a policy, competing claims stay
``contested``.

Claims reach the knowledge graph as edges between their ends, by standing: an
accepted claim is a plain edge, a contested one a separate contested edge, a
superseded one no edge. Each claim is also a node carrying its source, so a
query can go from an edge to the document behind it
(:ref:`reference-facility-graph`).

``classes.yaml`` adds device classes the shared vocabulary lacks, each as the
child of an existing class:

.. code-block:: yaml

   - class: Undulator
     parent: Magnet
     aliases: [insertion device, ID]

``signals.yaml`` adds signal roles the vocabulary lacks; one it already has is
refused:

.. code-block:: yaml

   - name: taper_setpoint
     aliases: [gap taper]
     description: Commanded gap difference between an undulator's two ends.
   - name: bootloader_version_readback
     description: Bootloader version a controller reports.
     nameplate: true
   - name: trip_level_readback
     description: Level at which a protection circuit trips.
     title: Trip

``nameplate: true`` marks a signal that names the unit it is read from rather
than its state; a device card shows such a signal with what the device is,
never as a key tile. The vocabulary's own ``model_readback``,
``serial_number_readback`` and ``firmware_version_readback`` are marked so.

``title`` is the short word a screen names the signal by, on a tile or a value
row; with none, the signal's name stands. The vocabulary titles its general
signals with the operator's word: ``Current``, ``Setpoint``, ``Voltage``,
``Ramp rate``, ``On command``, ``Strength`` (a multipole's normalized
strength), ``Kick`` (a corrector's angle), and ``Lolo``, ``Lo``, ``Hi``,
``Hihi`` for the temperature alarm limits. A ``channel_roles.csv`` row's ``label`` wins over it.

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
settles it (see `Fixes`_). A group's ``members`` are the exception: the group
carries every member any layer states, and layers that state different members
get a ``members-disagree`` warning naming which source states which member,
until they agree or a ``set`` fix names ``members``. A place's or device's
``scheme_names`` are joined too, so a layer can add a name to a record another
layer states; one name in one scheme that two layers state with different
content is still a ``layer-conflict``.
A channel's seed (``simulation``, which ``seeds.yaml`` states too) is joined
per key: one layer may state ``follows`` and another the seed's values, and
only one key two layers state differently is a ``layer-conflict``.

A place, device, channel or group record may say what the fields it states rest
on. ``basis: inferred`` with a ``rule`` says a named rule derived them, not a
source; with neither, they are stated. A producer that pairs channels by their
names writes such a layer:

.. code-block:: yaml

   # data/facility/imported/names/channels.yaml
   - id: BR:DIAG:TV:01:PADDLE:SP
     pair: BR:DIAG:TV:01:PADDLE:RB
     basis: inferred
     rule: an SP channel pairs with the RB channel of its name

The two words are no fields of the record: the build keeps them in the
provenance entry of that layer's file. Screens mark such a field inferred,
unless another layer states the same value outright. A record's fields all rest
on one basis, so a layer that states some fields and infers others writes them
as two layers.

The importers are ``osprey facility import mml``,
``osprey facility import list`` and ``osprey facility import relations``.

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

3. **Review the mapping.** Replace every ``null``. Four top-level blocks are
   required: ``models`` (what each exported system is called as a model),
   ``section_order``, ``families`` and ``directions`` (whether each field is
   ``read`` or ``write``). Each family states its ``aliases``,
   ``description``, ``provenance``, ``channels`` and ``fields``, and may
   state a ``class``, its ``devices``, a ``rename`` and the ``branch`` a new
   class extends. Three top-level blocks are optional: ``facility`` (the
   identity the first import seeds), ``branches`` (device classes the
   mapping adds, each with its ``parent``) and ``judgments``. Where the
   export leaves a family's shape open, ``judgments:`` carries one slot per
   question: a channel row beyond the family's devices (``drop``, ``device``
   or ``{field: <name>}``), a device no channel binds (``drop`` or ``keep``),
   and a channel shared across devices (``keep_all`` or an owning device).

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
``identity.yaml``, ``measurement/<model>.yaml`` and ``scenarios/readout.yaml``
when they do not exist, and never rewrites them afterwards. It creates
``classes.yaml`` when absent; when the file exists, the import appends only
the classes it lacks and leaves its rows as written. It replaces a bundled
facility, saying so in one line, and refuses only hand-written records of the
kinds it writes, naming each file. That refusal guards the first import: once
an imported layer exists, records you write by hand beside it stay through
every later import, and the build combines them with the layers.

A channel list
--------------

``osprey facility import list FILE`` reads a CSV of channel addresses into
``data/facility/imported/list/``. A header row names ``address`` and any of
``role``, ``pair``, ``device``, ``place``, ``unit``, ``description``,
``tags``, ``s`` and ``model``; a file with no ``address`` column holds one
address per line.

.. literalinclude:: examples/facility/channels.csv
   :language: text
   :caption: channels.csv

The files under ``imported/list/`` belong to the import and are replaced by
each run, so they are never edited by hand: a correction goes into the CSV or
into ``fixes.yaml``. Each device a row names is written to
``imported/list/devices.yaml``, with its ``s`` where a row gives one; a
``model`` column names the model whose deck ``s`` is measured along. A
misspelt ``device`` cell makes a device of its own: when other sources declare
devices too, the import prints a line naming those only the list declares. The
list import runs the same step as every format: it replaces a bundled facility
and refuses only hand-written channel and device lists, so it still positions
a list against your own deck, models and places.

.. _how-to-describe-your-facility-relations:

Relation tables
---------------

``osprey facility import relations TABLE --layer NAME`` reads relation claims,
from a cable schedule or a rack list, into ``data/facility/imported/<NAME>/``.
``TABLE`` holds the claims; three optional tables hold what the claims lean on.
Each is UTF-8 CSV with a header row naming its columns in any order; an empty
cell is absent.

.. list-table::
   :header-rows: 1
   :widths: 16 16 22 46

   * - Table
     - Option
     - Required columns
     - Optional columns
   * - claims
     - ``TABLE``
     - ``type``, ``subject``, ``object``
     - ``subject_kind``, ``object_kind`` (``device``, ``place`` or ``group``;
       empty means ``device``), ``subject_port``, ``object_port``,
       ``object_unit``, ``attrs`` (``key=value`` pairs separated by ``;``),
       ``basis``, ``rule``, ``subject_class``, ``object_class``
   * - names
     - ``--names``
     - ``thing``, ``name``, ``scheme``
     - ``basis``, ``rule``
   * - places
     - ``--places``
     - ``place``
     - ``within``, ``basis``, ``rule``
   * - members
     - ``--members``
     - ``group``, ``member``
     - ``member_kind``, ``group_kind`` (``family`` or ``section``)

Every table also takes the citation columns ``document``, ``version``,
``state``, ``locator`` and ``imported`` (an ISO date). ``basis: inferred``
needs a ``rule``. Against the demo preset, these three tables put two booster
focusing quadrupoles on one new supply in a new rack:

.. code-block:: text
   :caption: cables.csv

   type,subject,object,object_kind,object_class,document,version,locator
   powered_by,BR/QF01,BR/QF-PS,,PowerSupply,PS-LIST,C,p. 4
   powered_by,BR/QF02,BR/QF-PS,,,PS-LIST,C,p. 4
   located_in,BR/QF-PS,RK01,place,,PS-LIST,C,p. 2

.. code-block:: text
   :caption: racks.csv

   place,within,document,version
   RK01,BR,PS-LIST,C

.. code-block:: text
   :caption: names.csv

   thing,name,scheme,document
   BR/QF-PS,PS-QF-B,cable-schedule,PS-LIST

.. code-block:: bash

   osprey facility import relations cables.csv --layer cables --places racks.csv --names names.csv

Each end and name resolves against the whole facility except the layer being
replaced. The rules:

* **Stubs.** A device end the facility lacks is created as a stub only when its
  row states ``subject_class`` or ``object_class``, a class of the facility or
  the shared vocabulary; one row's class holds for every row naming that
  device. ``--stub-unknown`` stubs a class-less end as ``Unidentified``, and
  the build reports such stubs with ``relation-stub``. Without either, the row
  is refused, so a misspelt name stops the import. Place and group ends are
  never stubbed.
* **Places.** ``within`` names an authored place or one the places table
  creates, never one another imported layer creates, so where a layer puts a
  place does not depend on import order. A place's id is its path,
  ``BR/RK01`` above, so one name may repeat under several parents. The build
  reports a place created top-level with no citation with ``place-unplaced``.
* **Members.** The members table creates a group no other source has. A row
  naming a member another source's group lacks adds it: the layer then states
  that group too, with every member its rows name.
* **Names.** The names table names any device or place. For one the authored
  file or another layer states, the layer writes a record carrying only the
  id and the names, and the build joins them with the names the other sources
  state. A name for a group is refused: a group has no ``scheme_names``.
* **Ids.** A claim's id is ``<NAME>/<type>/<hash>``. The hash covers the type,
  both ends (in either order for a symmetric type), the ports, the type's key
  attribute and the cited document, never its version, locator or
  ``imported`` date: a new revision of the schedule keeps every id. A
  symmetric row written reversed, or a repeated row, is one claim; one fact
  cited from two documents is two claims. The import never stamps a date of
  its own.

The layer is replaced whole: ``relations.yaml``, ``devices.yaml`` (the stubs),
``places.yaml`` and ``groups.yaml``, each written only when it holds a record,
and a byte copy of each table. A file of the previous run this one does not
write is removed. Nothing outside ``imported/<NAME>/`` changes: unlike the
other importers, it neither replaces a bundled facility nor refuses
hand-written records. A refused row prints one ``source-invalid`` line, every
refusal is reported and the import writes nothing. The layer name matches
``[a-z0-9][a-z0-9_-]*`` and is not ``authored``, ``list`` or ``mml``.

Fixes
=====

``fixes.yaml`` corrects a record another source states, without editing that
source. Each fix names a record and gives the reason. These two correct the
channel list above:

.. code-block:: yaml

   schema: osprey.facility.fixes/1
   fixes:
     - op: set
       kind: channel
       id: SR:MAG:QF:01:CURRENT:SP
       fields: {description: Focusing quadrupole current setpoint}
       was: {description: {list: Focusing quadrupole current}}
       why: The list's description leaves out that this is the setpoint.
     - op: add
       kind: device
       id: QF02
       record: {class: Quadrupole}
       why: Missing from the list.

``set`` replaces the fields it names, each whole; ``add`` adds a record;
``drop`` removes one. ``kind`` is ``place``, ``device``, ``channel``,
``wiring``, ``group``, ``model`` or ``relation``. A ``set`` states under ``was`` the value
each layer held when the fix was written. When ``was`` is missing, or a later
import changes that value, the build stops with ``fix-stale`` and prints the
corrected fix to paste in its place, so a fix never silently outlives the fact it corrected. The result
does not depend on the order of the fixes.

``osprey facility show <id>`` lists the fixes applied to a record.

Limits
======

``limits.yaml`` holds the write limits, one record per channel. The
demo preset ships this one:

.. literalinclude:: ../../../src/osprey/templates/facilities/example/limits.yaml
   :language: yaml
   :caption: limits.yaml
   :start-after: Delete this line to make the file yours.

A float or int setpoint record with both ``min_value`` and ``max_value`` is
writable within them; ``writable: true`` on one without both bounds stops the
build. A bool, enum, string or waveform setpoint carries no bounds and is
writable when its record says ``writable: true``. For such a channel the limits
check decides only whether it may be written; which values it takes is decided
by the control system. ``max_step`` bounds a single change; ``writable: false``
blocks every write; ``confirm: false`` turns off the re-read after a write. A
record that does not state ``confirm`` is written ``confirm: true``, and
``confirm: false`` on a command that states ``timeout_s``, whose status confirms
the write. On such a command, stating ``confirm: true`` stops the build:
``osprey_connectors.command.fire`` writes such a command with
the re-read off and confirms it by waiting for its status
(:doc:`/reference/contracts/connectors`).

``osprey build`` renders the records into ``build/data/channel_limits.json``,
the file the connector's reference monitor and the runtime's limits check
enforce on every write. That file is an output: a profile that ships its own stops the build. The file
holds the records of ``limits.yaml`` and nothing else. What happens to a
channel with no record is the deployment's choice,
``control_system.limits_checking.mode``: under ``exclusive`` the limits file
is the complete list of writable channels and a channel without a record is
refused; under ``optional`` a channel without a record is written with no
limits. The hello-world and demo presets ship ``optional``.

Seeds
=====

``seeds.yaml`` says how a channel behaves in the simulator when no model
wires it. It is keyed by address. The preset's booster current starts at
1.2 mA with Gaussian noise:

.. code-block:: yaml

   BR:DIAG:DCCT:01:CURRENT:RB:
     nominal: 1.2
     noise:
       absolute: 0.01

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
the trace is flat zero. The preset's septum current waveform:

.. code-block:: yaml

   SR:MAG:SEPTUM:01:CURRENT:WAVEFORM:
     pulse:
       driver: SR:MAG:SEPTUM:01:CURRENT:SP
       enable: SR:MAG:SEPTUM:01:STATUS:ON

``gate`` says what a channel does while an interlock-like input blocks it.
``inputs`` maps each input's address to its **blocking label**; while any input
reads its blocking label the channel takes ``gated``, and otherwise it behaves
as its other seed keys say. ``gated`` is typed by the gated channel: a factor
greater than zero on a float channel (the value is scaled by it), a label on a
bool or enum channel, an integer on an int channel. In the preset, the
sum signal of the first beam position monitor in the gun-to-linac line drops
to 1% of full beam while the gun's heater or its high voltage is off:

.. code-block:: yaml

   GTL:DIAG:BPM:01:SUM:RB:
     nominal: 1.0
     gate:
       inputs:
         GTL:GUN:EGUN:01:HEATER_ON:RB: 'FALSE'
         GTL:GUN:EGUN:01:HV_ON:RB: 'FALSE'
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

``topoff`` makes a float channel a stored current under top-off: it starts
each ``period_s`` at ``ceiling`` and decays with the lifetime ``lifetime_h``
until the next refill. ``topoff_lifetime`` makes a channel read that lifetime,
named by the channel that carries the ``topoff`` seed, so current and lifetime
always agree.

.. code-block:: yaml

   SR:DIAG:DCCT:01:CURRENT:RB:
     topoff:
       ceiling: 500.0
       lifetime_h: 7.0
       period_s: 31.0
   SR:DIAG:DCCT:01:LIFETIME:RB:
     topoff_lifetime:
       of: SR:DIAG:DCCT:01:CURRENT:RB

The channel holds the sawtooth's mean, which is what its limits and the
archive's start state see. The build stops with ``value-invalid`` when
``topoff`` sits on a channel that is not float or ``lifetime_h`` or
``period_s`` is not above zero, and with ``seed-invalid`` when ``topoff`` has
a ``nominal`` beside it or ``topoff_lifetime.of`` names no channel with a
``topoff`` seed. Live reads and the archive show the same sawtooth.

``follows`` says whether the simulation copies a setpoint's writes into this
readback. By default it does when exactly one setpoint names the readback as
its ``pair`` and the two agree in ``value_type``, ``options`` and ``shape``,
and the readback states no ``idle``.
``follows: <setpoint>`` makes the copy where the default does not, and
``follows: none`` stops it where the default makes it. A followed readback
starts at its setpoint's value and takes each write at once, with its own noise
and drift on top.

A labelled readback (``bool`` or ``enum``) may follow a setpoint of another
type through ``follows: <setpoint>``, and then takes the state of the same
index. From a labelled setpoint of as many labels, ``Clear / Set`` into
``Off / On`` copies ``Set`` as ``On``. From a setpoint that holds a number,
``1`` (or ``1.0``) holds the readback's second label, and a fraction or a number
past its labels is refused and holds nothing. The default never copies by
index: only a seed asks for it.

The build stops with ``seed-invalid`` when ``follows`` sits on a channel no
setpoint names, on a readback several setpoints name, names a setpoint whose
``pair`` is another channel, or names a setpoint it cannot copy: one of another
type into an unlabelled readback, one with a different number of labels, or one
that holds neither labels nor a number. It stops when a number setpoint's start
value is no index of the readback's labels. It stops too when a followed
readback's seed gives it a value of its own (``topoff``, ``topoff_lifetime``,
``linear`` or ``pulse``): a write to the setpoint would replace that value, so
state ``follows: none`` on it. The simulator view's ``follows.json`` lists every
pair with whether the simulation follows it (``default``, ``seed`` or
``not followed``), why, and for a followed pair how it copies (``value`` or
``index``, with the label of each index), and counts each.

.. code-block:: yaml

   SR:VAC:VALVE:01:POSITION:OPEN:
     follows: none

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

   - name: BTS
     engine: pyat
     deck: decks/BTS.json
     settings:
       pyat:
         solve: single_pass
         twiss_in:
           beta: [0.63, 9.6]
           alpha: [-0.32, 2.42]
           dispersion: [0.074, 0.018, 0.0, 0.0]
     wiring:
     - address: BTS:DIAG:BPM:01:POSITION:X
       element: BPM1
       engine:
         axis: x
       calibration:
         curve:
           linear: {gain: 0.001, offset: 0.0}
         energy_scaling: none

The ``pyat`` settings may also carry ``error_elements``, a list of
``{element: <name>, multipoles: true|false}`` naming deck elements no wiring
record binds that a scenario may still misalign or give a multipole error; see
*Element errors* in :doc:`/architecture/virtual-accelerator`.

A wiring record's ``calibration`` converts between the channel's hardware
value and the engine's physics value, as a ``linear`` gain and offset or a
``table``.

A readback record reads the setpoint record that names the same element (or
slices), engine ``attribute`` and ``index``. A readback that states the
setpoint's calibration, or none, reads the setpoint's value on the setpoint's
way back. A readback that states another calibration reads the same physics
value through its own: the setpoint's ``curve`` gives the physics value, and
the readback's ``inverse`` (or, with none, the inverse of its ``linear``
curve) gives the reading. So one quantity can be read in a second unit. The
voltage of the quadrupole's circuit, 0.1 V per ampere, is a channel of its own:

.. code-block:: yaml

   - id: SR:MAG:QF:01:VOLTAGE:RB
     unit: V
     label: focusing quadrupole circuit voltage

and a third record on the element the current setpoint and its readback are
wired to, with the strength a volt stands for as its curve:

.. code-block:: yaml

   - address: SR:MAG:QF:01:VOLTAGE:RB
     element: QF01
     engine:
       attribute: PolynomB
       index: 1
     calibration:
       curve:
         linear: {gain: 0.06006155556990854, offset: 0.0}
       energy_scaling: brho

The readback's ``curve`` runs the way every curve does, from its own unit to
the physics value; a ``table`` needs an ``inverse``. State the setpoint's
``energy_scaling`` on it unless its unit does not move with the beam energy
as the setpoint's does. Its computed ``default`` is the value it is served
with at start, and its seed's noise and drift act in its own unit.

A setpoint may drive a property of the whole deck: a record that names no
element and the engine ``attribute: energy`` is the deck's energy, set through
the curve from its unit to electron-volts. The deck is at its own energy while
that setpoint holds its ``default``, and moves from there as the curve's value
does. A readback record with the same engine block reads that setpoint, and it
too may state its own calibration. The current of a string of bending magnets
and the string's voltage, 0.5 V per ampere, are two channels:

.. code-block:: yaml

   - id: SR:MAG:BEND:CURRENT:SP
     role: setpoint
     unit: A
     label: bending magnet string current
   - id: SR:MAG:BEND:VOLTAGE:RB
     unit: V
     label: bending magnet string voltage

and two records that name no element:

.. code-block:: yaml

   - address: SR:MAG:BEND:CURRENT:SP
     engine:
       attribute: energy
     calibration:
       curve:
         linear: {gain: 4.0e+6, offset: 0.0}
       energy_scaling: none
   - address: SR:MAG:BEND:VOLTAGE:RB
     engine:
       attribute: energy
     calibration:
       curve:
         linear: {gain: 8.0e+6, offset: 0.0}
       energy_scaling: none

The voltage reads the value of the setpoint's curve at the current the
setpoint holds, through its own way back: 250 V at 500 A here. Both curves end
in the same unit, the one the setpoint's curve states, so write the readback's
curve against the setpoint's curve and not against the deck, whose energy may
differ from the curve's by a constant factor. Such a readback states
``energy_scaling: none``, as the value of the setpoint's curve does not move
with the beam rigidity; the build stops on any other with ``engine-invalid``.

The model computes a wired readback from the setpoint record on its element,
so it reads that setpoint whether or not the channels ``pair`` the two, where
several setpoints name it, and where its seed states ``follows: none``; its
``default`` is the value it is served with in each case.

Channels no served model wires are served by the texture model from
their seeds. Which models a deployment serves, and what a failed one looks like, is
in :ref:`architecture-simulation-models`.

A value served under more than one address has one wiring record. Each
further address is a channel of its own:

.. code-block:: yaml

   - id: BTS:RACK:03:UNIT:01:X
     label: BPM 01 horizontal position, as the rack unit names it

and the record lists it under ``also``:

.. code-block:: yaml

   - address: BTS:DIAG:BPM:01:POSITION:X
     also: [BTS:RACK:03:UNIT:01:X]
     element: BPM1
     engine:
       axis: x

Each address under ``also`` is a channel of ``records/channels.yaml`` with the
``role``, ``value_type``, ``options`` and ``shape`` of the record's ``address``,
and the same ``unit`` when both state one. It reads what ``address`` reads at
the same instant, with the same calibration, motion and readout noise, and a
write to it is a write to ``address``. It reads back the ``pair`` of
``address`` and settles within its ``tolerance``, and on a command it has the
time bound and the status of ``address``; ``tolerance``, ``timeout_s`` and ``idle`` are stated on ``address`` only. It
carries no seed of its own, and a limit is still stated per address. A pair is stated between
records' ``address`` names: the build stops when an address under ``also``
states a ``pair``, and when a setpoint's ``pair`` names an address under
``also``. A fault variable is named by ``address``
only (``BTS:DIAG:BPM:01:POSITION:X/offset``); a scenario block keyed by channel address
(``overrides``, ``noise``, ``couple``, ``channel_faults``, ``still``) may name
either. An address belongs to one wiring record: the build stops when two
records name it, and when two records read the same axis of one monitor.

Measurement
===========

``measurement/<model>.yaml`` states which measurements a model allows and
what they use. The demo preset ships this one for its storage
ring:

.. literalinclude:: ../../../src/osprey/templates/facilities/example/measurement/SR.yaml
   :language: yaml
   :caption: measurement/SR.yaml
   :start-after: Delete this line to make the file yours.

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

A measurement OSPREY does not ship is a kind an installed package registers
(:doc:`register-a-measurement-kind`). It is listed under ``kinds`` like the
five, may name roles of its own under ``groups`` and ``instruments``, and
takes its own settings under ``kind_blocks.<kind>``.

A model solved in a single pass allows only the kinds that run single pass;
of the shipped ones, ``orm`` alone. A group plays one role;
only ``hcor`` and ``vcor`` may name the same group. A role naming several
groups is one pyAML array named after the role. The tools
that run these measurements are in :doc:`measure-with-pyaml`.

Scenarios
=========

Each file under ``scenarios/`` is one way the simulated machine departs from
its baseline: value overrides, model and channel faults, its own motion
(drivers, couplings, noise), archive history and logbook entries. The
simulator activates the scenarios marked ``start: true`` when it first comes
up; with such marks the profile states no ``simulation.default_scenarios``.
Writing and applying them is in :ref:`how-to-run-scenarios`.

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
login and image render.

.. seealso::

   :doc:`/architecture/facility-file`
      The facility file, each view the build writes from it and what reads
      it.

   :ref:`cli-osprey-facility`
      Every exit and message of ``osprey facility validate``, ``show`` and
      the importers.
