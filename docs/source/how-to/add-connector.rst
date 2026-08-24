Add a Connector
===============

**What you'll build:** Control system connectors for accessing hardware abstraction layers

Overview
--------

The Control System Integration system provides a **two-layer abstraction** for working with control systems and archivers. This enables development and R&D work using mock connectors (without hardware access) and migration to production by changing a single configuration line.

**Capabilities:**

- **Mock Mode**: Work with any channel names without hardware access
- **Production Mode**: EPICS and DOOCS ship in-tree; LabVIEW, Tango, and other stacks via user-registered custom connectors
- **One API**: the same code works with mock and production connectors
- **Custom connectors**: register your own via ``ConnectorFactory``

**Built-in Connectors:**

- **mock** / **mock_archiver**: Development/R&D mode (no hardware access required)
- **epics** / **epics_archiver**: EPICS Channel Access / Archiver Appliance (production)
- **virtual_accelerator**: the PyAT Virtual Accelerator's EPICS soft-IOC — behaves
  like ``epics`` but tracks setpoints through the simulated machine, so plans
  actually run (the mock connector can't do that); see :doc:`use-virtual-accelerator`
- **mongodb_archiver**: MongoDB time-series archiver
- **doocs** / **doocs_archiver**: DOOCS properties and DOOCS local histories
  (DESY, European XFEL). Both require ``doocs4py``, which is supplied by the
  DOOCS environment rather than installed from PyPI — the import is deferred to
  ``connect()``, so the names register everywhere and only fail where a DOOCS
  environment is genuinely absent.


Quick Start: Using Connectors
-----------------------------

Mock Mode (Development & R&D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory

   # Create mock connector - works with ANY channel names
   connector = await ConnectorFactory.create_control_system_connector({
       'type': 'mock',
       'connector': {
           'mock': {
               'response_delay_ms': 10,
               'noise_level': 0.01
           }
       }
   })

   channel_value = await connector.read_channel('ANY:MADE:UP:NAME')
   print(f"Value: {channel_value.value} {channel_value.metadata.units}")

   # A state channel (EPICS mbbi/bi/bo, PVAccess NTEnum) reads as its integer
   # state index, with the state names alongside it:
   mode = await connector.read_channel('SR:DIAG:MODE')
   print(f"{mode.value} means {mode.metadata.enum_label}")   # e.g. 2 means ACQUIRING
   print(mode.metadata.enum_labels)  # ['OFFLINE', 'STANDBY', 'ACQUIRING', 'FAULT']

   await connector.disconnect()

Production Mode (EPICS)
~~~~~~~~~~~~~~~~~~~~~~~

Switch to real hardware by changing ``type`` in ``config.yml``:

.. code-block:: yaml

   # Mock (default, for development):
   control_system:
     type: mock
     connector:
       mock: { response_delay_ms: 10, noise_level: 0.01 }

   # Production:
   control_system:
     type: epics
     connector:
       epics:
         gateways:
           # EPICS uses one process-wide CA context, so the connector points at a
           # single gateway. When control_system.writes_enabled is true and a
           # write_access gateway is set, writes route through it; otherwise the
           # connector uses read_only (so a read-only deployment rejects writes at
           # the network layer as well).
           read_only:   { address: cagw.facility.edu, port: 5064 }
           write_access: { address: cagw.facility.edu, port: 5084 }
         timeout: 5.0

The Python API is identical -- only the config changes.

**Archiver configuration** uses a parallel ``archiver:`` block. Switch from the mock
archiver (synthetic data) to the EPICS Archiver Appliance the same way:

.. code-block:: yaml

   # Mock archiver (default, for development):
   archiver:
     type: mock_archiver

   # Production:
   archiver:
     type: epics_archiver
     epics_archiver:
       url: https://archiver.facility.edu:8443   # required
       timeout: 60                                # seconds, default 60

.. note::

   Write operations require explicit opt-in. See :ref:`write-safety-config` below for the
   ``writes_enabled`` setting that controls write permissions.

PVAccess Channels (PVA)
~~~~~~~~~~~~~~~~~~~~~~~

The ``epics`` connector speaks Channel Access by default. Some data -- camera
frames above all -- is served only over pvAccess, EPICS' second protocol. List
those addresses as glob patterns under ``pva_channels`` and the connector reads
them through pvAccess, while every other address keeps using Channel Access. One
deployment covers both:

.. code-block:: yaml

   control_system:
     type: epics
     connector:
       epics:
         timeout: 5.0
         pva_channels:
           - "*:IMAGE*"
           - "BL:CAM?:ARRAY"
         pva_gateway:
           address: pvagw.facility.edu
           port: 5075               # default
           use_name_server: false
         gateways:
           read_only: { address: cagw.facility.edu, port: 5064 }

Patterns are matched with ``fnmatch.fnmatchcase`` -- case-sensitive, and the
same on every operating system. Addresses stay exactly as you write them: no
``pva://`` prefix is ever added, so the limits database, channel databases and
audit records keep matching the strings they already use. An absent or empty
``pva_channels`` list is a complete no-op -- the connector behaves exactly as it
did before and never even imports the pvAccess client.

``pva_gateway`` is a single flat block, unlike the ``read_only`` /
``write_access`` pair on the Channel Access side, because pvAccess is read-only
here. The block is only consulted when ``pva_channels`` is non-empty, and it becomes
``EPICS_PVA_ADDR_LIST`` in the form ``address:port`` -- or
``EPICS_PVA_NAME_SERVERS``, same form, when ``use_name_server`` is true, which
makes the client connect over TCP instead of searching by UDP broadcast (that is
what an SSH tunnel needs). Whenever the block is present,
``EPICS_PVA_AUTO_ADDR_LIST`` is forced to ``"NO"`` -- the same containment the
Channel Access side applies, so a deployment deliberately pinned to one gateway
cannot also broadcast-discover servers on the local subnet.

**PVA channels are read-only.** A write to an address matching a
``pva_channels`` pattern comes straight back as a blocked
``ChannelWriteResult`` naming the reason, with nothing sent on the network. That
is a deliberate safety decision rather than an unfinished feature: a supervised
pvAccess write path is separate work with its own review, and until it exists a
refusal is the honest answer.

pvAccess **RPC services** are refused too, and for a different reason. Code the
agent runs through the Python executor cannot call ``Context.rpc(...)``: it is
blocked at runtime and no approval can let it through, because an rpc payload
carries an arbitrary request -- there is no way to tell a read from a write, and
nothing for limits checking to validate. Use ``channel_read`` and
``channel_write`` for the operation you actually need.

A camera frame is almost always too large to return to the agent as raw
numbers; see `Reading Large Values`_ below for what comes back instead.

.. note::

   Compressed NTNDArray frames are not decoded. When a camera's areaDetector
   pvAccess plugin is configured with a codec (blosc, lz4, jpeg), the read fails
   with an error naming the codec and the remedy -- disable pvAccess compression
   for that channel -- rather than reshaping a compressed blob into a
   plausible-looking image full of meaningless statistics.

.. note::

   The pvAccess client (``p4p``) ships with OSPREY as an ordinary dependency, so
   there is normally nothing to install. The exception is a bare-metal install on
   arm64 Linux, where ``p4p``, ``pvxslibs`` and ``epicscorelibs`` publish no
   wheels and pip builds them from source: that host needs a C toolchain
   (a compiler and ``make``) present. OSPREY's own project images already stage
   one, so containerized deployments are unaffected.

MongoDB Archiver
~~~~~~~~~~~~~~~~

For facilities that store time-series PV data in MongoDB rather than EPICS Archiver
Appliance, configure the archiver block independently of the control-system choice:

.. code-block:: yaml

   archiver:
     type: mongodb_archiver
     mongodb_archiver:
       host: mongodb.facility.edu
       port: 27017
       name: archiver_db
       collection: pv_data
       auth: admin
       username: readonly
       password_env: MONGODB_READONLY_PASSWORD

Documents in the collection are expected to have a ``date`` field (``ISODate``) and
one or more PV names as top-level fields: ``{date: ISODate(...), PV1: value1, PV2:
value2, ...}``. A query matches any document that carries **at least one** of the
requested PVs (an ``$or`` across per-PV ``$exists`` checks) -- documents do not need
to carry every requested PV together, so channels archived at different cadences, or
written into separate documents by different collectors, are still returned
correctly, each on its own timestamp series. The connector's MongoDB client ships
with OSPREY, so there is nothing extra to install.

Production Mode (DOOCS)
~~~~~~~~~~~~~~~~~~~~~~~

DOOCS facilities select both connectors by name. Channel addresses are DOOCS
properties (``FACILITY/DEVICE/LOCATION/PROPERTY``), and the control-system
connector needs no options -- it reads its environment from the DOOCS
installation:

.. code-block:: yaml

   control_system:
     type: doocs

   archiver:
     type: doocs_archiver
     doocs_archiver:
       avg_window: 20    # optional moving average, in samples

The archiver reads DOOCS *local histories*, so it only makes sense alongside
``type: doocs``. Both connectors need ``doocs4py``, which the DOOCS environment
provides rather than PyPI; without it, ``connect()`` fails with a clear
``ImportError`` instead of silently degrading.

.. note::

   DOOCS supports the ``none`` and ``readback`` write-verification levels.
   ``callback`` is accepted but has no DOOCS equivalent, so it performs a
   readback and reports the level as ``readback``.


Reading Large Values
--------------------

Array-valued channels -- camera frames, waveforms, orbit vectors -- are often
far too big to hand to the agent as raw numbers. ``channel_read`` applies a size
rule to every read, whatever protocol answered it: values within an element
budget come back inline as JSON lists, and anything larger is saved to the
artifact gallery and reported as a summary plus a handle to it. A Channel Access
waveform takes exactly the same path as a pvAccess camera frame.

.. code-block:: yaml

   control_system:
     read_inline_max_elements: 2000        # per-value element budget
     channel_read_artifact_retention: 20   # readings kept per channel

``read_inline_max_elements`` (default 2000) is the per-value budget. The default
is the same inline budget OSPREY already plots against, which keeps ordinary
waveforms and orbit vectors inline while camera frames take the artifact path.
Only arrays are measured -- strings and single scalars are always inline,
however long they are.

One ``channel_read`` call also has an aggregate budget of four times that
number, spent in request order, so a batch of individually small arrays cannot
add up to a flood. Every withheld value says which of the two limits it hit, as
``artifact_reason``:

``per_value_threshold``
    The value on its own is over ``read_inline_max_elements``. It will never
    come back inline.

``aggregate_budget``
    The value is small enough by itself, but earlier channels in the same
    request had already spent the call's budget. Reading this channel in a
    smaller batch returns it inline.

The summary that replaces the value reports shape, dtype and element count, plus
min/max/mean for numeric data, along with the artifact's id and its
``data_file`` path. What gets saved depends on the shape:

- **1-D** -- an interactive chart in the artifact gallery, with the values as
  JSON in ``data_file``. The x axis is the sample index, not wall-clock time:
  one read of an array carries a single timestamp.
- **2-D and 3-D with a color axis** -- a PNG preview as the gallery image, with the raw array
  beside it as a ``.npy`` file (``data_file``). The preview is auto-scaled:
  brightness is relative to that frame's own minimum and maximum, so it carries
  no absolute units. ``data_file`` is the authoritative copy of the values.
- **Layouts with no honest rendering** -- four dimensions and up, or a 3-D stack
  with no color axis -- are still saved as a loadable ``.npy``, just without a
  preview image.

The agent reaches the numbers by loading ``data_file`` inside ``execute``:
``numpy.load(data_file)`` returns the array with its original shape and dtype,
and ``json.load`` opens a 1-D series file. If the artifact store cannot write at
all, the read is still reported as successful, with an ``artifact_error`` field
in place of the handle -- the machine did answer, and reporting the read as
failed would be a false alarm.

``channel_read_artifact_retention`` (default 20) bounds how far these readings
pile up: only the newest N **unpinned** artifacts per channel are kept, and
older ones are pruned as new readings are saved. Pinning a reading exempts it --
a pinned entry is never pruned and never occupies a slot in the window either.
Set the key to ``0`` to keep everything, remembering that an unattended polling
loop will then grow the gallery without bound.

Write Verification
------------------

All ``write_channel()`` calls return :class:`~osprey.connectors.control_system.base.ChannelWriteResult`:

.. code-block:: python

   connector = await ConnectorFactory.create_control_system_connector()

   result = await connector.write_channel("BEAM:CURRENT", 100.0)

   if result.verification and result.verification.verified:
       print(f"Write confirmed ({result.verification.level})")
   else:
       print(f"Verification failed: {result.verification.notes}")

   # Override verification level
   result = await connector.write_channel(
       "MOTOR:POSITION", 50.0,
       verification_level="readback",
       tolerance=0.1
   )

**Verification levels:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Level
     - Speed
     - Confidence
     - When to Use
   * - ``none``
     - Instant
     - Low
     - Development, non-critical writes
   * - ``callback``
     - Fast (~1-10ms)
     - Medium
     - Most production writes (default)
   * - ``readback``
     - Slow (~50-100ms)
     - High
     - Critical setpoints, safety-critical operations

**Configuration (global default):**

.. code-block:: yaml

   control_system:
     write_verification:
       default_level: "callback"
       default_tolerance_percent: 0.1   # interpreted as percent

**Per-channel configuration** (in limits database):

.. code-block:: json

   {
     "defaults": {
       "writable": true,
       "verification": { "level": "callback" }
     },
     "MOTOR:POSITION": {
       "min_value": -100.0,
       "max_value": 100.0,
       "max_step": 2.0,
       "writable": true,
       "verification": {
         "level": "readback",
         "tolerance_absolute": 0.1
       }
     }
   }

``tolerance_absolute`` takes priority over ``tolerance_percent`` (percentage of value).
Each channel inherits any field it does not set from the ``defaults`` block, and a
channel's own value always overrides it. ``writable`` defaults to ``true``; a channel's
verification falls back to the ``defaults`` block's verification. Set ``"writable": false``
-- on a channel, or in ``defaults`` to lock everything down by default -- to block writes.

**Where the level comes from.** Passing no ``verification_level`` means "no opinion",
not "no verification". The connector then resolves the level for that channel through
four layers, using the first one that supplies a value:

1. the channel's own ``verification`` entry in the limits database,
2. the limits database ``defaults.verification`` block,
3. ``control_system.write_verification`` in ``config.yml``,
4. the built-in fallback: ``callback``, with no tolerance.

The tolerance resolves through the same four layers, and it keeps doing so when the
level was passed explicitly, but a layer only supplies a tolerance when the level it
resolves to is ``readback``. Asking for ``readback`` on a channel whose limits entry
declares ``level: readback`` with ``tolerance_percent: 1.0`` verifies at 1%, not at
the built-in default -- pass ``tolerance`` as well to override that too.

Batch writes resolve the same way. ``write_multiple_channels()`` carries one level for
the whole batch, so when the caller omits it the keyword is left off each per-channel
write and every channel resolves its own. This is why ``osprey.runtime.write_channels``
verifies exactly like ``osprey.runtime.write_channel``.

.. _write-safety-config:

Write Safety Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Write operations are disabled by default and must be explicitly enabled at two levels:

**Global write permission** (in ``config.yml``):

.. code-block:: yaml

   control_system:
     writes_enabled: true          # Master switch for all write operations

If ``writes_enabled`` is omitted, it defaults to ``false`` and all writes are blocked.

``writes_enabled`` is a **launch-time deployment posture, not a live kill-switch.**
It is read from config and process-cached, so flipping it in ``config.yml`` does not
take effect in a running process. The enforced kill-switch lives at the harness layer
(a renderer ``permissions.deny`` on the write tool, then regenerate and relaunch the
agent); in-flight control of an active plan is the RunEngine's own ``abort`` / ``pause``.

The connector applies **per-write mechanical safety** — the ``writes_enabled`` gate,
limits validation, and the fail-closed validation path — on every Channel Access put.
This is a separate, complementary layer from the **per-intent human authorization**
enforced at the tool boundary (the PreToolUse approval hook, and the launch token for
plans), which gates the *intent* to write once per intent rather than once per put.
The approval layer cannot substitute for the connector's mechanical refusal.

.. _limits-checking-config:

Limits Checking
~~~~~~~~~~~~~~~

Automatic safety-limit validation for write operations:

.. code-block:: yaml

   control_system:
     limits_checking:
       enabled: true                     # Enable limits validation
       database_path: ./limits_db.json   # Path to the channel limits JSON
       allow_unlisted_channels: false    # Block writes to channels not in the database

When enabled, every ``write_channel()`` call is validated against the limits database
before the write is sent to hardware. See per-channel configuration above for the
database format.

.. seealso::

   :class:`~osprey.connectors.control_system.base.ChannelValue`
       Channel read result data model

   :class:`~osprey.connectors.control_system.base.ChannelWriteResult`
       Complete write operation result

   :class:`~osprey.connectors.control_system.base.WriteVerification`
       Verification result data model


Implementing Custom Connectors
------------------------------

Subclass :class:`~osprey.connectors.control_system.base.ControlSystemConnector` and implement the abstract methods: ``connect``, ``disconnect``, ``read_channel``, ``write_channel``, ``read_multiple_channels``, ``subscribe``, ``unsubscribe``, ``get_metadata``, ``validate_channel``.

You may also override the non-abstract ``write_multiple_channels()`` method if your backend benefits from atomic batch writes (e.g., disabling lattice recalculation between writes in a simulator). The default implementation writes sequentially via ``write_channel()``.

**The verification arguments are optional.** Declare them as::

   async def write_channel(
       self,
       channel_address: str,
       value: Any,
       timeout: float | None = None,
       verification_level: str | None = None,
       tolerance: float | None = None,
   ) -> ChannelWriteResult:

``None`` is a sentinel meaning "the caller did not specify a level" -- deliberately
different from the ``"none"`` level, which means "write, and do not verify". When
``verification_level`` arrives as ``None``, hand the channel and the value to the
inherited ``_get_verification_config()`` helper, which walks the four layers described
under `Write Verification`_ and returns the level and tolerance to use.

A connector written before the sentinel existed -- one declaring a concrete default such
as ``verification_level: str = "callback"`` -- keeps working unchanged. The batch path
omits the keyword rather than forwarding ``None``, so that connector's own default still
applies.

**Three verification fields are optional too.**
:class:`~osprey.connectors.control_system.base.WriteVerification` accepts, beyond the
level and the readback value:

- ``readback_alarm_status`` -- the alarm name the readback carried, as the control
  system spells it (for EPICS, ``"NO_ALARM"``, ``"HIHI"``, and so on).
- ``readback_alarm_severity`` -- the numeric severity: ``0`` healthy, ``1`` MINOR,
  ``2`` MAJOR, ``3`` INVALID.
- ``failure_kind`` -- a short tag for *why* verification failed, when the reason is
  more specific than "the value did not match". A readback that raised is reported as
  ``"readback_failed"``; a plain value mismatch leaves the field unset.

Leaving a field at ``None`` means "not reported", which is deliberately different from a
severity of ``0`` ("reported, and healthy"). A connector whose backend has no alarm
concept leaves all three unset and remains fully supported.

**Two metadata fields are enum-only.**
:class:`~osprey.connectors.control_system.base.ChannelMetadata` accepts, beside units
and precision:

- ``enum_labels`` -- every state the channel can report, as a list of names in index
  order (EPICS ``ZRST``..``FFST``, a PVAccess NTEnum's ``choices``).
- ``enum_label`` -- the name of the state *this* reading is in.

They belong to enum-typed records only (``mbbi``/``mbbo``/``bi``/``bo``, ``NTEnum``,
and whatever your control system calls the same thing) and stay ``None`` everywhere
else -- which is how a consumer tells a state channel from a numeric one. Report the
**integer index** as ``ChannelValue.value`` and put the name here, rather than making
the value a string: the reading then has one machine-readable type whichever protocol
served it, and the operator-facing name is carried rather than lost. Populate them
best-effort -- a control system that cannot report the label list, or an index outside
it, leaves the fields unset and the read still succeeds.

.. note::

   Archiver connectors are a different contract: historical enum samples arrive as
   **strings** in the ``value`` column and must not be coerced (see `Archiver
   Connectors`_ below). A live read reports the index plus these labels; an archived
   series reports what the archiver stored.

Your connector must return the standard data models from ``osprey.connectors.control_system.base``: :class:`~osprey.connectors.control_system.base.ChannelValue`, :class:`~osprey.connectors.control_system.base.ChannelMetadata`, :class:`~osprey.connectors.control_system.base.ChannelWriteResult`, and :class:`~osprey.connectors.control_system.base.WriteVerification`.

Archiver Connectors
~~~~~~~~~~~~~~~~~~~

.. versionchanged:: Unreleased

   ``get_data`` returns long-format data (below) instead of a shared-index wide
   ``DataFrame``. Out-of-tree connectors written against the old contract must be updated.

Subclass :class:`~osprey.connectors.archiver.base.ArchiverConnector` and implement
``connect``, ``disconnect``, ``get_data``, ``get_metadata``, ``check_availability``.

``get_data`` is the entire contract. It returns a **long-format** ``pandas.DataFrame``
with exactly three columns, sorted by ``channel`` then ``timestamp``:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Column
     - Dtype
     - Contents
   * - ``timestamp``
     - ``datetime64[ns, UTC]``
     - When the sample -- or, under a ``processing`` mode, the bin's aggregate --
       occurred.
   * - ``channel``
     - ``str``
     - The channel/PV name the row belongs to.
   * - ``value``
     - not dtype-constrained
     - ``float64`` when every requested channel's samples are numeric; pandas'
       natural mixed dtype (typically ``object``) once any channel is non-numeric.

An empty result is an empty frame with these same three columns (``value`` defaults
to ``float64``, since there is no data to infer a dtype from).

**Nothing is manufactured.** Channels are never placed on a shared index. Each
channel contributes only its own real samples -- never forward-filled, never
reindexed onto a regular grid, never padded with a row for a bin or timestamp
nothing was actually recorded at. A channel with no data in the requested range
simply contributes no rows; it never appears as an all-NaN column. Connector
correctness bugs trace back to violating this rule, so hold to it strictly: if a
custom connector finds itself building a shared ``DatetimeIndex`` and reindexing
per-channel series onto it, that is the bug.

**Per-channel aggregation.** ``get_data`` takes a trailing ``processing: str =
"raw"`` keyword -- one of ``raw``, ``mean``, ``min``, ``max``, ``median``, ``std``,
``count`` -- applied independently to each channel's own real samples, never across
channels and never onto a shared grid:

- ``raw`` decimates each ``precision_ms`` bin down to its **last real sample**,
  keeping that sample's own true timestamp -- never a timestamp invented at the
  bin's edge to hold it. This matches the EPICS Archiver Appliance's long-standing
  ``lastSample_N`` semantics, and every in-tree backend applies it the same way.
- Every other mode aggregates the real samples that landed in each ``precision_ms``
  bin. A bin with no samples is dropped, not emitted as ``NaN`` -- so a sparse
  channel returns *fewer* rows than it has samples, never more, and no bin-width
  floor is ever needed to avoid upsampling.
- ``precision_ms <= 0`` means full resolution: every real sample, undecimated. It is
  only valid with ``processing="raw"`` -- an aggregate has no bin to aggregate over,
  and requesting one must raise ``ValueError`` rather than silently falling back to
  raw.
- Aggregating a non-numeric channel with anything but ``raw`` must raise
  ``ValueError`` naming the channel -- never coerce it, drop it, or silently emit
  ``NaN``. Backends that bin client-side get this from ``aggregate_series``; a
  backend that pushes the aggregation to its server must call
  ``reject_non_numeric`` on what comes back, since it never reaches
  ``aggregate_series``.
- A bin width your backend cannot express must raise ``ValueError``, never round
  to one it can. The EPICS Archiver Appliance's operator syntax takes whole
  seconds, so that connector rejects any positive ``precision_ms`` that is not a
  multiple of 1000 rather than serving a different resolution than was asked
  for.

The shared helpers in ``osprey.connectors.archiver._timerange`` (``to_utc``,
``require_datetime``, ``resolve_processing``, ``long_frame``, ``decimate_raw``,
``aggregate_series``, ``reject_non_numeric``)
implement all of the above and are the easiest way to get it right -- every in-tree
connector (EPICS, MongoDB, DOOCS, mock) builds on them rather than reimplementing
binning.

**Why the ``value`` dtype rule matters.** Enum/status channels -- machine mode,
interlock state, RF state, anything archived as EPICS ``mbbi`` or DOOCS
``DBR_STRING`` -- carry string values, not numbers. ``get_data`` never coerces them:
a channel's own dtype flows straight through, and only combining a non-numeric
channel with a numeric one in the same query promotes the shared ``value`` column to
a mixed dtype. A custom connector must resist forcing ``value`` to ``float64`` "for
consistency" -- doing so silently corrupts every enum/status channel it touches.

This is deliberately not the live-read contract, where an enum reads as its index
and the label travels in ``enum_label``. An archiver reports what its backend
recorded, and what these backends recorded is the string; converting one form into
the other would require a label list the archive does not carry.

Query windows must also be normalized to UTC before touching the wire: a naive
(timezone-less) ``start_date``/``end_date`` is facility-local, matching how the rest
of the framework reads operator wall-clock times, and must be converted -- not
relabeled -- to your backend's UTC wire format. ``to_utc()`` in
``osprey.connectors.archiver._timerange`` does this.

Registering Custom Connectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Direct registration** (simplest approach):

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory

   ConnectorFactory.register_control_system("tango", TangoConnector)

After registration, use ``type: tango`` in ``config.yml`` and the factory will instantiate
your connector automatically.

**Registry-based registration** (for packaging as a reusable extension):

.. code-block:: python

   from osprey.registry.base import ConnectorRegistration
   from osprey.registry.helpers import extend_framework_registry

   registration = ConnectorRegistration(
       name="labview",
       connector_type="control_system",
       module_path="my_package.connectors.labview_connector",
       class_name="LabVIEWConnector",
       description="LabVIEW Web Services connector for NI systems",
   )

   config = extend_framework_registry(connectors=[registration])

**Dotted-module-path** (no registration call needed):

.. code-block:: yaml

   control_system:
     type: my_package.connectors.tango_connector.TangoConnector

When ``type`` contains a dot, the factory imports the module via ``importlib`` and
instantiates the named class directly -- useful for one-off custom connectors that
don't need a registry entry.

Testing Custom Connectors
~~~~~~~~~~~~~~~~~~~~~~~~~

Test in three phases:

1. **Capability logic** -- use ``type: mock`` connector, no hardware needed.
2. **Interface compliance** -- instantiate your connector against a local simulator.
3. **Integration** -- mark with ``@pytest.mark.integration``; run against real hardware.

Switch connectors via environment variables in ``conftest.py``:

.. code-block:: python

   @pytest.fixture
   def connector_config():
       if os.getenv('USE_REAL_CONNECTOR') == '1':
           return {'type': 'epics', 'connector': {'epics': {}}}
       return {'type': 'mock', 'connector': {'mock': {}}}
