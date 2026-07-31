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
  like ``epics`` but tracks setpoints through the simulated machine, so scans
  actually run (the mock connector can't do that); see :doc:`use-virtual-accelerator`
- **mongodb_archiver**: MongoDB time-series archiver (optional, ``pip install "osprey-framework[archiver-mongodb]"``)

A DOOCS connector and DOOCS archiver also ship in-tree; they are wired up via
their dotted class paths rather than a registered name.


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
correctly, each on its own timestamp series. The connector requires the optional
``archiver-mongodb`` extra:

.. code-block:: bash

   pip install "osprey-framework[archiver-mongodb]"


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
verification falls back to the ``defaults`` block's verification and then to the global
``control_system.write_verification.default_level``. Set ``"writable": false`` -- on a
channel, or in ``defaults`` to lock everything down by default -- to block writes.

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
agent); in-flight control of an active scan is the RunEngine's own ``abort`` / ``pause``.

The connector applies **per-write mechanical safety** — the ``writes_enabled`` gate,
limits validation, and the fail-closed validation path — on every Channel Access put.
This is a separate, complementary layer from the **per-intent human authorization**
enforced at the tool boundary (the PreToolUse approval hook, and the launch token for
scans), which gates the *intent* to write once per intent rather than once per put.
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
simply contributes no rows; it does not appear as an all-NaN column the way the old
wide, shared-index format required. Every connector correctness bug this contract
replaced traced back to violating this rule, so hold to it strictly: if a custom
connector finds itself building a shared ``DatetimeIndex`` and reindexing per-channel
series onto it, that is the bug.

**Per-channel aggregation.** ``get_data`` takes a trailing ``processing: str =
"raw"`` keyword -- one of ``raw``, ``mean``, ``min``, ``max``, ``median``, ``std``,
``count`` -- applied independently to each channel's own real samples, never across
channels and never onto a shared grid:

- ``raw`` decimates each ``precision_ms`` bin down to its **last real sample**,
  keeping that sample's own true timestamp -- never a timestamp invented at the
  bin's edge to hold it. This matches the EPICS Archiver Appliance's long-standing
  ``lastSample_N`` semantics, and every in-tree backend now applies it the same way.
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
