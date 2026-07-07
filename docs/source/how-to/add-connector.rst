Add a Connector
===============

**What you'll build:** Control system connectors for accessing hardware abstraction layers

Overview
========

The Control System Integration system provides a **two-layer abstraction** for working with control systems and archivers. This enables development and R&D work using mock connectors (without hardware access) and migration to production by changing a single configuration line.

**Capabilities:**

- **Mock Mode**: Work with any channel names without hardware access
- **Production Mode**: EPICS in-tree; LabVIEW, Tango, and other stacks via user-registered custom connectors
- **One API**: the same code works with mock and production connectors
- **Custom connectors**: register your own via ``ConnectorFactory``

**Built-in Connectors:**

- **mock** / **mock_archiver**: Development/R&D mode (no hardware access required)
- **epics** / **epics_archiver**: EPICS Channel Access / Archiver Appliance (production)
- **mongodb_archiver**: MongoDB time-series archiver (optional, ``pip install "osprey-framework[archiver-mongodb]"``)


Quick Start: Using Connectors
=============================

Mock Mode (Development & R&D)
------------------------------

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
-----------------------

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
----------------

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
PV names as top-level fields: ``{date: ISODate(...), PV1: value1, PV2: value2, ...}``.
The connector requires the optional ``archiver-mongodb`` extra:

.. code-block:: bash

   pip install "osprey-framework[archiver-mongodb]"


Write Verification
==================

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
--------------------------

Write operations are disabled by default and must be explicitly enabled at two levels:

**Global write permission** (in ``config.yml``):

.. code-block:: yaml

   control_system:
     writes_enabled: true          # Master switch for all write operations

If ``writes_enabled`` is omitted, it defaults to ``false`` and all writes are blocked.

.. _limits-checking-config:

Limits Checking
---------------

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
==============================

Subclass :class:`~osprey.connectors.control_system.base.ControlSystemConnector` and implement the abstract methods: ``connect``, ``disconnect``, ``read_channel``, ``write_channel``, ``read_multiple_channels``, ``subscribe``, ``unsubscribe``, ``get_metadata``, ``validate_channel``.

You may also override the non-abstract ``write_multiple_channels()`` method if your backend benefits from atomic batch writes (e.g., disabling lattice recalculation between writes in a simulator). The default implementation writes sequentially via ``write_channel()``.

Your connector must return the standard data models from ``osprey.connectors.control_system.base``: :class:`~osprey.connectors.control_system.base.ChannelValue`, :class:`~osprey.connectors.control_system.base.ChannelMetadata`, :class:`~osprey.connectors.control_system.base.ChannelWriteResult`, and :class:`~osprey.connectors.control_system.base.WriteVerification`.

Registering Custom Connectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-------------------------

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
