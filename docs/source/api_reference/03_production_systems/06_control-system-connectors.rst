=========================
Control System Connectors
=========================

Pluggable connector abstraction for control systems and archivers with mock and production implementations. Enables development without hardware access and seamless migration to production by changing configuration.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/06_control-system-integration`.

.. currentmodule:: osprey.connectors

Factory Classes
===============

.. autoclass:: osprey.connectors.factory.ConnectorFactory
   :members:
   :show-inheritance:

The factory provides centralized creation and configuration of connectors with plugin-style registration.

Registry Integration
====================

Connectors can be registered through the Osprey registry system for unified component management.

.. currentmodule:: osprey.registry

.. autoclass:: ConnectorRegistration
   :members:
   :show-inheritance:

Registration dataclass for control system and archiver connectors. Used to register connectors
through the registry system, providing lazy loading and unified management alongside other framework components.

**Usage Example:**

.. code-block:: python

   from osprey.registry import ConnectorRegistration, extend_framework_registry

   def get_registry_config(self):
       return extend_framework_registry(
           connectors=[
               ConnectorRegistration(
                   name="labview",
                   connector_type="control_system",
                   module_path="my_app.connectors.labview_connector",
                   class_name="LabVIEWConnector",
                   description="LabVIEW Web Services connector for NI systems"
               ),
               ConnectorRegistration(
                   name="tango",
                   connector_type="control_system",
                   module_path="my_app.connectors.tango_connector",
                   class_name="TangoConnector",
                   description="Tango control system connector"
               ),
               ConnectorRegistration(
                   name="tango_archiver",
                   connector_type="archiver",
                   module_path="my_app.connectors.tango_archiver",
                   class_name="TangoArchiverConnector",
                   description="Tango archiver connector"
               )
           ],
           capabilities=[...],
           context_classes=[...]
       )

.. currentmodule:: osprey.connectors

Control System Interfaces
=========================

Base Classes
------------

.. autoclass:: osprey.connectors.control_system.base.ControlSystemConnector
   :members:
   :show-inheritance:

Abstract base class defining the contract for all control system connectors (EPICS, LabVIEW, Tango, Mock, etc.).

Data Models
-----------

.. autoclass:: osprey.connectors.control_system.base.PVValue
   :members:
   :show-inheritance:

Result container for process variable reads with value, timestamp, and metadata.

.. autoclass:: osprey.connectors.control_system.base.PVMetadata
   :members:
   :show-inheritance:

Metadata about a process variable (units, precision, alarms, limits, etc.).

Built-in Implementations
------------------------

.. autoclass:: osprey.connectors.control_system.mock_connector.MockConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Development connector that accepts any PV names and generates realistic simulated data. Ideal for R&D when you don't have control room access.

**Key Features:**

- Accepts any PV name (no real control system required)
- Configurable response delays and noise levels
- Realistic units, timestamps, and metadata
- Optional write operation support

.. autoclass:: osprey.connectors.control_system.epics_connector.EPICSConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Production EPICS Channel Access connector using ``pyepics`` library. Supports gateway configuration for secure access.

**Key Features:**

- EPICS Channel Access protocol
- Read-only and read-write gateway support
- Configurable timeouts and retry logic
- Full metadata support (units, precision, alarms, limits)

**Requirements:**

- ``pyepics`` library: ``pip install pyepics``
- Access to EPICS gateway or IOCs

Archiver Interfaces
===================

Base Classes
------------

.. autoclass:: osprey.connectors.archiver.base.ArchiverConnector
   :members:
   :show-inheritance:

Abstract base class defining the contract for all archiver connectors.

.. autoclass:: osprey.connectors.archiver.base.ArchivedData
   :members:
   :show-inheritance:

Container for historical time series data with timestamps and values.

Built-in Implementations
------------------------

.. autoclass:: osprey.connectors.archiver.mock_archiver_connector.MockArchiverConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Development archiver that generates synthetic historical data. Ideal for R&D when you don't have archiver access.

**Key Features:**

- Generates realistic time series with trends and noise
- Configurable retention period and sample rates
- Works with any PV names
- Consistent with mock control system connector

.. autoclass:: osprey.connectors.archiver.epics_archiver_connector.EPICSArchiverConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Production connector for EPICS Archiver Appliance using ``archivertools`` library.

**Key Features:**

- EPICS Archiver Appliance integration
- Efficient bulk data retrieval
- Configurable precision and time ranges
- Connection pooling for performance

**Requirements:**

- ``archivertools`` library: ``pip install archivertools``
- Access to EPICS Archiver Appliance URL

Pattern Detection
=================

.. currentmodule:: osprey.services.python_executor.pattern_detection

Static code analysis for detecting control system operations in generated code. Used by approval system to identify reads and writes.

.. autofunction:: detect_control_system_operations

Analyzes Python code using configurable regex patterns to detect control system operations. Returns detection results including operation types and matched patterns.

**Example:**

.. code-block:: python

   from osprey.services.python_executor.pattern_detection import detect_control_system_operations

   code = """
   current = epics.caget('BEAM:CURRENT')
   if current < 400:
       epics.caput('ALARM:STATUS', 1)
   """

   result = detect_control_system_operations(code)
   # result['has_writes'] == True
   # result['has_reads'] == True

Configuration Schema
====================

Control System Configuration
----------------------------

Control system connector configuration in ``config.yml``:

.. code-block:: yaml

   control_system:
     type: mock | epics | labview | tango | custom

     # Pattern detection for approval system
     patterns:
       <control_system_type>:
         write:
           - '<regex_pattern_1>'
           - '<regex_pattern_2>'
         read:
           - '<regex_pattern_1>'
           - '<regex_pattern_2>'

     # Type-specific configurations
     connector:
       mock:
         response_delay_ms: 10
         noise_level: 0.01
         enable_writes: true

      epics:
        gateways:
          read_only:
            address: cagw.facility.edu
            port: 5064
            use_name_server: false    # Use EPICS_CA_NAME_SERVERS vs CA_ADDR_LIST (default: false)
          read_write:
            address: cagw-rw.facility.edu
            port: 5065
            use_name_server: false
        timeout: 5.0
        retry_count: 3
        retry_delay: 0.5

Archiver Configuration
----------------------

Archiver connector configuration in ``config.yml``:

.. code-block:: yaml

   archiver:
     type: mock_archiver | epics_archiver | custom_archiver

     # Mock archiver uses sensible defaults
     mock_archiver:
       sample_rate_hz: 1.0
       noise_level: 0.01

     epics_archiver:
       url: https://archiver.facility.edu:8443
       timeout: 60
       max_retries: 3
       verify_ssl: true
       pool_connections: 10
       pool_maxsize: 20

Usage Examples
==============

Basic Usage
-----------

Create and use a connector from global configuration:

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory

   # Create connector from config.yml
   connector = await ConnectorFactory.create_control_system_connector()

   try:
       # Read a PV
       result = await connector.read_pv('BEAM:CURRENT')
       print(f"Current: {result.value} {result.metadata.units}")

       # Read multiple PVs
       results = await connector.read_multiple_pvs([
           'BEAM:CURRENT',
           'BEAM:LIFETIME',
           'BEAM:ENERGY'
       ])

       # Get metadata
       metadata = await connector.get_metadata('BEAM:CURRENT')
       print(f"Units: {metadata.units}, Range: {metadata.min_value}-{metadata.max_value}")

   finally:
       await connector.disconnect()

Custom Configuration
--------------------

Create connector with inline configuration:

.. code-block:: python

   # Mock connector with custom settings
   config = {
       'type': 'mock',
       'connector': {
           'mock': {
               'response_delay_ms': 5,
               'noise_level': 0.02
           }
       }
   }
   connector = await ConnectorFactory.create_control_system_connector(config)

   # EPICS connector with specific gateway
   config = {
       'type': 'epics',
       'connector': {
           'epics': {
               'gateways': {
                   'read_only': {
                       'address': 'cagw.als.lbl.gov',
                       'port': 5064
                   }
               },
               'timeout': 3.0
           }
       }
   }
   connector = await ConnectorFactory.create_control_system_connector(config)

Archiver Usage
--------------

Retrieve historical data:

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory
   from datetime import datetime, timedelta

   # Create archiver connector
   connector = await ConnectorFactory.create_archiver_connector()

   try:
       # Define time range
       end_time = datetime.now()
       start_time = end_time - timedelta(hours=24)

       # Retrieve data for multiple PVs
       data = await connector.get_data(
           pv_list=['BEAM:CURRENT', 'BEAM:LIFETIME'],
           start_date=start_time,
           end_date=end_time,
           precision_ms=1000  # 1 second precision
       )

       # Process results
       for pv_name, pv_data in data.items():
           print(f"{pv_name}: {len(pv_data.timestamps)} data points")

   finally:
       await connector.disconnect()

Pattern Detection Usage
-----------------------

Detect control system operations in generated code:

.. code-block:: python

   from osprey.services.python_executor.pattern_detection import detect_control_system_operations

   code = """
   # Read beam current
   current = epics.caget('BEAM:CURRENT')

   # Adjust setpoint if needed
   if current < 400:
       epics.caput('BEAM:SETPOINT', 420.0)
   """

   # Detect operations
   result = detect_control_system_operations(code)

   if result['has_writes']:
       print("⚠️ Code performs write operations - requires approval")

   if result['has_reads']:
       print("✓ Code performs read operations")

   print(f"Control system: {result['control_system_type']}")
   print(f"Write patterns detected: {result['detected_patterns']['writes']}")
   print(f"Read patterns detected: {result['detected_patterns']['reads']}")

Custom Connector Registration
-----------------------------

Custom connectors are registered through the Osprey registry system:

.. code-block:: python

   # In your application's registry.py
   from osprey.registry import ConnectorRegistration, extend_framework_registry

   class MyAppRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self):
           return extend_framework_registry(
               connectors=[
                   # Control system connectors
                   ConnectorRegistration(
                       name="labview",
                       connector_type="control_system",
                       module_path="my_app.connectors.labview_connector",
                       class_name="LabVIEWConnector",
                       description="LabVIEW Web Services connector"
                   ),
                   ConnectorRegistration(
                       name="tango",
                       connector_type="control_system",
                       module_path="my_app.connectors.tango_connector",
                       class_name="TangoConnector",
                       description="Tango control system connector"
                   ),
                   # Archiver connectors
                   ConnectorRegistration(
                       name="tango_archiver",
                       connector_type="archiver",
                       module_path="my_app.connectors.tango_archiver",
                       class_name="TangoArchiverConnector",
                       description="Tango archiver connector"
                   ),
               ],
               capabilities=[...],
               context_classes=[...]
           )

After registration, connectors are available via configuration:

.. code-block:: yaml

   # config.yml
   control_system:
     type: labview  # or tango, epics, mock
     connector:
       labview:
         base_url: "http://labview-server:8080"
         api_key: "your-api-key"
       tango:
         device_name: "tango://host:10000/sys/dev/1"

   archiver:
     type: tango_archiver
     tango_archiver:
       url: "https://archiver.facility.edu"

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/06_control-system-integration`
       Complete implementation guide with step-by-step examples

   :doc:`../../../getting-started/control-assistant-part1-setup`
       See connectors in action in the Control Assistant tutorial

   :doc:`01_human-approval`
       How pattern detection integrates with approval workflows

   :doc:`03_python-execution`
       Pattern detection in secure Python code execution

