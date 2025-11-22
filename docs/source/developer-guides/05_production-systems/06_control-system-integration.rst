==========================
Control System Integration
==========================

**What you'll build:** Control system connectors for accessing hardware abstraction layers

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Implementing :class:`ControlSystemConnector` and :class:`ArchiverConnector` base classes
   - Using :class:`ConnectorFactory` for automatic connector instantiation
   - Configuring connector types and pattern detection for approval workflows
   - Building capabilities that work with both mock and production connectors
   - Creating custom connectors for new control systems (LabVIEW, Tango, custom protocols)

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/03_registry-and-discovery` and async programming

   **Time Investment:** 45-60 minutes for complete understanding

Overview
========

The Control System Integration system provides a **two-layer abstraction** for working with control systems and archivers. This enables development and R&D work using mock connectors (without hardware access) and seamless migration to production by changing a single configuration line.

**Key Features:**

- **Mock Mode**: Work with any channel names without hardware access (R&D, firewalled systems, development)
- **Production Mode**: Real control system connectors (EPICS, LabVIEW, Tango, custom)
- **Unified API**: Same code works with mock and production connectors
- **Pattern Detection**: Automatic identification of control system operations in generated code
- **Pluggable Architecture**: Register custom connectors via standalone ConnectorFactory

.. tab-set::

   .. tab-item:: Architecture

      The Two-Layer Architecture
      ---------------------------

      The connector system implements two complementary layers:

      **Layer 1: Pattern Detection (Config-Based)**
         - **Purpose**: Detect control system operations in generated code for approval workflow
         - **Implementation**: Regex patterns in YAML configuration
         - **Location**: ``osprey.services.python_executor.pattern_detection``
         - **Used by**: Approval system, domain analyzer, Python execution service

      **Layer 2: Runtime Connectors (Code-Based)**
         - **Purpose**: Actual I/O operations to control systems and archivers
         - **Implementation**: Abstract base classes with concrete implementations
         - **Location**: ``osprey.connectors``
         - **Used by**: Capabilities (pv_value_retrieval, archiver_retrieval, etc.)

      Both layers work together to provide complete control system abstraction with safety controls.

   .. tab-item:: Available Connectors

      Built-in Connectors
      -------------------

      The framework provides these built-in connectors:

      **Control System Connectors:**

      - **mock**: Development/R&D mode (accepts any PV names, no hardware access required)
      - **epics**: EPICS Channel Access (production, requires ``pyepics``)

      **Archiver Connectors:**

      - **mock_archiver**: Development/R&D mode (generates synthetic time series data)
      - **epics_archiver**: EPICS Archiver Appliance (production, requires ``archivertools``)

      Custom Connectors
      -----------------

      Need to integrate with other control systems (LabVIEW, Tango, custom facility systems)?
      See `Advanced: Implementing Custom Connectors`_ below for guidance on creating your own connectors.


Quick Start: Using Connectors
=============================

Mock Mode (Development & R&D)
------------------------------

The mock connector works without any hardware access - ideal for R&D, firewalled systems, or when developing remotely:

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

   # Use it - accepts any PV name!
   pv_value = await connector.read_pv('ANY:MADE:UP:NAME')
   print(f"Value: {pv_value.value} {pv_value.metadata.units}")

   await connector.disconnect()

Production Mode (EPICS Connector)
---------------------------------

Switch to real hardware by changing the ``type`` field:

.. code-block:: python

   # Create EPICS connector - requires real PVs
   connector = await ConnectorFactory.create_control_system_connector({
       'type': 'epics',
       'connector': {
           'epics': {
               'gateways': {
                   'read_only': {
                       'address': 'cagw.facility.edu',
                       'port': 5064
                   }
               }
           }
       }
   })

   # Same API as mock connector!
   pv_value = await connector.read_pv('REAL:BEAM:CURRENT')
   print(f"Beam current: {pv_value.value} {pv_value.metadata.units}")

   await connector.disconnect()

**The power:** Change configuration, not code. Your capabilities work in both modes!


Step-by-Step: Building a Capability with Connectors
===================================================

Step 1: Use ConnectorFactory in Capabilities
--------------------------------------------

Capabilities should use ``ConnectorFactory`` to create connectors from global configuration:

.. code-block:: python

   """
   Channel Value Retrieval Capability
   Works with both mock and production connectors
   """
   from osprey.connectors.factory import ConnectorFactory
   from osprey.base.decorators import capability_node
   from osprey.base.capability import BaseCapability
   from osprey.state import AgentState, StateManager

   @capability_node
   class ChannelValueRetrievalCapability(BaseCapability):
       """Retrieve current values from control system."""

       name = "channel_value_retrieval"
       description = "Retrieve current channel values"
       provides = ["CHANNEL_VALUES"]
       requires = [("CHANNEL_ADDRESSES", "single")]

       async def execute(self) -> dict:
           """Execute channel value retrieval."""

           # Get channel addresses from context (automatically extracted)
           # The "single" constraint in requires guarantees this is not a list
           channel_context, = self.get_required_contexts()

           # Create connector from global configuration
           # Automatically selects mock or production based on config
           connector = await ConnectorFactory.create_control_system_connector()

           try:
               # Read all channel values
               channel_values = {}
               for channel in channel_context.channels:
                   pv_result = await connector.read_pv(channel)
                   channel_values[channel] = {
                       'value': pv_result.value,
                       'units': pv_result.metadata.units,
                       'timestamp': pv_result.timestamp
                   }

               # Store results in context
               result = ChannelValuesContext(channel_values=channel_values)
               return self.store_output_context(result)

           finally:
               # Always disconnect
               await connector.disconnect()

**Key Patterns:**

- Use ``ConnectorFactory.create_control_system_connector()`` without arguments to load from global config
- Always use ``try/finally`` to ensure ``disconnect()`` is called
- The same code works with mock and production - no changes needed!

Step 2: Configure Connector Type
--------------------------------

Control which connector is used via ``config.yml``:

**Mock mode (default - for development/R&D):**

.. code-block:: yaml

   control_system:
     type: mock                    # ← Mock connector
     connector:
       mock:
         response_delay_ms: 10     # Simulate network latency
         noise_level: 0.01         # Add realistic noise
         enable_writes: true       # Allow write operations

**Production mode:**

.. code-block:: yaml

   control_system:
     type: epics                   # ← EPICS connector
     connector:
       epics:
         gateways:
           read_only:
             address: cagw.facility.edu
             port: 5064
             use_name_server: false  # EPICS CA config method (see EPICS docs)
           read_write:
             address: cagw-rw.facility.edu
             port: 5065
             use_name_server: false
         timeout: 5.0

**That's it!** Your capability automatically uses the configured connector.

Step 3: Configure Pattern Detection
-----------------------------------

Pattern detection enables the approval system to identify control system operations in generated Python code:

.. code-block:: yaml

   control_system:
     type: epics
     patterns:
       epics:
         write:
           - 'epics\.caput\('        # Matches: epics.caput(...)
           - '\.put\('               # Matches: pv.put(...)
         read:
           - 'epics\.caget\('        # Matches: epics.caget(...)
           - '\.get\('               # Matches: pv.get(...)

**Usage in capabilities:**

.. code-block:: python

   from osprey.services.python_executor.pattern_detection import detect_control_system_operations

   code = """
   current = epics.caget('BEAM:CURRENT')
   if current < 400:
       epics.caput('ALARM:STATUS', 1)
   """

   result = detect_control_system_operations(code)
   print(f"Has writes: {result['has_writes']}")  # True
   print(f"Has reads: {result['has_reads']}")    # True

This enables the approval system to require human review for code that performs writes.


Advanced: Implementing Custom Connectors
========================================

Implementing Custom Connectors
------------------------------


.. tab-set::

   .. tab-item:: Control System Connector

      Create custom connectors for new control systems (LabVIEW, Tango, proprietary protocols, etc.):

      **Example: LabVIEW Web Services Connector**

      This example demonstrates a connector for National Instruments LabVIEW systems, commonly used in industrial automation, research labs, and test/measurement environments.

      **Key Implementation Points:**

      - **Connection**: HTTP client to LabVIEW Web Services with optional authentication
      - **Read/Write**: REST API calls to ``/api/variables/{name}`` endpoints
      - **Batch Operations**: Optimized multi-variable reads with fallback to sequential
      - **Metadata**: Retrieves units, limits, precision from LabVIEW variable properties
      - **Subscriptions**: Polling-based pattern (can be upgraded to WebSocket/SSE)
      - **Error Handling**: Proper exception mapping for 404 (not found), 403 (permission), 400 (invalid value)

      .. dropdown:: Full LabVIEW Connector Implementation (Click to expand)
         :color: info
         :icon: code

         .. admonition:: Note
            :class: warning

            The following connector implementation serves as an example to demonstrate the pattern for creating custom connectors. This code has not been tested in production.

         .. code-block:: python

            """
            Custom LabVIEW Control System Connector

            LabVIEW is a widely-used graphical programming environment by National Instruments
            for data acquisition, instrument control, and industrial automation. This connector
            interfaces with LabVIEW systems via HTTP Web Services and Network Shared Variables.
            """
            from osprey.connectors.control_system.base import (
                ControlSystemConnector,
                PVValue,
                PVMetadata
            )
            from osprey.utils.logger import get_logger
            from datetime import datetime
            from typing import Any, Callable, Dict, List, Optional
            import httpx
            import uuid

            logger = get_logger("labview_connector")

            class LabVIEWConnector(ControlSystemConnector):
                """
                Connector for LabVIEW control systems.

                Interfaces with LabVIEW Web Services for reading/writing variables
                and retrieving metadata. Supports both REST API endpoints and
                Network Shared Variables (via HTTP).
                """

                def __init__(self):
                    self._connected = False
                    self._client: Optional[httpx.AsyncClient] = None
                    self._base_url = None
                    self._subscriptions: Dict[str, Dict] = {}

                async def connect(self, config: Dict[str, Any]) -> None:
                    """
                    Establish connection to LabVIEW Web Service.

                    Expected config:
                        - base_url: URL of LabVIEW Web Service (e.g., 'http://localhost:8080')
                        - api_key: Optional API key for authentication
                        - timeout: Request timeout in seconds (default: 5.0)
                        - verify_ssl: Whether to verify SSL certificates (default: True)
                    """
                    try:
                        self._base_url = config.get('base_url', 'http://localhost:8080')
                        timeout = config.get('timeout', 5.0)
                        api_key = config.get('api_key')
                        verify_ssl = config.get('verify_ssl', True)

                        # Setup HTTP client
                        headers = {}
                        if api_key:
                            headers['X-API-Key'] = api_key

                        self._client = httpx.AsyncClient(
                            base_url=self._base_url,
                            timeout=timeout,
                            headers=headers,
                            verify=verify_ssl
                        )

                        # Test connection with health check
                        response = await self._client.get('/api/health')
                        response.raise_for_status()

                        self._connected = True
                        logger.info(f"Connected to LabVIEW Web Service at {self._base_url}")

                    except Exception as e:
                        raise ConnectionError(f"Failed to connect to LabVIEW: {e}")

                async def disconnect(self) -> None:
                    """Close LabVIEW connection."""
                    if self._client:
                        await self._client.aclose()
                        self._client = None
                    self._subscriptions.clear()
                    self._connected = False

                async def read_pv(
                    self,
                    pv_address: str,
                    timeout: Optional[float] = None
                ) -> PVValue:
                    """
                    Read a LabVIEW variable or VI output.

                    Args:
                        pv_address: Variable path (e.g., 'System/Temperature/Sensor1')
                        timeout: Optional timeout override
                    """
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    try:
                        # Call LabVIEW Web Service endpoint to read variable
                        response = await self._client.get(
                            f'/api/variables/{pv_address}',
                            timeout=timeout
                        )
                        response.raise_for_status()
                        data = response.json()

                        # Parse response
                        metadata = PVMetadata(
                            units=data.get('units', ''),
                            description=data.get('description', ''),
                            min_value=data.get('min_value'),
                            max_value=data.get('max_value'),
                            precision=data.get('precision'),
                            alarm_status=data.get('alarm_status')
                        )

                        # Parse timestamp
                        timestamp_str = data.get('timestamp')
                        timestamp = (
                            datetime.fromisoformat(timestamp_str)
                            if timestamp_str
                            else datetime.now()
                        )

                        return PVValue(
                            value=data['value'],
                            timestamp=timestamp,
                            metadata=metadata
                        )

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            raise ValueError(f"Variable not found: {pv_address}")
                        raise RuntimeError(f"Failed to read {pv_address}: {e}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to read {pv_address}: {e}")

                async def write_pv(
                    self,
                    pv_address: str,
                    value: Any,
                    timeout: Optional[float] = None
                ) -> bool:
                    """
                    Write a value to a LabVIEW variable.

                    Args:
                        pv_address: Variable path
                        value: Value to write
                        timeout: Optional timeout override
                    """
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    try:
                        # Call LabVIEW Web Service endpoint to write variable
                        response = await self._client.put(
                            f'/api/variables/{pv_address}',
                            json={'value': value},
                            timeout=timeout
                        )
                        response.raise_for_status()
                        return True

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 403:
                            raise PermissionError(f"Write access denied for {pv_address}")
                        elif e.response.status_code == 400:
                            raise ValueError(f"Invalid value for {pv_address}: {value}")
                        raise RuntimeError(f"Failed to write {pv_address}: {e}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to write {pv_address}: {e}")

                async def get_metadata(self, pv_address: str) -> PVMetadata:
                    """Get metadata about a LabVIEW variable."""
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    try:
                        response = await self._client.get(
                            f'/api/variables/{pv_address}/metadata'
                        )
                        response.raise_for_status()
                        data = response.json()

                        return PVMetadata(
                            units=data.get('units', ''),
                            description=data.get('description', ''),
                            min_value=data.get('min_value'),
                            max_value=data.get('max_value'),
                            precision=data.get('precision'),
                            alarm_status=data.get('alarm_status')
                        )

                    except Exception as e:
                        raise RuntimeError(f"Failed to get metadata for {pv_address}: {e}")

                async def read_multiple_pvs(
                    self,
                    pv_addresses: List[str],
                    timeout: Optional[float] = None
                ) -> Dict[str, PVValue]:
                    """
                    Read multiple LabVIEW variables efficiently.

                    Uses batch endpoint if available, otherwise reads sequentially.
                    """
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    try:
                        # Try batch endpoint first (if LabVIEW Web Service supports it)
                        response = await self._client.post(
                            '/api/variables/batch/read',
                            json={'variables': pv_addresses},
                            timeout=timeout
                        )

                        if response.status_code == 200:
                            data = response.json()
                            results = {}
                            for var_name, var_data in data.items():
                                if 'error' not in var_data:
                                    metadata = PVMetadata(
                                        units=var_data.get('units', ''),
                                        description=var_data.get('description', '')
                                    )
                                    timestamp_str = var_data.get('timestamp')
                                    timestamp = (
                                        datetime.fromisoformat(timestamp_str)
                                        if timestamp_str
                                        else datetime.now()
                                    )
                                    results[var_name] = PVValue(
                                        value=var_data['value'],
                                        timestamp=timestamp,
                                        metadata=metadata
                                    )
                            return results

                        # Fall back to sequential reads
                        results = {}
                        for pv in pv_addresses:
                            try:
                                results[pv] = await self.read_pv(pv, timeout)
                            except Exception as e:
                                logger.warning(f"Failed to read {pv}: {e}")
                        return results

                    except Exception as e:
                        raise RuntimeError(f"Failed to read multiple PVs: {e}")

                async def subscribe(
                    self,
                    pv_address: str,
                    callback: Callable[[PVValue], None]
                ) -> str:
                    """
                    Subscribe to LabVIEW variable changes.

                    Note: This implementation uses polling. For production,
                    consider WebSocket connections or Server-Sent Events (SSE)
                    if supported by your LabVIEW Web Service.
                    """
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    subscription_id = str(uuid.uuid4())
                    self._subscriptions[subscription_id] = {
                        'pv_address': pv_address,
                        'callback': callback,
                        'last_value': None
                    }

                    # In production, implement actual subscription mechanism
                    # (WebSocket, SSE, or polling task)
                    logger.info(f"Subscribed to {pv_address} (subscription: {subscription_id})")

                    return subscription_id

                async def unsubscribe(self, subscription_id: str) -> None:
                    """Unsubscribe from variable changes."""
                    if subscription_id in self._subscriptions:
                        del self._subscriptions[subscription_id]

                async def validate_pv(self, pv_address: str) -> bool:
                    """Check if LabVIEW variable exists and is accessible."""
                    if not self._connected:
                        raise ConnectionError("Not connected to LabVIEW")

                    try:
                        response = await self._client.head(
                            f'/api/variables/{pv_address}'
                        )
                        return response.status_code == 200
                    except Exception:
                        return False

      **Registration:**

      Connectors are registered through the Osprey registry system for unified component management:

      .. code-block:: python

         # In your application's registry.py
         from osprey.registry import ConnectorRegistration, extend_framework_registry

         def get_registry_config(self):
             return extend_framework_registry(
                 connectors=[
                     ConnectorRegistration(
                         name="labview",
                         connector_type="control_system",
                         module_path="my_app.connectors.labview_connector",
                         class_name="LabVIEWConnector",
                         description="LabVIEW Web Services connector for NI control systems"
                     )
                 ],
                 capabilities=[...],
                 context_classes=[...]
             )

      The registry system automatically calls ``ConnectorFactory.register_control_system()`` during
      initialization, ensuring connectors are available before capabilities use them. Use descriptive
      names (``tango``, ``custom``, ``facility_name``).

   .. tab-item:: Archiver Connector

      Custom archiver connectors follow the same pattern:

      .. code-block:: python

         """
         Custom Archiver Connector
         """
         from osprey.connectors.archiver.base import ArchiverConnector, ArchivedData
         from datetime import datetime
         from typing import Dict, List
         import pandas as pd

         class CustomArchiverConnector(ArchiverConnector):
             """Connector for custom archiver system."""

             async def connect(self, config: Dict) -> None:
                 """Establish archiver connection."""
                 self._url = config.get('url')
                 self._connected = True

             async def disconnect(self) -> None:
                 """Close archiver connection."""
                 self._connected = False

             async def get_data(
                 self,
                 pv_list: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 precision_ms: int = 1000
             ) -> Dict[str, ArchivedData]:
                 """Retrieve historical data."""
                 if not self._connected:
                     raise ConnectionError("Not connected to archiver")

                 results = {}
                 for pv in pv_list:
                     # Implement your archiver API call here
                     timestamps, values = await self._fetch_from_archiver(
                         pv, start_date, end_date, precision_ms
                     )

                     results[pv] = ArchivedData(
                         timestamps=timestamps,
                         values=values
                     )

                 return results

             async def _fetch_from_archiver(
                 self,
                 pv: str,
                 start: datetime,
                 end: datetime,
                 precision: int
             ):
                 """Your archiver-specific implementation."""
                 # Implement API call to your archiver
                 pass

      **Register the archiver connector** in your application's registry:

      .. code-block:: python

         # In your application's registry.py
         from osprey.registry import ConnectorRegistration, extend_framework_registry

         def get_registry_config(self):
             return extend_framework_registry(
                 connectors=[
                     ConnectorRegistration(
                         name="custom_archiver",
                         connector_type="archiver",
                         module_path="my_app.connectors.custom_archiver",
                         class_name="CustomArchiverConnector",
                         description="Custom archiver connector"
                     )
                 ],
                 capabilities=[...],
                 context_classes=[...]
             )

   .. tab-item:: Testing Patterns


        Testing Custom Connectors
        --------------------------

        Testing happens in three phases, each serving a different purpose:

        **Phase 1: Capability Logic (No Hardware)**

        Test your capability integration logic with mock connectors:

        .. code-block:: python

            from osprey.connectors.factory import ConnectorFactory

            async def test_capability_logic():
                """Test capability logic with mock connector."""
                connector = await ConnectorFactory.create_control_system_connector({
                    'type': 'mock',
                    'connector': {'mock': {'response_delay_ms': 0}}
                })

                try:
                    result = await connector.read_pv('BEAM:CURRENT')
                    assert result.value is not None
                    assert result.metadata.units is not None
                finally:
                    await connector.disconnect()

        Mock connectors accept any PV names and return realistic data - perfect for fast iteration.

        **Phase 2: Interface Compliance (Local/Simulator)**

        Test your custom connector implements the protocol correctly:

        .. code-block:: python

            import pytest
            from my_app.connectors.tango_connector import TangoConnector

            @pytest.fixture
            async def tango_connector():
                connector = TangoConnector()
                await connector.connect({'host': 'localhost', 'port': 10000})
                yield connector
                await connector.disconnect()

            @pytest.mark.asyncio
            async def test_read_pv_structure(tango_connector):
                """Verify interface contract."""
                result = await tango_connector.read_pv('sys/tg_test/1/double_scalar')
                assert result.value is not None
                assert result.timestamp is not None

        **Phase 3: Integration (Production-Like)**

        Test complete stack with real hardware. Mark with ``@pytest.mark.integration`` to run separately:

        .. code-block:: python

            @pytest.mark.integration
            async def test_full_stack():
                """E2E test with real connector."""
                connector = await ConnectorFactory.create_control_system_connector({
                    'type': 'tango',
                    'connector': {'tango': {'host': 'tango.facility.edu'}}
                })

                try:
                    result = await connector.read_pv('domain/family/member/current')
                    assert 0 <= result.value <= 500  # Expected range
                finally:
                    await connector.disconnect()

        Best Practices
        --------------

        **Configuration-Driven Tests**: Switch connectors via environment variables:

        .. code-block:: python

            # conftest.py
            @pytest.fixture
            def connector_config():
                if os.getenv('USE_REAL_CONNECTOR') == '1':
                    return {'type': 'epics', 'connector': {'epics': {}}}
                return {'type': 'mock', 'connector': {'mock': {}}}

        Run with ``pytest`` (mock) or ``USE_REAL_CONNECTOR=1 pytest`` (real hardware).

        **When to Use What**: Use mocks for CI/CD and development. Use real connectors for pre-deployment validation
        and connector implementation testing. Your capability code stays the same - only configuration changes

Related Documentation
=====================

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🏗️ Getting Started
      :link: ../../getting-started/control-assistant-part1-setup
      :link-type: doc

      See connectors in action in the Control Assistant tutorial

   .. grid-item-card:: 🛡️ Approval Workflows
      :link: 01_human-approval-workflows
      :link-type: doc

      How pattern detection integrates with approval system

   .. grid-item-card:: 🐍 Python Execution
      :link: 03_python-execution-service
      :link-type: doc

      Pattern detection for secure code execution

   .. grid-item-card:: 📚 API Reference
      :link: ../../api_reference/03_production_systems/06_control-system-connectors
      :link-type: doc

      Complete API documentation for connectors