"""
Connector factory for creating control system and archiver connectors.

Provides centralized factory for instantiating and configuring connectors
based on configuration. Connectors are registered through the Osprey registry
system for unified component management and lazy loading.

Related to Issue #18 - Control System Abstraction (Layer 2 - Factory)
"""

from typing import Any

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.utils.logger import get_logger

logger = get_logger("connector_factory")


class ConnectorFactory:
    """
    Factory for creating control system and archiver connectors.

    Provides centralized management of available connectors. Connectors are
    registered automatically through the Osprey registry system during
    framework initialization.

    Example:
        >>> # Using default config
        >>> cs_connector = await ConnectorFactory.create_control_system_connector()
        >>>
        >>> # Using custom config
        >>> config = {'type': 'mock', 'connector': {'mock': {...}}}
        >>> cs_connector = await ConnectorFactory.create_control_system_connector(config)
    """

    _control_system_connectors: dict[str, type[ControlSystemConnector]] = {}
    _archiver_connectors: dict[str, type[ArchiverConnector]] = {}

    @classmethod
    def register_control_system(
        cls,
        name: str,
        connector_class: type[ControlSystemConnector]
    ) -> None:
        """
        Register a control system connector.

        Args:
            name: Unique name for the connector (e.g., 'epics', 'tango', 'mock')
            connector_class: Connector class implementing ControlSystemConnector
        """
        cls._control_system_connectors[name] = connector_class
        logger.debug(f"Registered control system connector: {name}")

    @classmethod
    def register_archiver(
        cls,
        name: str,
        connector_class: type[ArchiverConnector]
    ) -> None:
        """
        Register an archiver connector.

        Args:
            name: Unique name for the connector (e.g., 'epics_archiver', 'mock_archiver')
            connector_class: Connector class implementing ArchiverConnector
        """
        cls._archiver_connectors[name] = connector_class
        logger.debug(f"Registered archiver connector: {name}")

    @classmethod
    async def create_control_system_connector(
        cls,
        config: dict[str, Any] = None
    ) -> ControlSystemConnector:
        """
        Create and configure a control system connector.

        Args:
            config: Control system configuration dict with keys:
                - type: Connector type (e.g., 'epics', 'mock')
                - connector: Dict with connector-specific configs
                If None, loads from global config

        Returns:
            Initialized and connected ControlSystemConnector

        Raises:
            ValueError: If connector type is unknown or config is invalid
            ConnectionError: If connection fails

        Example:
            >>> config = {
            >>>     'type': 'epics',
            >>>     'connector': {
            >>>         'epics': {
            >>>             'timeout': 5.0,
            >>>             'gateways': {'read_only': {...}}
            >>>         }
            >>>     }
            >>> }
            >>> connector = await ConnectorFactory.create_control_system_connector(config)
        """
        # Load config if not provided
        if config is None:
            try:
                from osprey.utils.config import get_config_value
                config = get_config_value('control_system', {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        connector_type = config.get('type', 'epics')
        connector_class = cls._control_system_connectors.get(connector_type)

        if not connector_class:
            available = list(cls._control_system_connectors.keys())
            raise ValueError(
                f"Unknown control system type: '{connector_type}'. "
                f"Available types: {available}"
            )

        # Create connector instance
        connector = connector_class()

        # Get type-specific configuration
        connector_configs = config.get('connector', {})
        type_config = connector_configs.get(connector_type, {})

        # Connect with configuration
        await connector.connect(type_config)

        logger.debug(f"Created control system connector: {connector_type}")
        return connector

    @classmethod
    async def create_archiver_connector(
        cls,
        config: dict[str, Any] = None
    ) -> ArchiverConnector:
        """
        Create and configure an archiver connector.

        Args:
            config: Archiver configuration dict with keys:
                - type: Connector type (e.g., 'epics_archiver', 'mock_archiver')
                - [type]: Dict with type-specific configs
                If None, loads from global config

        Returns:
            Initialized and connected ArchiverConnector

        Raises:
            ValueError: If connector type is unknown or config is invalid
            ConnectionError: If connection fails

        Example:
            >>> config = {
            >>>     'type': 'epics_archiver',
            >>>     'epics_archiver': {
            >>>         'url': 'https://archiver.als.lbl.gov:8443',
            >>>         'timeout': 60
            >>>     }
            >>> }
            >>> connector = await ConnectorFactory.create_archiver_connector(config)
        """
        # Load config if not provided
        if config is None:
            try:
                from osprey.utils.config import get_config_value
                config = get_config_value('archiver', {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        connector_type = config.get('type', 'epics_archiver')
        connector_class = cls._archiver_connectors.get(connector_type)

        if not connector_class:
            available = list(cls._archiver_connectors.keys())
            raise ValueError(
                f"Unknown archiver type: '{connector_type}'. "
                f"Available types: {available}"
            )

        # Create connector instance
        connector = connector_class()

        # Get type-specific configuration
        type_config = config.get(connector_type, {})

        # Connect with configuration
        await connector.connect(type_config)

        logger.info(f"Created archiver connector: {connector_type}")
        return connector

    @classmethod
    def list_control_systems(cls) -> list:
        """List available control system connector types."""
        return list(cls._control_system_connectors.keys())

    @classmethod
    def list_archivers(cls) -> list:
        """List available archiver connector types."""
        return list(cls._archiver_connectors.keys())

