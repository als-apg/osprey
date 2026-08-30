"""
Connector factory for creating control system and archiver connectors.

Provides centralized factory for instantiating and configuring connectors
based on configuration. Connectors are registered through the Osprey registry
system for unified component management and lazy loading.

"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey_connectors import types
from osprey_connectors.control_system.base import ControlSystemConnector
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    # Import only for annotations: pulling in the archiver base eagerly would
    # make pandas a hard dependency of the lean control-system import chain.
    from osprey_connectors.archiver.base import ArchiverConnector

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
        cls, name: str, connector_class: type[ControlSystemConnector]
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
    def register_archiver(cls, name: str, connector_class: type[ArchiverConnector]) -> None:
        """
        Register an archiver connector.

        Args:
            name: Unique name for the connector (e.g., 'epics_archiver', 'mock_archiver')
            connector_class: Connector class implementing ArchiverConnector
        """
        cls._archiver_connectors[name] = connector_class
        logger.debug(f"Registered archiver connector: {name}")

    @classmethod
    def build_control_system_connector(
        cls, config: dict[str, Any] | None = None, *, control_target: str | None = None
    ) -> tuple[ControlSystemConnector, dict[str, Any]]:
        """
        Build a control system connector and its type config, without connecting.

        Args:
            config: Control system configuration dict with keys:
                - type: Connector type (e.g., 'epics', 'mock'). Defaults to
                  'mock' with a warning when unset, so an under-specified
                  config never reaches live hardware.
                - connector: Dict with connector-specific configs
                If None, loads from global config
            control_target: The session target this connector is being built
                for — ``live``, ``va`` or ``standin``. Stamped on the instance
                as ``_control_target`` and used to index the per-(session,
                target) posture store, and nothing else: the deployment half of
                the write posture stays keyed on the connector *type*. Callers
                that have no target to name leave it ``None``, which is the
                behaviour every caller had before the stamp existed.

        Returns:
            ``(connector, type_config)`` — the stamped but *unconnected*
            connector, and the type-specific config block ``connect()`` takes.
            Both halves are needed to finish the build, and the second is not
            derivable from the first, so they travel together.

        Raises:
            ValueError: If connector type is unknown or config is invalid

        Example:
            >>> connector, type_config = ConnectorFactory.build_control_system_connector(config)
            >>> await connector.connect(type_config)
        """
        # Load config if not provided
        if config is None:
            try:
                from osprey_connectors.config import get_config_value

                config = get_config_value("control_system", {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        # Fail closed: an under-specified config must never bring up a live
        # control system. A blank value is treated as absent for the same
        # reason. The fallback lives in types.resolve_control_system_type so
        # that anything deciding what this call WILL build — the honesty rule's
        # three refusals above all — resolves it identically.
        connector_type = types.resolve_control_system_type(config)
        if not config.get("type"):
            logger.warning(
                f"control_system.type is not set; defaulting to '{types.MOCK}'. "
                f"Set control_system.type explicitly to select a connector."
            )

        connector_class = cls._control_system_connectors.get(connector_type)

        if not connector_class:
            if "." in connector_type:
                module_path, class_name = connector_type.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    connector_class = getattr(module, class_name)
                except ImportError as e:
                    raise ValueError(
                        f"Could not import connector module '{module_path}': {e}"
                    ) from e
                except AttributeError as e:
                    raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e
                cls._control_system_connectors[connector_type] = connector_class
            else:
                available = list(cls._control_system_connectors.keys())
                raise ValueError(
                    f"Unknown control system type: '{connector_type}'. "
                    f"Available types: {available}. "
                    f"Use a dotted module path for custom connectors."
                )

        # Create connector instance
        connector = connector_class()
        # The one seam between construction and connect(): the write posture and
        # the limits posture are both per connector type, and connect() itself
        # already consults the stamp to load them.
        connector._connector_type = connector_type
        # Beside it, and for the same reason: the session half of the posture is
        # per target, so the target has to be on the instance before connect()
        # reads it. The two stamps answer different questions — the type selects
        # the deployment's connector block, the target indexes the session
        # posture store — so neither is derivable from the other.
        connector._control_target = control_target

        # Get type-specific configuration
        connector_configs = config.get("connector", {})
        type_config = connector_configs.get(connector_type, {})

        logger.debug(f"Built control system connector: {connector_type}")
        return connector, type_config

    @classmethod
    async def create_control_system_connector(
        cls, config: dict[str, Any] | None = None, *, control_target: str | None = None
    ) -> ControlSystemConnector:
        """
        Create, configure and connect a control system connector.

        The whole build in one call, and what almost every caller wants.
        :meth:`build_control_system_connector` is the same build stopped one
        step short, for the one caller that has to reach the instance between
        the type stamp and ``connect()``.

        Args:
            config: Control system configuration dict, as
                :meth:`build_control_system_connector` documents it.
            control_target: The session target this connector is being built
                for, stamped on the instance as
                :meth:`build_control_system_connector` documents it.

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
        connector, type_config = cls.build_control_system_connector(
            config, control_target=control_target
        )
        await connector.connect(type_config)
        logger.debug(f"Created control system connector: {connector._connector_type}")
        return connector

    @classmethod
    async def create_archiver_connector(cls, config: dict[str, Any] = None) -> ArchiverConnector:
        """
        Create and configure an archiver connector.

        Args:
            config: Archiver configuration dict with keys:
                - type: Connector type (e.g., 'epics_archiver', 'mock_archiver').
                  Defaults to 'mock_archiver' with a warning when unset, so a
                  config that omits the archiver section stays serviceable.
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
                from osprey_connectors.config import get_config_value

                config = get_config_value("archiver", {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        # Fail closed: a config with no `archiver:` section is under-specified,
        # not a declaration that a facility archiver exists. A blank value is
        # treated as absent for the same reason. The fallback lives in
        # types.resolve_archiver_type because the honesty rule's refusals have
        # to answer "what will this build?" with exactly this answer.
        connector_type = types.resolve_archiver_type(config)
        if not config.get("type"):
            logger.warning(
                f"archiver.type is not set; defaulting to '{types.MOCK_ARCHIVER}'. "
                f"Set archiver.type explicitly to select a connector."
            )

        connector_class = cls._archiver_connectors.get(connector_type)

        if not connector_class:
            if "." in connector_type:
                module_path, class_name = connector_type.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    connector_class = getattr(module, class_name)
                except ImportError as e:
                    raise ValueError(
                        f"Could not import connector module '{module_path}': {e}"
                    ) from e
                except AttributeError as e:
                    raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e
                cls._archiver_connectors[connector_type] = connector_class
            else:
                available = list(cls._archiver_connectors.keys())
                raise ValueError(
                    f"Unknown archiver type: '{connector_type}'. "
                    f"Available types: {available}. "
                    f"Use a dotted module path for custom connectors."
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


@dataclass(frozen=True)
class ConnectorRegistrySnapshot:
    """Captured contents of both :class:`ConnectorFactory` registries.

    Treat the two dicts as read-only. They are the restore point, not a working
    copy: mutating either one changes what
    :func:`restore_connector_registries` puts back, so the registries would be
    restored to a state the process was never in.
    """

    control_system: dict[str, type[ControlSystemConnector]] = field(default_factory=dict)
    archiver: dict[str, type[ArchiverConnector]] = field(default_factory=dict)


def snapshot_connector_registries() -> ConnectorRegistrySnapshot:
    """Copy the current control system and archiver registrations.

    Returns:
        Snapshot that :func:`restore_connector_registries` can replay.
    """
    return ConnectorRegistrySnapshot(
        control_system=dict(ConnectorFactory._control_system_connectors),
        archiver=dict(ConnectorFactory._archiver_connectors),
    )


def restore_connector_registries(snapshot: ConnectorRegistrySnapshot) -> None:
    """Replace both registries with the contents of ``snapshot``.

    The restore is total: registrations added after the snapshot are dropped
    and registrations removed after it are put back, so any mutation made in
    between is undone.

    Args:
        snapshot: Value returned by :func:`snapshot_connector_registries`.
    """
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._control_system_connectors.update(snapshot.control_system)
    ConnectorFactory._archiver_connectors.clear()
    ConnectorFactory._archiver_connectors.update(snapshot.archiver)


@contextmanager
def isolated_connector_registries(*, clear: bool = False) -> Iterator[ConnectorRegistrySnapshot]:
    """Bracket registry mutations so the surrounding process state survives.

    This is the sanctioned way for tests to mutate the factory registries.
    Clearing them directly is destructive: registrations made elsewhere (by
    ``initialize_registry()``, by another test module, by a facility preset)
    are gone for the rest of the process, and
    :func:`register_builtin_connectors` cannot tell a deliberately customized
    registry from a truncated one.

    Args:
        clear: Empty both registries on entry, for code that needs to observe
            registration from scratch. The clear is undone on exit like any
            other mutation.

    Yields:
        The snapshot taken on entry, before any ``clear``. It is the restore
        point for the ``finally`` below — read it, do not mutate it.

    Example:
        >>> with isolated_connector_registries(clear=True):
        ...     register_builtin_connectors()
    """
    snapshot = snapshot_connector_registries()
    if clear:
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._archiver_connectors.clear()
    try:
        yield snapshot
    finally:
        restore_connector_registries(snapshot)


_BUILTIN_CONTROL_SYSTEMS = (
    types.MOCK,
    types.EPICS,
    types.VIRTUAL_ACCELERATOR,
    types.DOOCS,
    types.LIVE_STANDIN,
)
_BUILTIN_ARCHIVERS = (
    types.MOCK_ARCHIVER,
    types.EPICS_ARCHIVER,
    types.MONGODB_ARCHIVER,
    types.DOOCS_ARCHIVER,
)


def register_builtin_connectors() -> None:
    """Register all built-in control system and archiver connectors.

    Safe to call multiple times and convergent: every call ensures each
    built-in name is registered, and an existing registration under one of
    those names is never replaced. A registry holding only part of the
    built-ins (e.g., one that was cleared and then partially repopulated)
    therefore heals on the next call instead of keeping the truncated set.
    """
    missing_control_systems = [
        name
        for name in _BUILTIN_CONTROL_SYSTEMS
        if name not in ConnectorFactory._control_system_connectors
    ]
    missing_archivers = [
        name for name in _BUILTIN_ARCHIVERS if name not in ConnectorFactory._archiver_connectors
    ]
    if not missing_control_systems and not missing_archivers:
        return

    # The DOOCS and MongoDB connectors import their optional drivers (doocs4py,
    # pymongo) inside connect(), not at module scope, so they register
    # unconditionally like any other built-in. A machine without the driver
    # only finds out when it tries to connect — which is the point at which it
    # would have failed anyway. Registering MongoDB conditionally instead used
    # to leave it out of _BUILTIN_ARCHIVERS, which broke convergence: a registry
    # holding the other built-ins satisfied the early return above and never
    # healed the missing mongodb_archiver entry.
    from osprey_connectors.archiver.doocs_archiver_connector import DOOCSArchiverConnector
    from osprey_connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
    from osprey_connectors.archiver.mock_archiver_connector import MockArchiverConnector
    from osprey_connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector
    from osprey_connectors.control_system.doocs_connector import DOOCSConnector
    from osprey_connectors.control_system.epics_connector import EPICSConnector
    from osprey_connectors.control_system.mock_connector import MockConnector
    from osprey_connectors.control_system.va_connector import VirtualAcceleratorConnector

    control_systems: list[tuple[str, type[ControlSystemConnector]]] = [
        (types.MOCK, MockConnector),
        (types.EPICS, EPICSConnector),
        (types.VIRTUAL_ACCELERATOR, VirtualAcceleratorConnector),
        (types.DOOCS, DOOCSConnector),
        # The live stand-in is a soft IOC, so the EPICS connector is what reaches
        # it — but it registers under its own name rather than sharing 'epics'.
        # The registry key is what build_control_system_connector() stamps onto
        # the instance as _connector_type, and that stamp is what selects the
        # connector block ('control_system.connector.live_standin') and the write
        # posture read from it. Registering the stand-in as 'epics' would give it
        # the facility's authored block and the facility's arming.
        (types.LIVE_STANDIN, EPICSConnector),
    ]
    archivers: list[tuple[str, type[ArchiverConnector]]] = [
        (types.MOCK_ARCHIVER, MockArchiverConnector),
        (types.EPICS_ARCHIVER, EPICSArchiverConnector),
        (types.MONGODB_ARCHIVER, MongoDBArchiverConnector),
        (types.DOOCS_ARCHIVER, DOOCSArchiverConnector),
    ]

    for name, connector_class in control_systems:
        if name not in ConnectorFactory._control_system_connectors:
            ConnectorFactory.register_control_system(name, connector_class)
    for name, archiver_class in archivers:
        if name not in ConnectorFactory._archiver_connectors:
            ConnectorFactory.register_archiver(name, archiver_class)
