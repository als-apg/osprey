"""
Connector factory for control system and archiver connectors and the service backends.

Provides centralized factory for instantiating and configuring connectors
based on configuration. Connectors are registered through the Osprey registry
system for unified component management and lazy loading. The four read-only
service kinds (``directory``, ``alarm_state``, ``alarm_log``, ``put_log``) are
registered and built through one kind-keyed pair, :meth:`ConnectorFactory.register_service`
and :meth:`ConnectorFactory.create_service`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from osprey_connectors import types
from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from osprey_connectors.alarm_log import AlarmLog
    from osprey_connectors.alarm_state import AlarmState

    # Import only for annotations: pulling in the archiver base eagerly would
    # make pandas a hard dependency of the lean control-system import chain.
    from osprey_connectors.archiver._base import ArchiverConnector
    from osprey_connectors.directory import ChannelDirectory
    from osprey_connectors.put_log import PutLog
    from osprey_connectors.service_answers import BackendSpec

    #: What ``create_service`` returns: the base of the kind the spec names.
    _ServiceBackend = ChannelDirectory | AlarmState | AlarmLog | PutLog

__all__ = [
    "BUILTIN_CONNECTORS",
    "BuiltinConnector",
    "ConnectorFactory",
    "register_builtin_connectors",
]

logger = get_logger("connector_factory")

#: A service kind's name, as ``BackendSpec.kind`` spells it.
_ServiceKind = Literal["directory", "alarm_state", "alarm_log", "put_log"]
_SERVICE_KINDS: tuple[_ServiceKind, ...] = ("directory", "alarm_state", "alarm_log", "put_log")


def _unknown_service_kind(kind: str) -> ValueError:
    return ValueError(f"Unknown service kind: {kind!r}. Service kinds: {list(_SERVICE_KINDS)}")


_Class = TypeVar("_Class")


class _NotASubclassError(TypeError):
    """An imported name is not a subclass of the base its registry holds."""

    def __init__(self, module_path: str, class_name: str, bases: str) -> None:
        super().__init__(f"{module_path}.{class_name} is not a subclass of {bases}")


def _imported(module_path: str, class_name: str) -> object:
    """The name *class_name* of the module *module_path*.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no such name.
    """
    found: object = getattr(importlib.import_module(module_path), class_name)
    return found


def _control_system_class(module_path: str, class_name: str) -> type[ControlSystemConnector]:
    """The control system connector class *class_name* of *module_path*.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no such name.
        _NotASubclassError: The name is not a ``ControlSystemConnector`` subclass.
    """
    found = _imported(module_path, class_name)
    if isinstance(found, type) and issubclass(found, ControlSystemConnector):
        return found
    raise _NotASubclassError(module_path, class_name, "ControlSystemConnector")


def _archiver_class(module_path: str, class_name: str) -> type[ArchiverConnector]:
    """The archiver connector class *class_name* of *module_path*.

    The base is imported here, not at module scope: it would make pandas a hard
    dependency of the lean control-system import chain.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no such name.
        _NotASubclassError: The name is not an ``ArchiverConnector`` subclass.
    """
    from osprey_connectors.archiver._base import ArchiverConnector

    found = _imported(module_path, class_name)
    if isinstance(found, type) and issubclass(found, ArchiverConnector):
        return found
    raise _NotASubclassError(module_path, class_name, "ArchiverConnector")


def _service_class(module_path: str, class_name: str) -> type[_ServiceBackend]:
    """A service backend class: a subclass of one of the four service bases.

    The four bases together are the type the service registries hold. They are
    imported here, not at module scope, so the lean control-system import chain
    does not load the service packages.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no such name.
        _NotASubclassError: The name subclasses none of the four bases.
    """
    from osprey_connectors.alarm_log import AlarmLog
    from osprey_connectors.alarm_state import AlarmState
    from osprey_connectors.directory import ChannelDirectory
    from osprey_connectors.put_log import PutLog

    found = _imported(module_path, class_name)
    if isinstance(found, type) and issubclass(
        found, (ChannelDirectory, AlarmState, AlarmLog, PutLog)
    ):
        return found
    raise _NotASubclassError(
        module_path, class_name, "ChannelDirectory, AlarmState, AlarmLog or PutLog"
    )


def _dotted_class(connector_type: str, load: Callable[[str, str], _Class]) -> _Class:
    """Import the class a dotted out-of-tree type names, through *load*.

    Raises:
        ValueError: The module does not import, has no such class, or the class
            is not a subclass of the base *load* requires.
    """
    module_path, class_name = connector_type.rsplit(".", 1)
    try:
        return load(module_path, class_name)
    except ImportError as e:
        raise ValueError(f"Could not import connector module '{module_path}': {e}") from e
    except AttributeError as e:
        raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e
    except _NotASubclassError as e:
        raise ValueError(f"Connector type '{connector_type}': {e}") from e


def _mapping_section(value: object, path: str) -> Mapping[str, object]:
    """*value* as the config mapping read from *path*.

    Raises:
        ValueError: *value* is not a mapping.
    """
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"'{path}' must be a mapping, got {type(value).__name__}")


class ConnectorFactory:
    """
    Factory for control system and archiver connectors and the service backends.

    Provides centralized management of available connectors. Connectors are
    registered automatically through the Osprey registry system during
    framework initialization. A control system or archiver is built from a
    config dict and connected; a service backend (kind ``directory``,
    ``alarm_state``, ``alarm_log`` or ``put_log``) is built from a ``BackendSpec``
    by :meth:`create_service` and connects nothing.

    Example:
        >>> # Using default config
        >>> cs_connector = await ConnectorFactory.create_control_system_connector()
        >>>
        >>> # Using custom config
        >>> config = {'type': 'epics', 'connector': {'epics': {...}}}
        >>> cs_connector = await ConnectorFactory.create_control_system_connector(config)
    """

    _control_system_connectors: dict[str, type[ControlSystemConnector]] = {}
    _archiver_connectors: dict[str, type[ArchiverConnector]] = {}
    #: Service backend classes by kind, then by the type name a spec selects them by.
    _service_backends: dict[str, dict[str, type[_ServiceBackend]]] = {
        kind: {} for kind in _SERVICE_KINDS
    }

    @classmethod
    def register_control_system(
        cls, name: str, connector_class: type[ControlSystemConnector]
    ) -> None:
        """
        Register a control system connector.

        Args:
            name: Unique name for the connector (e.g., 'epics', 'tango')
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
    def register_service(cls, kind: str, name: str, backend_class: type[_ServiceBackend]) -> None:
        """Register a service backend under the type name a ``BackendSpec`` selects it by.

        Args:
            kind: ``directory``, ``alarm_state``, ``alarm_log`` or ``put_log``.
            name: The type name.
            backend_class: A subclass of that kind's base.

        Raises:
            ValueError: *kind* is not one of the four.
        """
        registry = cls._service_backends.get(kind)
        if registry is None:
            raise _unknown_service_kind(kind)
        registry[name] = backend_class
        logger.debug(f"Registered {kind} backend: {name}")

    @classmethod
    def build_control_system_connector(
        cls, config: Mapping[str, object] | None = None, *, control_target: str | None = None
    ) -> tuple[ControlSystemConnector, dict[str, object]]:
        """
        Build a control system connector and its type config, without connecting.

        Args:
            config: Control system configuration dict, or None to read the
                ``control_system`` section of the config this process loads
                (see :func:`osprey_connectors.config.get_config_value`). Its
                keys:

                - type: Connector type (e.g., 'epics', 'virtual_accelerator').
                  Defaults to the simulator served in process, with a warning,
                  when unset, so an under-specified config never reaches live
                  hardware.
                - connector: Dict with connector-specific configs
            control_target: The control target this connector is being built
                for — ``live``, ``va`` or ``standin``. Stamped on the instance
                as ``_control_target`` and used to index the per-target
                narrowing in the control-context record, and nothing else: the
                deployment half of the write posture stays keyed on the
                connector *type*. Callers
                that have no target to name leave it ``None``, which is the
                behaviour every caller had before the stamp existed.

        Returns:
            ``(connector, type_config)`` — the stamped but *unconnected*
            connector, and the type-specific config block ``connect()`` takes.
            Both halves are needed to finish the build, and the second is not
            derivable from the first, so they travel together.

        Raises:
            ValueError: If connector type is unknown or config is invalid: a
                dotted type that names no ``ControlSystemConnector`` subclass,
                or a section, ``connector`` block or type block that is not a
                mapping.
            FileNotFoundError: If ``config`` is None and no config is found.
                The error names where it looked; nothing is built in its place.

        Example:
            >>> connector, type_config = ConnectorFactory.build_control_system_connector(config)
            >>> await connector.connect(type_config)
        """
        # Load config if not provided. A config that cannot be loaded is the
        # caller's answer, not an empty section: the unset-type fallback below is
        # for a config that was read and names no type.
        if config is None:
            from osprey_connectors.config import get_config_value

            config = _mapping_section(get_config_value("control_system", {}), "control_system")

        # Fail closed: an under-specified config must never bring up a live
        # control system. A blank value is treated as absent for the same
        # reason. The fallback lives in types.resolve_control_system_type so
        # that anything deciding what this call WILL build — the honesty rule's
        # three refusals above all — resolves it identically.
        connector_type = types.resolve_control_system_type(config)
        if not config.get("type"):
            logger.warning(types.UNSET_TYPE_WARNING)

        connector_class = cls._control_system_connectors.get(connector_type)
        # The venue is the factory's choice, made from the same function every
        # check reads the connector's wire from: the simulator served in process
        # is built here, and nowhere else decides it.
        if (
            connector_type == types.VIRTUAL_ACCELERATOR
            and types.resolve_serving(config) == types.IN_PROCESS
        ):
            from osprey_connectors.control_system._va_in_process_connector import (
                VAInProcessConnector,
            )

            connector_class = VAInProcessConnector

        if not connector_class:
            if "." in connector_type:
                connector_class = _dotted_class(connector_type, _control_system_class)
                cls._control_system_connectors[connector_type] = connector_class
            else:
                if connector_type in types.RETIRED_CONTROL_SYSTEM_TYPES:
                    raise ValueError(types.retired_type_message(connector_type))
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
        # On the same seam, the wire this instance speaks: computed once from the
        # type and the serving leaf, and echoed by a host child so its parent can
        # refuse a child that built the other venue.
        connector._transport = types.connector_transport(config, connector_type)
        # Beside it, and for the same reason: the session half of the posture is
        # per target, so the target has to be on the instance before connect()
        # reads it. The two stamps answer different questions — the type selects
        # the deployment's connector block, the target indexes the session
        # posture store — so neither is derivable from the other.
        connector._control_target = control_target

        # Get type-specific configuration
        connector_configs = _mapping_section(
            config.get("connector", {}), "control_system.connector"
        )
        type_config = connector_configs.get(connector_type, {})
        if not isinstance(type_config, dict):
            raise ValueError(
                f"'control_system.connector.{connector_type}' must be a mapping, "
                f"got {type(type_config).__name__}"
            )

        logger.debug(f"Built control system connector: {connector_type}")
        return connector, type_config

    @classmethod
    async def create_control_system_connector(
        cls, config: Mapping[str, object] | None = None, *, control_target: str | None = None
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
            control_target: The control target this connector is being built
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
            >>>             'timeout_s': 5.0,
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
    async def create_archiver_connector(
        cls, config: Mapping[str, object] | None = None
    ) -> ArchiverConnector:
        """
        Create and configure an archiver connector.

        Args:
            config: Archiver configuration dict, or None to read the
                ``archiver`` section of the config this process loads (see
                :func:`osprey_connectors.config.get_config_value`). Its keys:

                - type: Connector type (e.g., 'epics_archiver', 'mock_archiver').
                  Defaults to 'mock_archiver' with a warning when unset, so a
                  config that omits the archiver section stays serviceable.
                - settings: The selected connector's settings, handed to
                  connect() whole. A short-named connector's own-name block is
                  read when settings is absent; both at once raise ValueError.

        Returns:
            Initialized and connected ArchiverConnector

        Raises:
            ValueError: If connector type is unknown or config is invalid: a
                dotted type that names no ``ArchiverConnector`` subclass, or a
                section that is not a mapping.
            FileNotFoundError: If ``config`` is None and no config is found.
                The error names where it looked; nothing is built in its place.
            ConnectionError: If connection fails

        Example:
            >>> config = {
            >>>     'type': 'epics_archiver',
            >>>     'settings': {
            >>>         'url': 'https://archiver.example.org:8443',
            >>>         'timeout_s': 60
            >>>     }
            >>> }
            >>> connector = await ConnectorFactory.create_archiver_connector(config)
        """
        # Load config if not provided. A config that cannot be loaded is the
        # caller's answer, not an empty section: the mock would refuse to
        # connect for want of a simulator view, and say nothing of the config.
        if config is None:
            from osprey_connectors.config import get_config_value

            config = _mapping_section(get_config_value("archiver", {}), "archiver")

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
                connector_class = _dotted_class(connector_type, _archiver_class)
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

        # The selected archiver's settings block. The rule is in types so that the
        # build's refusal and this read give the same answer.
        type_config = types.resolve_archiver_settings(config)

        # Connect with configuration
        await connector.connect(type_config)

        logger.info(f"Created archiver connector: {connector_type}")
        return connector

    @classmethod
    async def create_service(
        cls, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> _ServiceBackend:
        """Build the service backend a spec names. Backends connect nothing here: no I/O.

        The kind is ``spec.kind``. The builtins are registered first: the panel and
        the directory MCP server never register them themselves, and registration
        converges, so every create call can afford to.

        Args:
            spec: The backend to build, from ``service_targets`` or written by hand.
            project_dir: The rendered config's directory.
            **seams: Test seams, passed on only when not ``None``.

        Raises:
            ValueError: The kind is not one of the four, the type is unknown, or a
                dotted type does not resolve to a subclass of one of the four
                service bases.
        """
        registry = cls._service_backends.get(spec.kind)
        if registry is None:
            raise _unknown_service_kind(spec.kind)
        register_builtin_connectors()
        backend_class = registry.get(spec.type)
        if backend_class is None:
            if "." not in spec.type:
                raise ValueError(
                    f"Unknown {spec.kind} type: '{spec.type}'. "
                    f"Available types: {list(registry)}. "
                    f"Use a dotted module path for custom connectors."
                )
            backend_class = _dotted_class(spec.type, _service_class)
            registry[spec.type] = backend_class
        given = {name: seam for name, seam in seams.items() if seam is not None}
        backend: _ServiceBackend = backend_class(spec, project_dir=project_dir, **given)
        return backend

    @classmethod
    def list_control_systems(cls) -> list[str]:
        """List available control system connector types."""
        return list(cls._control_system_connectors.keys())

    @classmethod
    def list_archivers(cls) -> list[str]:
        """List available archiver connector types."""
        return list(cls._archiver_connectors.keys())


@dataclass(frozen=True)
class BuiltinConnector:
    """One connector this package ships, as a registry names it.

    The module path names an implementation module of this package, which only
    this table spells: a registry that loads connectors lazily by path takes
    its entries from here rather than restating the private paths.
    """

    #: The type name a config selects it by, and the key it registers under.
    name: str
    #: Which registry it goes into, and the ``connector_type`` a registration names.
    kind: Literal["control_system", "archiver"]
    module_path: str
    class_name: str
    description: str

    def load(self) -> type[ControlSystemConnector] | type[ArchiverConnector]:
        """Import the module and return the connector class.

        Raises:
            TypeError: The class is not a subclass of the base its kind names.
        """
        if self.kind == "archiver":
            return _archiver_class(self.module_path, self.class_name)
        return _control_system_class(self.module_path, self.class_name)


#: Every built-in connector, in registration order.
BUILTIN_CONNECTORS: tuple[BuiltinConnector, ...] = (
    BuiltinConnector(
        types.EPICS,
        "control_system",
        "osprey_connectors.control_system._epics_connector",
        "EPICSConnector",
        "EPICS Channel Access control system connector",
    ),
    BuiltinConnector(
        types.VIRTUAL_ACCELERATOR,
        "control_system",
        "osprey_connectors.control_system._va_connector",
        "VirtualAcceleratorConnector",
        "Virtual Accelerator connector for PyAT-backed soft-IOC simulations",
    ),
    BuiltinConnector(
        types.DOOCS,
        "control_system",
        "osprey_connectors.control_system._doocs_connector",
        "DOOCSConnector",
        "DOOCS control system connector (requires doocs4py)",
    ),
    BuiltinConnector(
        types.TANGO,
        "control_system",
        "osprey_connectors.control_system._tango_connector",
        "TangoConnector",
        "TANGO Controls control system connector (requires PyTango)",
    ),
    # The live stand-in is a soft IOC, so the EPICS connector is what reaches
    # it — but it registers under its own name rather than sharing 'epics'.
    # The registry key is what build_control_system_connector() stamps onto
    # the instance as its connector type, and that stamp is what selects the
    # connector block ('control_system.connector.live_standin') and the write
    # posture read from it. Registering the stand-in as 'epics' would give it
    # the facility's authored block and the facility's arming.
    BuiltinConnector(
        types.LIVE_STANDIN,
        "control_system",
        "osprey_connectors.control_system._epics_connector",
        "EPICSConnector",
        "Live stand-in connector: the facility-shaped soft IOC a deployment runs "
        "for itself, served over Channel Access",
    ),
    BuiltinConnector(
        types.MOCK_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mock_archiver_connector",
        "MockArchiverConnector",
        "Mock archiver connector for development and testing",
    ),
    BuiltinConnector(
        types.EPICS_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._epics_archiver_connector",
        "EPICSArchiverConnector",
        "EPICS Archiver Appliance connector",
    ),
    BuiltinConnector(
        types.MONGODB_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mongodb_archiver_connector",
        "MongoDBArchiverConnector",
        "MongoDB archiver connector for time-series PV data",
    ),
    BuiltinConnector(
        types.DOOCS_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._doocs_archiver_connector",
        "DOOCSArchiverConnector",
        "DOOCS local history connector (requires doocs4py)",
    ),
    BuiltinConnector(
        types.MYA_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mya_archiver_connector",
        "MYAArchiverConnector",
        "MYA archiver connector, read over the myquery HTTP service "
        "(requires jlab-archiver-client)",
    ),
)

_BUILTIN_CONTROL_SYSTEMS = tuple(c.name for c in BUILTIN_CONNECTORS if c.kind == "control_system")
_BUILTIN_ARCHIVERS = tuple(c.name for c in BUILTIN_CONNECTORS if c.kind == "archiver")


@dataclass(frozen=True)
class _ServiceBuiltin:
    """One service backend this package ships.

    Apart from :class:`BuiltinConnector` because the framework registry reads
    that table, and these kinds must never reach a framework that predates them.
    """

    #: The type name a ``BackendSpec`` selects it by, and the key it registers under.
    name: str
    kind: _ServiceKind
    module_path: str
    class_name: str
    description: str

    def load(self) -> type[_ServiceBackend]:
        """Import the module and return the backend class.

        Raises:
            TypeError: The class subclasses none of the four service bases.
        """
        return _service_class(self.module_path, self.class_name)


#: Every built-in service backend, in registration order. ``simulator_view``
#: and ``alarm_replay`` are internal: ``service_targets`` selects them, and the
#: framework's section resolver refuses them as a configured ``type``.
_SERVICE_BUILTINS: tuple[_ServiceBuiltin, ...] = (
    _ServiceBuiltin(
        "channelfinder",
        "directory",
        "osprey_connectors.directory._channelfinder",
        "ChannelFinderDirectory",
        "ChannelFinder directory service",
    ),
    _ServiceBuiltin(
        "simulator_view",
        "directory",
        "osprey_connectors.directory._simulator",
        "SimulatorDirectory",
        "Every PV the virtual accelerator serves, from the build's simulator view",
    ),
    _ServiceBuiltin(
        "phoebus_alarm_logger",
        "alarm_log",
        "osprey_connectors.alarm_log._phoebus",
        "PhoebusAlarmLogger",
        "Phoebus alarm logger service",
    ),
    _ServiceBuiltin(
        "phoebus_alarm_server",
        "alarm_state",
        "osprey_connectors.alarm_state._phoebus_kafka",
        "PhoebusAlarmState",
        "Phoebus alarm server state, from its Kafka topic (requires aiokafka)",
    ),
    _ServiceBuiltin(
        "alarm_replay",
        "alarm_state",
        "osprey_connectors.alarm_state._replay",
        "ReplayAlarmState",
        "Alarm state replayed from a file, for the simulator",
    ),
    _ServiceBuiltin(
        "caputlog",
        "put_log",
        "osprey_connectors.put_log._caputlog",
        "CaPutLog",
        "caPutLog put history service",
    ),
)


def register_builtin_connectors() -> None:
    """Register all built-in control system and archiver connectors and service backends.

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
    missing_service_rows = [
        row
        for row in _SERVICE_BUILTINS
        if row.name not in ConnectorFactory._service_backends[row.kind]
    ]
    if not missing_control_systems and not missing_archivers and not missing_service_rows:
        return

    # The DOOCS and MongoDB connectors import their optional drivers (doocs4py,
    # pymongo) inside connect(), not at module scope, so they register
    # unconditionally like any other built-in. A machine without the driver
    # only finds out when it tries to connect — which is the point at which it
    # would have failed anyway. Registering MongoDB conditionally instead used
    # to leave it out of _BUILTIN_ARCHIVERS, which broke convergence: a registry
    # holding the other built-ins satisfied the early return above and never
    # healed the missing mongodb_archiver entry.
    for builtin in BUILTIN_CONNECTORS:
        if builtin.kind == "control_system":
            if builtin.name not in ConnectorFactory._control_system_connectors:
                ConnectorFactory.register_control_system(
                    builtin.name, _control_system_class(builtin.module_path, builtin.class_name)
                )
        elif builtin.kind == "archiver":
            if builtin.name not in ConnectorFactory._archiver_connectors:
                ConnectorFactory.register_archiver(
                    builtin.name, _archiver_class(builtin.module_path, builtin.class_name)
                )
        else:
            raise ValueError(
                f"built-in connector {builtin.name!r} has unknown kind {builtin.kind!r}"
            )

    # Their optional client (aiokafka) is imported on start, not at module scope.
    for row in missing_service_rows:
        ConnectorFactory._service_backends[row.kind][row.name] = row.load()
