"""The queueserver RE-worker startup script: devices, plans, and the document plane.

This file is mounted into the ``queueserver`` container and handed to
``start-re-manager`` as its ``--startup-script``. The manager spawns the RE
worker, which ``exec``s this file into the worker's namespace (with
``__name__`` patched to ``"__main__"``) and then scans that namespace for the
``RE`` instance, the plans, and the devices it will expose over its 0MQ
control channel. Everything the worker can ever run therefore has to be built
here, in this process — the bridge is a client of the manager, not its parent.

Three things are assembled, in this order:

1. **Devices** — connector-mediated ``ConnectorSettable``/``ConnectorReadable``
   instances built from the substrate env (``BLUESKY_EPICS_SUBSTRATE`` +
   ``BLUESKY_EPICS_MOTORS``/``BLUESKY_EPICS_DETECTORS``), exactly as the
   bridge's in-process wiring built them: every scan read goes through
   ``connector.read_channel`` and every scan write through
   ``connector.write_channel_checked``. Moving execution out of the bridge
   moves the reference monitor here with it — there is no raw Channel Access
   in this process either.
2. **Plans** — the bridge's own layered plan catalog (``plan_loader``, driven
   by the same ``BLUESKY_PLAN_DIRS`` the bridge reads), each catalog entry
   exposed as a named generator function whose ``**kwargs`` are validated
   back into that plan's ``PARAMS`` model and whose device names resolve
   against the device set built in step 1.
3. **The document plane** — a ``TiledWriter`` subscription (gated on
   ``BLUESKY_TILED_URI``) for durable run data, and a
   ``bluesky.callbacks.zmq`` ``Publisher`` (gated on
   ``BLUESKY_ZMQ_PUBLISH_ADDR``) that streams every document to the
   ``zmq.Proxy`` the bridge runs, which is how the bridge's live-row buffer
   sees a run it is no longer executing itself. Both subscriptions are
   fault-isolated: a Tiled outage or a dead proxy degrades telemetry, it never
   aborts a scan.

**Browse-only is the failure mode.** With no substrate env — the mock
connector case — no devices are built, no plans are registered, and the
namespace ends up holding only ``RE``. The script does not raise: the manager
comes up healthy with an empty allowed-plan list, so the queue surface is
fail-closed (nothing to enqueue) instead of offering plans that would fail at
execution time against devices that were never built.

Self-contained on purpose: this module reads its own env and builds its own
connector rather than importing the bridge's in-process runner wiring, which
the queue backend replaces. It runs in a different container from ``app.py``
and must keep working when that wiring is gone.

**NEVER use a relative import in this file — not even inside a function.** In
production this module is not imported at all: it is the manager's
``--startup-script``, and upstream's ``load_startup_script`` ``exec``s the file
with ``__name__`` patched to ``"__main__"`` and no ``__package__``, so ``from
.devices import ...`` raises ``ImportError: attempted relative import with no
known parent package`` and the worker environment never opens. Spell every
import absolutely (``from osprey.services.bluesky_bridge... import``). The unit
tests import this module normally, as a package member, where relative imports
resolve fine — so they cannot catch a regression here; the AST pin in
``tests/services/bluesky_bridge/test_qserver_startup.py`` is what does.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from typing import Any

logger = logging.getLogger("osprey.services.bluesky_bridge.qserver_startup")

SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
"""Opt-in flag: build real connector-mediated devices. Absent/false means
browse-only (see the module docstring)."""

TILED_URI_ENV = "BLUESKY_TILED_URI"
"""Tiled server URI. Absent means no ``TiledWriter`` subscription at all."""

TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"
"""Tiled API key. Grants catalog access only, never launch authority."""

ZMQ_PUBLISH_ADDR_ENV = "BLUESKY_ZMQ_PUBLISH_ADDR"
"""Address of the bridge's ``zmq.Proxy`` in-side, which this process's
``Publisher`` connects to. Absent means no document publishing — the bridge
then has no live rows for a run, and falls back to Tiled."""

ZMQ_CURVE_SECRET_KEY_ENV = "BLUESKY_ZMQ_CURVE_SECRET_KEY"
"""Path to *this* process's CURVE secret certificate (the ``ClientCurve``
client half). Set together with :data:`ZMQ_CURVE_SERVER_PUBLIC_KEY_ENV`."""

ZMQ_CURVE_SERVER_PUBLIC_KEY_ENV = "BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY"
"""Path to the bridge proxy's public CURVE certificate. Set together with
:data:`ZMQ_CURVE_SECRET_KEY_ENV`; setting exactly one of the two is a
misconfiguration and is refused rather than silently publishing unencrypted."""

_TRUTHY_VALUES = {"1", "true", "yes", "on"}

_EPICS_LIKE_CONNECTOR_TYPES = ("virtual_accelerator", "epics")
"""Connector types that get a gateway-less ``type_config`` — real Channel
Access, whether a virtual-accelerator soft-IOC or live hardware. A gateway-less
config makes ``connect()`` skip the block that sets process-wide ``EPICS_CA_*``
env, so the compose-inherited ``EPICS_CA_NAME_SERVERS`` survives untouched."""

CONNECT_TIMEOUT = 30.0
"""Seconds :func:`build_namespace` waits for the async device build to finish
on the RunEngine's loop. Generous enough for real Channel Access connects
(IOC startup, network hiccups), bounded so a dead IOC fails the environment
open instead of hanging the manager forever."""


def is_substrate_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True if ``BLUESKY_EPICS_SUBSTRATE`` is set to any of ``1/true/yes/on``.

    Absent, empty, or any other value (e.g. ``"false"``) is off. Deliberately
    liberal on the "on" spellings, but never guesses at "on" from an
    unrecognized value — matching how the bridge parses the same flag.
    """
    env = os.environ if env is None else env
    return env.get(SUBSTRATE_ENV, "").strip().lower() in _TRUTHY_VALUES


def resolve_control_system_type() -> str:
    """Read ``control_system.type`` from the mounted project config.

    Single source of truth: one config line flips the whole Bluesky stack
    between the mock connector and real Channel Access. Fail-SAFE default of
    ``"mock"`` whenever the config cannot be read at all (no project config
    context, or a transient lookup failure) — the mock connector never touches
    Channel Access, so an unreadable config can never silently connect this
    worker to real hardware.
    """
    from osprey.utils.config import get_config_value

    try:
        control_system_type = get_config_value("control_system.type", "mock")
    except (FileNotFoundError, KeyError, RuntimeError):
        return "mock"

    if not control_system_type or not isinstance(control_system_type, str):
        return "mock"
    return control_system_type


def build_connector_config(control_system_type: str) -> dict[str, Any]:
    """The ``type_config`` mapping ``ConnectorFactory`` consumes for ``control_system_type``.

    EPICS-like types get a gateway-less config (see
    :data:`_EPICS_LIKE_CONNECTOR_TYPES`) with a connect timeout; anything else
    is forwarded through with no type-specific config, so an unrecognized value
    surfaces as ``ConnectorFactory``'s own "Unknown control system type" error
    rather than being silently mis-wired to a connector nobody asked for.
    """
    if control_system_type in _EPICS_LIKE_CONNECTOR_TYPES:
        return {
            "type": control_system_type,
            "connector": {control_system_type: {"timeout": 5.0}},
        }
    return {"type": control_system_type, "connector": {control_system_type: {}}}


async def create_connector() -> Any:
    """Construct and connect the worker's single long-lived OSPREY connector.

    One connector per worker process, shared by every device: the same
    "connector is the single control-system interface" shape the bridge used
    when it executed plans in-process.
    """
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

    control_system_type = resolve_control_system_type()
    register_builtin_connectors()  # idempotent; must run before create
    connector = await ConnectorFactory.create_control_system_connector(
        build_connector_config(control_system_type)
    )
    logger.info(
        "qserver_startup: connected the worker's OSPREY connector (control_system.type=%s, %s)",
        control_system_type,
        type(connector).__name__,
    )
    return connector


async def build_devices(
    env: Mapping[str, str] | None = None, *, connector: Any = None
) -> dict[str, Any]:
    """Build the worker's connector-mediated device set from the substrate env.

    Returns an empty mapping — never raises — when the substrate is disabled
    (browse-only) or when the PV-list env vars name no devices at all. A caller
    treats an empty result as "this worker cannot execute plans", which is
    exactly what :func:`build_namespace` does with it.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.
        connector: An already-connected OSPREY connector to mediate every
            device's reads and writes. Defaults to constructing one via
            :func:`create_connector`.

    Returns:
        Mapping of device name to connected device.
    """
    env = os.environ if env is None else env
    if not is_substrate_enabled(env):
        logger.info(
            "qserver_startup: %s is not set — building no devices (browse-only worker)",
            SUBSTRATE_ENV,
        )
        return {}

    # Absolute, not relative — see the module docstring: this file is exec'd as
    # a script with no package context, so `from .devices import ...` cannot
    # resolve and takes the whole worker environment down with it.
    from osprey.services.bluesky_bridge.devices import connector as connector_devices
    from osprey.services.bluesky_bridge.devices._specs_from_env import specs_from_env

    motors, detectors = specs_from_env(env)
    if not motors and not detectors:
        logger.warning(
            "qserver_startup: %s is enabled but BLUESKY_EPICS_MOTORS / "
            "BLUESKY_EPICS_DETECTORS name no devices; this worker will expose no plans",
            SUBSTRATE_ENV,
        )
        return {}

    if connector is None:
        connector = await create_connector()

    devices = dict(await connector_devices.build_devices(motors, detectors, connector))
    logger.info(
        "qserver_startup: built %d connector-mediated device(s) (%d motor(s), %d detector(s))",
        len(devices),
        len(motors),
        len(detectors),
    )
    return devices


def _make_plan_function(spec: Any, devices: Mapping[str, Any]) -> Callable[..., Iterator[Any]]:
    """Wrap one catalog ``PlanSpec`` as a qserver-executable plan function.

    The wrapper is a *generator function* by necessity, not by style:
    queueserver's namespace scan (``existing_plans_and_devices_from_nspace``)
    only recognizes generator functions as plans and silently ignores everything
    else, so a plain function returning ``spec.plan(...)`` would make the plan
    invisible to the manager. The consequence is that ``**kwargs`` validation
    happens on first iteration — when the RunEngine starts the item — rather
    than at call time; queueserver reports that as a failed item with the
    pydantic error attached, which is the legible outcome either way.

    ``**kwargs`` (VAR_KEYWORD) is the whole parameter surface: the queue item's
    ``kwargs`` ARE the plan's ``PARAMS`` fields, reconstructed here via
    ``spec.schema.model_validate``. That keeps one schema — the plan's own
    pydantic model, already served to the panels by ``GET /plans`` — as the
    single validation authority, instead of restating it as a Python signature
    for queueserver to re-derive.

    Device names inside ``params`` are resolved against ``devices`` by
    ``spec.plan`` itself. A name that isn't in the namespace surfaces as a
    ``KeyError`` naming the missing device *and* listing what this worker
    actually has, rather than a bare key.
    """

    def plan_function(**kwargs: Any) -> Iterator[Any]:
        params = spec.schema.model_validate(kwargs)
        try:
            plan = spec.plan(dict(devices), params)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            raise KeyError(
                f"plan {spec.name!r} referenced device {missing!r}, which this worker "
                f"did not build; available devices: {sorted(devices)}"
            ) from exc
        return (yield from plan)

    plan_function.__name__ = spec.name
    plan_function.__qualname__ = spec.name
    plan_function.__doc__ = spec.description or f"OSPREY catalog plan {spec.name!r}."
    # REAL annotation objects, not the PEP 563 strings this module's
    # `from __future__ import annotations` would otherwise leave behind.
    # Upstream builds each plan's validation model straight off
    # `inspect.signature`, wrapping a VAR_KEYWORD parameter's annotation as
    # `typing.Dict[str, <annotation>]` and handing it to `pydantic.create_model`.
    # A STRING there is an unresolvable forward reference in pydantic's own
    # namespace, and the manager then refuses EVERY enqueue with
    # "`Model` is not fully defined; you should define `Any`" — a failure that
    # looks like a plan problem and is really an annotation-representation
    # problem. Rebinding here keeps the future-import (which the rest of the
    # file wants) without making this one wrapper uninspectable.
    plan_function.__annotations__ = {"kwargs": Any, "return": Iterator[Any]}
    return plan_function


def build_plan_functions(
    devices: Mapping[str, Any], plans: Mapping[str, Any] | None = None
) -> dict[str, Callable[..., Iterator[Any]]]:
    """Expose every catalog plan as a named, queueserver-executable plan function.

    Args:
        devices: The worker's device set; each plan function closes over it.
        plans: Catalog to expose, as ``plan_loader``'s ``{name: PlanSpec}``.
            Defaults to ``get_facility_plans().plans`` — the same layered
            catalog (``shipped``/``preset``/``facility``/``session``, driven by
            ``BLUESKY_PLAN_DIRS``) the bridge serves from ``GET /plans``, so the
            two surfaces cannot drift.

    Returns:
        Mapping of plan name to generator function, ready to drop into the
        worker namespace. A plan whose name is not a Python identifier is
        skipped with a warning: the namespace key IS the name queueserver
        exposes, and a non-identifier key would be unreachable anyway.
    """
    if plans is None:
        # Absolute, not relative — see the module docstring.
        from osprey.services.bluesky_bridge.plan_loader import get_facility_plans

        plans = get_facility_plans().plans

    functions: dict[str, Callable[..., Iterator[Any]]] = {}
    for name, spec in plans.items():
        if not name.isidentifier():
            logger.warning(
                "qserver_startup: skipping plan %r — not a valid Python identifier, "
                "so it cannot be exposed in the worker namespace",
                name,
            )
            continue
        functions[name] = _make_plan_function(spec, devices)
    return functions


class _FaultIsolatedCallback:
    """Wraps a ``(name, doc)`` RunEngine callback so its failure never aborts a run.

    The RunEngine does not swallow callback exceptions — one escaping
    ``__call__`` aborts the running plan. Both document-plane subscriptions
    (Tiled persistence and the 0MQ publisher) are telemetry: losing them
    degrades what an operator can see afterwards, which must never be worth
    killing a scan that is currently moving magnets. This wrapper catches any
    exception, latches ``degraded``, logs once with a traceback, and
    short-circuits every later document.
    """

    def __init__(self, inner: Callable[[str, dict[str, Any]], Any], label: str) -> None:
        self._inner = inner
        self._label = label
        self.degraded = False

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        if self.degraded:
            return
        try:
            self._inner(name, doc)
        except Exception:
            self.degraded = True
            logger.error(
                "qserver_startup: %s failed on %r document; that stream is now degraded",
                self._label,
                name,
                exc_info=True,
            )


def build_tiled_writer(env: Mapping[str, str] | None = None) -> Any | None:
    """Build the ``TiledWriter`` document callback, or ``None`` if Tiled is unconfigured.

    ``None`` when ``BLUESKY_TILED_URI`` is unset — identical to a deploy with
    no Tiled server, where run data lives only in the bridge's live-row buffer.
    Imports ``TiledWriter`` from ``bluesky.callbacks.tiled_writer``, not from
    ``tiled``.
    """
    env = os.environ if env is None else env
    uri = env.get(TILED_URI_ENV)
    if not uri:
        return None

    from bluesky.callbacks.tiled_writer import TiledWriter

    return TiledWriter.from_uri(uri, api_key=env.get(TILED_API_KEY_ENV))


def build_zmq_publisher(env: Mapping[str, str] | None = None) -> Any | None:
    """Build the 0MQ document ``Publisher``, or ``None`` if no address is configured.

    The publisher connects to the ``zmq.Proxy`` the bridge runs; the bridge's
    ``RemoteDispatcher`` on the other side is what lets it show live rows for a
    run this process is executing.

    CURVE key authentication is applied when *both*
    ``BLUESKY_ZMQ_CURVE_SECRET_KEY`` and
    ``BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY`` are set. Setting exactly one raises
    ``ValueError`` rather than falling back to an unencrypted socket: the
    queueserver container is dual-homed onto the shared network, so silently
    publishing in the clear because one env var was misspelled is precisely the
    failure this key-auth exists to prevent.

    Raises:
        ValueError: Exactly one of the two CURVE key env vars is set.
    """
    env = os.environ if env is None else env
    address = env.get(ZMQ_PUBLISH_ADDR_ENV)
    if not address:
        return None

    secret_key = env.get(ZMQ_CURVE_SECRET_KEY_ENV)
    server_public_key = env.get(ZMQ_CURVE_SERVER_PUBLIC_KEY_ENV)
    if bool(secret_key) != bool(server_public_key):
        raise ValueError(
            f"{ZMQ_CURVE_SECRET_KEY_ENV} and {ZMQ_CURVE_SERVER_PUBLIC_KEY_ENV} must be set "
            "together — refusing to publish run documents on an unauthenticated socket"
        )

    from bluesky.callbacks.zmq import ClientCurve, Publisher

    curve_config = None
    if secret_key and server_public_key:
        curve_config = ClientCurve(secret_path=secret_key, server_public_key=server_public_key)
    return Publisher(address, curve_config=curve_config)


def subscribe_document_callbacks(
    run_engine: Any, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Subscribe the env-gated document-plane callbacks to ``run_engine``.

    Each callback is wrapped in :class:`_FaultIsolatedCallback` and subscribed
    inside its own guard, so neither a Tiled server that is down at environment
    open nor a proxy that never came up can fail the environment open — the
    worker starts, and the missing stream is logged and degraded.

    Returns:
        Mapping of stream label (``"tiled"``, ``"zmq"``) to the wrapper that
        was subscribed. A stream the env did not configure is absent from the
        mapping entirely, which is how a caller tells "not configured" from
        "configured but degraded".
    """
    env = os.environ if env is None else env
    subscribed: dict[str, Any] = {}
    for label, builder in (("tiled", build_tiled_writer), ("zmq", build_zmq_publisher)):
        try:
            inner = builder(env)
        except Exception:
            logger.error(
                "qserver_startup: could not build the %s document callback; "
                "runs will execute without it",
                label,
                exc_info=True,
            )
            continue
        if inner is None:
            continue
        wrapper = _FaultIsolatedCallback(inner, label)
        try:
            run_engine.subscribe(wrapper)
        except Exception:
            wrapper.degraded = True
            logger.error(
                "qserver_startup: could not subscribe the %s document callback; "
                "runs will execute without it",
                label,
                exc_info=True,
            )
            continue
        subscribed[label] = wrapper
        logger.info("qserver_startup: subscribed the %s document callback", label)
    return subscribed


def _await_on_loop(coro: Any, loop: Any) -> Any:
    """Run ``coro`` to completion on ``loop`` from this thread, bounded by ``CONNECT_TIMEOUT``.

    Device construction is ``async`` and the devices it builds bind their I/O
    to whichever loop awaits them, so it has to be awaited on the RunEngine's
    own loop — the one the run itself will drive — not a throwaway one.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=CONNECT_TIMEOUT)
    except TimeoutError:
        future.cancel()
        raise


def build_namespace(
    env: Mapping[str, str] | None = None,
    *,
    run_engine: Any = None,
    devices: Mapping[str, Any] | None = None,
    plans: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the worker namespace: ``RE``, the devices, and the plan functions.

    The RunEngine is created here (queueserver takes ``RE`` from the namespace
    the startup script leaves behind) and created *first*, because its event
    loop is what the async device build is awaited on.

    Plan functions are registered only when at least one device was built. A
    device-less worker is the browse-only case, and exposing plans there would
    advertise executable work that could only fail at run time; leaving the
    namespace with just ``RE`` makes the manager report an empty allowed-plan
    list, so the queue is fail-closed by construction.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.
        run_engine: An existing RunEngine to populate instead of constructing
            one — for tests, and for a deploy that wants its own metadata.
        devices: Pre-built device mapping, skipping the substrate build.
        plans: Catalog override passed through to :func:`build_plan_functions`.

    Returns:
        The namespace mapping, ready to ``globals().update(...)``.
    """
    env = os.environ if env is None else env

    if run_engine is None:
        from bluesky import RunEngine

        run_engine = RunEngine()

    if devices is None:
        devices = _await_on_loop(build_devices(env), run_engine.loop)

    namespace: dict[str, Any] = {"RE": run_engine}
    namespace.update(devices)

    if not devices:
        logger.warning(
            "qserver_startup: no devices were built — registering no plans "
            "(browse-only worker; the queue has nothing to execute)"
        )
    else:
        plan_functions = build_plan_functions(devices, plans)
        namespace.update(plan_functions)
        logger.info(
            "qserver_startup: namespace ready with %d device(s) and %d plan(s)",
            len(devices),
            len(plan_functions),
        )

    subscribe_document_callbacks(run_engine, env)
    return namespace


# Queueserver `exec`s a startup script with `__name__` patched to "__main__"
# (see `bluesky_queueserver.manager.profile_ops.load_startup_script`), so this
# guard is exactly "am I being run as the startup script?" — importing this
# module normally, as the unit tests do, leaves the wiring untouched.
if __name__ == "__main__":  # pragma: no cover - exercised only by a real RE worker
    from osprey.utils.logger import configure_logging

    configure_logging()
    globals().update(build_namespace())
