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
   instances built from the device file named by ``BLUESKY_DEVICES_FILE``,
   exactly as the bridge's in-process wiring built them: every plan read goes
   through ``connector.read_channel`` and every plan write through
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
   aborts a plan.

**Browse-only is the failure mode.** With no readable device file — the mock
connector case — no devices are built, no plans are registered, and the
namespace ends up holding only ``RE``. The script does not raise: the manager
comes up healthy with an empty allowed-plan list, so the queue surface is
fail-closed (nothing to enqueue) instead of offering plans that would fail at
execution time against devices that were never built.

Self-contained on purpose: this module reads its own env and builds its own
connector rather than importing the bridge's in-process runner wiring, which
the queue backend replaces. It runs in a different container from ``app.py``
and must keep working when that wiring is gone. The one thing it does import
from the bridge's side is ``queue_backend``'s lane resolver — which machine
this worker binds to is the same question the bridge answers on its capability
record, and one ladder is what keeps the two containers from describing the
lane differently.

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
import inspect
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.services.bluesky_bridge.qserver_startup")

DEVICES_FILE_ENV = "BLUESKY_DEVICES_FILE"
"""Path to the worker's device file, as mounted into this container.

Its presence *is* the substrate switch. The file lists every channel this
worker exposes as a scan device, so a deploy that mounts one gets real
connector-mediated devices and a deploy that does not is browse-only — there is
no separate on/off flag to keep in step with it. Unset, or naming something
this process cannot read as a file, means no devices are built."""

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


def resolve_control_system_type() -> str:
    """The connector type this worker builds, by the lane resolution ladder.

    A single-lane deployment — every project rendered before the lane axis
    existed — declares no target, and the ladder's first rung is then exactly
    what this function has always answered: ``control_system.type``, fail-SAFE
    to ``"mock"`` when the config cannot be read at all, because the mock
    connector never touches Channel Access. A lane that DOES declare a target
    resolves through it instead, which is how two lanes over one mounted
    config.yml build two different connectors.

    Delegated to ``queue_backend`` rather than restated here even though this
    module is otherwise self-contained: the ladder decides which machine this
    worker's devices bind to, and the bridge beside it publishes the same
    answer on its capability record. Two spellings of that is one of them
    being wrong about a machine somebody can move.
    """
    from osprey.services.bluesky_bridge.queue_backend import resolve_lane_connector_type

    return resolve_lane_connector_type()[0]


def worker_writes_enabled() -> bool:
    """Whether this worker is armed to drive the machine its lane addresses.

    The same answer the reference monitor inside the connector reaches, from
    the same inputs — stated here because the type the ladder lands on is the
    whole point of the lane axis (a VA lane beside a live baseline is armed by
    the VA block alone), and a worker that comes up unarmed should say so at
    startup rather than leave an operator to discover it on a refused write.

    Which posture applies follows the rung the ladder stopped on:

    * a lane whose target resolved is armed by its own type's block,
      ``control_system.connector.<type>.writes_enabled``, inheriting the
      deployment-wide key when that block says nothing;
    * a DEGRADED lane (rung 3) is armed by ``control_system.writes_enabled``
      and nothing else. It was built as the deployment baseline while
      addressing a different machine, so the baseline type's block describes a
      machine this lane does not talk to — arming a facility gateway on the
      strength of "you may write to the simulator" is exactly the confusion the
      per-type posture exists to prevent. The deployment-wide key is the only
      thing such a config has ever said about this lane.

    False whenever the config cannot be read, and False in a readonly run
    whatever the deployment says.
    """
    from osprey.services.bluesky_bridge.queue_backend import resolve_lane_connector_type
    from osprey.utils.config import get_config_value
    from osprey_connectors.control_system.base import is_readonly_run
    from osprey_connectors.types import WRITES_ENABLED_KEY, type_writes_enabled

    connector_type, lane_degraded = resolve_lane_connector_type()
    try:
        if lane_degraded:
            armed = get_config_value(WRITES_ENABLED_KEY, False) is True
        else:
            armed = type_writes_enabled(get_config_value("control_system", {}), connector_type)
    except (FileNotFoundError, KeyError, RuntimeError):
        return False
    return armed and not is_readonly_run()


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
    from osprey.services.bluesky_bridge.queue_backend import resolve_lane_connector_type

    control_system_type, lane_degraded = resolve_lane_connector_type()
    if lane_degraded:
        logger.warning("qserver_startup: %s", lane_degraded)
    register_builtin_connectors()  # idempotent; must run before create
    connector = await ConnectorFactory.create_control_system_connector(
        build_connector_config(control_system_type)
    )
    if lane_degraded:
        # The factory stamps the type it built, and the reference monitor keys
        # this deployment's per-type write posture on that stamp. A degraded
        # lane was built as the baseline type while addressing a machine this
        # config never tied to that type, so the type's block does not describe
        # it: clearing the stamp is what makes the monitor read
        # `control_system.writes_enabled` — the only posture the config has
        # ever stated about this lane — which is also what
        # `worker_writes_enabled` reports.
        connector._connector_type = None
    logger.info(
        "qserver_startup: connected the worker's OSPREY connector (type=%s, %s, writes %s)",
        control_system_type,
        type(connector).__name__,
        "enabled" if worker_writes_enabled() else "disabled",
    )
    return connector


async def build_devices(
    env: Mapping[str, str] | None = None, *, connector: Any = None
) -> dict[str, Any]:
    """Build the worker's connector-mediated device set from the device file.

    The substrate is enabled exactly when :data:`DEVICES_FILE_ENV` is set *and*
    names a file this process can read — file presence is the switch, so there
    is no second flag that can disagree with it.

    Returns an empty mapping — never raises — when the substrate is disabled
    (browse-only) or when the device file names no devices at all. A caller
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
    devices_file = (env.get(DEVICES_FILE_ENV) or "").strip()
    if not devices_file:
        logger.info(
            "qserver_startup: %s is not set — building no devices (browse-only worker)",
            DEVICES_FILE_ENV,
        )
        return {}

    path = Path(devices_file)
    if not path.is_file() or not os.access(path, os.R_OK):
        logger.warning(
            "qserver_startup: %s names %r, which this worker cannot read as a file — "
            "building no devices (browse-only worker)",
            DEVICES_FILE_ENV,
            devices_file,
        )
        return {}

    # Absolute, not relative — see the module docstring: this file is exec'd as
    # a script with no package context, so `from .devices import ...` cannot
    # resolve and takes the whole worker environment down with it.
    from osprey.services.bluesky_bridge.devices import connector as connector_devices
    from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

    setpoints, readbacks = specs_from_file(path)
    if not setpoints and not readbacks:
        logger.warning(
            "qserver_startup: device file %r names no devices; this worker will expose no plans",
            devices_file,
        )
        return {}

    if connector is None:
        connector = await create_connector()

    devices = dict(await connector_devices.build_devices(setpoints, readbacks, connector))
    logger.info(
        "qserver_startup: built %d connector-mediated device(s) (%d setpoint(s), %d readback(s))",
        len(devices),
        len(setpoints),
        len(readbacks),
    )
    return devices


def _declared_devices(schema: Any, params: Any, devices: Mapping[str, Any]) -> dict[str, Any]:
    """The devices one plan may resolve: the channels its params *declare*.

    A plan's ``PARAMS`` model declares, per field, whether the channels named
    there are movable (the plan drives them) or readable (it records them).
    That declaration is the plan's whole claim on this worker, so it is also
    the bound: the mapping built here holds the declared channels and nothing
    else, and a plan reaching for a device it never declared finds it absent
    however many devices the worker actually built.

    Names the worker did not build are simply left out rather than raising —
    the wrapper's own handler is what turns that into a legible failure, and it
    treats both misses the same way.
    """
    # Absolute, not relative — see the module docstring.
    from osprey.services.bluesky_bridge.plan_fields import declared_channels

    return {name: devices[name] for name in declared_channels(schema, params) if name in devices}


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
    ``spec.plan`` itself — but only against the channels the params *declare*
    (see :func:`_declared_devices`), never the worker's whole device set. A
    name the plan cannot resolve surfaces as a ``KeyError`` saying which of the
    two reasons applies — the worker never built it, or the parameter naming it
    declares no channel role — and listing what this worker actually has,
    rather than a bare key.
    """

    def plan_function(**kwargs: Any) -> Iterator[Any]:
        params = spec.schema.model_validate(kwargs)
        try:
            plan = spec.plan(_declared_devices(spec.schema, params, devices), params)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            if missing in devices:
                raise KeyError(
                    f"plan {spec.name!r} referenced device {missing!r}, which its parameters "
                    f"do not declare as a movable or readable channel — a plan resolves only "
                    f"the channels it declares; available devices: {sorted(devices)}"
                ) from exc
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


PREVIEW_MOVE_CAP = 200
"""Most channel moves one :func:`preview_plan` response carries back.

A bound on the *payload*, not on the walk: past the cap the preview stops
collecting moves but keeps counting them, so ``total_moves`` is always the
exact number of moves the run would make and ``truncated`` says whether the
list is the whole trajectory or its opening slice.

Sized against what reads it. The one consumer is the approval prompt, which
names a handful of moves from each end of the trajectory and elides the middle
as a count, so a couple of hundred is already a generous slice and ten thousand
was payload nothing asked for. Keeping it small also keeps the *contract* cheap
to test: proving that the list truncates while the total stays exact needs a
plan of just over ``PREVIEW_MOVE_CAP`` moves, and the walk that proves it costs
in proportion. It says nothing about how large a plan can be previewed — the
walk runs to exhaustion whatever this is, and a plan too long to walk inside the
caller's budget comes back ``preview_timed_out`` regardless of the cap.
"""

PREVIEW_ERROR_CHARS = 2000
"""Length bound on a failed preview's ``error`` text. A pydantic validation
report over a deeply nested params model can run to many kilobytes, and this
payload crosses a 0MQ hop to end up in an operator's approval prompt; the head
of the message names the offending field, which is the part that is worth
carrying."""


def _preview_channel(obj: Any) -> str:
    """The channel label to report for a message's target object.

    ``name`` is what every device in this worker's namespace is keyed by, and
    it is the same string a plan's movable/readable fields name, so the
    trajectory joins directly to the params an operator is approving. Falls
    back to ``repr`` for a target that carries no name rather than dropping the
    move — a move whose channel cannot be labeled is still a move.
    """
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return repr(obj)


def _json_safe(value: Any) -> Any:
    """Reduce one move target to a plain JSON-serializable scalar.

    Targets arrive as whatever the plan computed — commonly a numpy scalar out
    of ``numpy.linspace``, which no JSON encoder accepts. ``.item()`` unwraps
    those to plain Python; anything left that is not already a JSON scalar is
    carried as its ``repr`` so an exotic target degrades to a readable string
    instead of failing the whole response at encode time.
    """
    unwrap = getattr(value, "item", None)
    if callable(unwrap):
        try:
            value = unwrap()
        except Exception:  # noqa: BLE001 - a non-scalar `.item()`; fall through to repr
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def collect_channel_moves(
    messages: Any, *, cap: int = PREVIEW_MOVE_CAP
) -> tuple[list[dict[str, Any]], int]:
    """Walk a plan's message stream read-only, collecting the moves it would make.

    Pure iteration, and that is the whole safety argument: a bluesky plan is a
    generator of *instructions*, and every instruction that changes machine
    state is carried out by the RunEngine when it consumes one. Walking the
    stream here — no RunEngine, nothing sent back into the generator, no device
    method ever called — reads the trajectory the plan would drive without
    driving any of it.

    Iteration continues past ``cap``: the plan's contract is an exact total, so
    the walk always runs the generator to exhaustion and only stops *collecting*
    at the cap. A plan that never terminates would therefore spin here; the
    walk is CPU-only and touches nothing, and the caller runs it as a background
    task under its own timeout, which is the bound that applies.

    Args:
        messages: The plan generator, or any iterable of bluesky ``Msg``\\ s.
        cap: Most moves to collect into the returned list.

    Returns:
        ``(moves, total)`` — the collected ``{"channel", "target"}`` mappings in
        the order the plan would make them, and the exact count of moves the
        whole stream carried.
    """
    moves: list[dict[str, Any]] = []
    total = 0
    for message in messages:
        if getattr(message, "command", None) != "set":
            continue
        total += 1
        if len(moves) >= cap:
            continue
        args = getattr(message, "args", ()) or ()
        moves.append(
            {
                "channel": _preview_channel(getattr(message, "obj", None)),
                "target": _json_safe(args[0]) if args else None,
            }
        )
    return moves, total


def preview_plan_in_namespace(
    name: Any,
    kwargs: Any,
    namespace: Mapping[str, Any],
    *,
    cap: int = PREVIEW_MOVE_CAP,
) -> dict[str, Any]:
    """Build ``name`` out of ``namespace`` and summarize the moves it would make.

    The plan is built through the namespace's own installed plan function — the
    same entry point the manager calls to execute it — so the preview validates
    the same kwargs against the same params schema and resolves the same live,
    mock-free devices. A preview that succeeds is evidence the enqueue would
    build; a preview that fails carries the reason the enqueue would fail.

    Only generator functions are previewable. That is what a plan is in this
    namespace, and requiring it means the preview entry point cannot be used to
    call anything else that happens to live in the worker's globals.

    Returns:
        Always a mapping, never an exception — see :func:`preview_plan` for the
        two shapes. On failure ``moves`` is empty and ``total_moves`` is 0:
        moves collected before a mid-stream failure describe a trajectory the
        run would never complete, so ``total_moves`` keeps one meaning (the
        exact total of a complete walk) rather than two.
    """
    result: dict[str, Any] = {
        "ok": True,
        "plan": name if isinstance(name, str) else repr(name),
        "moves": [],
        "total_moves": 0,
        "truncated": False,
        "move_cap": cap,
    }

    plan_function = namespace.get(name) if isinstance(name, str) else None
    if plan_function is None or not inspect.isgeneratorfunction(plan_function):
        available = sorted(
            key for key, value in namespace.items() if inspect.isgeneratorfunction(value)
        )
        result["ok"] = False
        result["error"] = (
            f"{result['plan']} is not a plan in this worker's namespace; "
            f"available plans: {available}"
        )
        return result

    plan = None
    try:
        plan = plan_function(**dict(kwargs or {}))
        moves, total = collect_channel_moves(plan, cap=cap)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, never raised
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:PREVIEW_ERROR_CHARS]
        logger.info(
            "qserver_startup: preview of plan %r failed: %s", result["plan"], result["error"]
        )
        return result
    finally:
        close = getattr(plan, "close", None)
        if callable(close):
            close()

    result["moves"] = moves
    result["total_moves"] = total
    result["truncated"] = total > len(moves)
    return result


def preview_plan(name: str, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read-only pre-flight: the channel moves running ``name`` with ``kwargs`` would make.

    Exposed to the manager as a namespace *function* (``function_execute``),
    not a plan: it is a plain function, so upstream's namespace scan classifies
    it as a callable to invoke directly rather than as a plan to hand the
    RunEngine, and it is the single deliberate exception in an otherwise
    deny-all function permission list. It moves nothing — see
    :func:`collect_channel_moves` for why walking a plan's message stream
    cannot.

    Reads the *live* worker namespace (this module's globals, which upstream
    ``exec``s the startup script into and :func:`build_namespace` populates), so
    the devices resolved here are the connected, mock-free devices the run
    itself would use.

    Args:
        name: Plan name, as it appears in the worker namespace.
        kwargs: The queue item's kwargs — the plan's ``PARAMS`` fields.

    Returns:
        A JSON-serializable mapping, always. On success::

            {"ok": True, "plan": "grid_scan", "move_cap": 10000,
             "moves": [{"channel": "corrector_01", "target": 0.5}, ...],
             "total_moves": 42, "truncated": False}

        ``moves`` is in the order the run would make them and holds at most
        ``move_cap`` entries; ``total_moves`` is the exact count regardless, and
        ``truncated`` is True when the list is only its opening slice. On
        failure — an unknown plan, kwargs the params schema rejects, a device
        this worker never built, or anything the plan raises while building —
        the same shape with ``ok`` False, an ``error`` string naming the
        failure, no moves, and a zero total.
    """
    return preview_plan_in_namespace(name, kwargs, globals())


class _FaultIsolatedCallback:
    """Wraps a ``(name, doc)`` RunEngine callback so its failure never aborts a run.

    The RunEngine does not swallow callback exceptions — one escaping
    ``__call__`` aborts the running plan. Both document-plane subscriptions
    (Tiled persistence and the 0MQ publisher) are telemetry: losing them
    degrades what an operator can see afterwards, which must never be worth
    killing a plan that is currently moving magnets. This wrapper catches any
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
