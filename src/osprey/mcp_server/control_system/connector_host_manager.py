"""Supervision of the connector-host child, and the spawn-then-swap switch.

The controls MCP server owns exactly one connector-host child at a time: one
process, one target, one Channel Access context. Changing the target is
therefore a process lifecycle operation, and this module is the only place that
performs it. :class:`ConnectorHostManager` holds the child, the target of
record, the generation counter, and the :class:`asyncio.Lock` that makes every
one of those a single-writer quantity.

Spawn-then-swap
---------------
A switch never begins by giving up what works. The order is fixed:

1. derive the destination's endpoints and read its ``probe_channel`` — a target
   that cannot be derived, or that names no probe channel, is refused before a
   process is spawned;
2. spawn child **B** and read its post-connect report;
3. **verify** that report against the derivation (:mod:`target_eligibility`);
4. **probe** B: a real read of the destination's ``probe_channel``, in B, over
   the control system B just connected to;
5. only then touch child **A**: refuse new work on it, drain it for at most
   ``control_system.target_switch.drain_timeout_s`` (default 5), kill it;
6. bump the generation **only when the target actually changed**, publish the
   outcome to the state file, and swap the active references.

Every failure in steps 1–4 kills B and returns, leaving A exactly as it was:
still active, still serving, still on its own generation. The no-child state is
consequently *not* reachable through a failed switch — it exists only when
child A dies on its own, and :meth:`ConnectorHostManager.has_child` reports it
honestly so the layer above can refuse with a reason rather than pretend.

Generations
-----------
The generation is what pins holders: an executor sandbox or an approved write
that was granted under generation *n* must refuse once the session has moved
past it. It therefore counts **target changes**, not process restarts. A
same-target respawn — :meth:`ConnectorHostManager.respawn_same_target`, which
is what the old in-process ``ConnectionError`` → invalidate path becomes in
child mode — replaces the process and leaves the generation alone, because
nothing a holder was promised has changed.

Why the launch handshake bypasses the proxy
-------------------------------------------
``init`` and ``spawn_probe`` are *supervisor* methods, not connector methods,
and :class:`~osprey_connectors.ipc.proxy.ConnectorHostProxy` deliberately
exposes only the connector surface. The handshake is spoken here on the child's
raw pipes with the public frame codec, and the proxy is constructed only once
the child has proven itself — so a child that fails to launch never becomes a
connector anything can call, and no half-initialized proxy has to be unwound.

Attributing a killed child's outstanding requests
-------------------------------------------------
When the drain deadline expires, the requests still in flight on child A have
to fail, and they have to say *why*. The proxy fails them with whatever ended
its read stream, so this module hands it a reader wrapper
(:class:`_AttributedReader`) that can be told the retirement reason before the
kill; the resulting :class:`ConnectionError` then names the switch rather than
an anonymous end-of-pipe.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.target_eligibility import (
    PROBE_CHANNEL_KEY,
    REASON_PROBE_CHANNEL_MISSING,
    REASON_TARGET_UNRESOLVABLE,
    ROLE_READ_ONLY,
    ROLE_WRITE_ACCESS,
    TargetDerivation,
    Verification,
    connector_block,
    derive_endpoints,
    effective_writes_for_target,
    endpoint_is_live_standin,
    verify_child_report,
)
from osprey_connectors.control_system.base import is_readonly_run
from osprey_connectors.ipc import frames
from osprey_connectors.ipc.proxy import ConnectorHostProxy
from osprey_connectors.types import (
    MOCK,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
)
from osprey_connectors.types import baseline_target as types_baseline_target
from osprey_connectors.types import switch_capable as types_switch_capable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.mcp_server.control_system.server_context import MCPServerConfig

logger = logging.getLogger("osprey.mcp_server.control_system.connector_host_manager")

__all__ = [
    "CHILD_MODULE",
    "DEFAULT_DRAIN_TIMEOUT_S",
    "DEFAULT_PROBE_TIMEOUT_S",
    "DEFAULT_SPAWN_TIMEOUT_S",
    "ConnectorHostManager",
    "NoConnectorHostError",
    "SwitchError",
    "baseline_target",
    "kill_orphans",
    "looks_like_a_connector_host",
    "reset_target_state",
    "switch_capable",
    "target_display_metadata",
]

#: The child is always this module, run with ``-m``. No arguments: everything
#: the child needs arrives on the wire, so nothing about a deployment shows up
#: in ``ps``.
CHILD_MODULE = "osprey_connectors.ipc.host"

#: Scrubbed from the environment handed to a child. The child scrubs again on
#: its own first line; this is the defense-in-depth half of the same rule, so
#: that an ambient gateway cannot even reach the process that might read it.
EPICS_ENV_PREFIXES = ("EPICS_CA_", "EPICS_PVA_")

#: ``control_system.target_switch.drain_timeout_s`` and its default.
DRAIN_TIMEOUT_KEY = "drain_timeout_s"
DEFAULT_DRAIN_TIMEOUT_S = 5.0

#: Bound on the readiness probe run against a freshly spawned child. Not a
#: config key: it is a property of the switch, not of a deployment, and the
#: deployment's own tuning surface is the drain timeout above.
DEFAULT_PROBE_TIMEOUT_S = 5.0

#: Bound on "process started and answered its init frame". Generous, because it
#: covers a cold import of a control-system client library.
DEFAULT_SPAWN_TIMEOUT_S = 30.0

#: How long a child gets between ``SIGTERM`` and ``SIGKILL``.
TERMINATE_GRACE_S = 2.0

#: How long the parent waits, after killing a child, for the proxy's reader to
#: turn the dead pipe into failures on the requests that were in flight.
SETTLE_TIMEOUT_S = 2.0

_READ_CHUNK = 65536

# -- switch stages and machine-readable reasons -----------------------------

STAGE_TARGET = "target"
STAGE_PROBE_CHANNEL = "probe_channel"
STAGE_SPAWN = "spawn"
STAGE_VERIFY = "verify"
STAGE_PROBE = "probe"

REASON_SPAWN_FAILED = "spawn_failed"
REASON_VERIFICATION_FAILED = "verification_failed"
REASON_PROBE_FAILED = "probe_failed"
#: Not a switch stage: the state a session is in when its child has died.
REASON_NO_CHILD = "no_connector_host"

#: Connector types that serve a machine nobody has to be careful around. Used
#: only to label the state file's per-target display metadata.
_SIMULATED_TYPES = (MOCK, VIRTUAL_ACCELERATOR)

#: The operator-facing name each display branch defaults to — one word per way
#: a target can be derived, not per target name, because that is what the name
#: describes: the simulator is a Simulator whichever target selected it, and a
#: ``live`` target with no connector configured is still the Real machine (the
#: "not set up" nuance is the reader's to render, never the name's to carry).
#: A deployment renames any of them per target via
#: ``control_system.target_display_names``; these words are what a config that
#: says nothing gets.
_DISPLAY_NAME_DEFAULTS = {
    "machine": "Real machine",
    "standin": "Rehearsal",
    "simulated": "Demo",
    "va": "Simulator",
}


class SwitchError(RuntimeError):
    """A switch that did not happen, and the stage it stopped at.

    Structured rather than prose because two different layers consume it: the
    ``control_target_set`` tool renders ``detail`` to the agent, and the roster
    matches ``reason`` against the eligibility reasons so the two never disagree
    about why a target is unusable.
    """

    def __init__(
        self,
        target: str,
        stage: str,
        reason: str,
        detail: str,
        *,
        verification: Verification | None = None,
        gateway: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.target = target
        self.stage = stage
        self.reason = reason
        self.detail = detail
        self.verification = verification
        #: ``{"role", "host", "port"}`` of the gateway a failed probe ran
        #: through. Both roles usually share a hostname and differ only by
        #: port, so a refusal that named only the probe channel would read as
        #: "the control system is down" when one gateway beside a healthy one
        #: is unserved.
        self.gateway = gateway

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": self.target,
            "stage": self.stage,
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.verification is not None:
            payload["verification"] = self.verification.as_dict()
        if self.gateway is not None:
            payload["gateway"] = self.gateway
        return payload


class NoConnectorHostError(ConnectionError):
    """No connector host is alive to serve a control-system operation.

    Reachable only by a child dying outside a switch — a failed switch leaves
    the previous child serving — and it is what every control-system-routed
    tool refuses with while it lasts (FR-1). Deliberately a
    :class:`ConnectionError`: the tools' shared handler already treats that as
    "the connector is gone, drop it and let the next call rebuild", which for a
    child-served deployment means respawning it on the same target. The refusal
    is therefore both the honest answer to this call and the trigger for the
    recovery of the next one.
    """

    def __init__(self, target: str, generation: int, detail: str | None = None) -> None:
        self.target = target
        self.generation = generation
        self.detail = detail or (
            f"No connector-host child is serving target {target!r} (generation {generation}): "
            "the process that held this session's control-system connection exited. "
            "Control-system operations refuse until one is running again."
        )
        super().__init__(self.detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "generation": self.generation,
            "reason": REASON_NO_CHILD,
            "detail": self.detail,
        }


class _AttributedReader:
    """A child's stdout, able to say why reading it stopped.

    The proxy fails every outstanding request with whatever ended its read
    stream. Left to itself that is an anonymous end-of-pipe, which tells an
    operator nothing about the switch that caused it — so the supervisor names
    the reason here before it kills the child, and the proxy's
    :class:`ConnectionError` carries it.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._reason: str | None = None

    def retire(self, reason: str) -> None:
        """Name the reason the stream is about to end. Call before the kill."""
        self._reason = reason

    async def read(self, count: int) -> bytes:
        if self._reason is not None:
            raise ConnectionError(self._reason)
        chunk = await self._stream.read(count)
        # The usual path: the reader was already blocked here when the child was
        # retired, and the kill it was told about is what ends the read.
        if not chunk and self._reason is not None:
            raise ConnectionError(self._reason)
        return chunk


class _LaunchChannel:
    """The launch handshake, spoken on a child's raw pipes.

    One request at a time, one reply expected, matched by request id. The child
    sends nothing unsolicited, so a reply that leaves bytes behind in the frame
    reader means the stream is not what it claims to be — and those bytes would
    be lost when the pipes are handed to the proxy, so the handshake refuses
    rather than hand over a stream it has already truncated.
    """

    def __init__(self, target: str, process: Any) -> None:
        self._target = target
        self._process = process
        self._frames = frames.FrameReader()

    async def request(self, method: str, kwargs: dict[str, Any], timeout: float, stage: str) -> Any:
        request_id = frames.new_request_id()
        try:
            self._process.stdin.write(frames.encode_request(request_id, method, kwargs))
            await self._process.stdin.drain()
        except (OSError, RuntimeError, ConnectionError, AttributeError) as exc:
            raise self._failure(stage, f"could not send its {method!r} request: {exc}") from exc

        frame = await self._reply(request_id, method, timeout, stage)
        if isinstance(frame, frames.ErrorFrame):
            raise self._failure(stage, f"failed {method!r}: {frame.message}")
        return frame.value

    async def _reply(self, request_id: str, method: str, timeout: float, stage: str) -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise self._failure(stage, f"did not answer {method!r} within {timeout}s")
            try:
                chunk = await asyncio.wait_for(self._process.stdout.read(_READ_CHUNK), remaining)
            except TimeoutError:
                raise self._failure(stage, f"did not answer {method!r} within {timeout}s") from None
            if not chunk:
                raise self._failure(
                    stage,
                    f"closed its output stream while answering {method!r} "
                    f"(exit code {self._process.returncode})",
                )
            decoded = self._frames.feed(chunk)
            if not decoded:
                continue
            # One request, one reply: a chunk carrying more than one whole frame
            # means the child said something nobody asked for, and the extra
            # frame would be silently dropped here rather than reaching anyone.
            if len(decoded) > 1 or getattr(decoded[0], "request_id", None) != request_id:
                raise self._failure(stage, f"answered a request nobody made during {method!r}")
            return decoded[0]

    def assert_stream_is_clean(self) -> None:
        """Refuse to hand the proxy a stream with an unread tail in it.

        Bytes left in the reader are a partial frame the proxy will never see
        the front of; whole unsolicited frames are caught in :meth:`_reply`,
        which is the only place they can arrive.
        """
        if len(self._frames):
            raise self._failure(
                STAGE_SPAWN, f"left {len(self._frames)} unsolicited bytes on its output stream"
            )

    def _failure(self, stage: str, what: str) -> SwitchError:
        return SwitchError(
            self._target,
            stage,
            REASON_PROBE_FAILED if stage == STAGE_PROBE else REASON_SPAWN_FAILED,
            f"The connector-host child for target {self._target!r} {what}.",
        )


@dataclass
class _Child:
    """One live connector-host child and everything the parent holds about it."""

    target: str
    connector_type: str
    probe_channel: str
    process: Any
    proxy: ConnectorHostProxy
    reader: _AttributedReader
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def is_alive(self) -> bool:
        return self.process.returncode is None


# ---------------------------------------------------------------------------
# Config-derived facts
# ---------------------------------------------------------------------------


def _control_system_section(config: Any) -> Any:
    """The ``control_system:`` section of a rendered config, or ``None``.

    The module's one door onto the config: every config-derived fact below is a
    question about this section, and a caller may hand in anything the config
    layer produced, including nothing at all.
    """
    return config.get("control_system") if isinstance(config, dict) else None


def baseline_target(config: Any) -> str:
    """The target this deployment's own config selects.

    The whole-config spelling of :func:`osprey_connectors.types.baseline_target`,
    for callers holding a rendered config rather than a ``control_system:``
    section.
    """
    return types_baseline_target(_control_system_section(config))


def switch_capable(config: Any) -> bool:
    """Whether this deployment gives a session more than one runtime target.

    The predicate itself lives in :func:`osprey_connectors.types.switch_capable`,
    beside the target resolution it is built from, so that the runtime and the
    build cannot answer it differently. This is the whole-config spelling of it,
    for callers holding a rendered config rather than a ``control_system:``
    section; it is what decides where the controls server's tools get their
    connector from.
    """
    return types_switch_capable(_control_system_section(config))


def _connector_block(config: Any, connector_type: str) -> dict[str, Any]:
    """The ``control_system.connector.<type>`` block, as a mapping.

    The traversal is eligibility's — one module reads the connector table — and
    the coercion is this module's: readers here want a mapping to ``.get()``
    from, not the distinction between an absent block and a malformed one.
    """
    block = connector_block(config, connector_type)
    return block if isinstance(block, dict) else {}


def target_display_metadata(config: Any) -> dict[str, dict[str, Any]]:
    """Per-target display metadata for the state file's single writer.

    Rendered once, here, from config: readers (the prompt hook, the roster)
    render the identity line straight from the state file and never re-derive
    it, so this is the only place the three targets get described.

    Each entry carries ``label``, ``display_name``, ``endpoint``,
    ``real_machine`` and ``probe_channel`` — the destination's probe channel
    travels with the metadata so a describer can name it without opening
    ``config.yml``.

    One entry per name in :data:`~osprey.mcp_server.control_system.target_state.TARGET_NAMES`,
    walked from that tuple rather than from a list spelled again here: the
    state file has a slot per target, this is its single writer, and a target
    described in one place and missing from the other is a reader rendering an
    empty identity line for a target an operator can select.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for target in target_state.TARGET_NAMES:
        try:
            derivation = derive_endpoints(config, target)
        except ValueError:
            # A deployment that has never named its real machine still needs a
            # slot: "unknown" is a truthful rendering, an absent key is not.
            metadata[target] = {
                "label": _label(target, None),
                "display_name": _display_name(config, target, None),
                "endpoint": "",
                "real_machine": False,
                "probe_channel": "",
            }
            continue
        endpoint = derivation.selected_endpoint()
        block = _connector_block(config, derivation.connector_type)
        standin = _is_live_standin(config, target, endpoint)
        metadata[target] = {
            "label": _label(target, derivation.connector_type, standin=standin),
            "display_name": _display_name(
                config, target, derivation.connector_type, standin=standin
            ),
            "endpoint": f"{endpoint.host}:{endpoint.port}" if endpoint else "",
            # True for the stand-in as well as the facility's machine, and the
            # question is the connector type alone rather than which target
            # asked: a stand-in is a real-machine posture, so every strict
            # limit, approval prompt and banner hardware gets, it gets. Only
            # the name on the label moves.
            "real_machine": derivation.connector_type not in _SIMULATED_TYPES,
            "probe_channel": str(block.get(PROBE_CHANNEL_KEY) or ""),
        }
    return metadata


def _is_live_standin(config: Any, target: str, endpoint: Any) -> bool:
    """Whether *endpoint* is this deployment's stand-in container.

    Derived here because here is where every other display fact is derived:
    this function is the state file's single writer, and its readers — the
    prompt hook, the roster, the approval banner, the web badge — render the
    label they are handed and never ask the config again. Two derivations of
    "is this the real machine" can disagree, and a disagreement about *that*
    is an operator being told they are somewhere they are not.

    The predicate itself is
    :func:`~osprey.mcp_server.control_system.target_eligibility.endpoint_is_live_standin`
    over :func:`osprey_connectors.standin.live_standin_active`, shared with the
    archiver's enablement gate and the build derivation that points the gateways
    at the stand-in in the first place. An SSH tunnel is the case it must not
    claim: forwarding a real gateway to ``localhost:5064`` is loopback and
    nothing more, so the port conjunct fails and the label stays
    ``LIVE MACHINE`` — which is the truth, because the operator is one hop from
    hardware.

    Asked of the live family — ``live`` and ``standin``, the two targets that
    dial a Channel Access gateway. ``va`` is not asked at all: the simulator is
    described by its own branch, and a sandbox that happened to be configured
    on the stand-in's port is still a simulator. Membership in the family is
    not the same as carrying the parenthesis, which is :func:`_label`'s to
    decide and is the ``standin`` target's alone — see there.
    """
    if target not in (TARGET_LIVE, TARGET_STANDIN):
        return False
    return endpoint_is_live_standin(config, endpoint)


def _label(target: str, connector_type: str | None, *, standin: bool = False) -> str:
    if target == TARGET_VA:
        return "virtual accelerator (simulation)"
    if connector_type is None:
        return "live machine (not configured)"
    if connector_type in _SIMULATED_TYPES:
        return f"live target on a simulated connector ({connector_type})"
    # The parenthesis is the whole of what an operator is told differently, and
    # it belongs to the 'standin' target alone. Two conjuncts, both required:
    # the target is the one an operator selected by name, and its endpoint
    # really is the stand-in container this deployment stood up. A 'standin'
    # whose endpoint fails the predicate is labelled plain LIVE MACHINE and
    # refused by the eligibility gate, and 'live' never carries the
    # parenthesis at all — it names the facility's authored machine, and
    # renaming it as a stand-in on the strength of an endpoint that merely
    # looks like one (a facility gateway forwarded to loopback on that port)
    # is the direction this stack must never fail in: telling an operator the
    # machine in front of them is only a stand-in when it is not.
    return "LIVE MACHINE (stand-in)" if (target == TARGET_STANDIN and standin) else "LIVE MACHINE"


def _display_name(
    config: Any, target: str, connector_type: str | None, *, standin: bool = False
) -> str:
    """The operator-facing name for *target*: configured, or the branch default.

    Walks the same branches as :func:`_label`, from the same inputs, so the two
    names cannot describe different machines — the label is the identity line's
    truth, this is the word an operator reads on the chip. A non-empty
    ``control_system.target_display_names.<target>`` string wins verbatim
    (stripped); empty or absent falls to :data:`_DISPLAY_NAME_DEFAULTS`. The
    override is per target name rather than per branch on purpose: the operator
    names the thing they select, and the derivation still decides what that
    thing defaults to.
    """
    section = _control_system_section(config)
    configured = section.get("target_display_names") if isinstance(section, dict) else None
    override = configured.get(target) if isinstance(configured, dict) else None
    if isinstance(override, str) and override.strip():
        return override.strip()
    if target == TARGET_VA:
        return _DISPLAY_NAME_DEFAULTS["va"]
    if connector_type in _SIMULATED_TYPES:
        return _DISPLAY_NAME_DEFAULTS["simulated"]
    if target == TARGET_STANDIN and standin:
        return _DISPLAY_NAME_DEFAULTS["standin"]
    return _DISPLAY_NAME_DEFAULTS["machine"]


# ---------------------------------------------------------------------------
# Startup sweep
# ---------------------------------------------------------------------------


def looks_like_a_connector_host(pid: int) -> bool:
    """Whether *pid*'s command line still names the connector-host module.

    A PID recorded by a server that has since died may have been reused by the
    operating system for something else entirely, and killing whatever now
    holds that number would be far worse than leaving one orphan behind. This
    is a best-effort identity check through ``ps``: an answer that cannot be
    obtained at all — no ``ps``, a platform that spells it differently, a
    process owned by another user — is treated as "yes", because the recorded
    PID is still the only evidence there is and the sweep is what the design
    relies on to clear a dead server's children.
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - platform oddity
        logger.debug("Could not read the command line of pid %s: %s", pid, exc)
        return True
    command = completed.stdout.strip()
    if completed.returncode != 0 or not command:
        # ps found nothing: the process is gone, and the kill below will say so.
        return True
    return CHILD_MODULE in command


def kill_orphans(pids: list[int], *, grace_s: float = TERMINATE_GRACE_S) -> list[int]:
    """Kill connector-host children left behind by a dead predecessor.

    ``SIGTERM`` first, ``SIGKILL`` after *grace_s*. A PID that is already gone
    is success: the point is that nothing is left holding a Channel Access
    context this server did not spawn. A PID that is alive but does not look
    like a connector host is skipped — see :func:`looks_like_a_connector_host`.

    Returns:
        The PIDs that were signalled, in the order they were given.
    """
    signalled: list[int] = []
    for pid in pids:
        if not looks_like_a_connector_host(pid):
            logger.warning(
                "Stale state file records connector-host child %s, but that pid now belongs "
                "to something else; leaving it alone",
                pid,
            )
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except (PermissionError, OSError) as exc:
            logger.warning("Could not signal orphaned connector host %s: %s", pid, exc)
            continue
        signalled.append(pid)
        deadline = time.monotonic() + max(grace_s, 0.0)
        while time.monotonic() < deadline:
            if not target_state.is_process_alive(pid):
                break
            time.sleep(0.05)
        else:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        logger.warning("Killed orphaned connector-host child %s from a stale state file", pid)
    return signalled


def reset_target_state(
    config: Any,
    *,
    baseline: str | None = None,
    targets_meta: dict[str, Any] | None = None,
    grace_s: float = TERMINATE_GRACE_S,
) -> list[int]:
    """Reset this server's state file to the baseline and kill any orphans.

    Called once at server start, before anything can switch: the state file is
    reset (no target selection survives the process that made it) and the child
    PIDs recorded by dead predecessors are killed, because a connector host
    outliving its server holds a gateway nobody is talking to.

    Returns:
        The orphan PIDs that were signalled.
    """
    orphans = target_state.write_on_start(
        baseline or baseline_target(config),
        target_display_metadata(config) if targets_meta is None else targets_meta,
    )
    return kill_orphans(orphans, grace_s=grace_s)


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


class ConnectorHostManager:
    """The connector-host child's supervisor: one child, one target, one lock.

    Args:
        config: The server's parsed configuration. The whole rendered mapping
            is used for derivation and the ``control_system`` section travels
            to the child verbatim, so parent and child cannot disagree about
            what was configured.
        drain_timeout_s: Overrides ``control_system.target_switch.drain_timeout_s``.
        probe_timeout_s: Bound on the readiness probe.
        spawn_timeout_s: Bound on "spawned and answered its init frame".
        python_executable: Interpreter used for the child; defaults to this one.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        drain_timeout_s: float | None = None,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        spawn_timeout_s: float = DEFAULT_SPAWN_TIMEOUT_S,
        terminate_grace_s: float = TERMINATE_GRACE_S,
        python_executable: str | None = None,
    ) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._child: _Child | None = None
        self._baseline = baseline_target(config.raw)
        #: The target of record. It survives the death of a child: a session on
        #: 'va' whose child died is still a session on 'va', and saying
        #: otherwise would be inventing a switch nobody made.
        self._target = self._baseline
        self._generation = 0
        self._started = False
        self._drain_override = drain_timeout_s
        self._probe_timeout_s = probe_timeout_s
        self._spawn_timeout_s = spawn_timeout_s
        self._terminate_grace_s = terminate_grace_s
        self._python = python_executable or sys.executable

    # -- accessors ---------------------------------------------------------

    @property
    def baseline(self) -> str:
        """The deployment's own baseline target."""
        return self._baseline

    def is_started(self) -> bool:
        """Whether a child has ever been started by this manager."""
        return self._started

    def has_child(self) -> bool:
        """Whether a connector host is alive right now.

        ``False`` on a started manager is the no-child state: child A died
        outside a switch. It is reported rather than papered over — the layer
        above refuses control-system work with that as the reason.
        """
        return self._live_child() is not None

    def active_target(self) -> str:
        """The target of record, alive child or not."""
        return self._target

    def active_generation(self) -> int:
        """The generation, which counts target changes and not respawns."""
        return self._generation

    def active_binding(self) -> tuple[str, int]:
        """``(target, generation)`` — what a write binds itself to."""
        return self._target, self._generation

    def active_proxy(self) -> ConnectorHostProxy | None:
        """The live child's connector-shaped handle, or ``None`` if there is none."""
        child = self._live_child()
        return child.proxy if child is not None else None

    def child_env(self) -> dict[str, str]:
        """The environment a child is launched with: this one's, minus EPICS.

        The child scrubs ``EPICS_CA_*``/``EPICS_PVA_*`` again on its own first
        line, and that is the scrub the design depends on. This one is the
        defense-in-depth half: an ambient gateway never reaches the process that
        could act on it, so no window exists between exec and scrub.
        """
        return {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(EPICS_ENV_PREFIXES)
        }

    def status(self) -> dict[str, Any]:
        """Everything the roster needs about the running host, in one mapping."""
        child = self._live_child()
        return {
            "target": self._target,
            "generation": self._generation,
            "baseline_target": self._baseline,
            "started": self._started,
            "child_alive": child is not None,
            "child_pid": child.pid if child is not None else None,
            "connector_type": child.connector_type if child is not None else None,
            "probe_channel": child.probe_channel if child is not None else None,
            "selected_role": child.report.get("selected_role") if child is not None else None,
            "drain_timeout_s": self._drain_timeout(),
        }

    def _live_child(self) -> _Child | None:
        child = self._child
        if child is None:
            return None
        if not child.is_alive():
            logger.warning(
                "Connector-host child %s for target %r exited (%s) outside a switch; "
                "the session has no connector host",
                child.pid,
                child.target,
                child.process.returncode,
            )
            self._child = None
            return None
        return child

    # -- lifecycle ---------------------------------------------------------

    def reset_state(self, *, grace_s: float | None = None) -> list[int]:
        """Reset the state file to the baseline and kill inherited orphans.

        Synchronous on purpose: it runs during server start, before the event
        loop that will own the children exists.
        """
        return reset_target_state(
            self._config.raw,
            baseline=self._baseline,
            grace_s=self._terminate_grace_s if grace_s is None else grace_s,
        )

    async def start(self, target: str | None = None) -> dict[str, Any]:
        """Bring up the first connector host, on *target* or on the baseline."""
        async with self._lock:
            wanted = self._target if target is None else target
            return await self._switch_locked(
                wanted, cause=f"starting the connector host on target {wanted!r}"
            )

    async def ensure_started(self, target: str | None = None) -> bool:
        """Start the deployment's first child if one has never run.

        This is the serving path's entry point: the first tool call on a
        switch-capable deployment brings the baseline child up, and every call
        after that finds it already there. It starts a child exactly once —

        * **A child that has died is not restarted here.** That is the no-child
          state, and it is a refusal with a reason (FR-1), not something to
          paper over on the way to a read. Recovery runs through the error
          path: the refusal is a :class:`ConnectionError`, the tool handler
          invalidates the connector, and that is what respawns the child.
        * **The baseline start is not probed.** The probe is the gate that
          protects a *working* session from being swapped onto a target that
          cannot answer; on the first start there is no session to protect and
          no traffic to swap, and refusing to bring the server up over it would
          leave the deployment with no control system at all rather than one
          whose failures are reported per call — which is exactly what the
          in-process connector this replaces did.

        Returns:
            Whether a live child exists now.

        Raises:
            SwitchError: The baseline child could not be spawned or verified.
                Nothing is cached from a failed start, so the next call retries.
        """
        async with self._lock:
            if self._started:
                return self._live_child() is not None
            wanted = self._target if target is None else target
            await self._switch_locked(
                wanted,
                cause=f"starting the connector host on target {wanted!r}",
                probe=False,
            )
            return True

    async def switch(self, target: str, *, force: bool = False) -> dict[str, Any]:
        """Move the session to *target*, spawn-then-swap, under the lock.

        A switch whose destination is already active *and* served is already
        done: nothing is spawned, and the result says so with
        ``target_changed`` false. Every gate above this one is evaluated before
        the lock is taken, so two callers can both be told to go to the same
        place and both be right when they were told — and the one that arrives
        second would otherwise replace a working child with an identical one,
        costing the session every connection it holds to change nothing.

        Args:
            target: The destination.
            force: Replace the child even when it already serves *target*. The
                deliberate respawn (:meth:`respawn_same_target`) is the one
                caller that means "a new process" rather than "be on this
                target".

        Raises:
            SwitchError: The target could not be derived, names no probe
                channel, or the new child failed to spawn, verify or probe. In
                every case the previous child is untouched and still active.
        """
        async with self._lock:
            return await self._switch_locked(
                target,
                cause=(f"the control-system target switch from {self._target!r} to {target!r}"),
                force=force,
            )

    async def respawn_same_target(self) -> dict[str, Any]:
        """Replace the child with a fresh one on the same target.

        This is what the in-process ``ConnectionError`` → invalidate path
        becomes once the connector lives in a child: the process is replaced,
        the target is not, and the generation therefore does not move — nothing
        a pinned holder was promised has changed.
        """
        async with self._lock:
            return await self._switch_locked(
                self._target,
                cause=f"respawning the connector host on target {self._target!r}",
                force=True,
            )

    async def shutdown(self) -> None:
        """Retire the child and clear the state file's record of it.

        The manager stops counting as started: a later invalidation must not
        respawn a child into a server that is on its way out.
        """
        async with self._lock:
            child, self._child = self._child, None
            if child is not None:
                await self._retire(child, "the controls server is shutting down")
            if self._started:
                self._started = False
                try:
                    target_state.record_child_pids([])
                except OSError as exc:
                    # Shutting down is not a moment to raise: the children are
                    # already gone, and the worst a stale record costs is one
                    # skipped kill in the next server's sweep.
                    logger.warning("Could not clear the child PIDs from the state file: %s", exc)

    # -- the switch --------------------------------------------------------

    async def _switch_locked(
        self, target: str, *, cause: str, probe: bool = True, force: bool = False
    ) -> dict[str, Any]:
        derivation = self._derive(target)
        if not force:
            settled = self._already_served(target, derivation)
            if settled is not None:
                return settled
        # ``probe`` is false only for the deployment's very first child, where
        # there is no working session to protect — see ensure_started().
        probe_channel = self._probe_channel(target, derivation) if probe else ""

        fallback: dict[str, Any] | None = None
        try:
            candidate = await self._launch(target, derivation, probe_channel)
        except SwitchError as exc:
            read_derivation = self._read_role_fallback(derivation, exc)
            if read_derivation is None:
                raise
            dead = derivation.selected_endpoint()
            logger.warning(
                "The %r gateway for target %r at %s:%s failed its readiness probe; "
                "retrying through the %r gateway so the session can reach the target. "
                "A write on this session routes to the read-only gateway, which is "
                "expected to refuse it, until the write gateway is served again.",
                ROLE_WRITE_ACCESS,
                target,
                dead.host,
                dead.port,
                ROLE_READ_ONLY,
            )
            derivation = read_derivation
            try:
                candidate = await self._launch(
                    target, derivation, probe_channel, without_write_gateway=True
                )
            except SwitchError as retry_error:
                if retry_error.stage != STAGE_PROBE:
                    raise
                # Both probes failed. The surfaced error is the read-role one —
                # the last thing actually probed — but the operator gets the
                # whole story: the write gateway failed first, so this is not
                # one flaky endpoint but a target with no reachable gateway.
                raise SwitchError(
                    retry_error.target,
                    retry_error.stage,
                    retry_error.reason,
                    f"{retry_error.detail} The {ROLE_WRITE_ACCESS!r} gateway at "
                    f"{dead.host}:{dead.port} was probed first and also failed, so no "
                    f"configured gateway for target {target!r} is answering.",
                    verification=retry_error.verification,
                    gateway=retry_error.gateway,
                ) from None
            fallback = {
                "host": dead.host,
                "port": dead.port,
                "detail": (
                    f"The {ROLE_WRITE_ACCESS!r} gateway at {dead.host}:{dead.port} failed "
                    f"its readiness probe, so the session reached target {target!r} through "
                    f"the {ROLE_READ_ONLY!r} gateway instead. Reads are unaffected; a write "
                    "routes to the read-only gateway and is expected to be refused there "
                    "until the write gateway is served again."
                ),
            }

        # The active child stays active right through the retirement: a tool
        # that calls in while A is draining is refused *by A*, with the switch
        # as the reason, which is a different and truer answer than the
        # no-child state it would see if the reference were cleared first.
        previous = self._child
        drained = True
        if previous is not None:
            drained = await self._retire(previous, cause)

        if target != self._target:
            self._generation += 1
        previous_target = self._target
        self._target = target
        self._child = candidate
        self._started = True

        self._publish(target, candidate.pid)

        endpoint = derivation.selected_endpoint()
        logger.info(
            "Connector host now on target %r (generation %s, pid %s, type %r)",
            target,
            self._generation,
            candidate.pid,
            derivation.connector_type,
        )
        result: dict[str, Any] = {
            "target": target,
            "generation": self._generation,
            "previous_target": previous_target,
            "target_changed": target != previous_target,
            "connector_type": derivation.connector_type,
            "selected_role": derivation.selected_role,
            "endpoint": endpoint.as_dict() if endpoint is not None else None,
            "probe_channel": probe_channel,
            "child_pid": candidate.pid,
            "previous_drained": drained,
            "drain_timeout_s": self._drain_timeout(),
        }
        if fallback is not None:
            result["write_gateway_fallback"] = fallback
        return result

    def _already_served(self, target: str, derivation: TargetDerivation) -> dict[str, Any] | None:
        """The answer for a switch whose destination is already being served.

        Both halves of the condition matter. The **target of record** alone is
        not enough: a child that died outside a switch leaves the session
        pointed at its target with nothing serving it, and answering "already
        there" would strand it with no connector host and call that success. A
        **live child** on that target, though, is the whole of what the caller
        asked for, so the switch is reported as done rather than performed
        again — the destination is not re-derived into a second process, the
        generation does not move, and the connections the session holds survive.

        The fields describe the child that is already running rather than one
        that was just launched: its connector type, the role it reported having
        connected on, and the channel it proved itself with — empty for the
        deployment's first child, which was started without a probe, because
        claiming a probe that never ran would be worse than saying so.

        Returns:
            The normal switch result for the running child, or ``None`` when
            this is not that case and the caller should go on and spawn.
        """
        if target != self._target:
            return None
        child = self._live_child()
        if child is None:
            return None
        logger.info(
            "The connector host is already on target %r (generation %s, pid %s); "
            "nothing was spawned and nothing was retired",
            target,
            self._generation,
            child.pid,
        )
        reported = child.report.get("selected_role")
        has_role = isinstance(reported, str) and bool(reported)
        selected_role = reported if has_role else derivation.selected_role
        endpoint = derivation.endpoints.get(selected_role)
        return {
            "target": target,
            "generation": self._generation,
            "previous_target": self._target,
            "target_changed": False,
            "connector_type": child.connector_type,
            "selected_role": selected_role,
            "endpoint": endpoint.as_dict() if endpoint is not None else None,
            "probe_channel": child.probe_channel,
            "child_pid": child.pid,
            "previous_drained": True,
            "drain_timeout_s": self._drain_timeout(),
        }

    def _publish(self, target: str, child_pid: int) -> bool:
        """Record the completed switch in the state file. Never raises.

        The swap has already happened by the time this runs: child A is dead,
        the generation has moved, and the session *is* on the new target. A
        state file that cannot be written — an unwritable data root, a full
        disk — must therefore not turn a switch that succeeded into an
        exception that says it failed, because the caller would then report the
        old target while every tool call went to the new one. The in-memory
        truth is authoritative and is what the accessors report; the file is
        the copy readers outside this process get, and its failure to update is
        logged loudly and left stale rather than pretended away.
        """
        try:
            if target_state.publish_switch(self._target, self._generation, children=[child_pid]):
                return True
            logger.warning(
                "No target-state record to publish the switch to %r into; "
                "the server did not reset its state file at start",
                target,
            )
        except OSError as exc:
            logger.error(
                "The switch to %r (generation %s) succeeded, but its state file could not be "
                "written: %s. Readers outside this server will keep seeing the previous "
                "target until the next successful write",
                target,
                self._generation,
                exc,
            )
        return False

    def _derive(self, target: str) -> TargetDerivation:
        """The destination's derivation, or a refusal naming what is missing.

        The write posture handed in is this SESSION's, not the config's: the
        child selects its gateway from the per-(session, target) posture store
        as well as from config, so a parent deriving the configured posture
        would expect ``write_access`` from a child the operator has narrowed to
        ``read_only`` — and ``verify_child_report``, comparing the two, would
        abort the switch over a disagreement neither side got wrong.
        """
        try:
            return derive_endpoints(
                self._config.raw,
                target,
                writes_enabled=effective_writes_for_target(self._config.control_system, target),
            )
        except ValueError as exc:
            raise SwitchError(target, STAGE_TARGET, REASON_TARGET_UNRESOLVABLE, str(exc)) from exc

    def _read_role_fallback(
        self, derivation: TargetDerivation, error: SwitchError
    ) -> TargetDerivation | None:
        """The read-role derivation a failed write-role probe falls back to.

        Reaching a target at all is a stronger requirement than reaching it
        write-capable: a ``write_access`` row that is configured but served by
        nothing must not make the target unreachable — least of all the
        deployment baseline, whose return leg exists so a session can always
        come home (issue #718). The fallback therefore applies to every probed
        switch, not only the return leg, and it mirrors the ``connect()``
        fallback that already lands a write-armed deployment *without* a write
        gateway on ``read_only`` with a warning.

        Only a **probe**-stage failure qualifies: the probe is the reachability
        check, so its failure is evidence about the gateway. A spawn or
        verification failure says the child or the config is wrong, and
        retrying those through another gateway would paper over a different
        disease.

        Returns:
            The same derivation with ``read_only`` selected, or ``None`` when
            the failure is not one a read-role retry could answer.
        """
        if error.stage != STAGE_PROBE:
            return None
        if derivation.selected_role != ROLE_WRITE_ACCESS:
            return None
        if ROLE_READ_ONLY not in derivation.endpoints:
            return None
        return replace(derivation, selected_role=ROLE_READ_ONLY)

    def _probe_channel(self, target: str, derivation: TargetDerivation) -> str:
        """The channel the destination proves itself with, or a refusal.

        Eligibility already reports a missing ``probe_channel``; re-checking it
        here costs one dictionary lookup and keeps the manager's own contract
        closed — a target with nothing to probe must never reach a spawn.
        """
        block = _connector_block(self._config.raw, derivation.connector_type)
        channel = block.get(PROBE_CHANNEL_KEY)
        if not isinstance(channel, str) or not channel.strip():
            raise SwitchError(
                target,
                STAGE_PROBE_CHANNEL,
                REASON_PROBE_CHANNEL_MISSING,
                f"'control_system.connector.{derivation.connector_type}.{PROBE_CHANNEL_KEY}' "
                f"is not set. The switch reads that channel to prove target {target!r} is "
                "reachable before making it active, so a target without one is never "
                "switched to.",
            )
        return channel.strip()

    async def _launch(
        self,
        target: str,
        derivation: TargetDerivation,
        probe_channel: str,
        *,
        without_write_gateway: bool = False,
    ) -> _Child:
        """Spawn, verify and probe a child — or leave nothing behind.

        With ``without_write_gateway`` the child's init payload omits the
        connector's ``write_access`` gateway row, so the child runs
        ``connect()``'s documented absent-row fallback — ``read_only`` with a
        warning — and never even learns where the write gateway is. The caller
        passes a derivation whose selected role is ``read_only`` to match.
        """
        process = await self._spawn(target)
        channel = _LaunchChannel(target, process)
        try:
            report = await channel.request(
                "init",
                self._init_kwargs(
                    target,
                    derivation.connector_type,
                    without_write_gateway=without_write_gateway,
                ),
                self._spawn_timeout_s,
                STAGE_SPAWN,
            )
            if not isinstance(report, dict):
                raise SwitchError(
                    target,
                    STAGE_SPAWN,
                    REASON_SPAWN_FAILED,
                    f"The connector-host child for target {target!r} answered its init frame "
                    f"with {type(report).__name__}, not the post-connect report.",
                )
            verification = _verify(derivation, report)
            if not verification.ok:
                raise SwitchError(
                    target,
                    STAGE_VERIFY,
                    REASON_VERIFICATION_FAILED,
                    f"Refusing target {target!r}: {verification.detail}",
                    verification=verification,
                )
            if probe_channel:
                try:
                    await channel.request(
                        "spawn_probe",
                        {"channel": probe_channel, "timeout": self._probe_timeout_s},
                        self._probe_timeout_s + 1.0,
                        STAGE_PROBE,
                    )
                except SwitchError as exc:
                    # The probe channel alone does not say which endpoint would
                    # not answer; the derivation this child was launched from
                    # does. Raised richer, not merely logged, because the
                    # refusal is the only thing the operator sees.
                    raise _name_probed_gateway(exc, derivation) from None
            else:
                logger.info(
                    "Connector host for target %r started without a readiness probe: "
                    "this is the deployment's first child, so there is no session to protect",
                    target,
                )
            channel.assert_stream_is_clean()
        except BaseException:
            # Nothing survives a failed launch: the previous child is still the
            # active one, and a spare process on the destination's gateway is
            # exactly the thing this design exists to prevent.
            with contextlib.suppress(Exception):
                process.stdin.close()
            await self._kill_process(process)
            raise

        # One reader object, held by both the proxy and this record: retiring it
        # is how the parent names the reason the proxy's stream ended.
        reader = _AttributedReader(process.stdout)
        return _Child(
            target=target,
            connector_type=derivation.connector_type,
            probe_channel=probe_channel,
            process=process,
            proxy=ConnectorHostProxy(reader, process.stdin),
            reader=reader,
            report=report,
        )

    async def _spawn(self, target: str) -> Any:
        env = self.child_env()
        try:
            return await asyncio.create_subprocess_exec(
                self._python,
                "-m",
                CHILD_MODULE,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise SwitchError(
                target,
                STAGE_SPAWN,
                REASON_SPAWN_FAILED,
                f"Could not spawn a connector-host child for target {target!r}: {exc}",
            ) from exc

    def _init_kwargs(
        self, target: str, connector_type: str, *, without_write_gateway: bool = False
    ) -> dict[str, Any]:
        """The init payload: the section as resolved here, plus this run's posture."""
        control_system = self._config.control_system
        if without_write_gateway:
            control_system = _without_write_gateway(control_system, connector_type)
        kwargs: dict[str, Any] = {
            "control_system": control_system,
            "target": target,
        }
        config_path = getattr(self._config, "config_path", None)
        if config_path:
            kwargs["config_file"] = str(Path(config_path).resolve())
        if is_readonly_run():
            # Restriction only. The child cannot be granted writes by launch
            # payload; ``control_system.writes_enabled`` is the only thing that
            # ever enables them.
            kwargs["execution_mode"] = "readonly"
        return kwargs

    # -- retirement --------------------------------------------------------

    async def _retire(self, child: _Child, cause: str) -> bool:
        """Stop new work, drain, and kill. Never raises.

        Returns:
            ``True`` when the child was drained before the deadline, ``False``
            when it was killed with requests still in flight — those requests
            complete with a :class:`ConnectionError` naming *cause*.
        """
        timeout = self._drain_timeout()
        try:
            child.proxy.refuse_new_requests(cause)
            drained = await child.proxy.drain(timeout)
            if drained:
                await child.proxy.disconnect()
                await self._kill_process(child.process)
                return True

            logger.warning(
                "Drain deadline of %ss expired on the connector-host child %s for target %r; "
                "killing it with requests still in flight (%s)",
                timeout,
                child.pid,
                child.target,
                cause,
            )
            child.reader.retire(
                f"{cause} killed the connector-host child serving target {child.target!r} "
                f"after its {timeout}s drain deadline expired with this request in flight"
            )
            await self._kill_process(child.process)
            # Give the proxy's reader the chance to turn the dead pipe into the
            # attributed failures before anything else touches the proxy.
            await child.proxy.drain(SETTLE_TIMEOUT_S)
            await child.proxy.disconnect()
            return False
        except Exception:  # pragma: no cover - defensive
            logger.warning("Error retiring connector-host child %s", child.pid, exc_info=True)
            # The child still has to go, and whatever was in flight on it still
            # has to be told why — an unexpected failure in the orderly path is
            # not a reason to leave a caller waiting on a pipe nobody will
            # answer, nor to leave it guessing at the cause.
            child.reader.retire(cause)
            await self._kill_process(child.process)
            with contextlib.suppress(Exception):
                await child.proxy.drain(SETTLE_TIMEOUT_S)
                await child.proxy.disconnect()
            return False

    async def _kill_process(self, process: Any) -> None:
        """``SIGTERM``, then ``SIGKILL`` after the grace period."""
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), self._terminate_grace_s)
            return
        except TimeoutError:
            pass
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), self._terminate_grace_s)

    # -- config ------------------------------------------------------------

    def _drain_timeout(self) -> float:
        if self._drain_override is not None:
            return max(float(self._drain_override), 0.0)
        section = self._config.control_system
        switch = section.get("target_switch") if isinstance(section, dict) else None
        value = switch.get(DRAIN_TIMEOUT_KEY) if isinstance(switch, dict) else None
        if value is None:
            return DEFAULT_DRAIN_TIMEOUT_S
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            logger.warning(
                "control_system.target_switch.%s is %r, which is not a number; using %ss",
                DRAIN_TIMEOUT_KEY,
                value,
                DEFAULT_DRAIN_TIMEOUT_S,
            )
            return DEFAULT_DRAIN_TIMEOUT_S


def _name_probed_gateway(error: SwitchError, derivation: TargetDerivation) -> SwitchError:
    """The same probe failure, naming the gateway the probe ran through.

    A target that derives no endpoint (the mock, a gatewayless deployment) has
    nothing to add, and the error passes through unchanged.
    """
    endpoint = derivation.selected_endpoint()
    if endpoint is None:
        return error
    role = derivation.selected_role
    return SwitchError(
        error.target,
        error.stage,
        error.reason,
        f"{error.detail} The probe ran through the {role!r} gateway at "
        f"{endpoint.host}:{endpoint.port}.",
        verification=error.verification,
        gateway={"role": role, "host": endpoint.host, "port": endpoint.port},
    )


def _without_write_gateway(section: dict[str, Any], connector_type: str) -> dict[str, Any]:
    """A deep copy of the ``control_system`` section, minus one write gateway.

    A child launched with this payload sees a deployment whose write gateway
    was never configured, which is the case ``connect()`` already handles by
    routing through ``read_only`` with a warning. Stripping the row — rather
    than teaching the child a role override — keeps the fallback on that
    single documented path, and means the fallback child cannot route anything
    to the write gateway even by accident: it never learns the endpoint.
    """
    stripped = copy.deepcopy(section)
    connector = stripped.get("connector")
    block = connector.get(connector_type) if isinstance(connector, dict) else None
    gateways = block.get("gateways") if isinstance(block, dict) else None
    if isinstance(gateways, dict):
        gateways.pop(ROLE_WRITE_ACCESS, None)
    return stripped


def _verify(derivation: TargetDerivation, report: dict[str, Any]) -> Verification:
    """Assert the child came up where the derivation said it would.

    :func:`~osprey.mcp_server.control_system.target_eligibility.verify_child_report`
    answers this for every target whose config names a gateway. A deployment
    can also select a connector that talks to no gateway at all — the mock is
    one, and it is the generic template's default — and for that one the
    derivation has no endpoint and the child reports none. Nothing is verified
    there because there is no endpoint to get wrong, but the *symmetry* is:
    a child that configured Channel Access where the config derived nothing has
    inherited an environment from somewhere, and that is a mismatch as serious
    as any other.

    The unverified branch is entered only when the derivation has **no endpoint
    rows at all**. A target with rows whose *selected* role is missing — a
    write-only gateway table on a deployment that selects ``read_only``, say —
    is a gateway deployment with a hole in it, not a gatewayless one: its child
    connects to a real control system over whatever default broadcast address
    it finds, and that is precisely the unpinned-CA case verification exists to
    catch. It goes to ``verify_child_report``, which refuses it either for
    reporting no gateway or for having no derived endpoint to compare against.
    """
    if derivation.endpoints:
        return verify_child_report(derivation, report)

    if report.get("_epics_configured") or report.get("mode") or report.get("host"):
        return Verification(
            False,
            "_epics_configured",
            False,
            report.get("_epics_configured"),
            f"Target {derivation.target!r} derives no gateway on this deployment, but the "
            f"child reports it configured {report.get('mode')!r} routing to "
            f"{report.get('host')!r} — an endpoint this config never described.",
        )
    return Verification(
        True,
        detail=(
            f"Target {derivation.target!r} derives no gateway (connector type "
            f"{derivation.connector_type!r}) and the child configured none."
        ),
    )
