"""Core ``reach`` health category.

Reports whether each shared service this render's clients are switched on for
answers at the address **the client itself dials** — one row per service,
all knocked on concurrently.

This is the run-time half of the Reach Contract
(:mod:`osprey.deployment.reach`). The build refuses a consumer switched on
with nothing to resolve; this category takes the consumers that passed and
asks the only question left: from *here*, does anything answer? "Here" is
what makes the row worth having. The ``ariel``, ``graphdb`` and
``openobserve`` categories probe a service where its config block says it is;
this one resolves the endpoint through the consumer's own resolver — the
bridge URL the bluesky MCP server builds, the DSN the panel server derives,
the OTLP endpoint the exporter posts to, env overrides included — in the
process that runs the check. Run inside a per-user container by the health
MCP server, the rows say what that container's agent will reach; run by
``osprey health`` on the host, what the host's own processes will.

A knock is a TCP connect and nothing more: every shared service is a TCP
listener, and what this category owes the operator is "the port the client
dials is open", not a second implementation of each service's health
contract. Rows are advisory (``ok``/``warning``), matching ``web_panels``: a
service that is down is a warning, never a suite error, and a consumer whose
client has nothing to dial — the state the build refuses, met at run time —
is a warning naming the switch and the key. With no live consumer the
category contributes no rows at all.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from osprey.deployment.reach import Consumer, Dial, ReachContract, reach_dials
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "reach"

_KNOCK_TIMEOUT_S = 3.0

#: Opens a TCP connection to ``(host, port)`` and closes it, raising
#: :class:`OSError` (or :class:`TimeoutError`) when nothing answers.
Knock = Callable[[str, int], Awaitable[None]]


class _Target(NamedTuple):
    """One shared service to knock on, for every live consumer that dials it.

    Attributes:
        contract: The service's contract, for the row name and the remedy.
        consumers: The live consumers, in registry order; all share one dial.
        dial: What they connect to, or ``None`` when the client resolves
            nothing.
    """

    contract: ReachContract
    consumers: tuple[Consumer, ...]
    dial: Dial | None


def reach(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    knock: Knock | None = None,
) -> CategoryCallable:
    """Build the ``reach`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Every consumer switch and resolver in the registry reads it.
        context: Health runtime. Unused — the knock is a plain TCP connect.
        knock: Optional connect function for dependency injection in tests;
            ``None`` opens a real connection.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        targets = _targets(cfg)
        if not targets:
            return []
        rows = await asyncio.gather(*(_probe(t, knock or _tcp_knock) for t in targets))
        return sorted(rows, key=lambda row: row.name)

    return _run


def _targets(cfg: Mapping[str, Any]) -> list[_Target]:
    """One target per service with a live consumer.

    Consumers of one service dial through one resolver (hybrid search and the
    OKF panel both ask :func:`osprey.deployment.qmd_service.resolve_qmd_service_config`),
    so a service is knocked on once and its row names every consumer that
    depends on the answer.
    """
    by_service: dict[str, _Target] = {}
    for contract, consumer, dial in reach_dials(cfg):
        target = by_service.get(contract.service)
        if target is None:
            by_service[contract.service] = _Target(contract, (consumer,), dial)
        else:
            by_service[contract.service] = target._replace(
                consumers=(*target.consumers, consumer), dial=target.dial or dial
            )
    return list(by_service.values())


async def _tcp_knock(host: str, port: int) -> None:
    """Open and close one TCP connection, within the knock timeout."""
    _reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port), timeout=_KNOCK_TIMEOUT_S
    )
    writer.close()
    await writer.wait_closed()


async def _probe(target: _Target, knock: Knock) -> CheckResult:
    """Knock on one target and turn the outcome into a row."""
    # `<category>.<service>`: the dashboard's fmtName() strips the leading
    # category segment, so the row reads "Qmd" / "Postgresql" / "Bluesky".
    name = f"{CATEGORY}.{target.contract.service}"
    who = " / ".join(consumer.name for consumer in target.consumers)
    switches = ", ".join(consumer.switch_key for consumer in target.consumers)
    keys = ", ".join(projected.key for projected in target.contract.projected) or (
        f"services.{target.contract.service}"
    )

    if target.dial is None:
        degrades = all(not consumer.refuse for consumer in target.consumers)
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{who}: nothing to dial",
            value="unresolved",
            details=(
                f"Switched on ({switches}) but this config resolves no endpoint for it: "
                f"no {keys}. "
                + (
                    "The consumer degrades without one, by design. "
                    if degrades
                    else "Every use will fail. "
                )
                + "An attached render is told the key by the build from its hosting "
                "deployment; name it under `config:` or switch the consumer off."
            ),
        )

    host, port = target.dial
    address = f"{host}:{port}"
    start = time.perf_counter()
    try:
        await knock(host, port)
    except (OSError, TimeoutError) as exc:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{who}: unreachable at {address}",
            value="offline",
            details=(
                f"{address} — {exc or 'no answer'}. This is the address the client "
                f"itself resolves from this config ({keys}) and would connect to from "
                f"here; check that the service is up and published on that port on "
                f"the host it shares with this render."
            ),
        )

    return CheckResult(
        name,
        CATEGORY,
        Status.OK,
        f"{who}: reachable at {address}",
        value="up",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
