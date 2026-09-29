"""Core ``web_terminals`` health category.

Reports, for every web-terminal card and every dispatch worker, whether the
process runs under the account name the render intended — the name the control
system sees its writes arrive under. One row per card, one per worker, all
probed concurrently.

Each process answers ``GET /health`` with ``ca_user`` (the name uid 1000
resolves to inside the container, read in-process) and
``control_identity_skipped`` (set when the entrypoint could not apply the
identity). The expectation is derived from the same config keys the compose
templates read:

* **A card** expects its roster ``control_identity`` only behind a login wall
  (``auth.method`` ``password``/``oidc``) — without a wall there is nobody
  signed in to attribute writes to, so compose emits no identity and the card
  keeps the image account ``osprey``. A card with no ``control_identity``
  expects ``osprey`` too. The card is probed on its own terminal port on
  loopback (every terminal runs on the host network).
* **Dispatch worker ``i``** always expects ``osprey-dispatch-<i>``. Only a
  host-networked worker has a host port to probe; a bridge-networked worker
  publishes nothing, so it gets a ``skip`` row saying so.

Status mapping:

* ``control_identity_skipped`` present -> ``warning`` (a non-root start or a
  failed rewrite left uid 1000 as ``osprey``; the deployment runs,
  unattributed).
* ``ca_user`` differs from the expectation -> ``error``.
* ``ca_user`` absent -> ``ok`` when ``osprey`` is expected (an image that
  predates the field runs as ``osprey`` anyway), else ``error`` (that image
  ignores ``OSPREY_CONTROL_IDENTITY``, so the name is not applied).
* unreachable, non-2xx or a non-JSON body -> ``warning``, never an exception.

With web terminals disabled and no dispatch worker deployed the category
contributes no rows.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from osprey.deployment.web_terminals.personas import normalize_users
from osprey.deployment.web_terminals.ports import allocate_ports, base_ports_from_config
from osprey.deployment.web_terminals.render import _auth_tls_context
from osprey.health.models import CheckResult, Status
from osprey.port_layout import default_port, resolve_port_base

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "web_terminals"

_PROBE_TIMEOUT_S = 5.0

#: The account every image runs uid 1000 as when no identity is applied.
_IMAGE_ACCOUNT = "osprey"

#: Every terminal and every host-networked worker binds loopback.
_LOOPBACK = "127.0.0.1"

_WORKER_IDENTITY_PREFIX = "osprey-dispatch-"
_DEFAULT_WORKER_COUNT = 1
_DEFAULT_WORKER_PORT_STRIDE = 1


class _Target(NamedTuple):
    """One process to probe.

    Attributes:
        row_id: Row name suffix (``<category>.<row_id>``).
        label: Display label for the row message.
        url: The ``/health`` URL.
        expected: The account name ``ca_user`` must report.
    """

    row_id: str
    label: str
    url: str
    expected: str


def web_terminals(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CategoryCallable:
    """Build the ``web_terminals`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Read for ``modules.web_terminals``, ``services.dispatch_worker``,
            ``deployed_services`` and the deployment port base.
        context: Health runtime. Unused — processes are probed over HTTP.
        transport: Optional httpx transport for dependency injection in tests.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        targets, config_rows = _resolve_targets(cfg)
        probed = list(await asyncio.gather(*(_probe(t, transport) for t in targets)))
        return sorted(probed + config_rows, key=lambda row: row.name)

    return _run


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_name(row_id: str) -> str:
    return f"{CATEGORY}.{row_id.replace('-', '_')}"


def _resolve_targets(cfg: Mapping[str, Any]) -> tuple[list[_Target], list[CheckResult]]:
    """Enumerate the cards and workers to probe, plus rows that need no probe."""
    targets: list[_Target] = []
    rows: list[CheckResult] = []
    base = resolve_port_base(cfg)

    card_targets, card_rows = _card_targets(cfg, base)
    targets.extend(card_targets)
    rows.extend(card_rows)

    worker_targets, worker_rows = _worker_targets(cfg, base)
    targets.extend(worker_targets)
    rows.extend(worker_rows)
    return targets, rows


def _card_targets(cfg: Mapping[str, Any], base: int) -> tuple[list[_Target], list[CheckResult]]:
    wt = _as_dict(_as_dict(cfg.get("modules")).get("web_terminals"))
    if not wt.get("enabled"):
        return [], []
    try:
        walled = bool(_auth_tls_context(wt, base=base)["walled"])
    except ValueError as exc:
        # The render refuses this config outright; there is no deployment whose
        # identities could be compared, only the key to fix.
        return [], [
            CheckResult(
                _row_name("auth"),
                CATEGORY,
                Status.WARNING,
                "Web terminals: auth misconfigured",
                value="misconfigured",
                details=str(exc),
            )
        ]

    base_ports = base_ports_from_config(wt, base=base)
    targets: list[_Target] = []
    for entry in normalize_users(wt.get("users"), strict=False):
        name = entry["name"]
        try:
            port = allocate_ports(base_ports, entry["index"])["web"]
        except ValueError:
            # The web-terminal lint names this entry; there is no port to probe.
            continue
        identity = entry.get("control_identity")
        expected = identity if walled and identity else _IMAGE_ACCOUNT
        targets.append(_Target(name, f"Card {name}", f"http://{_LOOPBACK}:{port}/health", expected))
    return targets, []


def _worker_targets(cfg: Mapping[str, Any], base: int) -> tuple[list[_Target], list[CheckResult]]:
    deployed = cfg.get("deployed_services")
    if not isinstance(deployed, list) or "dispatch_worker" not in deployed:
        return [], []
    worker = _as_dict(_as_dict(cfg.get("services")).get("dispatch_worker"))
    count = _int_or_default(worker.get("worker_count"), _DEFAULT_WORKER_COUNT)
    host_network = str(worker.get("network") or "").strip() == "host"

    if not host_network:
        return [], [
            CheckResult(
                _row_name(f"dispatch_worker_{i}"),
                CATEGORY,
                Status.SKIP,
                f"Dispatch worker {i}: not probed — bridge network publishes no host port",
                details=(
                    f"Expected account {_WORKER_IDENTITY_PREFIX}{i}; check it with "
                    "`docker exec <worker> whoami`."
                ),
            )
            for i in range(1, count + 1)
        ]

    worker_base = _int_or_default(
        worker.get("worker_port_base"), default_port("worker", 1, base=base)
    )
    stride = _int_or_default(worker.get("worker_port_stride"), _DEFAULT_WORKER_PORT_STRIDE)
    targets = [
        _Target(
            f"dispatch_worker_{i}",
            f"Dispatch worker {i}",
            f"http://{_LOOPBACK}:{worker_base + (i - 1) * stride}/health",
            f"{_WORKER_IDENTITY_PREFIX}{i}",
        )
        for i in range(1, count + 1)
    ]
    return targets, []


async def _probe(target: _Target, transport: httpx.AsyncBaseTransport | None) -> CheckResult:
    """Fetch one ``/health`` and compare its ``ca_user`` with the expectation."""
    name = _row_name(target.row_id)
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(target.url)
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{target.label}: unreachable",
            value="offline",
            details=f"{target.url} — {exc}. Expected account {target.expected!r}.",
        )
    latency_ms = (time.perf_counter() - start) * 1000.0

    if not resp.is_success:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{target.label}: HTTP {resp.status_code}",
            value="degraded",
            latency_ms=latency_ms,
            details=f"{target.url} answered {resp.status_code}.",
            probed=True,
        )
    try:
        body = resp.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{target.label}: /health answered no JSON object",
            value="degraded",
            latency_ms=latency_ms,
            details=f"{target.url} did not return a JSON object.",
            probed=True,
        )

    return _classify(name, target, body, latency_ms)


#: Why the entrypoint left a service identity unapplied, keyed by the
#: ``OSPREY_CONTROL_IDENTITY_SKIPPED`` value it exports.
_SKIP_CAUSES = {
    "non-root-start": "The container started without root; start it as root to apply the identity",
    "apply-failed": (
        "The entrypoint could not rewrite /etc/passwd (a read-only root filesystem, a "
        "missing module or interpreter); the container log names the cause"
    ),
}


def _classify(name: str, target: _Target, body: dict[str, Any], latency_ms: float) -> CheckResult:
    ca_user = body.get("ca_user")
    skipped = body.get("control_identity_skipped")

    if skipped:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{target.label}: control identity {target.expected!r} not applied ({skipped})",
            value=str(ca_user or ""),
            latency_ms=latency_ms,
            details=(
                f"{_SKIP_CAUSES.get(skipped, _SKIP_CAUSES['apply-failed'])}, so uid 1000 "
                f"keeps the name {ca_user or _IMAGE_ACCOUNT!r} and control-system writes "
                f"are not attributed to {target.expected!r}."
            ),
            probed=True,
        )

    if not isinstance(ca_user, str) or not ca_user:
        if target.expected == _IMAGE_ACCOUNT:
            return CheckResult(
                name,
                CATEGORY,
                Status.OK,
                f"{target.label}: runs as {_IMAGE_ACCOUNT}",
                value=_IMAGE_ACCOUNT,
                latency_ms=latency_ms,
                details="/health reports no ca_user; the image predates the field.",
                probed=True,
            )
        return CheckResult(
            name,
            CATEGORY,
            Status.ERROR,
            f"{target.label}: /health reports no ca_user, expected {target.expected!r}",
            value="unknown",
            latency_ms=latency_ms,
            details=(
                "The running image predates control identity and ignores "
                "OSPREY_CONTROL_IDENTITY; rebuild and redeploy it."
            ),
            probed=True,
        )

    if ca_user != target.expected:
        return CheckResult(
            name,
            CATEGORY,
            Status.ERROR,
            f"{target.label}: runs as {ca_user!r}, expected {target.expected!r}",
            value=ca_user,
            latency_ms=latency_ms,
            details=(
                "The control system attributes this process's writes to the name it "
                "runs under. Re-render and redeploy so the container and the config agree."
            ),
            probed=True,
        )

    return CheckResult(
        name,
        CATEGORY,
        Status.OK,
        f"{target.label}: runs as {ca_user}",
        value=ca_user,
        latency_ms=latency_ms,
        probed=True,
    )
