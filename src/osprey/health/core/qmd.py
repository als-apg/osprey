"""Core ``qmd`` health category.

One row per corpus sidecar, **only when ``services.qmd`` is configured** — the
same block the compose fragment, the port sweep and the clients read, so the
category stays a valid ``--category`` name at all times while contributing no
rows to a project that runs no sidecar.

Each row asks the sidecar's own ``status`` tool how large its index is and how
many of its documents have no vectors yet:

* ``ok`` — every indexed document is embedded (``value`` is the document count);
* ``warning`` — some documents have no vectors, so they are found by keyword
  search only; the message says how many. A sidecar still embedding after a
  large change reads this way for a while, and a count that never falls names
  chunks that fail on every retry;
* ``warning`` — the sidecar did not answer. A sidecar builds its index before it
  opens its port, so a first boot reads as unreachable for as long as the build
  takes.

Every row is advisory: a sidecar that is down or behind degrades search, it does
not break the deployment.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "qmd"

#: Seconds one sidecar may take to answer. Its status tool counts rows in the
#: index, which is quick even at hundreds of thousands of documents.
_STATUS_TIMEOUT_S = 10.0


def qmd(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
) -> CategoryCallable:
    """Build the ``qmd`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Read for the ``services.qmd`` block and the corpora it serves.
        context: Health runtime. Unused — the sidecars are dialed over HTTP.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        from osprey.deployment.compose_generator import _resolve_qmd_corpora
        from osprey.deployment.qmd_service import resolve_qmd_service_config

        try:
            resolved = resolve_qmd_service_config(cfg)
            if resolved is None:
                return []
            corpora = [corpus["collection"] for corpus in _resolve_qmd_corpora(dict(cfg), None)]
        except ValueError as exc:
            return [
                CheckResult(
                    f"{CATEGORY}_config",
                    CATEGORY,
                    Status.WARNING,
                    f"Cannot resolve the qmd sidecars: {exc}",
                    details="Fix the key named above in config.yml.",
                )
            ]
        rows = await asyncio.gather(
            *(asyncio.to_thread(_probe, name, resolved.for_corpus(name)) for name in corpora)
        )
        return list(rows)

    return _run


def _probe(corpus: str, settings: Any) -> CheckResult:
    """Ask one corpus's sidecar for its index counts and turn them into a row."""
    from osprey.services.qmd import QMDClient, QMDClientError

    name = f"{CATEGORY}_{corpus}"
    client = QMDClient(settings, timeout=_STATUS_TIMEOUT_S, client_name="osprey-health")
    start = perf_counter()
    try:
        status = client.status()
    except QMDClientError as exc:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{corpus}: sidecar did not answer at {settings.base_url}",
            value="offline",
            probed=True,
            details=(
                f"{exc}. A sidecar opens its port only once its index is built, so a "
                "first boot reads as unreachable until the build finishes; its "
                "container log reports the build's progress."
            ),
        )
    latency_ms = (perf_counter() - start) * 1000.0
    if status.pending:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{corpus}: {status.pending} of {status.documents} documents have no vectors",
            value=str(status.documents),
            latency_ms=latency_ms,
            probed=True,
            details=(
                "Those documents are found by keyword search only. The sidecar embeds "
                "pending documents on every update; a count that stays put names chunks "
                "that fail on every retry, which its container log lists."
            ),
        )
    return CheckResult(
        name,
        CATEGORY,
        Status.OK,
        f"{corpus}: {status.documents} documents, all embedded",
        value=str(status.documents),
        latency_ms=latency_ms,
        probed=True,
    )
