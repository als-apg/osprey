"""Async-native health-suite runner.

Executes a merged set of :class:`~osprey.health.config.CategoryRecord`\\ s and
returns a single :class:`~osprey.health.models.CheckReport`.

Execution model
---------------

* **Categories run concurrently** via :func:`asyncio.gather`.
* **Within a declarative category, only ``requires:`` chains serialize.** Checks
  are executed in topological batches: every check whose dependencies have all
  resolved runs in the same batch, concurrently. Independent checks therefore
  overlap; only a genuine ``A → B`` dependency forces ordering.
* **``requires`` gating.** A dependency *passes* iff its status is ``ok`` or
  ``warning`` — composing literally with ``timeout_status`` (a timeout
  classified ``warning`` counts as passed). A check with any non-passing
  dependency is emitted as ``skip`` without running, and that ``skip`` in turn
  fails its own dependents (the cascade).
* **Isolation.** A probe or callable that raises becomes a single ``error``
  result; it never aborts sibling checks or categories.
* **Per-check timeout.** Each declarative check is awaited under
  :func:`asyncio.wait_for` with a small backstop margin over its ``timeout_s``
  (so a probe's own internal timeout — which yields a richer result — fires
  first). A backstop expiry synthesizes the check's ``timeout_status``.
* **Sync off-loading.** Callable categories that are plain functions run on a
  daemon thread via :func:`osprey.health.offload.run_sync`, so a hung sync check
  can never wedge process exit.
* **Cost gating.** ``on_demand`` categories run only when ``full`` is set;
  otherwise they are emitted as ``skip`` rows carrying a "run with --full" hint.
  ``--category`` selection never elevates cost class — ``full`` is the sole gate.
* **Cost-class deadlines.** Poll-class categories are collectively bounded by
  ``suite_timeout_s``; on_demand categories by ``on_demand_timeout_s`` — the two
  budgets are independent absolute deadlines. At a deadline, unfinished checks
  are *synthesized* rather than dropped: for a declarative category, one row per
  configured check always (a pending check whose dependency failed → ``skip``;
  an eligible pending check → ``error`` "suite deadline exceeded"; a synthesized
  deadline-``error`` fails its own dependents, cascading in topological order).
  A callable-backed category that hits its budget yields exactly one synthesized
  ``error`` row named after the category. A run that synthesizes any such row
  sets :attr:`CheckReport.deadline_hit`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from time import perf_counter

from .config import DEFAULT_SUITE_TIMEOUT_S, CategoryRecord, Cost
from .models import CheckReport, CheckResult, Status
from .offload import run_sync
from .probes import ProbeContext, get_probe
from .runtime import HealthRuntime

#: Extra seconds added to a declarative check's ``timeout_s`` for the runner's
#: backstop ``wait_for``. Keeping the backstop slightly longer than the check's
#: own timeout lets the probe's internal timeout fire first (producing a richer
#: result with measured latency) while still bounding a probe that fails to
#: self-limit.
_TIMEOUT_BACKSTOP_MARGIN_S: float = 0.5

_PASSING = (Status.OK, Status.WARNING)
_DEADLINE_MESSAGE = "suite deadline exceeded"


def _on_demand_hint(category: str) -> str:
    return (
        f"on_demand category; run with --full (e.g. `osprey health --full --category {category}`)"
    )


def _skip_on_demand(record: CategoryRecord) -> list[CheckResult]:
    """Emit skip rows for an on_demand category that was not elevated by ``full``."""
    hint = _on_demand_hint(record.name)
    if record.checks is not None:
        return [CheckResult(c.name, record.name, Status.SKIP, hint) for c in record.checks]
    return [CheckResult(record.name, record.name, Status.SKIP, hint)]


def _skip_for_failed_dep(check_name: str, category: str, failed_dep: str) -> CheckResult:
    return CheckResult(
        check_name,
        category,
        Status.SKIP,
        f"skipped: dependency '{failed_dep}' did not pass",
    )


async def _run_check(
    spec_name: str,
    category: str,
    params: dict,
    timeout_s: float,
    timeout_status: Status,
    ctx: ProbeContext,
    probe_type: str,
) -> CheckResult:
    """Run a single declarative check under a per-check timeout backstop."""
    spec = {
        **params,
        "name": spec_name,
        "category": category,
        "timeout_s": timeout_s,
        "timeout_status": timeout_status.value,
    }
    try:
        probe = get_probe(probe_type)
        return await asyncio.wait_for(
            probe(spec, ctx), timeout=timeout_s + _TIMEOUT_BACKSTOP_MARGIN_S
        )
    except TimeoutError:
        return CheckResult(
            spec_name,
            category,
            timeout_status,
            f"{spec_name} timed out after {timeout_s:g}s",
        )
    except Exception as exc:  # noqa: BLE001 - isolation: any failure is one error row
        return CheckResult(
            spec_name,
            category,
            Status.ERROR,
            f"{spec_name} raised {type(exc).__name__}",
            details=str(exc),
        )


def _synthesize_pending(
    record: CategoryRecord,
    results: dict[str, CheckResult],
) -> None:
    """Fill in rows for checks left pending at a deadline, in topological order.

    Declared order is topological (``requires`` only references earlier checks),
    so a dependency's row — real or already synthesized — is present by the time
    its dependent is reached. A pending check whose dependency did not pass is
    ``skip`` (requires rule wins); an otherwise-eligible pending check is an
    ``error`` deadline row, which in turn fails its own dependents.
    """
    for check in record.checks or []:
        if check.name in results:
            continue
        failed_dep = next(
            (d for d in check.requires if results[d].status not in _PASSING),
            None,
        )
        if failed_dep is not None:
            results[check.name] = _skip_for_failed_dep(check.name, record.name, failed_dep)
        else:
            results[check.name] = CheckResult(
                check.name, record.name, Status.ERROR, _DEADLINE_MESSAGE
            )


async def _run_declarative(
    record: CategoryRecord,
    ctx: ProbeContext,
    deadline: float | None,
) -> tuple[list[CheckResult], bool]:
    """Execute a declarative category in topological batches under a deadline.

    Returns the results (in declared order) and whether a deadline synthesis
    occurred.
    """
    checks = record.checks or []
    results: dict[str, CheckResult] = {}

    while len(results) < len(checks):
        if deadline is not None and perf_counter() >= deadline:
            break

        ready = [
            c for c in checks if c.name not in results and all(d in results for d in c.requires)
        ]
        if not ready:  # defensive: config validation forbids cycles/forward refs
            break

        to_run = []
        for check in ready:
            failed_dep = next(
                (d for d in check.requires if results[d].status not in _PASSING),
                None,
            )
            if failed_dep is not None:
                results[check.name] = _skip_for_failed_dep(check.name, record.name, failed_dep)
            else:
                to_run.append(check)

        if not to_run:
            continue

        remaining = None if deadline is None else max(deadline - perf_counter(), 0.0)
        tasks = {
            asyncio.ensure_future(
                _run_check(
                    c.name,
                    record.name,
                    dict(c.params),
                    c.timeout_s,
                    c.timeout_status,
                    ctx,
                    c.type,
                )
            ): c
            for c in to_run
        }
        done, still_pending = await asyncio.wait(tasks.keys(), timeout=remaining)
        for task in done:
            check = tasks[task]
            results[check.name] = task.result()
        if still_pending:
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
            break

    deadline_hit = len(results) < len(checks)
    if deadline_hit:
        _synthesize_pending(record, results)

    return [results[c.name] for c in checks], deadline_hit


async def _run_callable(
    record: CategoryRecord,
    deadline: float | None,
) -> tuple[list[CheckResult], bool]:
    """Execute a callable-backed category (core/plugin), off-loading sync funcs.

    The effective budget is the smaller of the category's ``timeout_s`` and the
    time left before the cost-class deadline. A budget expiry synthesizes one
    ``error`` row named after the category and flags a deadline hit; a raise
    synthesizes one ``error`` row without flagging a deadline (isolation).
    """
    func = record.func
    if deadline is None:
        effective = record.timeout_s
    else:
        effective = min(record.timeout_s, max(deadline - perf_counter(), 0.0))

    try:
        if asyncio.iscoroutinefunction(func):
            result = await asyncio.wait_for(func(), timeout=effective)
        else:
            result = await run_sync(func, timeout_s=effective)
        return list(result), False
    except TimeoutError:
        return (
            [
                CheckResult(
                    record.name,
                    record.name,
                    Status.ERROR,
                    f"{record.name} exceeded its {effective:g}s budget",
                )
            ],
            True,
        )
    except Exception as exc:  # noqa: BLE001 - isolation: one error row per failing category
        return (
            [
                CheckResult(
                    record.name,
                    record.name,
                    Status.ERROR,
                    f"{record.name} raised {type(exc).__name__}",
                    details=str(exc),
                )
            ],
            False,
        )


async def run_health_suite(
    records: Sequence[CategoryRecord],
    *,
    runtime: HealthRuntime,
    full: bool = False,
    categories: Iterable[str] | None = None,
    suite_timeout_s: float = DEFAULT_SUITE_TIMEOUT_S,
    on_demand_timeout_s: float | None = None,
) -> CheckReport:
    """Run the merged health categories and return an aggregated report.

    Args:
        records: The merged category set (core + YAML + plugin), already
            resolved to :class:`~osprey.health.config.CategoryRecord`\\ s.
        runtime: The suite's :class:`~osprey.health.runtime.HealthRuntime`,
            passed to every probe via :class:`~osprey.health.probes.ProbeContext`.
        full: When ``True`` on_demand categories execute; otherwise they emit
            ``skip`` rows. ``--category`` selection never elevates cost class.
        categories: Optional set of category names to run; ``None`` runs all.
        suite_timeout_s: Wall-clock budget bounding the poll-class categories
            collectively.
        on_demand_timeout_s: Wall-clock budget bounding the on_demand categories
            collectively; when ``None`` it defaults to the sum of the selected
            on_demand categories' resolved budgets.

    Returns:
        A :class:`~osprey.health.models.CheckReport` whose results preserve
        category order and, within a category, declared check order, with
        ``elapsed_ms`` and ``deadline_hit`` populated.
    """
    t0 = perf_counter()
    if categories is None:
        selected = list(records)
    else:
        wanted = set(categories)
        selected = [r for r in records if r.name in wanted]

    ctx = ProbeContext(runtime=runtime)
    suite_deadline = t0 + suite_timeout_s

    on_demand_deadline: float | None = None
    if full:
        if on_demand_timeout_s is None:
            od_budgets = [r.timeout_s for r in selected if r.cost is Cost.ON_DEMAND]
            on_demand_timeout_s = sum(od_budgets) if od_budgets else suite_timeout_s
        on_demand_deadline = t0 + on_demand_timeout_s

    async def _run_category(record: CategoryRecord) -> tuple[list[CheckResult], bool]:
        if record.cost is Cost.ON_DEMAND and not full:
            return _skip_on_demand(record), False
        deadline = on_demand_deadline if record.cost is Cost.ON_DEMAND else suite_deadline
        if record.func is not None:
            return await _run_callable(record, deadline)
        return await _run_declarative(record, ctx, deadline)

    grouped = await asyncio.gather(*(_run_category(r) for r in selected))
    results = [result for rows, _ in grouped for result in rows]
    deadline_hit = any(hit for _, hit in grouped)
    elapsed_ms = (perf_counter() - t0) * 1000.0
    return CheckReport(results=results, elapsed_ms=elapsed_ms, deadline_hit=deadline_hit)
