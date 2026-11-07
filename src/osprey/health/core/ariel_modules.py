"""Core ``ariel_modules`` health category (on_demand).

Each enabled ARIEL enhancement module (``ariel.enhancement_modules``) is built
alone and its own ``health_check()`` awaited (5 s bound, all concurrently) —
the check ``osprey ariel status`` runs. That check is not free: a semantic
processor sends a chat completion, an embedding module sends an embedding, an
export module creates its mirror root. So the category is ``on_demand``: the
poll never runs it; ``osprey health --full`` or a re-probe click does.

* ``ariel.enhancement.<module>`` — ``ok`` when reachable, ``warning`` with the
  module's reason when not, ``skip`` when the module has no check of its own.
  A module that cannot be built is a ``warning`` ("config error").

No enabled module, no rows. Every row is advisory.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "ariel_modules"

#: Per-module bound in seconds.
_TIMEOUT_S = 5.0

#: Builds the enabled modules from the config mapping; injectable for tests.
ModuleFactory = Callable[[Mapping[str, Any]], list[Any]]


def ariel_modules(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    modules: ModuleFactory | None = None,
) -> CategoryCallable:
    """Build the ``ariel_modules`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Health runtime. Unused.
        modules: Optional factory of the enabled modules, for tests; ``None``
            builds them from the config.

    Returns:
        A no-argument async callable returning one row per enabled module.
    """
    cfg: Mapping[str, Any] = config or {}
    build = modules or _build_ariel_modules

    async def _run() -> list[CheckResult]:
        return await _ariel_rows(cfg, build)

    return _run


def _enabled_module_names(cfg: Mapping[str, Any]) -> list[str]:
    ariel = cfg.get("ariel")
    modules = ariel.get("enhancement_modules") if isinstance(ariel, Mapping) else None
    if not isinstance(modules, Mapping):
        return []
    return [
        str(name)
        for name, block in modules.items()
        if isinstance(block, Mapping) and block.get("enabled")
    ]


class _Unbuildable:
    """Stands in for a module whose construction failed, so it reports as one row."""

    def __init__(self, name: str, exc: Exception) -> None:
        self.name = name
        self.config_error = exc


def _build_ariel_modules(cfg: Mapping[str, Any]) -> list[Any]:
    """Build each enabled module alone, as ``osprey ariel status`` does."""
    from osprey.port_layout import resolve_port_base
    from osprey.services.ariel_search import ARIELConfig
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config

    services = cfg.get("services")
    postgresql = services.get("postgresql") if isinstance(services, Mapping) else None
    config = ARIELConfig.from_dict(
        dict(cfg.get("ariel") or {}), dict(postgresql or {}), base=resolve_port_base(cfg)
    )
    built: list[Any] = []
    for name in _enabled_module_names(cfg):
        try:
            built.extend(
                m
                for m in create_enhancers_from_config(config, stage="all", names=[name])
                if m.name == name
            )
        except Exception as exc:  # one misconfigured module never hides the others
            built.append(_Unbuildable(name, exc))
    return built


async def _ariel_rows(cfg: Mapping[str, Any], build: ModuleFactory) -> list[CheckResult]:
    if not _enabled_module_names(cfg):
        return []
    try:
        modules = await asyncio.to_thread(build, cfg)
    except Exception as exc:
        return [
            CheckResult(
                "ariel.enhancement",
                CATEGORY,
                Status.WARNING,
                "ARIEL enhancement modules could not be built",
                value="config error",
                details=str(exc),
            )
        ]
    return list(await asyncio.gather(*(_module_row(m) for m in modules)))


async def _module_row(module: Any) -> CheckResult:
    from osprey.services.ariel_search.enhancement.base import as_health_result

    name = f"ariel.enhancement.{module.name}"
    error = getattr(module, "config_error", None)
    if error is not None:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"ARIEL module {module.name} cannot be configured",
            value="config error",
            details=str(error),
        )
    started = time.perf_counter()
    try:
        result = as_health_result(await asyncio.wait_for(module.health_check(), _TIMEOUT_S))
    except TimeoutError:
        result = None
        failure = f"no answer within {_TIMEOUT_S:g} s"
        value = "no answer"
    except Exception as exc:
        result = None
        failure = str(exc) or type(exc).__name__
        value = "check failed"
    elapsed = (time.perf_counter() - started) * 1000.0
    if result is None:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"ARIEL module {module.name} health check failed",
            value=value,
            details=failure,
            latency_ms=elapsed,
            probed=True,
        )
    if result.reachable is None:
        return CheckResult(
            name,
            CATEGORY,
            Status.SKIP,
            f"ARIEL module {module.name} has no health check",
            value="no health check",
        )
    if result.reachable:
        return CheckResult(
            name,
            CATEGORY,
            Status.OK,
            result.message or f"ARIEL module {module.name} reachable",
            value=f"reachable · {elapsed:.0f} ms",
            latency_ms=elapsed,
            probed=True,
        )
    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        result.message or f"ARIEL module {module.name} unreachable",
        value=result.reason or "unreachable",
        latency_ms=elapsed,
        probed=True,
    )
