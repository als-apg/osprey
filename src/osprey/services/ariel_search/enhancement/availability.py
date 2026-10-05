"""Whether a picture module's service can be used, and the one log of when that changes.

Three pieces every picture path shares:

* :func:`unavailable_reason` — the one classification of an exception into an
  availability reason. No other module maps exceptions to reasons.
* :func:`preflight` — the one pre-pass check (schema, required relations, then
  the module's own ``health_check`` under 5 s), so a catch-up pass and
  ``status`` always agree.
* The per-process tracker — :func:`report_unavailable` and
  :func:`report_available` log one WARNING when a module becomes unavailable or
  its reason changes, DEBUG while it stays so, and one INFO when it is
  available again, so a deployment running with no model server logs one line
  per module, not one per poll.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

from osprey.models.providers.health import failure_reason
from osprey.services.ariel_search.enhancement.base import HealthResult, as_health_result
from osprey.services.ariel_search.exceptions import ModuleConfigError, ModuleUnavailable
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule

__all__ = [
    "HEALTH_TIMEOUT_S",
    "ModuleConfigError",
    "ModuleUnavailable",
    "exception_chain",
    "fix_for",
    "has_success",
    "library_exception_classes",
    "note_success",
    "preflight",
    "report_available",
    "report_unavailable",
    "reset_availability",
    "unavailable_reason",
]

logger = get_logger("ariel")

AvailabilityReason = Literal["unreachable", "auth", "model", "config"]

#: Seconds the pre-pass check gives a module's ``health_check()``.
HEALTH_TIMEOUT_S = 5.0

_MIGRATE = "run osprey ariel migrate"

# module name -> current unavailability reason (None: available).
_state: dict[str, str | None] = {}
# (module, marker) pairs with at least one stored success in this process.
_successes: set[tuple[str, str]] = set()
# pool -> relations already seen to exist.
_relations: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and the exceptions it was raised from, outermost first."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None


def library_exception_classes(module_name: str, *names: str) -> tuple[type[BaseException], ...]:
    """The named exception classes of a library, or none when it is not importable."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return ()
    found = []
    for name in names:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


def unavailable_reason(exc: BaseException) -> AvailabilityReason | None:
    """Classify an exception into an availability reason.

    ==========================================================  ===============
    Exception (itself or one it was raised from)                Reason
    ==========================================================  ===============
    :class:`ModuleUnavailable`                                  its ``reason``
    :class:`ModuleConfigError`, ``EmbeddingDimensionError``     ``config``
    psycopg ``UndefinedTable`` / ``UndefinedColumn``            ``config``
    psycopg ``QueryCanceled``, psycopg_pool ``PoolTimeout``     ``unreachable``
    anything :func:`~osprey.models.providers.health.failure_reason`
    classifies (connections, HTTP 401/403/404)                  its reason
    anything else                                               None
    ==========================================================  ===============

    Classes are matched, never message text.

    Args:
        exc: The exception to classify.

    Returns:
        The reason, or None when the exception says nothing about availability.
    """
    from osprey.models.providers.base import EmbeddingDimensionError

    config_classes = (ModuleConfigError, EmbeddingDimensionError) + library_exception_classes(
        "psycopg.errors", "UndefinedTable", "UndefinedColumn"
    )
    unreachable_classes = library_exception_classes(
        "psycopg.errors", "QueryCanceled"
    ) + library_exception_classes("psycopg_pool", "PoolTimeout")
    for item in exception_chain(exc):
        if isinstance(item, ModuleUnavailable):
            return item.reason  # type: ignore[return-value]
        provider_reason = failure_reason(item)
        if provider_reason is not None:
            return provider_reason
        if isinstance(item, config_classes):
            return "config"
        if unreachable_classes and isinstance(item, unreachable_classes):
            return "unreachable"
    return None


def fix_for(module: str, reason: str, exc: BaseException | None = None) -> str:
    """Return the configuration key or command that fixes ``reason`` for ``module``.

    Args:
        module: Module name.
        reason: Availability reason.
        exc: The exception behind it, when one carries its own fix.

    Returns:
        The key or command to name in the log line.
    """
    for item in exception_chain(exc) if exc is not None else ():
        if isinstance(item, ModuleConfigError):
            return item.key
        if isinstance(item, ModuleUnavailable) and item.fix:
            return item.fix
    block = f"ariel.enhancement_modules.{module}"
    return {
        "unreachable": f"start the model server named by {block}.provider",
        "auth": f"the API key of the provider named by {block}.provider",
        "model": f"{block}.model (a model the server serves that accepts pictures)",
        "config": f"{block}, then {_MIGRATE}",
    }.get(reason, block)


def report_unavailable(module: str, reason: str, message: str = "", fix: str | None = None) -> None:
    """Record that ``module`` is unavailable for ``reason``.

    Logs one WARNING when the reason first appears or changes, DEBUG on repeats.

    Args:
        module: Module name.
        reason: Why the module cannot run.
        message: Human-readable detail.
        fix: Key or command that fixes it; derived from ``reason`` when omitted.
    """
    previous = _state.get(module)
    _state[module] = reason
    detail = f": {message}" if message else ""
    if previous == reason:
        logger.debug(f"{module}: still unavailable ({reason}); skipped this pass")
        return
    logger.warning(
        f"{module}: unavailable ({reason}){detail}; skipped until fixed "
        f"(fix: {fix or fix_for(module, reason)})"
    )


def report_available(module: str) -> None:
    """Record that a pass of ``module`` finished with no unavailability.

    Logs one INFO when the module was unavailable before.

    Args:
        module: Module name.
    """
    previous = _state.get(module)
    _state[module] = None
    if previous is not None:
        logger.info(f"{module}: available again")


def current_reason(module: str) -> str | None:
    """Return the reason ``module`` was last recorded unavailable for, or None."""
    return _state.get(module)


def note_success(module: str, marker: str) -> None:
    """Record one stored success of ``module`` under ``marker`` in this process."""
    _successes.add((module, marker))


def has_success(module: str, marker: str) -> bool:
    """Return whether ``module`` stored a success under ``marker`` in this process."""
    return (module, marker) in _successes


def reset_availability() -> None:
    """Forget every recorded availability, success and relation. For tests."""
    _state.clear()
    _successes.clear()
    _relations.clear()


async def _relation_exists(pool: Any, relation: str) -> bool:
    """Return whether ``relation`` exists in ``pool``'s database; positive answers are cached."""
    try:
        cached = _relations.get(pool)
    except TypeError:  # a pool that cannot be weakly referenced is never cached
        cached = None
    if cached is not None and relation in cached:
        return True
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT to_regclass(%(t)s)", {"t": relation})
        row = await cursor.fetchone()
    if row is None:
        value = None
    elif isinstance(row, dict):
        value = next(iter(row.values()), None)
    else:
        value = row[0]
    if value is None:
        return False
    try:
        _relations.setdefault(pool, set()).add(relation)
    except TypeError:
        pass
    return True


async def preflight(module: BaseEnhancementModule, repository: ARIELRepository) -> HealthResult:
    """The one pre-pass check of a picture module.

    In order: the attachment copy state exists, every relation the module
    requires exists, then the module's ``health_check()`` answers within
    :data:`HEALTH_TIMEOUT_S`. The first failure is the verdict; the schema and
    relation checks issue no statement beyond their own lookups.

    Args:
        module: The module to check.
        repository: Repository of the database the module works on.

    Returns:
        The verdict; ``reason`` is set whenever ``reachable`` is False.
    """
    if not (await repository.schema_facts()).has_copy_state:
        return HealthResult(False, f"attachment copy state not migrated: {_MIGRATE}", "config")
    for relation in module.required_relations():
        if not await _relation_exists(repository.pool, relation):
            return HealthResult(
                False,
                f"{relation} missing: pgvector unavailable or the migration was skipped: "
                f"{_MIGRATE}",
                "config",
            )
    try:
        result = as_health_result(await asyncio.wait_for(module.health_check(), HEALTH_TIMEOUT_S))
    except TimeoutError:
        return HealthResult(
            False, f"health check gave no answer within {HEALTH_TIMEOUT_S:g} s", "unreachable"
        )
    except Exception as exc:
        return HealthResult(
            False, f"{type(exc).__name__}: {exc}", unavailable_reason(exc) or "unreachable"
        )
    if result.reachable is False and result.reason is None:
        return HealthResult(False, result.message, "unreachable")
    return result
