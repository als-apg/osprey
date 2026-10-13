"""Shared helpers for loading environments and reading/writing variables.

Extracted from ``routers/environments.py`` so multiple routers (environments,
run-config, results/restore) share one machine-access path and error mapping.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from osprey.services.optimizer.api.job_tracker import JobTracker

from osprey.services.optimizer.models.api import EnvironmentDetails, EnvironmentStatusResponse
from osprey.services.optimizer.models.exceptions import ConnectionError as ALSConnectionError
from osprey.services.optimizer.models.exceptions import TimeoutError as ALSTimeoutError
from osprey.services.optimizer.models.exceptions import VariableOutOfBoundsError
from osprey.services.optimizer.models.protocols import EnvironmentProtocol
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry

logger = logging.getLogger(__name__)


def require_known_environment(name: str, registry: EnvironmentRegistry) -> None:
    """422 unless ``name`` is a discoverable environment."""
    known = {e.name for e in registry.list_environments()}
    if name not in known:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown environment '{name}'.",
        )


def environment_details(name: str, registry: EnvironmentRegistry) -> EnvironmentDetails:
    """Assemble an environment's metadata — the single details builder.

    Shared by the environments router and the current-environment reads so the
    field set can't drift between endpoints.
    """
    env = load_environment(name, registry)
    with env:
        return EnvironmentDetails.from_environment(name, env)


def probe_environment_health(name: str, registry: EnvironmentRegistry) -> EnvironmentStatusResponse:
    """Probe an environment's connectivity — the single health probe.

    404 for an unknown name; any other probe failure degrades to
    ``connected=None`` with the error message rather than a 5xx.
    """
    try:
        env = lookup_environment(name, registry)
        with env:
            connected = env.check_connection()
        return EnvironmentStatusResponse(name=name, connected=connected, message=None)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Connectivity check failed for '%s': %s", name, e)
        return EnvironmentStatusResponse(name=name, connected=None, message=str(e))


def is_sim_environment(env: str) -> bool:
    """True for simulator environments (no real-hardware safety gating).

    Name-based sim classification is the contract the reliability spec assumes:
    it gates restore-token safety and the health probe.
    """
    return env.endswith("_sim")


def _environment_connected(name: str, registry: EnvironmentRegistry) -> bool:
    return probe_environment_health(name, registry).connected is True


def _assert_selected(name: str, tracker: "JobTracker") -> None:
    if name != tracker.get_current_environment():  # fail-closed: unset -> reject
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=f"Environment '{name}' is not the current environment.",
        )


def _assert_healthy(name: str, registry: EnvironmentRegistry, override_unhealthy: bool) -> None:
    if is_sim_environment(name) or override_unhealthy:
        return
    if not _environment_connected(name, registry):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Environment '{name}' is not reachable; pass override_unhealthy to proceed.",
        )


def guard_environment_write(
    name: str, tracker: "JobTracker", registry: EnvironmentRegistry, *, override_unhealthy: bool
) -> None:
    """Point-in-time precondition for a manual hardware write.

    Requires no active job, name==current, and healthy-or-override. NOT a lock
    held across the EPICS write — a job may claim the slot after this returns;
    that residual same-env race is the accepted best-effort behavior (a write
    only ever targets its own name).
    """
    if tracker.get_active_job() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A job is active; manual writes are locked.",
        )
    _assert_selected(name, tracker)
    _assert_healthy(name, registry, override_unhealthy)


def guard_environment_start(
    name: str, tracker: "JobTracker", registry: EnvironmentRegistry, *, override_unhealthy: bool
) -> None:
    """Point-in-time precondition for a job start: name==current and healthy-or-override.

    The active-slot claim (claim_active_job) enforces the single-running-job
    rule, so this does not check the active-job lock.
    """
    # Claim-window subtlety: a concurrent switch between this guard and the active-slot
    # claim can leave current != running job.env; this is benign because the job drives
    # its own immutable config env and manual writes are 409-locked during the job.
    _assert_selected(name, tracker)
    _assert_healthy(name, registry, override_unhealthy)


def lookup_environment(
    name: str, registry: EnvironmentRegistry, env_config: dict[str, Any] | None = None
) -> EnvironmentProtocol:
    """Resolve an environment by name, mapping only the unknown-name case to 404.

    Use this when the caller wants to handle other load failures itself (e.g.
    ``/status`` degrades them to ``connected=False`` rather than a 500).
    """
    try:
        return registry.load_environment(name, env_config or {})
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment '{name}' not found",
        ) from e


def load_environment(
    name: str, registry: EnvironmentRegistry, env_config: dict[str, Any] | None = None
) -> EnvironmentProtocol:
    """Load an environment by name, mapping failures to HTTP errors."""
    try:
        return lookup_environment(name, registry, env_config)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load environment '%s'", name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load environment '{name}': {e}",
        ) from e


@contextmanager
def read_errors(name: str, action: str) -> Iterator[None]:
    """Map control-system read failures to HTTP errors.

    ``action`` names the read (e.g. ``"get variables"``) and is interpolated
    into the 500 detail. The 503 path is the catastrophic case where the CA
    worker thread never answered (wedged or never started).

    Wrap only the control-system call; keep name validation before the block
    and value transforms after it so their errors are not misclassified.
    """
    try:
        yield
    except ALSTimeoutError as e:
        logger.exception("CA worker unresponsive while reading from '%s'", name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Control system worker unresponsive: {e}",
        ) from e
    except Exception as e:
        logger.exception("Failed to %s from '%s'", action, name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {action}: {e}",
        ) from e


@contextmanager
def write_errors(name: str, action: str) -> Iterator[None]:
    """Map control-system write failures to HTTP errors.

    ``action`` names the write (e.g. ``"set variables"``) and is interpolated
    into the 500 detail. An out-of-bounds/unknown/non-finite value is a client
    error (422); an unreachable or unresponsive control system is a 503.

    Wrap only the control-system call; keep name validation before the block
    and value transforms after it so their errors are not misclassified.
    """
    try:
        yield
    except VariableOutOfBoundsError as e:
        # Unknown variable, out-of-bounds, or non-finite value: client error.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
    except (ALSConnectionError, ALSTimeoutError) as e:
        logger.exception("Control system unreachable while trying to %s on '%s'", action, name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Control system unavailable: {e}",
        ) from e
    except Exception as e:
        logger.exception("Failed to %s on '%s'", action, name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {action}: {e}",
        ) from e


def validate_observable_names(names: list[str], env: EnvironmentProtocol) -> None:
    """Raise 422 if any observable name is not in the environment metadata."""
    unknown = [n for n in names if n not in env.observables]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown observable(s): {unknown}",
        )


def validate_variable_names(
    names: list[str],
    env: EnvironmentProtocol,
    field: str = "variable",
) -> None:
    """Raise 422 if any name is not in the environment metadata.

    ``field`` labels the offending input so callers passing different fields
    (e.g. ``overrides`` vs ``current_values``) get distinguishable messages.
    """
    unknown = [n for n in names if n not in env.variables]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown {field}(s): {unknown}",
        )


def apply_variables_to_environment(
    env: EnvironmentProtocol, name: str, variables: dict[str, float]
) -> None:
    """Write variables to the machine, mapping control errors to HTTP errors.

    Owns the environment context: pass a freshly-loaded (not already-open)
    ``env`` from :func:`load_environment`; this opens and closes it.
    """
    with write_errors(name, "set variables"), env:
        env.set_variables(variables)
