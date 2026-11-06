"""Core ``service_capabilities`` health category.

Reads the capability surfaces three services already publish and that no other
health row reads. Each part contributes rows only when its subject is
configured; with none configured the category contributes no rows.

**Bluesky bridge** — present when ``services.bluesky`` or ``bluesky.bridge_url``
is configured. The bridge is dialed where its clients dial it
(``BLUESKY_BRIDGE_URL``, else :func:`osprey.bluesky_bridge_connection.bridge_url_from_config`).

* ``bluesky.capability`` — ``GET /health``'s ``capability`` record: ``ok`` when
  plans can execute (with the lane's target) or when the deployment is browse
  only by design (``browse_only_connector``); ``warning`` for every other
  reason the bridge cannot execute, and when it does not answer.

**OKF bundle** — present when ``facility_knowledge.bundle_path`` is set; the
path is resolved the way the knowledge panel resolves it
(:func:`osprey.services.facility_knowledge.bundle_path.resolve_bundle_path`,
against ``cwd``, the render directory).

* ``okf.bundle`` — the concept count the bundle lists and a digest of its
  Markdown files; when the render's build manifest tracks those files, whether
  they still match what the build wrote (``warning`` with ``osprey build`` as
  the fix when not). ``warning`` when the bundle directory is missing.
* ``okf.bundle_age`` — age of the newest Markdown file in the bundle (``ok``).

**ARIEL enhancement modules** — present when ``ariel.enhancement_modules``
enables at least one module. Each enabled module is built alone and its own
``health_check()`` awaited (5 s bound, all concurrently), the check ``osprey
ariel status`` runs:

* ``ariel.enhancement.<module>`` — ``ok`` when reachable, ``warning`` with the
  module's reason when not, ``skip`` when the module has no check of its own.
  A module that cannot be built is a ``warning`` ("config error").

Every row is advisory: a capability that is missing is a service that is
degraded, not a broken build.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "service_capabilities"

#: Per-request / per-module bound in seconds.
_TIMEOUT_S = 5.0

#: Capability reasons that are a deployment's choice rather than a fault.
_BY_DESIGN = frozenset({"executable", "browse_only_connector"})

_FIX = "osprey build"
_DIGEST_LEN = 8

#: Builds the enabled modules from the config mapping; injectable for tests.
ModuleFactory = Callable[[Mapping[str, Any]], list[Any]]


def service_capabilities(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    cwd: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: Callable[[], float] | None = None,
    ariel_modules: ModuleFactory | None = None,
) -> CategoryCallable:
    """Build the ``service_capabilities`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Health runtime. Unused.
        cwd: The render directory, against which a relative
            ``facility_knowledge.bundle_path`` resolves and whose build
            manifest the bundle is compared with. ``None`` resolves the bundle
            path the way the knowledge panel does without one.
        transport: Optional httpx transport, for tests.
        now: Epoch-seconds clock, for tests; defaults to :func:`time.time`.
        ariel_modules: Optional factory of the enabled ARIEL enhancement
            modules, for tests; ``None`` builds them from the config.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}
    clock = now or time.time
    build_modules = ariel_modules or _build_ariel_modules

    async def _run() -> list[CheckResult]:
        parts = await asyncio.gather(
            _bluesky_rows(cfg, transport),
            asyncio.to_thread(_okf_rows, cfg, cwd, clock()),
            _ariel_rows(cfg, build_modules),
            return_exceptions=True,
        )
        rows: list[CheckResult] = []
        for part, label in zip(parts, ("bluesky", "okf", "ariel.enhancement"), strict=True):
            if isinstance(part, BaseException):
                # Isolation: one part's surprise never hides the others.
                rows.append(
                    CheckResult(
                        label,
                        CATEGORY,
                        Status.WARNING,
                        f"{label} capability check raised {type(part).__name__}",
                        value="check failed",
                        details=str(part),
                    )
                )
            else:
                rows.extend(part)
        return rows

    return _run


# -- Bluesky bridge ---------------------------------------------------------------


def _bridge_url(cfg: Mapping[str, Any]) -> str | None:
    from osprey.bluesky_bridge_connection import BRIDGE_URL_ENV_VAR, bridge_url_from_config

    services = cfg.get("services")
    bluesky = cfg.get("bluesky")
    configured = (
        isinstance(services, Mapping) and isinstance(services.get("bluesky"), Mapping)
    ) or (isinstance(bluesky, Mapping) and bool(bluesky.get("bridge_url")))
    if not configured:
        return None
    override = os.environ.get(BRIDGE_URL_ENV_VAR)
    return (override or bridge_url_from_config(cfg)).rstrip("/")


async def _bluesky_rows(
    cfg: Mapping[str, Any], transport: httpx.AsyncBaseTransport | None
) -> list[CheckResult]:
    url = _bridge_url(cfg)
    if url is None:
        return []
    name = "bluesky.capability"
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, transport=transport) as client:
            response = await client.get(f"{url}/health")
        elapsed = (time.perf_counter() - started) * 1000.0
        capability = response.json().get("capability") if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return [
            CheckResult(
                name,
                CATEGORY,
                Status.WARNING,
                f"Bluesky bridge unreachable at {url}",
                value="unreachable",
                details=str(exc) or type(exc).__name__,
                latency_ms=elapsed,
                probed=True,
            )
        ]
    if not isinstance(capability, Mapping):
        return [
            CheckResult(
                name,
                CATEGORY,
                Status.WARNING,
                f"Bluesky bridge /health carries no capability record (HTTP {response.status_code})",
                value=f"HTTP {response.status_code}",
                latency_ms=elapsed,
                probed=True,
            )
        ]
    reason = str(capability.get("reason") or "unknown")
    detail = str(capability.get("detail") or "")
    if capability.get("can_execute"):
        target = capability.get("lane_target")
        value = f"can execute · {target}" if target else "can execute"
        status = Status.OK
    elif reason in _BY_DESIGN:
        value, status = f"browse only · {reason}", Status.OK
    else:
        value, status = f"cannot execute · {reason}", Status.WARNING
    return [
        CheckResult(
            name,
            CATEGORY,
            status,
            detail or f"Bluesky bridge capability: {reason}",
            value=value,
            latency_ms=elapsed,
            probed=True,
        )
    ]


# -- OKF bundle -------------------------------------------------------------------


def _okf_rows(cfg: Mapping[str, Any], cwd: Path | None, current: float) -> list[CheckResult]:
    block = cfg.get("facility_knowledge")
    raw = block.get("bundle_path") if isinstance(block, Mapping) else None
    if not raw:
        return []
    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path

    root = resolve_bundle_path(str(raw), config_dir=cwd)
    if not root.is_dir():
        return [
            CheckResult(
                "okf.bundle",
                CATEGORY,
                Status.WARNING,
                f"OKF bundle directory {root} does not exist",
                value="missing",
                fix=_FIX,
            )
        ]
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    docs = len(OKFBundle(root).list_concepts())
    files = sorted(p for p in root.rglob("*.md") if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    value = f"{docs} {'doc' if docs == 1 else 'docs'} · digest {digest.hexdigest()[:_DIGEST_LEN]}"

    status, fix, message = Status.OK, "", f"OKF bundle at {root}: {docs} concepts"
    differing = _render_drift(files, cwd)
    if differing is not None:
        if differing:
            noun = "file differs" if differing == 1 else "files differ"
            value += f" · {differing} {noun} from render"
            status, fix = Status.WARNING, _FIX
            message += f"; {differing} Markdown file(s) differ from what the build rendered"
        else:
            value += " = render"
    rows = [CheckResult("okf.bundle", CATEGORY, status, message, value=value, fix=fix)]
    if files:
        newest = max(path.stat().st_mtime for path in files)
        rows.append(
            CheckResult(
                "okf.bundle_age",
                CATEGORY,
                Status.OK,
                "Age of the newest document in the OKF bundle",
                value=_age(current - newest),
            )
        )
    return rows


def _render_drift(files: list[Path], cwd: Path | None) -> int | None:
    """How many bundle files differ from the build manifest, or ``None`` when untracked.

    The manifest keys are relative to the render; the bundle is served from the
    project root, where the same relative path names the file the build copied.
    """
    if cwd is None:
        return None
    from osprey.build.manifest import sha256_file
    from osprey.cli.templates.manifest import load_project_manifest
    from osprey.utils.config_paths import project_root_for_config_dir

    manifest = load_project_manifest(Path(cwd))
    checksums = manifest.get("file_checksums") if manifest else None
    if not isinstance(checksums, dict):
        return None
    project_root = project_root_for_config_dir(Path(cwd)).resolve()
    tracked = 0
    differing = 0
    for path in files:
        try:
            key = str(path.resolve().relative_to(project_root))
        except ValueError:
            continue
        expected = checksums.get(key)
        if expected is None:
            continue
        tracked += 1
        if f"sha256:{sha256_file(path)}" != expected:
            differing += 1
    return differing if tracked else None


def _age(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 3600:
        return f"{seconds / 60:.0f} m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


# -- ARIEL enhancement modules ----------------------------------------------------


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
