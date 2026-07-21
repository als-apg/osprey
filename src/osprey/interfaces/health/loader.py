"""Synchronous config-load phase of the health web view's refresh cycle.

The web health surface re-runs the same suite the ``osprey health`` CLI runs,
but from a long-lived process that must pick up edits to ``config.yml`` / ``.env``
without a restart and without entangling itself with the CLI's process-global
config singleton. This module owns the *synchronous* half of one refresh cycle:
resolve the config path, cheaply skip work when nothing on disk changed (an
mtime/size gate), reload ``.env`` only when it actually changed, load and parse
``config.yml`` through a private :class:`~osprey.utils.config.ConfigBuilder`
(never the shared ``get_config_builder`` singleton), and assemble the merged
category records via :mod:`osprey.health.records`.

Divergences from the CLI's one-shot load
(:func:`osprey.health.records.load_config`), each required by the
persistent-process setting:

* **No singleton.** Constructs ``ConfigBuilder(..., load_env=False)`` directly so
  an in-place config edit is observed on the next cycle and the view neither
  mutates nor is perturbed by the CLI's cached default config.
* **Explicit ``.env`` control.** ``load_env=False`` keeps the builder from
  touching ``os.environ``; this module instead calls ``load_dotenv`` against the
  project ``.env`` only on the first cycle or when that file changed, so an
  unchanged ``.env`` never re-mutates the process environment.
* **Always-usable settings.** A missing or broken config degrades to the default
  :class:`~osprey.health.config.HealthSettings` (``suite_timeout_s=30``,
  ``interval_s=60``) instead of ``None``, so the refresh scheduler always has an
  interval to sleep on. Record assembly stays identical to the CLI's degraded
  path — the default settings carry no overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from osprey.health.config import CategoryRecord, HealthSettings
    from osprey.health.core.configuration import ConfigState


class LoadedHealthConfig(NamedTuple):
    """One refresh cycle's resolved health inputs.

    Field order matches the loader contract so callers may unpack positionally.
    ``settings`` is never ``None`` — a degraded load yields default settings — so
    a caller can always read ``interval_s`` / ``suite_timeout_s``.
    """

    records: list[CategoryRecord]
    extra_rows: list[Any]
    settings: HealthSettings
    expanded: dict[str, Any] | None
    control_system: dict[str, Any]
    config_ok: bool


def _stat_signature(path: Path) -> tuple[int, int] | None:
    """Return a change-detecting ``(mtime_ns, size)`` signature, or ``None``.

    ``None`` means the file is absent. Comparing signatures across cycles
    detects a content edit (mtime and/or size change) as well as the appearance
    or disappearance of the file; pairing size with mtime catches an edit that a
    coarse filesystem clock would leave at the same timestamp.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _load_project_env(env_path: Path) -> None:
    """Load *env_path* into ``os.environ`` with override semantics (best-effort).

    A missing file or a missing ``python-dotenv`` is silently ignored, matching
    the CLI's ``.env`` handling.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if env_path.exists():
        load_dotenv(env_path, override=True)


def _load_config(
    config_path: Path,
) -> tuple[ConfigState, dict[str, Any] | None, HealthSettings | None, bool]:
    """Load and parse ``config.yml`` through a private ``ConfigBuilder``.

    Reuses :func:`osprey.health.records._load_config_result` — the single owner
    of the degradation-and-parse contract (a missing file, bad YAML, empty
    document, or invalid ``health:`` section all degrade to ``config_ok=False``
    and never raise) — but supplies a fresh ``ConfigBuilder(load_env=False)``
    instead of the shared singleton so the long-lived view observes edits and
    performs no ``.env`` side effect here.
    """
    from osprey.health.records import _load_config_result

    def _load() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        from osprey.utils.config import ConfigBuilder

        builder = ConfigBuilder(str(config_path), load_env=False)
        return builder.raw_config, builder.get_unexpanded_config()

    return _load_config_result(config_path, config_path.parent, _load)


class HealthConfigLoader:
    """Stateful synchronous loader for the health view's refresh cycle.

    A single instance is reused across refresh cycles. Each :meth:`load` call
    resolves the config path and returns its cached result unchanged when neither
    ``config.yml`` nor ``.env`` moved since the last cycle — so an idle view
    neither re-reads YAML nor re-mutates ``os.environ``.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Args:
            config_path: Explicit ``config.yml`` path. ``None`` (the default)
                resolves per :func:`osprey.utils.workspace.resolve_config_path`
                on every cycle (``OSPREY_CONFIG`` env, else ``./config.yml``), so
                resolution tracks the process cwd/env like the CLI does.
        """
        self._config_path_override = config_path
        self._config_sig: tuple[int, int] | None = None
        self._env_sig: tuple[int, int] | None = None
        self._cached: LoadedHealthConfig | None = None

    def load(self) -> LoadedHealthConfig:
        """Run one synchronous refresh phase and return the resolved inputs."""
        config_path = self._resolve_path()
        env_path = config_path.parent / ".env"

        config_sig = _stat_signature(config_path)
        env_sig = _stat_signature(env_path)

        first_run = self._cached is None
        env_changed = first_run or env_sig != self._env_sig
        config_changed = first_run or config_sig != self._config_sig

        if not first_run and not env_changed and not config_changed:
            # Nothing on disk moved: reuse the last cycle's records verbatim and,
            # crucially, do not touch os.environ.
            assert self._cached is not None  # narrowed by ``first_run`` above
            return self._cached

        # A changed ``.env`` must precede builder construction so ``${VAR}``
        # placeholders expand against the fresh environment.
        if env_changed:
            _load_project_env(env_path)

        result = self._build(config_path)

        self._config_sig = config_sig
        self._env_sig = env_sig
        self._cached = result
        return result

    def _resolve_path(self) -> Path:
        if self._config_path_override is not None:
            return self._config_path_override
        from osprey.utils.workspace import resolve_config_path

        return resolve_config_path()

    def _build(self, config_path: Path) -> LoadedHealthConfig:
        from osprey.health.config import parse_health_config
        from osprey.health.records import build_records

        config_state, expanded, settings, config_ok = _load_config(config_path)

        # A degraded load has no settings; fall back to framework defaults
        # (suite_timeout_s=30, interval_s=60) so downstream always has a usable
        # cadence. build_records with config_ok=False ignores overrides, so this
        # leaves record assembly identical to the CLI's degraded path.
        if settings is None:
            settings = parse_health_config(None)

        project_path = config_path.parent
        records, extra_rows = build_records(
            config_state,
            expanded,
            settings,
            config_ok,
            project_path,
            settings.suite_timeout_s,
        )
        control_system = (expanded or {}).get("control_system", {}) or {}
        return LoadedHealthConfig(
            records=records,
            extra_rows=extra_rows,
            settings=settings,
            expanded=expanded,
            control_system=control_system,
            config_ok=config_ok,
        )
