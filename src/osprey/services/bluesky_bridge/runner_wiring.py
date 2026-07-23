"""Deploy-time configuration resolution for the bridge's runner factory.

The stateless half of the runner-wiring seam: read the opt-in env flags, the
project's ``control_system.type``, and the Tiled env, and turn them into the
inputs ``app.py``'s ``_lifespan`` uses to pick and build a ``PlanRunner``. The
mutable process state (``_runner_factory``, ``_connector``) and the orchestration
that mutates it (``_lifespan``, ``_wire_epics_substrate_runner``,
``set_runner_factory``) stay in ``app.py``; everything here is pure lookups with
no module-level side effects.

Import-clean of the bluesky stack (``_BRIDGE_ONLY_MODULES``): the only bluesky/
tiled import lives lazily inside the closure ``_build_tiled_writer_factory``
returns, so importing this module — and therefore ``app.py`` — never pulls in
``bluesky``/``ophyd``/``tiled``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Root package names the demo-runner lifespan hook (in `app.py`) is allowed to
# fail on — i.e. the bridge running without the bluesky stack importable. The
# stack is a core dependency, so this normally succeeds; the guard stays for
# slimmed images or partial installs. An ImportError naming anything else (e.g. a
# module missing an expected attribute, or an unrelated third-party import broke)
# is a genuine bug and must not be swallowed as "bluesky is just absent".
_BRIDGE_ONLY_MODULES = {"bluesky", "ophyd", "ophyd_async", "tiled"}

# Opt-in flag (task 2.14a): when truthy (see `_is_demo_runner_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyPlanRunner` against mock devices — the deploy smoke demo's PlanRunner
# (PLAN.md), never a facility's real-hardware wiring. The deploy template
# only renders this var at all when the demo runner is wanted (house
# convention, matching `container_lifecycle.py`'s `DEV_MODE="true"`), so
# "absent" is the off state — but the check itself accepts a few equivalent
# truthy spellings rather than one exact string, so neither half of this
# seam (this hook vs. whatever template/generator sets the var) can drift
# out of sync with the other again.
_DEMO_RUNNER_ENV = "BLUESKY_DEMO_RUNNER"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Opt-in flag (task 2.3): when truthy (see `_is_epics_substrate_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyPlanRunner` against real EPICS devices — Channel Access clients of
# whatever IOC the deploy points at (a virtual accelerator, or real
# hardware), never mock devices. The PV list comes entirely from
# `BLUESKY_EPICS_MOTORS`/`BLUESKY_EPICS_DETECTORS` (see
# `devices/_specs_from_env.py`), never the VA manifest — this process cannot
# import that. If both this flag and `_DEMO_RUNNER_ENV` are set, this one
# wins (see `_lifespan`): an operator who explicitly asked for real EPICS
# must never silently get the mock demo instead.
_EPICS_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"

# Task 2.5: when set, both runner-factory branches subscribe a
# `TiledWriter` (via `_FaultIsolatedTiledWriter`, see `plan_runner_bluesky.py`) so
# run data survives a bridge restart. Orthogonal to `_DEMO_RUNNER_ENV`/
# `_EPICS_SUBSTRATE_ENV` — it augments whichever runner those two flags pick,
# rather than picking one itself. `BLUESKY_TILED_API_KEY` grants catalog
# access only, never launch authority — see `container_lifecycle.py`'s
# `_SERVICE_TOKEN_VARS`.
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"

# Connector types the EPICS-substrate branch knows how to build a gateway-less
# `type_config` for — real Channel Access, whether against a virtual
# accelerator soft-IOC or live hardware.
_EPICS_LIKE_CONNECTOR_TYPES = ("virtual_accelerator", "epics")


def _is_demo_runner_enabled() -> bool:
    """True if `BLUESKY_DEMO_RUNNER` is set to any of `_TRUTHY_VALUES` (case/whitespace-insensitive).

    Absent, empty, or any other value (e.g. "false") is off — deliberately
    liberal on the "on" spellings, but never guesses at "on" from an
    unrecognized value.
    """
    return os.environ.get(_DEMO_RUNNER_ENV, "").strip().lower() in _TRUTHY_VALUES


def _is_epics_substrate_enabled() -> bool:
    """True if `BLUESKY_EPICS_SUBSTRATE` is set to any of `_TRUTHY_VALUES`.

    Same liberal-on-"on"-spellings parsing as `_is_demo_runner_enabled` —
    see that function's docstring for why.
    """
    return os.environ.get(_EPICS_SUBSTRATE_ENV, "").strip().lower() in _TRUTHY_VALUES


def _resolve_control_system_type() -> str:
    """Read `control_system.type` from the bridge's mounted project config.

    Single source of truth (Connector = the single control-system interface):
    one config line flips the whole Bluesky stack between the mock connector and
    real Channel Access (virtual accelerator or live hardware) — see the
    `control-assistant` preset's `config.control_system.type` comment.

    Fail-SAFE default: `"mock"` whenever the config can't be read at all (no
    project config context — most unit-test environments — or a transient
    lookup failure), never `"virtual_accelerator"`/`"epics"` — the mock
    connector never touches Channel Access, so an unreadable config can never
    silently connect to real hardware. Mirrors
    `_assert_limits_readable_if_writable`'s "no project config context ->
    treat as absent, don't block" handling of the same exception set.
    """
    from osprey.utils.config import get_config_value

    try:
        control_system_type = get_config_value("control_system.type", "mock")
    except (FileNotFoundError, KeyError, RuntimeError):
        return "mock"

    if not control_system_type or not isinstance(control_system_type, str):
        return "mock"
    return control_system_type


def _build_tiled_writer_factory() -> Callable[[], Any] | None:
    """Build the `tiled_writer_factory` `BlueskyPlanRunner` accepts, or `None` if Tiled is unconfigured.

    Reads `BLUESKY_TILED_URI` fresh on every call (never cached), so each
    launch's `BlueskyPlanRunner` picks up the current env — matching
    `do_launch`'s "fresh runner per launch" contract (`plan_runner_bluesky.py`'s
    `_FaultIsolatedTiledWriter` docstring: "no cross-run state to reset").
    `None` when the URI is unset: `BlueskyPlanRunner.__init__` treats that as "no
    Tiled subscription", identical to Phase 1's no-Tiled-server behavior.

    The returned closure imports `TiledWriter` from
    `bluesky.callbacks.tiled_writer` — NOT from `tiled` (TR2) — lazily,
    inside itself, so this module stays import-clean of both `bluesky` and
    `tiled` (`_BRIDGE_ONLY_MODULES`) even when a caller holds onto the
    returned factory without ever invoking it.
    """
    uri = os.environ.get(_TILED_URI_ENV)
    if not uri:
        return None

    def factory() -> Any:
        from bluesky.callbacks.tiled_writer import TiledWriter

        return TiledWriter.from_uri(uri, api_key=os.environ[_TILED_API_KEY_ENV])

    return factory
