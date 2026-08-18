"""Coverage for `PlanSpec.render`: the optional, typed
``render(window, params) -> Figure`` a plan may declare, and the loader rule
that decides whether it is honored.

Two halves, because the field has two contracts:

- **Runtime** — `plan_loader._resolve_render`'s resolution rule. It is total:
  every failure mode (no ``render``, a non-callable one, an attribute access
  that raises, an untrusted tier, a render still written against the old row
  list) yields ``render=None`` and the plan still registers. A drawing concern
  must never cost an operator a launchable plan.
- **Static** — the field is typed against the plan's *own* ``SchemaT``, so a
  ``render()`` that takes ``dict[str, Any]`` instead of its PARAMS is a mypy
  error rather than a surprise at the first poll tick. Nothing at runtime can
  observe that, so it is pinned by running mypy over a fixture snippet.

The snippet check carries its own vacuity guard (see
`test_the_snippet_check_actually_resolves_planspec`): mypy has
``ignore_missing_imports = true``, so a snippet that cannot resolve ``osprey``
silently degrades `PlanSpec` to ``Any`` and every type assertion below it
passes for the wrong reason. The positive control asserts the revealed type is
the real parameterized class, not ``Any``.

All plan files here are pure pydantic/stdlib — no bluesky import — so this
suite runs in the bluesky-less lane alongside `test_plan_loader_layered.py`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

import osprey
from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.figure import (
    REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
    Figure,
    RowWindow,
)
from osprey.services.bluesky_bridge.plan_types import PlanSpec, Provenance
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records

_LOGGER_NAME = "osprey.services.bluesky_bridge.plan_loader"

# `osprey.__file__` is the running interpreter's answer, so these stay correct
# inside a worktree (where a repo-relative path would point at the main checkout).
_SRC_ROOT = Path(osprey.__file__).parents[1]
_REPO_ROOT = _SRC_ROOT.parent

# The trusted tiers, the ones whose `render()` the bridge is willing to run.
_TRUSTED_TIERS: tuple[Provenance, ...] = ("shipped", "preset", "facility")


@pytest.fixture(autouse=True)
def _isolated_plan_loader(monkeypatch: pytest.MonkeyPatch):
    """Clean loader caches and no leftover layer env vars, per test."""
    monkeypatch.delenv("BLUESKY_PLAN_DIRS", raising=False)
    monkeypatch.delenv("BLUESKY_PLAN_MODULE", raising=False)
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


@pytest.fixture
def _isolated_validation_records():
    """Swap in an empty record store so a session-tier fixture can be validated
    without leaking its hash into whatever the rest of the suite recorded."""
    with validation_records.lock:
        saved = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()
    yield validation_records
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved)


# ==========================================================================
# Plan-file fixtures
# ==========================================================================


def _plan_source(name: str, *, tail: str = "") -> str:
    """A minimal, well-formed directory-layer plan file, plus optional ``tail``."""
    return (
        "from osprey.services.bluesky_bridge.figure import Figure\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A render-contract test plan.",\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        f'    return {{"plan": {name!r}}}\n'
        f"{tail}"
    )


_GOOD_RENDER = '\n\ndef render(window, params):\n    return Figure(partial=False, source="live")\n'

_NON_CALLABLE_RENDER = "\n\nrender = 42\n"

_VAR_POSITIONAL_RENDER = (
    '\n\ndef render(*args, **kwargs):\n    return Figure(partial=False, source="live")\n'
)

_RAISING_RENDER = (
    "\n\n"
    "def __getattr__(name):\n"
    '    if name == "render":\n'
    '        raise RuntimeError("render attribute exploded")\n'
    "    raise AttributeError(name)\n"
)


def _write_plan(directory: Path, name: str, *, tail: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(_plan_source(name, tail=tail))
    return path


def _load_one(path: Path, provenance: Provenance) -> dict[str, tuple[Provenance, str, PlanSpec]]:
    registry: dict[str, tuple[Provenance, str, PlanSpec]] = {}
    plan_loader._load_plan_file(path, provenance, registry)
    return registry


def _spec(registry: dict[str, tuple[Provenance, str, PlanSpec]], name: str) -> PlanSpec:
    assert name in registry, f"{name!r} was not registered: {sorted(registry)}"
    return registry[name][2]


def _empty_window() -> RowWindow:
    """A window over a run that has produced nothing — complete, and vacuously so."""
    return RowWindow(rows=[], columns=[], rows_complete=True, total_seen=0)


# ==========================================================================
# The field itself
# ==========================================================================


class _DemoParams(BaseModel):
    amplitude: float = 1.0


def test_render_defaults_to_none() -> None:
    """A plan that says nothing about drawing gets no render — not a stub."""
    spec = PlanSpec(name="demo", plan=lambda devices, params: None, schema=_DemoParams)
    assert spec.render is None


def test_to_dict_does_not_serialize_render() -> None:
    """`GET /plans` is unchanged by this field. Callables do not serialize, and a
    plan's *view* is served by the figure route, not by the catalog."""
    spec = PlanSpec(
        name="demo",
        plan=lambda devices, params: None,
        schema=_DemoParams,
        render=lambda window, params: Figure(partial=False, source="live"),
    )
    assert set(spec.to_dict()) == {
        "name",
        "description",
        "schema",
        "metadata",
        "provenance",
    }


# ==========================================================================
# Loader resolution — trusted tiers
# ==========================================================================


@pytest.mark.parametrize("provenance", _TRUSTED_TIERS)
def test_trusted_tier_render_is_resolved_onto_the_spec(
    tmp_path: Path, provenance: Provenance
) -> None:
    """Every operator-supplied tier gets its `render()`, not just the shipped one."""
    path = _write_plan(tmp_path, "drawable", tail=_GOOD_RENDER)

    spec = _spec(_load_one(path, provenance), "drawable")

    assert spec.render is not None
    figure = spec.render(_empty_window(), _DemoParams())
    assert isinstance(figure, Figure)
    assert figure.source == "live"


def test_plan_without_render_loads_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Most plans have no view of their own. That is ordinary, not a defect —
    warning about it would train operators to ignore the log."""
    path = _write_plan(tmp_path, "plain")

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        spec = _spec(_load_one(path, "shipped"), "plain")

    assert spec.render is None
    assert "render" not in caplog.text


def test_non_callable_render_warns_and_still_registers_the_plan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stray ``render = 42`` reads as an author mistake, so it warns — but the
    plan stays launchable. Quarantining it would take a working plan away over
    a drawing concern."""
    path = _write_plan(tmp_path, "botched", tail=_NON_CALLABLE_RENDER)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        registry = _load_one(path, "shipped")

    spec = _spec(registry, "botched")
    assert spec.render is None
    assert "non-callable render" in caplog.text
    assert "quarantining" not in caplog.text


def test_render_attribute_that_raises_warns_and_still_registers_the_plan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A module ``__getattr__`` (PEP 562) can raise from the attribute lookup
    itself. That is still not a reason to lose the plan."""
    path = _write_plan(tmp_path, "explosive", tail=_RAISING_RENDER)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        registry = _load_one(path, "shipped")

    spec = _spec(registry, "explosive")
    assert spec.render is None
    assert "could not read render" in caplog.text
    assert "render attribute exploded" in caplog.text
    assert "quarantining" not in caplog.text


def test_a_var_positional_render_is_honored(tmp_path: Path) -> None:
    """``*args`` accepts whatever it is handed; any callable render is stored."""
    path = _write_plan(tmp_path, "splat", tail=_VAR_POSITIONAL_RENDER)

    spec = _spec(_load_one(path, "shipped"), "splat")

    assert spec.render is not None
    assert spec.render(_empty_window(), _DemoParams()).source == "live"


# ==========================================================================
# Loader resolution — untrusted tiers
# ==========================================================================


def test_session_tier_render_is_ignored_and_names_the_route_reason(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    _isolated_validation_records,
) -> None:
    """A session plan's `render()` runs nowhere. `render()` executes in the bridge
    API process on every poll tick, so an agent-authored one is an unbounded-work
    seam the validation gate does not close.

    The warning quotes the route's own reason constant so a plan author reads the
    same words the panel will show them, instead of two vocabularies for one rule.
    """
    path = _write_plan(tmp_path, "session_drawable", tail=_GOOD_RENDER)
    validation_records.record(hash_plan_body(path.read_text()))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        registry = _load_one(path, "session")

    spec = _spec(registry, "session_drawable")
    assert spec.render is None
    assert REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS in caplog.text


def test_unreviewed_tier_render_is_ignored_too(
    tmp_path: Path, _isolated_validation_records
) -> None:
    """The gate is on the tier, not on the one tier name that exists today."""
    path = _write_plan(tmp_path, "unreviewed_drawable", tail=_GOOD_RENDER)
    validation_records.record(hash_plan_body(path.read_text()))

    spec = _spec(_load_one(path, "unreviewed"), "unreviewed_drawable")

    assert spec.render is None


def test_session_tier_plan_without_render_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    _isolated_validation_records,
) -> None:
    """Nothing was ignored, so there is nothing to tell the author about."""
    path = _write_plan(tmp_path, "session_plain")
    validation_records.record(hash_plan_body(path.read_text()))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        spec = _spec(_load_one(path, "session"), "session_plain")

    assert spec.render is None
    assert REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS not in caplog.text


# ==========================================================================
# The legacy single-module contract
# ==========================================================================


def test_legacy_module_render_survives_provenance_normalization(tmp_path: Path) -> None:
    """`load_facility_plans` rewrites provenance with `dataclasses.replace`, which
    would drop `render` if the field were ever handled specially there."""
    module = tmp_path / "facility_plans.py"
    module.write_text(
        "from pydantic import BaseModel\n\n"
        "from osprey.services.bluesky_bridge.figure import Figure\n"
        "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n\n\n"
        "class Params(BaseModel):\n"
        "    amplitude: float = 1.0\n\n\n"
        "def render(window, params):\n"
        '    return Figure(partial=True, source="tiled")\n\n\n'
        "PLANS = {\n"
        '    "legacy": PlanSpec(\n'
        '        name="legacy",\n'
        "        plan=lambda devices, params: None,\n"
        "        schema=Params,\n"
        '        provenance="shipped",\n'
        "        render=render,\n"
        "    )\n"
        "}\n\n\n"
        "def get_devices():\n"
        "    return {}\n"
    )

    loaded = plan_loader.load_facility_plans(module_path=str(module))

    spec = loaded.plans["legacy"]
    assert spec.provenance == "facility"
    assert spec.render is not None
    assert spec.render(_empty_window(), spec.schema()).source == "tiled"


# ==========================================================================
# The static contract
# ==========================================================================

_SNIPPET_PREAMBLE = (
    "from __future__ import annotations\n\n"
    "from typing import Any\n\n"
    "from pydantic import BaseModel\n\n"
    "from osprey.services.bluesky_bridge.figure import Figure, RowWindow\n"
    "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n\n\n"
    "class DemoParams(BaseModel):\n"
    "    amplitude: float = 1.0\n\n\n"
    "def build_plan(devices: dict[str, Any], params: DemoParams) -> Any:\n"
    "    return None\n\n\n"
)

_WELL_TYPED_SNIPPET = (
    _SNIPPET_PREAMBLE + "def render(window: RowWindow, params: DemoParams) -> Figure:\n"
    '    return Figure(partial=False, source="live")\n\n\n'
    "SPEC = PlanSpec(name='demo', plan=build_plan, schema=DemoParams, render=render)\n"
    "reveal_type(SPEC)  # noqa: F821\n"
)

_MIS_TYPED_SNIPPET = (
    _SNIPPET_PREAMBLE + "def render(window: RowWindow, params: dict[str, Any]) -> Figure:\n"
    '    return Figure(partial=False, source="live")\n\n\n'
    "SPEC = PlanSpec(name='demo', plan=build_plan, schema=DemoParams, render=render)\n"
)

_ROW_LIST_SNIPPET = (
    _SNIPPET_PREAMBLE + "def render(rows: list[dict[str, Any]], params: DemoParams) -> Figure:\n"
    '    return Figure(partial=False, source="live")\n\n\n'
    "SPEC = PlanSpec(name='demo', plan=build_plan, schema=DemoParams, render=render)\n"
)


@pytest.fixture(scope="module")
def mypy_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Both snippets checked in one mypy pass — the process start dominates the cost.

    ``MYPYPATH`` is set explicitly: the snippets live outside the source tree, and
    with ``ignore_missing_imports = true`` an unresolved ``osprey`` would make
    `PlanSpec` ``Any`` and every assertion here vacuous.
    """
    snippet_dir = tmp_path_factory.mktemp("render_typing")
    (snippet_dir / "well_typed.py").write_text(_WELL_TYPED_SNIPPET)
    (snippet_dir / "mis_typed.py").write_text(_MIS_TYPED_SNIPPET)
    (snippet_dir / "row_list.py").write_text(_ROW_LIST_SNIPPET)

    env = dict(os.environ)
    env["MYPYPATH"] = str(_SRC_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(_REPO_ROOT / "pyproject.toml"),
            "--no-incremental",
            str(snippet_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    assert "No module named mypy" not in result.stderr, (
        "mypy is not installed in this environment, so the static half of this "
        f"contract cannot be checked at all:\n{result.stderr}"
    )
    return result.stdout + result.stderr


def test_the_snippet_check_actually_resolves_planspec(mypy_output: str) -> None:
    """The vacuity guard, and the reason this test exists at all.

    Without `MYPYPATH`, mypy resolves nothing and reports ``Revealed type is
    "Any"`` — under which the mis-typed snippet below type-checks clean and the
    whole static contract silently stops being enforced.
    """
    assert "PlanSpec[well_typed.DemoParams]" in mypy_output, (
        "mypy did not resolve PlanSpec to the real parameterized class — every "
        f"static assertion in this module would be vacuous:\n{mypy_output}"
    )


def test_a_render_typed_against_its_own_params_type_checks(mypy_output: str) -> None:
    """The positive control: the contract is satisfiable, not merely strict."""
    offending = [
        line for line in mypy_output.splitlines() if "well_typed.py" in line and "error:" in line
    ]
    assert not offending, "\n".join(offending)


def test_a_render_taking_dict_instead_of_its_params_fails_mypy(mypy_output: str) -> None:
    """The point of typing the field: `render()` is handed a validated PARAMS
    instance, and a plan that annotates it ``dict[str, Any]`` would fail at the
    first poll tick of a live run. mypy is where that gets caught."""
    offending = [
        line
        for line in mypy_output.splitlines()
        if "mis_typed.py" in line and "error:" in line and 'Argument "render"' in line
    ]
    assert offending, (
        "a render() annotated dict[str, Any] instead of its PARAMS type-checked "
        f"clean — the field's SchemaT parameterization is not biting:\n{mypy_output}"
    )


def test_a_render_taking_a_row_list_instead_of_a_window_fails_mypy(mypy_output: str) -> None:
    """The static half of the row-list refusal. `plan_loader._resolve_render`
    catches this at load time by inspecting the signature, but that costs the
    operator the figure; mypy catches it while the plan is still being written,
    which is where a plan author would rather hear it."""
    offending = [
        line
        for line in mypy_output.splitlines()
        if "row_list.py" in line and "error:" in line and 'Argument "render"' in line
    ]
    assert offending, (
        "a render() still taking list[dict[str, Any]] type-checked clean against "
        f"PlanSpec.render — the RowWindow cutover is not pinned statically:\n{mypy_output}"
    )
