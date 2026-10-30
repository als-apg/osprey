"""Every case-table row through every surface that answers "may this write?" (D6).

The case table (``osprey_connectors_tests/_harness/fixtures/write_verdict_cases.json``)
states one write verdict per row. The registry
(:mod:`tests.integration._write_verdict_surfaces_registry`) names every src
surface that asks that verdict, with the projection of a row onto what the
surface returns. This module holds the assertions:

* the registry covers every src module that calls ``write_verdict`` or
  ``backstop_verdict`` (an AST scan), so a new caller left out fails here;
* every row in a surface's domain reads on that surface as the row says —
  write rows on write surfaces, backstop rows on backstop surfaces;
* the switch off with no stamp reads ``switch_off`` on every stage-1 surface;
* the headless invariant: a headless read-only run blocks every framework and
  declared write tool, whatever the switch says.

A disagreement is fixed in src, never by loosening a projection.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import isolate_verdict_environment, reset_verdict_state
from osprey_connectors_tests._harness._write_verdict_cases import CASES, Case, host_root, stage
from tests.integration._write_verdict_surfaces_registry import (
    BY_NAME,
    OUT_OF_DOMAIN,
    SURFACES,
    Staged,
    Surface,
    write_config,
)

REPO = Path(__file__).resolve().parents[2]
SRC_ROOTS = (REPO / "src", REPO / "packages" / "osprey-connectors" / "src")
VERDICT_NAMES = frozenset({"write_verdict", "backstop_verdict"})

#: The modules that define a verdict rather than ask one.
DEFINING_MODULES = frozenset(
    {
        "osprey_connectors.posture_store",
        "osprey.templates.claude_code.claude.hooks.osprey_target_state",
    }
)

#: The row with the switch off, no stamped owner and no launch pin.
SWITCH_OFF_NO_STAMP = "switch-off-no-stamp"


# -- staging -------------------------------------------------------------------


def _project(surface: Surface, case: Case, tmp_path: Path) -> Any:
    """Stage *case* in its own directory and ask *surface*; restore everything after."""
    row_dir = tmp_path / surface.name / case.id
    row_dir.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_verdict_environment(monkeypatch, host_root(row_dir))
        config_path = write_config(row_dir / "config.yml", case.section)
        monkeypatch.setenv("CONFIG_FILE", str(config_path))
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.chdir(row_dir)
        reset_config_cache()
        try:
            stage(case, monkeypatch, row_dir)
            return surface.project(Staged(case, row_dir, monkeypatch, config_path))
        finally:
            reset_verdict_state()
            reset_config_cache()


def _rows_for(surface: Surface) -> list[Case]:
    return [case for case in CASES if case.verdict == surface.verdict]


# -- the registry covers every caller ----------------------------------------


def _module_name(path: Path) -> str:
    for root in SRC_ROOTS:
        if path.is_relative_to(root):
            return ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
    raise AssertionError(path)


def _verdict_callers() -> set[str]:
    callers = set()
    for root in SRC_ROOTS:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            # A call spells the name it calls (an identifier has no escape
            # sequence), and parsing every module is what the scan spends its
            # time on: a module that spells no verdict name is passed over.
            if not any(name in source for name in VERDICT_NAMES):
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in VERDICT_NAMES:
                    callers.add(_module_name(path))
                    break
    return callers


def test_the_registry_covers_every_src_caller():
    callers = _verdict_callers()
    registered = {surface.module for surface in SURFACES}

    assert callers, "the scan found no caller: it is looking in the wrong place"
    assert callers - DEFINING_MODULES <= registered, sorted(callers - registered)
    assert registered <= callers | DEFINING_MODULES, sorted(registered - callers)


def test_the_registry_names_are_unique():
    assert len(BY_NAME) == len(SURFACES)


def test_every_verdict_kind_has_a_surface_and_a_row():
    for surface in SURFACES:
        assert surface.verdict in {"write", "backstop"}, surface.name
        assert _rows_for(surface), surface.name


# -- every row through every surface -------------------------------------------


@pytest.mark.parametrize("surface", SURFACES, ids=[surface.name for surface in SURFACES])
def test_the_surface_reads_every_row_in_its_domain(surface: Surface, tmp_path):
    mismatches = []
    asked = 0
    for case in _rows_for(surface):
        answer = _project(surface, case, tmp_path)
        if answer is OUT_OF_DOMAIN:
            continue
        asked += 1
        observed, expected = answer
        if observed != expected:
            mismatches.append(f"{case.id}: {surface.name} read {observed!r}, row says {expected!r}")

    assert asked, f"{surface.name} was asked no row"
    assert not mismatches, "\n".join(mismatches)


def test_the_table_has_the_switch_off_no_stamp_row():
    (case,) = [case for case in CASES if case.id == SWITCH_OFF_NO_STAMP]
    assert case.verdict == "write"
    assert not case.env, "no stamped owner, no launch pin, no run mode"
    assert not case.has_owner
    assert case.record is not None and not case.record.get("switch")
    assert case.expect == {"allowed": False, "reason": "switch_off"}


STAGE1 = [surface for surface in SURFACES if surface.stage1]


@pytest.mark.parametrize("surface", STAGE1, ids=[surface.name for surface in STAGE1])
def test_switch_off_with_no_stamp_reads_switch_off_at_stage_1(surface: Surface, tmp_path):
    (case,) = [case for case in CASES if case.id == SWITCH_OFF_NO_STAMP]

    answer = _project(surface, case, tmp_path)

    assert answer is not OUT_OF_DOMAIN, f"{surface.name} cannot be asked the switch-off row"
    observed, expected = answer
    assert observed == expected
    if surface.reports_reason:
        assert observed == (False, "switch_off")
    else:
        assert observed is False


# -- the headless invariant ----------------------------------------------------

_DECLARING_SERVER = (
    "claude_code:\n  servers:\n    plc:\n      command: plc-server\n      writes: [set_value]\n"
)


@pytest.mark.parametrize("switch", [None, {"live": "on"}], ids=["switch-off", "switch-on"])
def test_a_headless_run_blocks_every_write_tool_whatever_the_switch(switch, monkeypatch, tmp_path):
    """Framework and declared write tools are blocked headless, switch on or off."""
    from osprey.agent_runner.write_tools import _FALLBACK_WRITE_TOOLS, read_only_disallowed_tools
    from osprey_connectors_tests._harness import write_control_context

    isolate_verdict_environment(monkeypatch, host_root(tmp_path))
    write_control_context(host_root(tmp_path), switch=switch, owned_by=None)
    (tmp_path / "config.yml").write_text(_DECLARING_SERVER, encoding="utf-8")

    blocked = set(read_only_disallowed_tools(tmp_path))

    assert set(_FALLBACK_WRITE_TOOLS) <= blocked
    assert "mcp__plc__set_value" in blocked
    assert "Bash" in blocked
