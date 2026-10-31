"""The hook's stdlib write verdict against the package's, row by row.

``osprey_target_state.write_verdict`` restates
:func:`osprey_connectors.posture_store.write_verdict` for the hooks, which run
outside the osprey venv. Both read one case table
(``osprey_connectors_tests/_harness/fixtures/write_verdict_cases.json``); this module drives its ``twin``
rows through the hook, and pins the restated words — ``REASONS``, every phrase
and ``STORE_PATH_MARKERS`` — key by key against
:mod:`osprey_connectors.refusals`. A drift between the two is a write refused
for one reason by the connector and another by the hook, or allowed by one of
them only.

The package is imported here and only here: a test may import what the code
under test may not. The last test checks that the hook module itself still
imports nothing beyond the standard library and its own directory.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey_connectors import posture_store, refusals
from osprey_connectors_tests._harness._write_verdict_cases import (
    Case,
    case_environment,
    cases_for,
    owner_argument,
    stage,
)
from tests.hooks.conftest import _HOOK_VARS, HOOKS_DIR

__all__ = ["case_environment"]

TWIN_CASES = cases_for("twin")


def test_the_table_has_twin_rows():
    assert TWIN_CASES
    assert all(case.verdict == "write" for case in TWIN_CASES)


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", TWIN_CASES, ids=[case.id for case in TWIN_CASES])
def test_twin_verdict_matches_the_row(case: Case, monkeypatch, tmp_path):
    stage(case, monkeypatch, tmp_path)

    verdict = reader.write_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case, no_owner=reader.NO_OWNER),
    )

    expect = case.expect
    assert verdict.allowed is expect["allowed"]
    assert verdict.reason == expect["reason"]
    assert verdict.target == (case.target if isinstance(case.target, str) else None)
    if "target" in expect:
        assert verdict.target == expect["target"]
    if "kind" in expect:
        assert verdict.kind == expect["kind"]


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", TWIN_CASES, ids=[case.id for case in TWIN_CASES])
def test_twin_and_package_agree_on_kind(case: Case, monkeypatch, tmp_path):
    """The kind is a word in the phrase, so it must match even where no row names it."""
    stage(case, monkeypatch, tmp_path)
    kwargs = {"ignore_launch_pin": case.ignore_launch_pin}

    twin = reader.write_verdict(
        case.section, case.target, **kwargs, **owner_argument(case, no_owner=reader.NO_OWNER)
    )
    package = posture_store.write_verdict(
        case.section, case.target, **kwargs, **owner_argument(case)
    )

    assert (twin.allowed, twin.reason, twin.target, twin.kind) == (
        package.allowed,
        package.reason,
        package.target,
        package.kind,
    )


def test_a_dispatch_run_resolves_no_owner_on_both_sides(monkeypatch, tmp_path):
    """The dispatch rung sits below the stamp and above the tree and the account."""
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("OSPREY_CONTROL_CONTEXT_TREE", raising=False)
    monkeypatch.delenv("OSPREY_CONTROL_OWNER", raising=False)
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    posture_store._owner_var.set(posture_store.NO_OWNER)
    assert reader.current_owner() is reader.NO_OWNER
    assert posture_store.current_owner() is posture_store.NO_OWNER

    monkeypatch.setenv("OSPREY_CONTROL_OWNER", "alice")
    assert reader.current_owner() == posture_store.current_owner() == "alice"

    monkeypatch.delenv("OSPREY_CONTROL_OWNER")
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "0")
    assert reader.current_owner() is not reader.NO_OWNER
    assert posture_store.current_owner() is not posture_store.NO_OWNER


def test_the_verdict_never_raises_on_a_hostile_section(monkeypatch):
    class Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __contains__(self, key):
            raise RuntimeError("boom")

    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    verdict = reader.write_verdict(Hostile(), "live", owner="alice")
    assert verdict.allowed is False
    assert verdict.reason in reader.REASONS


def test_a_verdict_carries_the_four_fields():
    verdict = reader.WriteVerdict(allowed=False, reason="switch_off", target="va", kind="simulator")
    assert (verdict.allowed, verdict.reason, verdict.target, verdict.kind) == (
        False,
        "switch_off",
        "va",
        "simulator",
    )


def test_the_no_owner_sentinel_has_no_spelling():
    assert str(reader.NO_OWNER) == "<no owner>"
    assert reader.NO_OWNER != "<no owner>"
    assert reader.current_owner(reader.NO_OWNER) is reader.NO_OWNER


# -- the restated words ------------------------------------------------------


def test_reasons_match_in_order():
    assert reader.REASONS == refusals.REASONS


def test_phrase_table_matches_key_by_key():
    assert list(reader._PHRASES) == list(refusals._PHRASES)
    for reason in refusals.REASONS:
        assert reader._PHRASES[reason] == refusals._PHRASES[reason], reason


@pytest.mark.parametrize("kind", [None, *refusals.KIND_WORDS, "unknown"])
@pytest.mark.parametrize("reason", refusals.REASONS)
def test_refusal_phrase_matches(reason, kind):
    assert reader.refusal_phrase(reason, kind) == refusals.refusal_phrase(reason, kind)


def test_an_unknown_reason_raises_on_both_sides():
    with pytest.raises(ValueError, match="unknown refusal reason"):
        refusals.refusal_phrase("not_a_reason")
    with pytest.raises(ValueError, match="unknown refusal reason"):
        reader.refusal_phrase("not_a_reason")


def test_kind_words_match_key_by_key():
    assert list(reader.KIND_WORDS) == list(refusals.KIND_WORDS)
    for kind, words in refusals.KIND_WORDS.items():
        assert reader.KIND_WORDS[kind] == words, kind


def test_store_path_markers_match():
    assert reader.STORE_PATH_MARKERS == refusals.STORE_PATH_MARKERS
    assert reader.STORE_PATH_MARKERS == (reader.STATE_DIR_NAME, reader.RECORD_FILENAME)


def test_restated_store_names_match_the_package():
    assert reader.CONTROL_OWNER_ENV_VAR == posture_store.CONTROL_OWNER_ENV_VAR
    assert reader.DISPATCH_RUN_ENV_VAR == posture_store.DISPATCH_RUN_ENV_VAR
    assert reader.CONTROL_CONTEXT_TREE_ENV_VAR == posture_store.CONTROL_CONTEXT_TREE_ENV_VAR
    assert reader.CONTROL_TREE_MARKER_NAME == posture_store.CONTROL_TREE_MARKER_NAME
    assert reader.LAUNCH_POSTURE_ENV_VAR == posture_store.LAUNCH_POSTURE_ENV_VAR
    assert reader.LAUNCH_POSTURE_ALL_TARGETS == posture_store.LAUNCH_POSTURE_ALL_TARGETS
    assert reader.RECORD_READ_LIMIT == posture_store._RECORD_READ_LIMIT


def test_new_names_are_exported():
    for name in (
        "DISPATCH_RUN_ENV_VAR",
        "KIND_WORDS",
        "NO_OWNER",
        "REASONS",
        "STORE_PATH_MARKERS",
        "WriteVerdict",
        "current_owner",
        "launch_pin_term",
        "machine_kind",
        "owner_tier_term",
        "refusal_phrase",
        "tier_allows",
        "tier_writes",
        "write_verdict",
    ):
        assert name in reader.__all__, name
        assert hasattr(reader, name), name


# -- the forwarded environment ----------------------------------------------


@pytest.mark.parametrize(
    "name", ["OSPREY_CONTROL_OWNER", "OSPREY_CONTROL_CONTEXT_TREE", "OSPREY_DISPATCH_RUN"]
)
def test_the_owner_ladder_vars_reach_a_hook_subprocess(name):
    assert name in _HOOK_VARS


@pytest.mark.parametrize(
    "name", ["OSPREY_CONTROL_OWNER", "OSPREY_CONTROL_CONTEXT_TREE", "OSPREY_DISPATCH_RUN"]
)
def test_the_owner_ladder_vars_are_blank_by_default(name):
    assert not os.environ.get(name)


# -- stdlib only -------------------------------------------------------------


def _hook_stems() -> set[str]:
    return {path.stem for path in HOOKS_DIR.glob("*.py")}


def test_the_module_source_imports_only_the_stdlib_and_the_hook_directory():
    tree = ast.parse((HOOKS_DIR / "osprey_target_state.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in a hook module"
            roots.add((node.module or "").split(".")[0])
    roots.discard("__future__")
    foreign = {root for root in roots if root not in sys.stdlib_module_names} - _hook_stems()
    assert not foreign


def test_the_module_imports_and_decides_without_site_packages(tmp_path):
    """Outside the venv — no site-packages at all — it imports and still decides.

    ``-S`` drops site-packages, which is where a hook stands at run time: the
    sibling ``osprey_hook_log``'s one optional package import then falls back,
    and everything that loads has to be the standard library or this directory.
    """
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(HOOKS_DIR)!r})\n"
        "before = set(sys.modules)\n"
        "import osprey_target_state\n"
        "added = {name.split('.')[0] for name in set(sys.modules) - before}\n"
        "foreign = sorted(name for name in added if name not in sys.stdlib_module_names)\n"
        "verdict = osprey_target_state.write_verdict(\n"
        "    {'type': 'epics', 'writes': 'allowed'}, 'live', owner='alice'\n"
        ")\n"
        "print(json.dumps([foreign, verdict.reason]))\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OSPREY_AGENT_DATA_ROOT": str(tmp_path),
        # The host's own login, so the owner is not refused as foreign.
        "OSPREY_AUDIT_IDENTITY": "alice",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=120,
    )

    foreign, reason = json.loads(result.stdout)
    assert set(foreign) <= _hook_stems()
    assert reason == "switch_off"
