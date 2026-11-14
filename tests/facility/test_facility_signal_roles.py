"""Facility-added signal roles through ``osprey build``.

``signals.yaml`` declares signal roles beyond the vocabulary, each a
``{name, aliases?, description?}`` row. A channel may carry a declared role,
``facility.json`` ``signals`` holds the rows verbatim, sorted by name, and the
facts and the channel finder describe the role from its row. A role that
shadows a vocabulary role or another declared role stops the build naming both.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from click.testing import Result

from osprey.facility.records import FacilityRecords
from tests.facility._synthetic_trees import plain_tree, write_tree

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

Build = Callable[..., tuple["BuiltProject", Result]]

FIXTURE_ROLE = {
    "name": "fixture_role",
    "aliases": ["fixture signal"],
    "description": "A signal the fixture facility adds beside the vocabulary.",
}


def _tree(signals: list[dict[str, Any]] | None, signal: str = "fixture_role") -> dict[str, Any]:
    tree = plain_tree()
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == "Q1:RB":
            channel["signal"] = signal
    if signals is not None:
        tree["signals.yaml"] = signals
    return tree


def test_a_declared_role_builds_and_is_carried_verbatim(build_project: Build) -> None:
    project, result = build_project(_tree([FIXTURE_ROLE]))

    assert result.exit_code == 0, result.output
    assert project.facility["signals"] == [FIXTURE_ROLE]
    [readback] = [c for c in project.facility["channels"] if c["id"] == "Q1:RB"]
    assert readback["signal"] == "fixture_role"


def test_the_declared_role_resolves_in_the_channel_finder(build_project: Build) -> None:
    from osprey.facility.views.channel_finder import middle_layer_document

    project, result = build_project(_tree([FIXTURE_ROLE]))
    assert result.exit_code == 0, result.output

    document, _left_out, _by_address = middle_layer_document(FacilityRecords(project.facility))
    fields = [
        family[FIXTURE_ROLE["name"]]
        for system, families in document.items()
        if isinstance(families, dict)
        for family in families.values()
        if isinstance(family, dict) and FIXTURE_ROLE["name"] in family
    ]
    assert fields == [{"ChannelNames": ["Q1:RB"], "_description": FIXTURE_ROLE["description"]}]


def test_the_facts_describe_the_declared_role(build_project: Build) -> None:
    from osprey.facility.views.facts import facts_document

    project, result = build_project(_tree([FIXTURE_ROLE]))
    assert result.exit_code == 0, result.output

    facts = facts_document(FacilityRecords(project.facility), [])
    described = facts["vocabulary"]["signals"]["fixture_role"]["description"]
    assert described == FIXTURE_ROLE["description"]


def test_signal_roles_joins_the_declared_rows_to_the_vocabulary() -> None:
    from osprey.facility.validate import signal_roles

    assert "fixture_role" not in signal_roles()
    joined = signal_roles([FIXTURE_ROLE])
    assert joined == signal_roles() | {"fixture_role"}


def test_a_tree_without_signals_yaml_carries_no_signals(build_project: Build) -> None:
    project, result = build_project(_tree(None, signal="current_readback"))

    assert result.exit_code == 0, result.output
    assert project.facility["signals"] == []


def test_an_undeclared_role_stops_the_build_naming_signals_yaml(build_project: Build) -> None:
    _project, result = build_project(_tree(None))

    assert result.exit_code != 0
    assert (
        "class-unknown: channel Q1:RB — signal fixture_role is in neither the vocabulary "
        "nor signals.yaml" in result.output
    )


def test_a_role_shadowing_a_vocabulary_role_is_refused_naming_both(build_project: Build) -> None:
    from osprey.facility.validate import signal_roles

    assert "current_readback" in signal_roles()
    shadow = {"name": "current_readback", "description": "a second readback"}
    _project, result = build_project(_tree([shadow], signal="current_readback"))

    assert result.exit_code != 0
    assert "source-invalid" in result.output
    assert (
        "signals.yaml role current_readback shadows the vocabulary signal role current_readback"
        in result.output
    )


def test_a_role_declared_twice_is_refused_naming_both(build_project: Build) -> None:
    _project, result = build_project(_tree([FIXTURE_ROLE, {"name": "fixture_role"}]))

    assert result.exit_code != 0
    assert (
        "signals.yaml.1 role fixture_role shadows signals.yaml.0 role fixture_role" in result.output
    )


def test_an_unknown_key_in_a_row_is_refused(build_project: Build) -> None:
    _project, result = build_project(_tree([{**FIXTURE_ROLE, "unit": "mm"}]))

    assert result.exit_code != 0
    assert "unknown key `unit`" in result.output


def test_facility_signals_reads_the_rows_of_signals_yaml(tmp_path: Path) -> None:
    from osprey.facility.sources import facility_signals

    write_tree(tmp_path, {"signals.yaml": [FIXTURE_ROLE]})
    assert facility_signals(tmp_path) == [FIXTURE_ROLE]
    assert facility_signals(tmp_path / "absent") == []


def test_check_mapping_accepts_a_declared_role_as_a_field_signal() -> None:
    from osprey.facility.layers.mml.mapping import check_mapping, draft_mapping, parse_mapping
    from tests.facility.test_layer_mml_mapping import _typed_export

    document = draft_mapping(_typed_export())
    document["families"]["QF"]["fields"]["Setpoint"]["signal"] = "fixture_role"
    mapping = parse_mapping(document)
    assert [str(problem) for problem in check_mapping(mapping)] == [
        "families.QF.fields.Setpoint.signal: 'fixture_role' is no vocabulary signal role"
    ]
    assert check_mapping(mapping, signals=[FIXTURE_ROLE]) == []
