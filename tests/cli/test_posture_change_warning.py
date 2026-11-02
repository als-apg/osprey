"""A merged render warns where the tier key's posture differs from today's reading.

The retired write-posture leaves are rewritten per document to
``control_system.writes`` (PR-3's per-layer rule). Today's reading was per
machine over the merged section (the retired reader), where a per-type ``false``
overrides a flat ``true`` and a leaf on a type no configured target reaches
arms nothing (R2 CF-1 / BE-3). Where the two differ, the render warns "was X,
now Y" once per resolved layer path, and never when resolve runs with
``warn=False`` (BE-22). An unchanged ``allowed`` that today armed only some of
the configured machines warns, naming the machines that become armable (CC-2).
A machine configured today only because a retired leaf was its block's one key
is refused (SK2-M1 / BD-4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli import build_profile_document as doc
from osprey.cli import build_profile_merge as merge
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    _hash_resolved_profile,
    render_document,
    resolve_profile_document,
)
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.cli.profile_root import ProfileKind
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import tier_writes

_LOGGER_NAME = merge.__name__


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()


# An EPICS deployment that also configures the virtual accelerator: two machines.
_TWO_MACHINES = {
    "control_system.type": "epics",
    "control_system.connector.epics.timeout_s": 5,
    "control_system.connector.virtual_accelerator.port": 8000,
}
# An EPICS-only deployment.
_EPICS_ONLY = {
    "control_system.type": "epics",
    "control_system.connector.epics.timeout_s": 5,
}


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root_config: dict, layer_config: dict | None, layer="tiers/ops.yml"):
    """A deployment repo; returns the layer's path (or the root's when no layer)."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True, exist_ok=True)
    root = _write_yaml(
        repo / "profile.yml", {"name": "Facility", "data": "data", "config": root_config}
    )
    if layer_config is None:
        return root
    return _write_yaml(repo / layer, {"name": "Ops", "config": layer_config})


def _resolve(path: Path, *, warn: bool = True):
    raw = _read_profile_document(path)
    return resolve_profile_document(raw, path, warn=warn)


def _posture_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING and "write" in r.getMessage()
    ]


def _tier(document) -> str:
    section = merge._rendered_control_system(document.raw["config"])
    return tier_writes(section)


# ── the posture-change warning ───────────────────────────────────────────────


def test_epics_false_tier_under_flat_true_root_with_a_second_machine_warns(tmp_path, caplog):
    """Today the tier still arms the virtual accelerator; the tier key reads never."""
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": True},
        {"control_system.connector.epics.writes_enabled": False},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == "never"
    [message] = _posture_warnings(caplog)
    assert "tiers/ops.yml" in message
    assert "was allowed, now never" in message


def test_true_leaf_on_a_type_no_target_reaches_warns_the_reverse(tmp_path, caplog):
    """A tango ``true`` on an EPICS-only deployment arms nothing today; the tier allows."""
    path = _repo(
        tmp_path,
        {**_EPICS_ONLY, "control_system.writes_enabled": False},
        {"control_system.connector.tango.writes_enabled": True},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == "allowed"
    [message] = _posture_warnings(caplog)
    assert "tiers/ops.yml" in message
    assert "was never, now allowed" in message


@pytest.mark.parametrize(
    ("root", "layer"),
    [
        pytest.param(
            {**_TWO_MACHINES, "control_system.writes_enabled": False},
            {"control_system.writes_enabled": False},
            id="flat-false-over-flat-false",
        ),
        pytest.param(
            {**_EPICS_ONLY, "control_system.writes_enabled": True},
            {"control_system.connector.epics.writes_enabled": False},
            id="epics-false-tier-epics-only",
        ),
        pytest.param(
            {
                **_TWO_MACHINES,
                "control_system.writes_enabled": True,
                "control_system.connector.epics.writes_enabled": False,
                "control_system.connector.virtual_accelerator.writes_enabled": False,
            },
            {"control_system.writes": "never"},
            id="flat-true-every-type-false-new-never",
        ),
        pytest.param(
            {**_TWO_MACHINES, "control_system.writes_enabled": True},
            {"control_system.writes_enabled": True},
            id="flat-true-allowed-both-armed",
        ),
    ],
)
def test_no_change_from_today_stays_silent(tmp_path, caplog, root, layer):
    path = _repo(tmp_path, root, layer)
    with caplog.at_level(logging.WARNING):
        _resolve(path)
    assert _posture_warnings(caplog) == []


def test_render_with_no_retired_origin_is_skipped(tmp_path, caplog):
    """Only the new key anywhere: nothing to compare against."""
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes": "allowed"},
        {"control_system.writes": "never"},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)
    assert _tier(document) == "never"
    assert _posture_warnings(caplog) == []


def test_root_only_render_reads_its_extends_chain(tmp_path, caplog):
    """A root whose base spells flat ``true`` with its one machine ``false`` was never."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _write_yaml(
        repo / "base.yml",
        {
            "config": {
                **_EPICS_ONLY,
                "control_system.writes_enabled": True,
                "control_system.connector.epics.writes_enabled": False,
            }
        },
    )
    root = _write_yaml(repo / "profile.yml", {"name": "Facility", "extends": "base.yml"})
    with caplog.at_level(logging.WARNING):
        _resolve(root)
    [message] = _posture_warnings(caplog)
    assert "was never, now allowed" in message


def test_an_app_never_warns(tmp_path, caplog):
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": True},
        {"control_system.connector.epics.writes_enabled": False},
        layer="apps/logbook.yml",
    )
    with caplog.at_level(logging.WARNING):
        _resolve(path)
    assert _posture_warnings(caplog) == []


def test_render_document_without_files_skips_the_check(tmp_path, caplog):
    """The materialization and lint callers hold no files: no origin, no warning."""
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": True},
        {"control_system.connector.epics.writes_enabled": False},
    )
    host = {"config": {**_TWO_MACHINES, "control_system.writes_enabled": True}}
    layer = _read_profile_document(path)
    with caplog.at_level(logging.WARNING):
        render_document(ProfileKind.TIER, host, layer, "tiers/ops.yml")
    assert _posture_warnings(caplog) == []


# ── the widening warning ─────────────────────────────────────────────────────


def test_previous_release_readwrite_tier_warns_widening(tmp_path, caplog):
    """Today the old readwrite layer arms the simulator only; the tier allows EPICS too."""
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": False},
        {
            "control_system.writes_enabled": False,
            "control_system.connector.epics.writes_enabled": False,
            "control_system.connector.virtual_accelerator.writes_enabled": True,
        },
        layer="tiers/readwrite.yml",
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == "allowed"
    [message] = _posture_warnings(caplog)
    assert "tiers/readwrite.yml" in message
    assert "become armable" in message
    assert "`live` (`epics`)" in message.split("so", 1)[1]
    assert "write posture was" not in message


@pytest.mark.parametrize("word", ["allowed", "asked"])
def test_a_tier_word_over_a_partly_armed_root_warns_widening_and_names_the_word(
    tmp_path, caplog, word
):
    """Both arming words widen: a retired root armed the simulator, the tier reaches EPICS too."""
    path = _repo(
        tmp_path,
        {
            **_TWO_MACHINES,
            "control_system.writes_enabled": False,
            "control_system.connector.virtual_accelerator.writes_enabled": True,
        },
        {"control_system.writes": word},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == word
    [message] = _posture_warnings(caplog)
    assert "tiers/ops.yml" in message
    assert f"`control_system.writes: {word}` lets a session write" in message
    assert "become armable" in message
    assert "`live` (`epics`)" in message.split("so", 1)[1]
    assert "write posture was" not in message


def test_an_asked_tier_over_an_unarmed_retired_root_warns_the_change(tmp_path, caplog):
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": False},
        {"control_system.writes": "asked"},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == "asked"
    [message] = _posture_warnings(caplog)
    assert "was never, now asked" in message
    assert "`control_system.writes` reads `asked`" in message


def test_an_asked_tier_over_a_fully_armed_retired_root_stays_silent(tmp_path, caplog):
    """Every configured machine was armed: asked reaches nothing new."""
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": True},
        {"control_system.writes": "asked"},
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)

    assert _tier(document) == "asked"
    assert _posture_warnings(caplog) == []


def test_a_never_tier_over_an_armed_retired_root_still_warns_the_change(tmp_path, caplog):
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": True},
        {"control_system.writes": "never"},
    )
    with caplog.at_level(logging.WARNING):
        _resolve(path)

    [message] = _posture_warnings(caplog)
    assert "was allowed, now never" in message


def test_flat_true_with_every_type_false_and_new_never_does_not_widen(tmp_path, caplog):
    path = _repo(
        tmp_path,
        {
            **_TWO_MACHINES,
            "control_system.writes_enabled": True,
            "control_system.connector.epics.writes_enabled": False,
            "control_system.connector.virtual_accelerator.writes_enabled": False,
        },
        {"control_system.writes": "never"},
    )
    with caplog.at_level(logging.WARNING):
        _resolve(path)
    assert not [m for m in _posture_warnings(caplog) if "armable" in m]


# ── the emptied connector block ──────────────────────────────────────────────


def test_block_held_only_by_a_retired_leaf_is_refused(tmp_path):
    path = _repo(
        tmp_path,
        {**_EPICS_ONLY, "control_system.writes_enabled": False},
        {"control_system.connector.virtual_accelerator.writes_enabled": True},
    )
    with pytest.raises(BuildProfileError) as excinfo:
        _resolve(path, warn=False)
    message = str(excinfo.value)
    assert "tiers/ops.yml" in message
    assert "control_system.connector.virtual_accelerator" in message
    assert "connection key" in message and "delete the block" in message


def test_unarmed_block_held_only_by_a_retired_leaf_warns_and_builds(tmp_path, caplog):
    """The previous-release readwrite layer on a simulator-only deployment keeps building."""
    path = _repo(
        tmp_path,
        {
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.port": 8000,
            "control_system.writes_enabled": False,
        },
        {
            "control_system.writes_enabled": False,
            "control_system.connector.epics.writes_enabled": False,
            "control_system.connector.virtual_accelerator.writes_enabled": True,
        },
        layer="tiers/readwrite.yml",
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)
    assert _tier(document) == "allowed"
    [message] = [m for m in _posture_warnings(caplog) if "no longer configured" in m]
    assert "control_system.connector.epics" in message
    assert not [m for m in _posture_warnings(caplog) if "armable" in m or "posture was" in m]


def test_block_with_a_connection_key_is_not_refused(tmp_path):
    path = _repo(
        tmp_path,
        {**_TWO_MACHINES, "control_system.writes_enabled": False},
        {"control_system.connector.virtual_accelerator.writes_enabled": True},
    )
    assert _tier(_resolve(path, warn=False)) == "allowed"


# ── once (BE-22) ─────────────────────────────────────────────────────────────

_CHANGING = (
    {**_TWO_MACHINES, "control_system.writes_enabled": True},
    {"control_system.connector.epics.writes_enabled": False},
)


def test_warns_once_per_layer_path(tmp_path, caplog):
    path = _repo(tmp_path, *_CHANGING)
    with caplog.at_level(logging.WARNING):
        _resolve(path)
        _resolve(path)
    assert len(_posture_warnings(caplog)) == 1


def test_no_warning_when_resolve_runs_without_warn(tmp_path, caplog):
    path = _repo(tmp_path, *_CHANGING)
    with caplog.at_level(logging.WARNING):
        _resolve(path, warn=False)
    assert _posture_warnings(caplog) == []


def test_hashing_prints_no_posture_warning(tmp_path, caplog):
    path = _repo(tmp_path, *_CHANGING)
    with caplog.at_level(logging.WARNING):
        _hash_resolved_profile(_read_profile_document(path), path)
    assert _posture_warnings(caplog) == []


def test_build_load_then_hash_warns_once(tmp_path, caplog):
    """What one ``osprey build`` does per render: load the profile, then hash it."""
    path = _repo(tmp_path, *_CHANGING)
    with caplog.at_level(logging.WARNING):
        resolve_build_document(path, None)
        resolve_build_document(path, None)
        _hash_resolved_profile(_read_profile_document(path), path)
    messages = _posture_warnings(caplog)
    assert len(messages) == 1
    assert "was allowed, now never" in messages[0]


def test_a_leaf_spelled_inside_a_mapping_is_read_where_it_stood(tmp_path, caplog):
    """The normalizer leaves the block in the mapping, so today's reading keeps the leaf."""
    path = _repo(
        tmp_path,
        {
            "control_system.type": "epics",
            "control_system.connector": {"epics": {"timeout_s": 5, "writes_enabled": True}},
        },
        None,
    )
    with caplog.at_level(logging.WARNING):
        document = _resolve(path)
    assert _tier(document) == "allowed"
    assert not [m for m in _posture_warnings(caplog) if "posture was" in m]


def test_a_later_mapping_replacing_the_block_takes_its_leaf_with_it(tmp_path):
    """A tier's whole ``control_system`` mapping replaces the root's leaf-only block."""
    path = _repo(
        tmp_path,
        {**_EPICS_ONLY, "control_system.connector.virtual_accelerator.writes_enabled": True},
        {"control_system": {"type": "epics", "writes": "never"}},
    )
    assert _tier(_resolve(path, warn=False)) == "never"
