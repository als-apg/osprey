"""The login tier's write ceiling (``control_system.writes``) and the machine kind of a target.

``tier_writes`` is the ceiling a login tier sets on writes: ``never``, ``asked``
or ``allowed``, target-blind. What is pinned here is the reading rule. When the
new key is present it decides alone, and only the literal strings ``"asked"`` and
``"allowed"`` arm; every other value is ``never``. ``tier_allows`` is "the word
is not ``never``", so it is true of both arming words. When the key is absent the
tier is ``never``: the retired arming leaves are not read at run time, so a
render built before the tier key existed (an unrebuilt render) writes nowhere
until the build's normalizer rewrites it. A section nobody can read is ``never``.

``machine_kind`` names what a target is by its name, for the words a refusal
uses: ``live`` and ``standin`` are the real machine, ``va`` the simulator, and
a target that does not resolve has no kind. Only ``published_machine_kind``,
which reads the controls server's report (C-KIND), calls a stand-in the
rehearsal machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import types
from osprey_connectors.types import (
    EPICS,
    LIVE_STANDIN,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    WRITES_KEY,
    machine_kind,
    published_machine_kind,
    tier_allows,
    tier_writes,
)


def _section(writes: Any = ..., writes_enabled: Any = ..., connector: Any = ...) -> dict[str, Any]:
    """A ``control_system:`` section with only the leaves a test names."""
    section: dict[str, Any] = {"type": EPICS}
    if writes is not ...:
        section["writes"] = writes
    if writes_enabled is not ...:
        section["writes_enabled"] = writes_enabled
    if connector is not ...:
        section["connector"] = connector
    return section


def test_writes_key_is_the_dotted_control_system_writes_key() -> None:
    assert WRITES_KEY == "control_system.writes"


def test_new_names_are_exported() -> None:
    for name in ("WRITES_KEY", "tier_writes", "tier_allows", "machine_kind"):
        assert name in types.__all__


# -- the new key present: only the two literal arming words arm ----------------


def test_literal_allowed_arms() -> None:
    assert tier_writes(_section(writes="allowed")) == "allowed"
    assert tier_allows(_section(writes="allowed")) is True


def test_literal_asked_arms() -> None:
    assert tier_writes(_section(writes="asked")) == "asked"
    assert tier_allows(_section(writes="asked")) is True


@pytest.mark.parametrize(
    "value",
    [
        "never",
        "Allowed",
        "ALLOWED",
        " allowed",
        "allowed ",
        "Asked",
        "ASKED",
        " asked",
        "asked ",
        "on",
        "off",
        "yes",
        "",
        True,
        1,
        None,
        ["allowed"],
        ["asked"],
        {"allowed": True},
        {"asked": True},
        b"allowed",
        b"asked",
    ],
)
def test_any_other_value_of_the_new_key_is_never(value: Any) -> None:
    assert tier_writes(_section(writes=value)) == "never"
    assert tier_allows(_section(writes=value)) is False


@pytest.mark.parametrize("section", [None, "allowed", True, 1, ["writes"], ("allowed",), object()])
def test_a_section_that_is_not_a_mapping_is_never(section: Any) -> None:
    assert tier_writes(section) == "never"
    assert tier_allows(section) is False


# -- the new key absent: never, whatever the retired leaves say ------------------
#
# These replace the removed fallback cases (absent key with a flat or per-type
# ``writes_enabled: true`` read as ``allowed``): the package reads the tier key
# alone, and the case table's ``tier-absent-old-armed-*`` rows pin the same
# answer through the write verdict.


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({"type": EPICS, "writes_enabled": True}, id="flat-leaf"),
        pytest.param(
            {"type": EPICS, "connector": {EPICS: {"writes_enabled": True}}}, id="connector-leaf"
        ),
        pytest.param(
            {"type": EPICS, "connector": {VIRTUAL_ACCELERATOR: {"writes_enabled": True}}},
            id="simulator-leaf-only",
        ),
        pytest.param(
            {
                "type": LIVE_STANDIN,
                "writes_enabled": True,
                "connector": {
                    LIVE_STANDIN: {"writes_enabled": True},
                    "mypackage.Moat": {"writes_enabled": True},
                },
            },
            id="every-leaf",
        ),
    ],
)
def test_an_unrebuilt_render_reads_never(section: dict[str, Any]) -> None:
    """A render from before the tier key existed is ``never`` until it is rebuilt."""
    assert tier_writes(section) == "never"
    assert tier_allows(section) is False


@pytest.mark.parametrize(
    "section",
    [
        {},
        _section(),
        _section(writes_enabled=False),
        _section(writes_enabled="true"),
        _section(writes_enabled=1),
        _section(writes_enabled=None),
        _section(connector=None),
        _section(connector=["epics"]),
        _section(connector={EPICS: None}),
        _section(connector={EPICS: {}}),
        _section(connector={EPICS: {"writes_enabled": "true"}}),
        _section(connector={EPICS: {"writes_enabled": 1}}),
        _section(
            connector={EPICS: {"writes_enabled": False}, VIRTUAL_ACCELERATOR: ["writes_enabled"]}
        ),
        _section(connector={EPICS: True}),
    ],
)
def test_absent_key_with_nothing_stated_is_never(section: dict[str, Any]) -> None:
    assert tier_writes(section) == "never"
    assert tier_allows(section) is False


# -- both present: the new key decides ------------------------------------------


@pytest.mark.parametrize(
    ("writes", "expected"),
    [
        ("never", "never"),
        ("garbage", "never"),
        (None, "never"),
        ("asked", "asked"),
        ("allowed", "allowed"),
    ],
)
def test_new_key_decides_when_both_are_present(writes: Any, expected: str) -> None:
    armed = _section(
        writes=writes, writes_enabled=True, connector={EPICS: {"writes_enabled": True}}
    )
    assert tier_writes(armed) == expected
    unarmed = _section(
        writes=writes, writes_enabled=False, connector={EPICS: {"writes_enabled": False}}
    )
    assert tier_writes(unarmed) == expected


def test_tier_reading_never_mutates_the_section() -> None:
    section = _section(writes_enabled=True, connector={EPICS: {"writes_enabled": True}})
    snapshot = repr(section)
    tier_writes(section)
    assert repr(section) == snapshot


# -- machine_kind -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "kind"),
    [(TARGET_LIVE, "real"), (TARGET_STANDIN, "real"), (TARGET_VA, "simulator")],
)
def test_machine_kind_per_target(target: str, kind: str) -> None:
    assert machine_kind(_section(), target) == kind


@pytest.mark.parametrize("target", [None, "", "LIVE", "Live", " live", "sim", "epics", 0, True])
def test_machine_kind_of_an_unknown_target_is_none(target: Any) -> None:
    assert machine_kind(_section(), target) is None


def test_machine_kind_of_live_with_no_real_machine_is_none() -> None:
    """Baselined on the simulator with no real block: ``live`` does not resolve."""
    section = {"type": VIRTUAL_ACCELERATOR}
    assert machine_kind(section, TARGET_LIVE) is None
    assert machine_kind(section, TARGET_VA) == "simulator"
    assert machine_kind(section, TARGET_STANDIN) == "real"


def test_machine_kind_of_live_derived_from_the_connector_table_is_real() -> None:
    section = {"type": VIRTUAL_ACCELERATOR, "connector": {EPICS: {}}}
    assert machine_kind(section, TARGET_LIVE) == "real"


@pytest.mark.parametrize("section", [None, "epics", 3])
def test_machine_kind_on_a_section_that_is_not_a_mapping_never_raises(section: Any) -> None:
    assert machine_kind(section, TARGET_VA) == "simulator"
    assert machine_kind(section, "nope") is None
    # live on an unreadable section either resolves or is None; it never raises.
    assert machine_kind(section, TARGET_LIVE) in ("real", None)


# -- published_machine_kind -------------------------------------------------------

_DEAD_PID = 2_147_483_646


def _report(directory: Path, pid: int, targets: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"server_pid": pid, "targets": targets}
    (directory / f"server_{pid}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_published_kind_of_a_live_standin_is_rehearsal(tmp_path: Path) -> None:
    _report(tmp_path, os.getpid(), {"standin": {"machine_kind": "rehearsal"}})
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path) == "rehearsal"


@pytest.mark.parametrize(
    "meta",
    [{"machine_kind": "real"}, {}, {"machine_kind": "REHEARSAL"}, {"machine_kind": True}, "x"],
)
def test_published_kind_of_a_standin_not_vouched_for_is_real(tmp_path: Path, meta: Any) -> None:
    _report(tmp_path, os.getpid(), {"standin": meta, "live": {"machine_kind": "real"}})
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path) == "real"


def test_published_kind_with_no_report_is_real(tmp_path: Path) -> None:
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path) == "real"
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path / "missing") == "real"


def test_published_kind_ignores_a_dead_servers_report(tmp_path: Path) -> None:
    _report(tmp_path, _DEAD_PID, {"standin": {"machine_kind": "rehearsal"}})
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path) == "real"


def test_published_kind_reads_the_first_live_report_with_targets(tmp_path: Path) -> None:
    """Name order, a dead or target-less report skipped, the first live one wins."""
    _report(tmp_path, 1, {})  # PID 1 is alive and states nothing: skipped
    _report(tmp_path, _DEAD_PID, {"standin": {"machine_kind": "real"}})
    (tmp_path / "server_2.json").write_text("{not json", encoding="utf-8")
    _report(tmp_path, os.getpid(), {"standin": {"machine_kind": "rehearsal"}})
    assert published_machine_kind(TARGET_STANDIN, state_dir=tmp_path) == "rehearsal"


@pytest.mark.parametrize(
    ("target", "kind"),
    [(TARGET_LIVE, "real"), (TARGET_VA, "simulator"), ("nope", "real"), (None, "real")],
)
def test_published_kind_of_other_targets_is_the_name_table(
    tmp_path: Path, target: Any, kind: str
) -> None:
    """A report can never make ``live`` (or anything but ``standin``) a rehearsal."""
    _report(tmp_path, os.getpid(), {"live": {"machine_kind": "rehearsal"}})
    assert published_machine_kind(target, state_dir=tmp_path) == kind


def test_published_kind_reads_the_state_dir_by_default(tmp_path: Path, monkeypatch) -> None:
    from osprey_connectors import posture_store

    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(tmp_path))
    _report(tmp_path, os.getpid(), {"standin": {"machine_kind": "rehearsal"}})
    assert published_machine_kind(TARGET_STANDIN) == "rehearsal"


def test_tier_word_is_the_tier_words_as_a_type():
    from typing import get_args

    assert get_args(types.TierWord) == types.TIER_WORDS


@pytest.mark.parametrize(
    ("tree_bound", "section", "ceiling"),
    [
        # A host caps by its tier; the rung there only refuses.
        (False, {"writes": "allowed", "login_tiers": {"alice": "asked"}}, "allowed"),
        (False, {"writes": "asked", "login_tiers": {"alice": "allowed"}}, "asked"),
        (False, {"writes": "allowed"}, "allowed"),
        # Under a tree the owner's rung is the ceiling.
        (True, {"writes": "allowed", "login_tiers": {"alice": "asked"}}, "asked"),
        (True, {"writes": "asked", "login_tiers": {"alice": "allowed"}}, "allowed"),
        (True, {"writes": "allowed", "login_tiers": {"alice": "never"}}, "never"),
        (True, {"writes": "allowed", "login_tiers": {"bob": "allowed"}}, "unknown"),
        (True, {"writes": "allowed"}, "unknown"),
    ],
)
def test_write_ceiling_is_the_tier_on_a_host_and_the_rung_under_a_tree(
    monkeypatch, tree_bound, section, ceiling
):
    from osprey_connectors import posture_store

    if tree_bound:
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/nonexistent")
    else:
        monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    assert posture_store.write_ceiling(section, "alice") == ceiling
    assert posture_store.write_ceiling(section, "alice", tree_bound=tree_bound) == ceiling
