"""Tests for the TTL generator's read/write direction resolution.

Two halves: the path-resolution branches (which limits file, if any, do we
read?) and the direction rules themselves (defaults-aware writability, the
group-constant invariant, and the PV-grammar fallback).

The last test is the one that matters most — it runs both rules over the real
shipped demo data and asserts they agree, which is what turns "writable iff
``:SP``" from an assumption into a checked property of the committed files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.facility_knowledge.ttl_generator import direction as direction_mod
from osprey.services.facility_knowledge.ttl_generator.direction import (
    DIRECTION_READ,
    DIRECTION_WRITE,
    LIMITS_DATABASE_CONFIG_KEY,
    DirectionConflictError,
    DirectionSource,
    LimitsFileMissing,
    assign_directions,
    load_writable_set,
    resolve_and_assign,
    resolve_limits_path,
)
from osprey.services.facility_knowledge.ttl_generator.model import build_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTROL_ASSISTANT_DATA = _REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
_TIER3_HIERARCHICAL = _CONTROL_ASSISTANT_DATA / "channel_databases/tiers/tier3/hierarchical.json"
_CHANNEL_LIMITS = _CONTROL_ASSISTANT_DATA / "channel_limits.json"

#: The demo machine's writable channel count, pinned by the shipped limits file.
_EXPECTED_WRITABLE = 396


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_limits(path: Path, entries: dict[str, dict], defaults: dict | None = None) -> Path:
    """Write a synthetic channel_limits.json-shaped file."""
    payload: dict[str, object] = {
        "_comment": "synthetic fixture",
        "_version": "4.0",
        "defaults": {"writable": True} if defaults is None else defaults,
    }
    payload.update(entries)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _lookup(values: dict[str, object]):
    """Build a ``get_config_value``-shaped callable answering from *values*."""

    def lookup(key: str, default=None):
        return values.get(key, default)

    return lookup


def _model(addresses):
    """Build a GraphModel from bare addresses (only the keys are read)."""
    return build_model({address: {} for address in addresses})


def _directions(model) -> dict[tuple[str, str, str], str | None]:
    """Direction per signal-group key."""
    return {group.key: group.direction for group in model.signal_groups}


@pytest.fixture
def no_ambient_config(monkeypatch):
    """Neutralize the two ambient path anchors so branches are testable in isolation."""
    monkeypatch.setattr(direction_mod, "default_config_path", lambda: None)
    monkeypatch.delenv("CONFIG_FILE", raising=False)


# ---------------------------------------------------------------------------
# resolve_limits_path
# ---------------------------------------------------------------------------


class TestResolveLimitsPath:
    """Which limits file the generator reads, and how it says so."""

    def test_explicit_path_wins_over_config(self, tmp_path, no_ambient_config):
        """An explicit path is used even when the config names a different file."""
        explicit = _write_limits(tmp_path / "explicit.json", {})
        other = _write_limits(tmp_path / "config.json", {})

        path, message = resolve_limits_path(
            explicit,
            config_lookup=_lookup({LIMITS_DATABASE_CONFIG_KEY: str(other)}),
        )

        assert path == explicit
        assert str(explicit) in message

    def test_explicit_missing_path_is_a_hard_error(self, tmp_path):
        """A named-but-absent file never falls back to the grammar."""
        missing = tmp_path / "nope.json"

        with pytest.raises(LimitsFileMissing) as excinfo:
            resolve_limits_path(missing)

        assert str(missing) in str(excinfo.value)

    def test_absolute_config_path_is_used_as_is(self, tmp_path, no_ambient_config):
        """An absolute database_path needs no anchoring."""
        limits = _write_limits(tmp_path / "limits.json", {})

        path, message = resolve_limits_path(
            None,
            config_lookup=_lookup({LIMITS_DATABASE_CONFIG_KEY: str(limits)}),
        )

        assert path == limits
        assert LIMITS_DATABASE_CONFIG_KEY in message

    def test_relative_path_anchors_on_the_loaded_config_dir(self, tmp_path, monkeypatch):
        """The primary anchor is the directory of the config actually in use."""
        render = tmp_path / "build"
        (render / "data").mkdir(parents=True)
        limits = _write_limits(render / "data" / "channel_limits.json", {})
        monkeypatch.setattr(
            direction_mod, "default_config_path", lambda: str(render / "config.yml")
        )
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "decoy" / "config.yml"))

        path, _message = resolve_limits_path(
            None,
            config_lookup=_lookup(
                {
                    LIMITS_DATABASE_CONFIG_KEY: "data/channel_limits.json",
                    "project_root": str(tmp_path / "decoy"),
                }
            ),
        )

        assert path == limits

    def test_relative_path_falls_back_to_config_file_env(self, tmp_path, monkeypatch):
        """Without a loaded config, CONFIG_FILE names the container's config."""
        container = tmp_path / "app"
        (container / "data").mkdir(parents=True)
        limits = _write_limits(container / "data" / "channel_limits.json", {})
        monkeypatch.setattr(direction_mod, "default_config_path", lambda: None)
        monkeypatch.setenv("CONFIG_FILE", str(container / "config.yml"))

        path, _message = resolve_limits_path(
            None,
            config_lookup=_lookup(
                {
                    LIMITS_DATABASE_CONFIG_KEY: "data/channel_limits.json",
                    "project_root": str(tmp_path / "decoy"),
                }
            ),
        )

        assert path == limits

    def test_relative_path_falls_back_to_project_root(self, tmp_path, no_ambient_config):
        """project_root is the last resort when nothing else names a config."""
        root = tmp_path / "project"
        (root / "data").mkdir(parents=True)
        limits = _write_limits(root / "data" / "channel_limits.json", {})

        path, _message = resolve_limits_path(
            None,
            config_lookup=_lookup(
                {
                    LIMITS_DATABASE_CONFIG_KEY: "data/channel_limits.json",
                    "project_root": str(root),
                }
            ),
        )

        assert path == limits

    def test_unset_key_returns_no_path_and_names_the_key(self, no_ambient_config):
        """Nothing configured: no path, and the message says which key was read."""
        path, message = resolve_limits_path(None, config_lookup=_lookup({}))

        assert path is None
        assert LIMITS_DATABASE_CONFIG_KEY in message

    def test_configured_but_absent_file_returns_no_path_and_names_it(
        self, tmp_path, no_ambient_config
    ):
        """A configured path that does not exist degrades to the grammar, legibly."""
        missing = tmp_path / "gone" / "channel_limits.json"

        path, message = resolve_limits_path(
            None,
            config_lookup=_lookup({LIMITS_DATABASE_CONFIG_KEY: str(missing)}),
        )

        assert path is None
        assert str(missing) in message


# ---------------------------------------------------------------------------
# load_writable_set
# ---------------------------------------------------------------------------


class TestLoadWritableSet:
    """Defaults-aware writability, the rule the runtime write path uses."""

    def test_entry_without_writable_inherits_the_default(self, tmp_path):
        """The demo file grants writability by omission — that must be honored."""
        limits = _write_limits(
            tmp_path / "limits.json",
            {"SR:MAG:DIPOLE:01:CURRENT:SP": {"min_value": 0.0, "max_value": 1.0}},
            defaults={"writable": True},
        )

        assert load_writable_set(limits) == frozenset({"SR:MAG:DIPOLE:01:CURRENT:SP"})

    def test_explicit_false_overrides_a_true_default(self, tmp_path):
        """An explicit writable:false wins over defaults.writable:true."""
        limits = _write_limits(
            tmp_path / "limits.json",
            {
                "SR:MAG:DIPOLE:01:CURRENT:SP": {},
                "SR:VAC:VALVE:01:CONTROL:OPEN": {"writable": False},
            },
            defaults={"writable": True},
        )

        assert load_writable_set(limits) == frozenset({"SR:MAG:DIPOLE:01:CURRENT:SP"})

    def test_explicit_true_overrides_a_false_default(self, tmp_path):
        """The merge runs both ways: an entry can opt in under a false default."""
        limits = _write_limits(
            tmp_path / "limits.json",
            {
                "SR:MAG:DIPOLE:01:CURRENT:SP": {"writable": True},
                "SR:MAG:DIPOLE:01:CURRENT:RB": {},
            },
            defaults={"writable": False},
        )

        assert load_writable_set(limits) == frozenset({"SR:MAG:DIPOLE:01:CURRENT:SP"})

    def test_metadata_and_defaults_keys_are_not_channels(self, tmp_path):
        """Underscore keys and 'defaults' never become addresses."""
        limits = _write_limits(
            tmp_path / "limits.json",
            {"SR:MAG:DIPOLE:01:CURRENT:SP": {}},
            defaults={"writable": True},
        )

        writable = load_writable_set(limits)

        assert writable == frozenset({"SR:MAG:DIPOLE:01:CURRENT:SP"})
        assert not any(key.startswith("_") or key == "defaults" for key in writable)

    def test_missing_defaults_block_falls_back_to_writable_true(self, tmp_path):
        """No defaults block at all still matches the validator's ``get(..., True)``."""
        path = tmp_path / "limits.json"
        path.write_text(json.dumps({"SR:MAG:DIPOLE:01:CURRENT:SP": {}}), encoding="utf-8")

        assert load_writable_set(path) == frozenset({"SR:MAG:DIPOLE:01:CURRENT:SP"})

    def test_non_object_payload_is_a_legible_error(self, tmp_path):
        """A JSON list is refused by name rather than raising AttributeError later."""
        path = tmp_path / "limits.json"
        path.write_text(json.dumps(["nope"]), encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            load_writable_set(path)


# ---------------------------------------------------------------------------
# assign_directions
# ---------------------------------------------------------------------------


class TestAssignDirections:
    """Group-constant direction, from limits or from the grammar."""

    def test_limits_source_marks_setpoints_write_and_the_rest_read(self, tmp_path):
        """The happy path: one write group, one read group, source == limits."""
        addresses = ["SR:MAG:DIPOLE:01:CURRENT:SP", "SR:MAG:DIPOLE:01:CURRENT:RB"]
        limits = _write_limits(
            tmp_path / "limits.json",
            {
                "SR:MAG:DIPOLE:01:CURRENT:SP": {},
                "SR:MAG:DIPOLE:01:CURRENT:RB": {"writable": False},
            },
        )

        annotated, report = assign_directions(_model(addresses), limits)

        assert _directions(annotated) == {
            ("DIPOLE", "CURRENT", "SP"): DIRECTION_WRITE,
            ("DIPOLE", "CURRENT", "RB"): DIRECTION_READ,
        }
        assert report.source is DirectionSource.LIMITS
        assert report.limits_path == limits
        assert str(limits) in report.message

    def test_limits_source_honors_a_non_setpoint_write(self, tmp_path):
        """The limits file, not the grammar, decides — even when they would differ."""
        addresses = ["SR:VAC:VALVE:01:CONTROL:OPEN"]
        limits = _write_limits(tmp_path / "limits.json", {"SR:VAC:VALVE:01:CONTROL:OPEN": {}})

        annotated, report = assign_directions(_model(addresses), limits)

        assert _directions(annotated) == {("VALVE", "CONTROL", "OPEN"): DIRECTION_WRITE}
        assert report.source is DirectionSource.LIMITS

    def test_grammar_fallback_uses_the_subfield_and_reports_the_lookup(self):
        """Without a limits file: subfield == SP writes, and the message says why."""
        addresses = ["SR:MAG:DIPOLE:01:CURRENT:SP", "SR:VAC:VALVE:01:CONTROL:OPEN"]

        annotated, report = assign_directions(
            _model(addresses),
            None,
            lookup_message="no limits file found at /tmp/absent.json",
        )

        assert _directions(annotated) == {
            ("DIPOLE", "CURRENT", "SP"): DIRECTION_WRITE,
            ("VALVE", "CONTROL", "OPEN"): DIRECTION_READ,
        }
        assert report.source is DirectionSource.GRAMMAR
        assert report.limits_path is None
        assert "PV grammar" in report.message
        assert "/tmp/absent.json" in report.message

    def test_mixed_group_raises_naming_the_group_and_dissenters(self, tmp_path):
        """Two devices, one group, disagreeing limits: refuse and say who dissents."""
        addresses = ["SR:MAG:DIPOLE:01:CURRENT:SP", "SR:MAG:DIPOLE:02:CURRENT:SP"]
        limits = _write_limits(
            tmp_path / "limits.json",
            {
                "SR:MAG:DIPOLE:01:CURRENT:SP": {},
                "SR:MAG:DIPOLE:02:CURRENT:SP": {"writable": False},
            },
        )

        with pytest.raises(DirectionConflictError) as excinfo:
            assign_directions(_model(addresses), limits)

        error = excinfo.value
        assert error.group_key == ("DIPOLE", "CURRENT", "SP")
        assert set(error.dissenting) | set(error.by_direction[error.majority]) == set(addresses)
        message = str(error)
        assert "DIPOLE" in message and "CURRENT" in message and "SP" in message
        assert error.dissenting[0] in message

    def test_conflict_message_caps_the_dissenter_list(self, tmp_path):
        """A wide conflict names ten channels and counts the rest."""
        writable = [f"SR:MAG:DIPOLE:{index:02d}:CURRENT:SP" for index in range(1, 21)]
        blocked = [f"SR:MAG:DIPOLE:{index:02d}:CURRENT:SP" for index in range(21, 33)]
        limits = _write_limits(
            tmp_path / "limits.json",
            {**{address: {} for address in writable}, **{a: {"writable": False} for a in blocked}},
        )

        with pytest.raises(DirectionConflictError) as excinfo:
            assign_directions(_model(writable + blocked), limits)

        error = excinfo.value
        assert error.majority == DIRECTION_WRITE
        assert error.dissenting == tuple(sorted(blocked))
        assert "+2 more" in str(error)

    def test_the_receiver_model_is_not_mutated(self, tmp_path):
        """with_directions is non-mutating and this pass must stay that way."""
        model = _model(["SR:MAG:DIPOLE:01:CURRENT:SP"])
        limits = _write_limits(tmp_path / "limits.json", {"SR:MAG:DIPOLE:01:CURRENT:SP": {}})

        annotated, _report = assign_directions(model, limits)

        assert _directions(model) == {("DIPOLE", "CURRENT", "SP"): None}
        assert _directions(annotated) == {("DIPOLE", "CURRENT", "SP"): DIRECTION_WRITE}


# ---------------------------------------------------------------------------
# resolve_and_assign
# ---------------------------------------------------------------------------


class TestResolveAndAssign:
    """The one call the CLI verb makes."""

    def test_chains_resolution_into_assignment(self, tmp_path, no_ambient_config):
        """A configured, existing limits file produces a limits-sourced report."""
        limits = _write_limits(tmp_path / "limits.json", {"SR:MAG:DIPOLE:01:CURRENT:SP": {}})

        annotated, report = resolve_and_assign(
            _model(["SR:MAG:DIPOLE:01:CURRENT:SP"]),
            None,
            config_lookup=_lookup({LIMITS_DATABASE_CONFIG_KEY: str(limits)}),
        )

        assert report.source is DirectionSource.LIMITS
        assert _directions(annotated) == {("DIPOLE", "CURRENT", "SP"): DIRECTION_WRITE}

    def test_unresolvable_config_degrades_to_the_grammar(self, no_ambient_config):
        """No configured file: grammar, with the key named in the message."""
        annotated, report = resolve_and_assign(
            _model(["SR:MAG:DIPOLE:01:CURRENT:SP"]),
            None,
            config_lookup=_lookup({}),
        )

        assert report.source is DirectionSource.GRAMMAR
        assert LIMITS_DATABASE_CONFIG_KEY in report.message
        assert _directions(annotated) == {("DIPOLE", "CURRENT", "SP"): DIRECTION_WRITE}

    def test_explicit_missing_path_still_hard_errors(self, tmp_path):
        """The hard error survives the convenience wrapper."""
        with pytest.raises(LimitsFileMissing):
            resolve_and_assign(_model(["SR:MAG:DIPOLE:01:CURRENT:SP"]), tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Real shipped data
# ---------------------------------------------------------------------------


class TestShippedDemoData:
    """The invariant this task exists to assert, checked against the real files."""

    @pytest.fixture(scope="class")
    def demo_model(self):
        database = HierarchicalChannelDatabase(str(_TIER3_HIERARCHICAL))
        return build_model(database.channel_map)

    def test_limits_direction_matches_the_setpoint_set_exactly(self, demo_model):
        """Exactly 396 channels write, and every one of them is a ``:SP`` address."""
        annotated, report = assign_directions(demo_model, _CHANNEL_LIMITS)

        write_addresses = [
            address
            for group in annotated.signal_groups
            if group.direction == DIRECTION_WRITE
            for address in group.members
        ]

        assert report.source is DirectionSource.LIMITS
        assert report.limits_path == _CHANNEL_LIMITS
        assert len(write_addresses) == _EXPECTED_WRITABLE
        assert all(address.endswith(":SP") for address in write_addresses)
        read_addresses = [
            address
            for group in annotated.signal_groups
            if group.direction == DIRECTION_READ
            for address in group.members
        ]
        assert len(write_addresses) + len(read_addresses) == len(annotated.bindings)

    def test_valve_control_commands_model_as_read_only(self, demo_model):
        """The 24 CONTROL:OPEN/CLOSE channels read as commands but are not writable."""
        annotated, _report = assign_directions(demo_model, _CHANNEL_LIMITS)
        by_address = {
            address: group.direction
            for group in annotated.signal_groups
            for address in group.members
        }

        valve_commands = [
            address
            for address in by_address
            if address.startswith("SR:VAC:VALVE:")
            and (address.endswith(":CONTROL:OPEN") or address.endswith(":CONTROL:CLOSE"))
        ]

        assert len(valve_commands) == 24
        assert {by_address[address] for address in valve_commands} == {DIRECTION_READ}

    def test_every_group_has_a_direction_and_none_conflict(self, demo_model):
        """No group is left undecided — a conflict would have raised instead."""
        annotated, _report = assign_directions(demo_model, _CHANNEL_LIMITS)

        assert annotated.signal_groups
        assert all(
            group.direction in {DIRECTION_READ, DIRECTION_WRITE}
            for group in annotated.signal_groups
        )

    def test_grammar_fallback_reproduces_the_limits_directions(self, demo_model):
        """Writable-iff-``:SP`` — asserted over the shipped data, not assumed."""
        from_limits, _limits_report = assign_directions(demo_model, _CHANNEL_LIMITS)
        from_grammar, grammar_report = assign_directions(demo_model, None)

        assert grammar_report.source is DirectionSource.GRAMMAR
        assert _directions(from_grammar) == _directions(from_limits)

    def test_shipped_limits_file_declares_the_expected_writable_count(self):
        """The 396 is a property of the committed file, pinned here directly."""
        writable = load_writable_set(_CHANNEL_LIMITS)

        assert len(writable) == _EXPECTED_WRITABLE
        assert all(address.endswith(":SP") for address in writable)
