"""Tests for the control-system LimitsValidator.

Covers two contracts:

1. `defaults` block inheritance - a channel inherits values from the top-level
   `defaults` block unless it overrides them, including the safety-critical
   `writable` lockdown and the per-channel `confirm` write policy.
2. Fail-closed invariant - a limits violation always raises; there is no policy
   value that turns enforcement off. (`on_violation` was removed as a knob; this
   guards against re-introducing a fail-open path.)
"""

import json
import sys
from pathlib import Path

import pytest

from osprey.connectors.control_system.limits_validator import (
    DEFAULTS_FIELD,
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError
from osprey_connectors.types import EPICS, VIRTUAL_ACCELERATOR, LimitsPosture


def _make_validator(tmp_path, db: dict, policy: dict | None = None) -> LimitsValidator:
    """Load a LimitsValidator from an on-disk JSON limits database."""
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(db))
    limits_db, raw_db = LimitsValidator._load_limits_database(str(limits_file))
    return LimitsValidator(limits_db, policy or {"allow_unlisted_channels": False}, raw_db)


# ---------------------------------------------------------------------------
# `defaults` block inheritance
# ---------------------------------------------------------------------------


def test_defaults_writable_lockdown_is_inherited(tmp_path):
    """A channel that omits `writable` inherits `defaults.writable = false`.

    Safety: a defaults-level read-only lockdown must block writes to channels
    that do not re-declare `writable`, instead of silently defaulting to True.
    """
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"writable": False},
            "FOO": {"min_value": 0.0, "max_value": 10.0},  # omits `writable`
        },
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 5.0)

    assert exc.value.violation_type == "READ_ONLY_CHANNEL"


def test_channel_writable_overrides_defaults(tmp_path):
    """A channel may override `defaults.writable = false` with its own `true`."""
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"writable": False},
            "FOO": {"writable": True, "min_value": 0.0, "max_value": 10.0},
        },
    )

    # Should not raise: channel's explicit writable=True wins over defaults.
    validator.validate("FOO", 5.0)


def test_defaults_min_max_are_inherited(tmp_path):
    """A channel omitting `max_value` inherits the `defaults` bound."""
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"min_value": 0.0, "max_value": 10.0},
            "FOO": {},  # inherits both bounds
        },
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 999.0)

    assert exc.value.violation_type == "MAX_EXCEEDED"


def test_defaults_confirm_is_inherited(tmp_path):
    """A channel omitting `confirm` inherits the `defaults` block's `confirm`.

    The defaults value is deliberately `false`, the opposite of the fleet
    default, so the test proves true inheritance rather than a coincidental
    fallback.
    """
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"confirm": False},
            "FOO": {"min_value": 0.0, "max_value": 100.0},  # omits `confirm`
        },
    )

    assert validator.resolve_confirm("FOO") is False


# ---------------------------------------------------------------------------
# Fail-closed invariant (on_violation removed as a behavioral knob)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("on_violation_value", ["skip", "error", "warn", None])
def test_violation_always_raises_regardless_of_policy(tmp_path, on_violation_value):
    """A limits violation always raises; no policy value makes it fail-open.

    `on_violation` was never honoured for control flow and has been removed as a
    config knob. This locks the fail-closed invariant: even if a policy dict
    carries a legacy `on_violation` value, enforcement still blocks.
    """
    validator = _make_validator(
        tmp_path,
        {"FOO": {"min_value": 0.0, "max_value": 10.0}},
        policy={"allow_unlisted_channels": False, "on_violation": on_violation_value},
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 999.0)  # exceeds max

    assert exc.value.violation_type == "MAX_EXCEEDED"


# ---------------------------------------------------------------------------
# resolve_database_path
# ---------------------------------------------------------------------------


class TestResolveDatabasePath:
    def test_absolute_path_is_returned_unchanged(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        abs_path = str(Path("/etc/osprey/limits.json"))

        assert LimitsValidator.resolve_database_path(abs_path, "/some/root") == abs_path

    def test_config_file_directory_wins(self, monkeypatch):
        """A relative path resolves against CONFIG_FILE's directory when set."""
        monkeypatch.setenv("CONFIG_FILE", "/app/project/config.yml")

        resolved = LimitsValidator.resolve_database_path("limits.json", "/host/build/path")

        assert resolved == str(Path("/app/project/limits.json"))

    def test_project_root_fallback_when_no_config_file(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        resolved = LimitsValidator.resolve_database_path("limits.json", "/proj/root")

        assert resolved == str(Path("/proj/root/limits.json"))

    def test_relative_path_unchanged_when_no_bases(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        assert LimitsValidator.resolve_database_path("limits.json", None) == "limits.json"

    def test_loaded_config_directory_beats_env_and_project_root(self, monkeypatch):
        """The config actually loaded is the anchor, ahead of CONFIG_FILE and project_root."""
        monkeypatch.setenv("CONFIG_FILE", "/somewhere/else/config.yml")

        resolved = LimitsValidator.resolve_database_path(
            "data/limits.json", "/repo", config_path="/repo/build/config.yml"
        )

        assert resolved == str(Path("/repo/build/data/limits.json"))

    def test_loaded_config_anchor_ignores_absolute_paths(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        abs_path = str(Path("/etc/osprey/limits.json"))

        resolved = LimitsValidator.resolve_database_path(
            abs_path, "/repo", config_path="/repo/build/config.yml"
        )

        assert resolved == abs_path


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def _nested_control_system(values: dict) -> dict:
    """The nested ``control_system:`` mapping the dotted keys in *values* spell.

    The real config singleton answers both spellings of one line: a dotted
    ``get_config_value("control_system.limits_checking.enabled")`` and the
    nested section ``get_config_value("control_system")`` read the same YAML.
    ``from_config`` now asks for the section — the posture resolvers walk it —
    while the database path is still read dotted, so a shim that only answered
    the dotted keys would hand the resolvers an empty deployment. Deriving the
    section from the same map keeps the two spellings describing one config
    instead of two, which is what the tests below are written against.

    Nesting on every dot is only correct because these keys name built-in
    leaves. A ``control_system.connector.*`` key would be nested on the dots of
    the connector type too — ``mypackage.TangoConnector`` becoming two levels —
    which is exactly the mistake the resolvers refuse to make, so a test that
    needs a per-type block passes ``control_system`` itself rather than adding
    one here.
    """
    section: dict = {}
    for key, value in values.items():
        if not key.startswith("control_system."):
            continue
        if key.startswith("control_system.connector."):
            pytest.fail(
                f"{key!r} cannot be spelled dotted in this shim: a connector type may "
                "itself contain dots, and nesting on every dot would build a section no "
                "config ever has. Pass an explicit 'control_system' entry instead."
            )
        node = section
        parts = key.split(".")[1:]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return section


def _patch_config(
    monkeypatch,
    values: dict,
    raise_exc: Exception | None = None,
    config_path: str | None = None,
):
    """Patch get_config_value with a key->value map (default fallback otherwise).

    Dotted ``control_system.*`` entries also answer the nested
    ``control_system`` section, so one map describes one deployment. A test that
    needs a section the dotted spelling cannot reach — a per-type connector
    block, or a section that is present but empty — passes ``control_system``
    itself as an entry, and that value is returned verbatim.

    ``config_path`` stands in for ``default_config_path()`` — the path of the
    config the singleton actually loaded. ``None`` (the default) models a
    process whose config came from somewhere the singleton can't name, which
    keeps the env-var/project_root fallback branches testable in isolation.
    """
    if "control_system" in values:
        section = values["control_system"]
    else:
        section = _nested_control_system(values) or None

    def fake_get_config_value(key, default=None):
        if raise_exc is not None:
            raise raise_exc
        if key == "control_system" and section is not None:
            return section
        return values.get(key, default)

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
    monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: config_path)


class TestFromConfig:
    def test_returns_none_when_disabled(self, monkeypatch):
        _patch_config(monkeypatch, {"control_system.limits_checking.enabled": False})

        assert LimitsValidator.from_config() is None

    @pytest.mark.parametrize("value", ["true", 1, "${OSPREY_LIMITS_ENABLED}"])
    def test_unreadable_enabled_blocks_rather_than_switching_checking_off(
        self, monkeypatch, tmp_path, value
    ):
        """A deployment-wide ``enabled`` nobody can read blocks every write.

        The dangerous reading is the quiet one. ``enabled`` unset means "this
        deployment configured no limits checking", so a leaf written as
        ``'true'``, as ``1``, or as an environment variable nothing expanded —
        expansion always yields strings — would otherwise return no validator
        at all, and every write would go unchecked on a deployment whose config
        says, in the only spelling its author knew, that checking is on.
        """
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": value,
                "control_system.limits_checking.database_path": str(db_file),
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        assert validator.limits == {}
        assert validator.failsafe_reason == (
            "control_system.limits_checking does not state enabled as true/false"
        )
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"

    def test_a_leaf_the_deployment_wide_block_never_carried_still_disables(self, monkeypatch):
        """Absent is not unreadable: a deployment that said nothing gets no validator.

        The compatibility half of the same rule — every deployment that never
        wrote a limits block must keep behaving as it did, or the fail-closed
        reading above would block the fleet instead of the typo.
        """
        _patch_config(monkeypatch, {"control_system.limits_checking.database_path": "/x.json"})

        assert LimitsValidator.from_config() is None

    def test_missing_db_path_yields_blocking_failsafe_validator(self, monkeypatch):
        """Enabled but no database path -> an empty validator that blocks all writes.

        The refusal must say the database is unavailable, not that the channel
        is unlisted — an agent reading "not in limits database" narrates a data
        problem when the actual failure is configuration (#636).
        """
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": None,
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"
        assert "failsafe" in exc.value.violation_reason

    def test_unreadable_db_failsafe_is_distinguishable_from_unlisted_channel(
        self, monkeypatch, tmp_path
    ):
        """A database that fails to load blocks with a load-failure refusal (#636)."""
        db_file = tmp_path / "limits.json"
        db_file.write_text("{not valid json")
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"
        assert "not in limits database" not in exc.value.violation_reason

    def test_genuinely_unlisted_channel_keeps_the_unlisted_refusal(self, monkeypatch, tmp_path):
        """A loaded database still refuses unlisted channels with the unlisted message."""
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
            },
        )

        validator = LimitsValidator.from_config()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:LISTED", 1.0)
        assert exc.value.violation_type == "UNLISTED_CHANNEL"

    def test_loads_absolute_database(self, monkeypatch, tmp_path):
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
                "project_root": None,
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        assert "FOO" in validator.limits

    def test_resolves_relative_path_against_project_root(self, monkeypatch, tmp_path):
        """A relative database_path is resolved via project_root (debug branch)."""
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        (tmp_path / "limits.json").write_text(json.dumps({"BAR": {"max_value": 5.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "limits.json",
                "project_root": str(tmp_path),
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert "BAR" in validator.limits

    def test_resolves_relative_path_against_config_file(self, monkeypatch, tmp_path):
        """CONFIG_FILE's directory takes priority for relative paths (debug branch)."""
        (tmp_path / "limits.json").write_text(json.dumps({"BAZ": {"max_value": 5.0}}))
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "limits.json",
                "project_root": "/nonexistent/host/path",
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert "BAZ" in validator.limits

    def test_returns_none_when_config_unavailable(self, monkeypatch):
        _patch_config(monkeypatch, {}, raise_exc=RuntimeError("no config"))

        assert LimitsValidator.from_config() is None

    def test_four_zone_hook_resolves_database_beside_loaded_config(self, monkeypatch, tmp_path):
        """Regression (#636): a relative database_path anchors on the config actually loaded.

        The four-zone hook scenario: the render lives at <repo>/build (config +
        data/ side by side), the config's ``project_root`` names <repo>, and the
        hook process has no CONFIG_FILE in its environment. The database must be
        found next to the loaded config — not resolved against project_root,
        where it does not exist and every write would hit the empty-DB failsafe.
        """
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        build = tmp_path / "build"
        (build / "data").mkdir(parents=True)
        (build / "data" / "channel_limits.json").write_text(
            json.dumps({"SR:C1:HCM:SP": {"min_value": -1.0, "max_value": 1.0}})
        )
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "data/channel_limits.json",
                "project_root": str(tmp_path),
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
            config_path=str(build / "config.yml"),
        )

        validator = LimitsValidator.from_config()

        assert "SR:C1:HCM:SP" in validator.limits
        validator.validate("SR:C1:HCM:SP", 0.5)  # listed and in range: allowed


DEPLOYMENT_WIDE_ALLOW_KEY = "control_system.limits_checking.allow_unlisted_channels"
VA_ALLOW_KEY = (
    "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
)


def _limits_db(tmp_path) -> Path:
    """A one-channel limits database, so a validator loads rather than failsafes."""
    db_file = tmp_path / "limits.json"
    db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
    return db_file


def _va_permissive_section() -> dict:
    """Deployment-wide strict, virtual accelerator relaxed in a block of its own.

    The configuration the feature exists for: ``live`` (EPICS here) keeps the
    deployment-wide refusal of unlisted channels while the simulator relaxes it,
    and both connector blocks are present so all three call shapes resolve.
    """
    return {
        "type": VIRTUAL_ACCELERATOR,
        "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
        "connector": {
            EPICS: {"gateway_address": "live.example"},
            VIRTUAL_ACCELERATOR: {
                "gateway_address": "va.example",
                "limits_checking": {"enabled": True, "allow_unlisted_channels": True},
            },
        },
    }


class TestFromConfigCallShapes:
    """The three shapes ``from_config`` answers, and the one it refuses.

    No argument is the deployment-wide question every caller asked before per-type
    blocks existed; ``connector_type=`` is what a connector holding its own type
    asks; ``target=`` is what a tool, hook or roster following the session's target
    asks. Passing both states the posture twice, and is a caller bug rather than a
    posture.
    """

    def test_no_arg_returns_none_when_the_section_is_absent(self, monkeypatch):
        """No ``control_system:`` at all is a deployment that stated no posture."""
        _patch_config(monkeypatch, {"project_root": None})

        assert LimitsValidator.from_config() is None

    def test_no_arg_returns_none_when_the_section_is_empty(self, monkeypatch):
        """An empty section states no ``enabled``, which is not a ``true``."""
        _patch_config(monkeypatch, {"control_system": {}})

        assert LimitsValidator.from_config() is None

    def test_no_arg_reads_the_deployment_wide_block(self, monkeypatch, tmp_path):
        """A deployment with only the deployment-wide block behaves as it always has."""
        _patch_config(
            monkeypatch,
            {
                "control_system": {
                    "limits_checking": {"enabled": True, "allow_unlisted_channels": True}
                },
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config()

        assert validator.policy["allow_unlisted_channels"] is True
        assert validator.policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY

    def test_no_arg_does_not_pick_up_a_per_type_block(self, monkeypatch, tmp_path):
        """The targetless question is deployment-wide and resolves no target.

        ``registry/manager.py`` and the executor's config helper call
        ``from_config()`` holding neither a type nor a target; folding the
        simulator's relaxation into their answer would report a posture no
        machine of theirs runs under.
        """
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config()

        assert validator.policy["allow_unlisted_channels"] is False
        assert validator.policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY

    def test_connector_type_reads_that_type_s_block(self, monkeypatch, tmp_path):
        """A connector asks about its own type and gets its own block's answer."""
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config(connector_type=VIRTUAL_ACCELERATOR)

        assert validator.policy["allow_unlisted_channels"] is True
        assert validator.policy["allow_unlisted_key"] == VA_ALLOW_KEY

    def test_connector_type_without_a_block_inherits_deployment_wide(self, monkeypatch, tmp_path):
        """The live machine wrote no block, so the deployment-wide one answers for it."""
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config(connector_type=EPICS)

        assert validator.policy["allow_unlisted_channels"] is False
        assert validator.policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY

    def test_target_reads_the_block_of_the_type_it_resolves_to(self, monkeypatch, tmp_path):
        """``va`` resolves to ``virtual_accelerator``, so the two shapes agree."""
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config(target="va")

        assert validator.policy["allow_unlisted_channels"] is True
        assert validator.policy["allow_unlisted_key"] == VA_ALLOW_KEY

    def test_target_live_keeps_the_deployment_wide_refusal(self, monkeypatch, tmp_path):
        """The relaxation written for the simulator must not reach the machine."""
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config(target="live")

        assert validator.policy["allow_unlisted_channels"] is False
        assert validator.policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:LISTED", 1.0)
        assert exc.value.violation_type == "UNLISTED_CHANNEL"

    def test_both_arguments_is_a_type_error(self, monkeypatch):
        """A target already names a type; stating both leaves the caller's intent unclear."""
        _patch_config(monkeypatch, {"control_system": _va_permissive_section()})

        with pytest.raises(TypeError):
            LimitsValidator.from_config(connector_type=VIRTUAL_ACCELERATOR, target="va")

    def test_both_arguments_raise_even_when_the_config_is_unreadable(self, monkeypatch):
        """The caller bug is not swallowed by the config-unavailable envelope.

        The executor wraps its load in a blanket ``except``; a ``TypeError`` that
        the missing-config branch turned into ``None`` would leave that call site
        silently unchecked instead of failing loudly in its own tests.
        """
        _patch_config(monkeypatch, {}, raise_exc=RuntimeError("no config"))

        with pytest.raises(TypeError):
            LimitsValidator.from_config(connector_type=EPICS, target="live")

    def test_incomplete_per_type_block_yields_the_blocking_failsafe(self, monkeypatch, tmp_path):
        """A half-written block answers nothing, so the write path blocks.

        The build refuses such a config, but a hand-edited or older render can
        still carry one and it still has to reach a machine safely.
        """
        section = _va_permissive_section()
        section["connector"][VIRTUAL_ACCELERATOR]["limits_checking"] = {"enabled": True}
        _patch_config(
            monkeypatch,
            {
                "control_system": section,
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config(target="va")

        assert "allow_unlisted_channels" in validator.failsafe_reason
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"


class TestFromConfigMostRestrictive:
    """The posture a caller with no target of its own has to assume.

    The stdlib hook's fallback: with the session's control target unreadable, the
    machine the write is about could be any of the reachable ones, so the answer
    holds across all of them and names the deployment-wide keys — no per-type
    line decides a union.
    """

    def test_one_strict_target_makes_the_answer_strict(self, monkeypatch, tmp_path):
        """The relaxed simulator does not relax what a targetless caller may assume."""
        _patch_config(
            monkeypatch,
            {
                "control_system": _va_permissive_section(),
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config_most_restrictive()

        assert validator.policy == {
            "allow_unlisted_channels": False,
            "allow_unlisted_key": DEPLOYMENT_WIDE_ALLOW_KEY,
        }

    def test_all_permissive_targets_answer_permissive(self, monkeypatch, tmp_path):
        """Not a fail-closed constant: it reports what every reachable target wrote."""
        section = _va_permissive_section()
        section["limits_checking"] = {"enabled": True, "allow_unlisted_channels": True}
        _patch_config(
            monkeypatch,
            {
                "control_system": section,
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config_most_restrictive()

        assert validator.policy["allow_unlisted_channels"] is True
        assert validator.policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY

    @pytest.mark.parametrize("value", ["true", 1, "${OSPREY_LIMITS_ENABLED}"])
    def test_an_unreadable_reachable_leaf_yields_the_failsafe_and_not_none(
        self, monkeypatch, tmp_path, value
    ):
        """One reachable machine with an unreadable block blocks the whole fold.

        This is the branch that must not fail open. Both leaf folds send an
        incomplete posture's `None` to `False`, so a fold that dropped the
        incompleteness would answer "checking off" -- and no validator at all is
        built for that, leaving the caller with the least information about
        which machine it is touching the one waved through.
        """
        section = _va_permissive_section()
        section["limits_checking"] = {"enabled": value, "allow_unlisted_channels": False}
        _patch_config(
            monkeypatch,
            {
                "control_system": section,
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        validator = LimitsValidator.from_config_most_restrictive()

        assert isinstance(validator, LimitsValidator)
        assert validator.limits == {}
        assert validator.failsafe_reason == (
            "control_system.limits_checking does not state enabled as true/false"
        )
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"

    def test_limits_disabled_everywhere_is_no_validator(self, monkeypatch, tmp_path):
        """No reachable target checks limits, so there is nothing to enforce."""
        section = _va_permissive_section()
        section["limits_checking"] = {"enabled": False, "allow_unlisted_channels": False}
        section["connector"][VIRTUAL_ACCELERATOR]["limits_checking"] = {
            "enabled": False,
            "allow_unlisted_channels": True,
        }
        _patch_config(
            monkeypatch,
            {
                "control_system": section,
                "control_system.limits_checking.database_path": str(_limits_db(tmp_path)),
            },
        )

        assert LimitsValidator.from_config_most_restrictive() is None

    def test_returns_none_when_config_unavailable(self, monkeypatch):
        """Same envelope as ``from_config``: no config, no checking."""
        _patch_config(monkeypatch, {}, raise_exc=RuntimeError("no config"))

        assert LimitsValidator.from_config_most_restrictive() is None


# ---------------------------------------------------------------------------
# _from_posture
# ---------------------------------------------------------------------------


class TestFromPosture:
    """Building a validator from an already-resolved :class:`LimitsPosture`.

    ``from_config`` reads the deployment-wide block and nothing else.
    ``_from_posture`` takes the posture a caller resolved for the target or
    connector type it is acting on, so a deployment that relaxed unlisted
    channels for its simulator alone gets the simulator's answer on the
    simulator's write path — and carries the key that answered it, so a refusal
    names the line an operator can actually edit.

    The database path stays deployment-wide: one file is mounted per
    deployment, so every posture loads the same database and differs only in
    policy.
    """

    def test_returns_none_when_posture_says_disabled(self, monkeypatch):
        """``enabled: false`` is limits checking off — no validator at all."""
        _patch_config(monkeypatch, {})
        posture = LimitsPosture(enabled=False, allow_unlisted=True, connector_type=EPICS)

        assert LimitsValidator._from_posture(posture) is None

    def test_returns_none_when_posture_is_unstated(self, monkeypatch):
        """An unstated ``enabled`` is not a ``true`` — same answer as ``false``."""
        _patch_config(monkeypatch, {})
        posture = LimitsPosture(enabled=None, allow_unlisted=None, connector_type=None)

        assert LimitsValidator._from_posture(posture) is None

    def test_deployment_wide_posture_names_the_deployment_wide_key(self, monkeypatch, tmp_path):
        """No connector type in the posture -> the deployment-wide key answers."""
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=None)

        validator = LimitsValidator._from_posture(posture)

        assert validator.policy == {
            "allow_unlisted_channels": False,
            "allow_unlisted_key": "control_system.limits_checking.allow_unlisted_channels",
        }
        assert "FOO" in validator.limits

    def test_per_type_posture_names_the_per_type_key(self, monkeypatch, tmp_path):
        """A per-type block answered, so the per-type key is what a refusal quotes.

        Naming the deployment-wide key here would send an operator to flip a
        line this block overrides — a change that does nothing.
        """
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(
            enabled=True, allow_unlisted=True, connector_type=VIRTUAL_ACCELERATOR
        )

        validator = LimitsValidator._from_posture(posture)

        assert validator.policy["allow_unlisted_channels"] is True
        assert validator.policy["allow_unlisted_key"] == (
            "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
        )
        validator.validate("NOT:LISTED", 1.0)  # permissive posture allows unlisted

    def test_strict_posture_still_refuses_unlisted_channels(self, monkeypatch, tmp_path):
        """The permissive answer belongs to one type; another type stays strict."""
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=EPICS)

        validator = LimitsValidator._from_posture(posture)

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:LISTED", 1.0)
        assert exc.value.violation_type == "UNLISTED_CHANNEL"

    def test_unstated_allow_unlisted_refuses_unlisted_channels(self, monkeypatch, tmp_path):
        """Tri-state: ``None`` is nobody's permission, so unlisted stays refused.

        The value is carried into the policy verbatim (``channel_limits``
        reports it as ``null``) rather than collapsed to ``False``, but the
        write path allows only on an explicit ``True``.
        """
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(enabled=True, allow_unlisted=None, connector_type=None)

        validator = LimitsValidator._from_posture(posture)

        assert validator.policy["allow_unlisted_channels"] is None
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:LISTED", 1.0)
        assert exc.value.violation_type == "UNLISTED_CHANNEL"

    def test_incomplete_block_yields_failsafe_naming_block_and_leaf(self, monkeypatch, tmp_path):
        """A half-written per-type block blocks every write and says which line is missing.

        The build refuses such a block, but a hand-edited deployed config, an
        older render or a fixture can still carry one. The posture then answered
        nothing, so guessing which half was meant is not available: block
        everything and name the block plus the missing leaf.
        """
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(
            enabled=None,
            allow_unlisted=None,
            connector_type=VIRTUAL_ACCELERATOR,
            incomplete=("allow_unlisted_channels",),
        )

        validator = LimitsValidator._from_posture(posture)

        assert isinstance(validator, LimitsValidator)
        assert validator.limits == {}
        assert validator.failsafe_reason == (
            "control_system.connector.virtual_accelerator.limits_checking "
            "does not state allow_unlisted_channels as true/false"
        )
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"

    def test_incomplete_block_names_every_missing_leaf(self, monkeypatch):
        """Both leaves missing under a present block: both are named."""
        _patch_config(monkeypatch, {})
        posture = LimitsPosture(
            enabled=None,
            allow_unlisted=None,
            connector_type="mypackage.TangoConnector",
            incomplete=("enabled", "allow_unlisted_channels"),
        )

        validator = LimitsValidator._from_posture(posture)

        assert validator.failsafe_reason == (
            "control_system.connector.mypackage.TangoConnector.limits_checking "
            "does not state enabled, allow_unlisted_channels as true/false"
        )

    def test_missing_database_path_yields_blocking_failsafe(self, monkeypatch):
        """Enabled with nowhere to read limits from -> block, don't wave writes through."""
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": None},
        )
        posture = LimitsPosture(enabled=True, allow_unlisted=True, connector_type=EPICS)

        validator = LimitsValidator._from_posture(posture)

        assert isinstance(validator, LimitsValidator)
        assert "database_path" in validator.failsafe_reason
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"

    def test_unreadable_database_yields_blocking_failsafe(self, monkeypatch, tmp_path):
        """An unparseable database blocks even a permissive posture."""
        db_file = tmp_path / "limits.json"
        db_file.write_text("{not valid json")
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(
            enabled=True, allow_unlisted=True, connector_type=VIRTUAL_ACCELERATOR
        )

        validator = LimitsValidator._from_posture(posture)

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"

    def test_database_path_is_read_deployment_wide(self, monkeypatch, tmp_path):
        """A per-type posture reads the one deployment-wide database path.

        There is no per-type ``database_path``: the deployment mounts a single
        limits file, so the per-type block changes policy only.
        """
        (tmp_path / "limits.json").write_text(json.dumps({"BAR": {"max_value": 5.0}}))
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.database_path": "limits.json",
                "project_root": str(tmp_path),
            },
        )
        posture = LimitsPosture(
            enabled=True, allow_unlisted=False, connector_type=VIRTUAL_ACCELERATOR
        )

        validator = LimitsValidator._from_posture(posture)

        assert "BAR" in validator.limits

    def test_policy_is_json_serialisable(self, monkeypatch, tmp_path):
        """The executor embeds ``policy`` into the sandbox as JSON verbatim.

        A posture value that does not survive ``json.dumps`` would break the
        sandbox's rebuilt validator, so the tri-state must ride as ``null``
        rather than as anything richer.
        """
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {"control_system.limits_checking.database_path": str(db_file)},
        )
        posture = LimitsPosture(
            enabled=True, allow_unlisted=None, connector_type=VIRTUAL_ACCELERATOR
        )

        validator = LimitsValidator._from_posture(posture)

        assert json.loads(json.dumps(validator.policy)) == validator.policy
        assert json.loads(json.dumps(validator.policy))["allow_unlisted_channels"] is None

    def test_returns_none_when_config_unavailable(self, monkeypatch):
        """The config-unavailable envelope is unchanged: no config, no checking."""
        _patch_config(monkeypatch, {}, raise_exc=RuntimeError("no config"))
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=EPICS)

        assert LimitsValidator._from_posture(posture) is None


# ---------------------------------------------------------------------------
# get_limits_config
# ---------------------------------------------------------------------------


class TestGetLimitsConfig:
    def test_known_channel_returns_full_dict(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"FOO": {"min_value": 1.0, "max_value": 9.0, "max_step": 2.0, "writable": True}},
        )

        config = validator.get_limits_config("FOO")

        assert config == {
            "channel_address": "FOO",
            "min_value": 1.0,
            "max_value": 9.0,
            "max_step": 2.0,
            "writable": True,
        }

    def test_unknown_channel_returns_none(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 9.0}})

        assert validator.get_limits_config("MISSING") is None


# ---------------------------------------------------------------------------
# resolve_confirm (channel entry -> defaults -> True)
# ---------------------------------------------------------------------------


class TestResolveConfirm:
    def test_no_raw_db_confirms(self):
        # No policy to read is not a licence to skip the check.
        validator = LimitsValidator({}, {}, raw_db=None)

        assert validator.resolve_confirm("FOO") is True

    def test_channel_entry_wins_over_defaults(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {
                "defaults": {"confirm": True},
                "FOO": {"max_value": 100.0, "confirm": False},
            },
        )

        assert validator.resolve_confirm("FOO") is False

    def test_channel_entry_can_re_enable_what_defaults_switched_off(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {
                "defaults": {"confirm": False},
                "FOO": {"max_value": 100.0, "confirm": True},
            },
        )

        assert validator.resolve_confirm("FOO") is True

    def test_defaults_apply_when_the_channel_is_silent(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"defaults": {"confirm": False}, "FOO": {"max_value": 100.0}},
        )

        assert validator.resolve_confirm("FOO") is False

    def test_silence_everywhere_confirms(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 100.0}})

        assert validator.resolve_confirm("FOO") is True

    def test_unlisted_channel_confirms(self, tmp_path):
        validator = _make_validator(
            tmp_path, {"defaults": {"confirm": True}, "FOO": {"max_value": 100.0}}
        )

        assert validator.resolve_confirm("MISSING") is True

    def test_unlisted_channel_still_inherits_defaults(self, tmp_path):
        validator = _make_validator(
            tmp_path, {"defaults": {"confirm": False}, "FOO": {"max_value": 100.0}}
        )

        assert validator.resolve_confirm("MISSING") is False

    def test_non_dict_defaults_block_is_ignored(self):
        """A malformed (non-dict) defaults block does not crash the lookup."""
        validator = LimitsValidator({}, {}, raw_db={"defaults": "oops", "FOO": {}})

        assert validator.resolve_confirm("FOO") is True

    def test_non_dict_channel_entry_is_ignored(self):
        validator = LimitsValidator({}, {}, raw_db={"defaults": {"confirm": False}, "FOO": "oops"})

        assert validator.resolve_confirm("FOO") is False


# ---------------------------------------------------------------------------
# _validate_channel_config
# ---------------------------------------------------------------------------


class TestValidateChannelConfig:
    def test_unknown_field_raises_and_names_the_key(self):
        with pytest.raises(ValueError, match="bogus_field"):
            LimitsValidator._validate_channel_config("FOO", {"bogus_field": 1})

    def test_underscore_prefixed_metadata_is_accepted(self):
        # Any '_'-prefixed key is documentation, not a closed set of four.
        LimitsValidator._validate_channel_config(
            "FOO", {"max_value": 1.0, "_units": "mA", "_owner": "APG"}
        )

    def test_retired_verification_block_raises_with_the_migration_message(self):
        with pytest.raises(ValueError) as exc:
            LimitsValidator._validate_channel_config("FOO", {"verification": {"level": "none"}})

        assert "'verification' was replaced by 'confirm: true|false'" in str(exc.value)

    def test_non_numeric_bound_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            LimitsValidator._validate_channel_config("FOO", {"min_value": "low"})

    def test_non_bool_writable_raises(self):
        with pytest.raises(ValueError, match="must be boolean"):
            LimitsValidator._validate_channel_config("FOO", {"writable": "yes"})

    def test_non_bool_confirm_raises(self):
        with pytest.raises(ValueError, match="'confirm' must be boolean"):
            LimitsValidator._validate_channel_config("FOO", {"confirm": "yes"})

    def test_bool_confirm_is_accepted(self):
        LimitsValidator._validate_channel_config("FOO", {"confirm": False})


# ---------------------------------------------------------------------------
# _load_limits_database (error + skip branches)
# ---------------------------------------------------------------------------


class TestLoadDatabase:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="database not found"):
            LimitsValidator._load_limits_database(str(tmp_path / "nope.json"))

    def test_non_dict_root_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps(["not", "a", "dict"]))

        with pytest.raises(ValueError, match="must be a JSON object"):
            LimitsValidator._load_limits_database(str(f))

    def test_non_dict_defaults_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({DEFAULTS_FIELD: 5}))

        with pytest.raises(ValueError, match="must be a dictionary"):
            LimitsValidator._load_limits_database(str(f))

    def test_invalid_defaults_config_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({DEFAULTS_FIELD: {"min_value": "not-numeric"}}))

        with pytest.raises(ValueError, match="Invalid 'defaults' configuration"):
            LimitsValidator._load_limits_database(str(f))

    def test_metadata_fields_are_skipped(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"_comment": "ignored metadata", "GOOD": {"max_value": 10.0}}))

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert set(limits_db) == {"GOOD"}

    def test_non_dict_channel_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"BADCHAN": 42, "GOOD": {"max_value": 10.0}}))

        with pytest.raises(ValueError, match="BADCHAN"):
            LimitsValidator._load_limits_database(str(f))

    def test_channel_with_invalid_field_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"BADFIELD": {"min_value": "x"}, "GOOD": {"max_value": 10.0}}))

        with pytest.raises(ValueError, match="BADFIELD"):
            LimitsValidator._load_limits_database(str(f))

    def test_max_step_channel_loads(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"STEP": {"max_value": 100.0, "max_step": 5.0}}))

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert limits_db["STEP"].max_step == 5.0

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text("{ this is not valid json ")

        with pytest.raises(ValueError, match="Invalid JSON"):
            LimitsValidator._load_limits_database(str(f))

    def test_unreadable_path_raises_generic(self, tmp_path):
        # A directory exists but cannot be opened as a file -> generic failure branch.
        with pytest.raises(ValueError, match="Failed to load"):
            LimitsValidator._load_limits_database(str(tmp_path))


# ---------------------------------------------------------------------------
# max_step check (Check 4) — the I/O safety path
# ---------------------------------------------------------------------------


def _step_validator(max_step: float = 5.0) -> LimitsValidator:
    """A validator with one channel that has max_step configured (triggers caget)."""
    limits = {
        "FOO": ChannelLimitsConfig(
            channel_address="FOO", min_value=0.0, max_value=100.0, max_step=max_step
        )
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False}, {})


class TestMaxStepCheck:
    def test_current_none_blocks_write(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: None)
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"

    def test_step_exceeded_blocks_with_details(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: 10.0)
        validator = _step_validator(max_step=5.0)

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 100.0)  # step of 90 >> 5

        assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
        assert exc.value.current_value == 10.0
        assert exc.value.max_step == 5.0

    def test_step_within_limit_passes(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: 48.0)
        validator = _step_validator(max_step=5.0)

        # Step of 2.0 is within max_step=5.0 -> no raise.
        validator.validate("FOO", 50.0)

    def test_non_numeric_current_skips_step_check(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: "not-a-number")
        validator = _step_validator(max_step=1.0)

        # Current value can't be coerced to float -> step check is skipped, write allowed.
        validator.validate("FOO", 50.0)

    def test_missing_pyepics_blocks_write(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "epics", None)  # import epics -> ImportError
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"
        assert "pyepics not available" in exc.value.violation_reason

    def test_caget_error_blocks_write(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("CA timeout")

        monkeypatch.setattr("epics.caget", boom)
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"
        assert "CA timeout" in exc.value.violation_reason


# ---------------------------------------------------------------------------
# validate() — non-step branches (unlisted policy, non-numeric, min bound)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_unlisted_allowed_when_policy_permits(self, tmp_path):
        """allow_unlisted_channels=True lets an unknown channel through."""
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={"allow_unlisted_channels": True},
        )

        # No raise: the channel is unlisted but policy allows it.
        validator.validate("NOT:IN:DB", 5.0)

    def test_non_numeric_value_skips_numeric_checks(self, tmp_path):
        """A non-coercible value skips min/max/step checks rather than crashing."""
        validator = _make_validator(tmp_path, {"FOO": {"min_value": 0.0, "max_value": 10.0}})

        # No raise: "on" can't be floated, so numeric bounds are not applied.
        validator.validate("FOO", "on")

    def test_below_minimum_raises(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"min_value": 0.0, "max_value": 10.0}})

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", -5.0)

        assert exc.value.violation_type == "MIN_EXCEEDED"


# ---------------------------------------------------------------------------
# validate() — the unlisted refusal names the key that answered
# ---------------------------------------------------------------------------


class TestUnlistedRefusalNamesKey:
    """An unlisted-channel refusal quotes the config line an operator can edit.

    A deployment may answer `allow_unlisted_channels` per connector type, so
    the deployment-wide key is not always the one in force: quoting it on a
    deployment whose per-type block overrides it would send an operator to
    flip a line that changes nothing. The validator carries the answering key
    in its policy (`_from_posture`) and the refusal repeats it.
    """

    DEPLOYMENT_WIDE_KEY = "control_system.limits_checking.allow_unlisted_channels"
    PER_TYPE_KEY = (
        "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
    )

    def test_refusal_names_the_per_type_key_the_policy_carries(self, tmp_path):
        """A per-type block answered, so its key is the one worth quoting."""
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={
                "allow_unlisted_channels": False,
                "allow_unlisted_key": self.PER_TYPE_KEY,
            },
        )

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:IN:DB", 5.0)

        assert exc.value.violation_type == "UNLISTED_CHANNEL"
        assert self.PER_TYPE_KEY in exc.value.violation_reason
        assert "not in limits database" in exc.value.violation_reason

    def test_refusal_falls_back_to_the_deployment_wide_key(self, tmp_path):
        """A hand-built validator carries no key — the deployment-wide one answers.

        Validators built outside `_from_posture` (tests, and any caller that
        passes a bare policy dict) keep working and still name a real key.
        """
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={"allow_unlisted_channels": False},
        )

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:IN:DB", 5.0)

        assert exc.value.violation_type == "UNLISTED_CHANNEL"
        assert self.DEPLOYMENT_WIDE_KEY in exc.value.violation_reason

    def test_empty_policy_still_refuses_and_names_the_deployment_wide_key(self, tmp_path):
        """No policy at all is nobody's permission, not a permissive default."""
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 10.0}}, policy={})

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:IN:DB", 5.0)

        assert exc.value.violation_type == "UNLISTED_CHANNEL"
        assert self.DEPLOYMENT_WIDE_KEY in exc.value.violation_reason

    def test_unstated_policy_value_refuses(self, tmp_path):
        """Tri-state: a stored `None` is unset, and unset refuses.

        The posture carries the unstated answer verbatim so `channel_limits`
        can report it as `null`; the write path allows only an explicit `True`.
        """
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={
                "allow_unlisted_channels": None,
                "allow_unlisted_key": self.PER_TYPE_KEY,
            },
        )

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:IN:DB", 5.0)

        assert exc.value.violation_type == "UNLISTED_CHANNEL"
        assert self.PER_TYPE_KEY in exc.value.violation_reason

    @pytest.mark.parametrize("truthy", ["true", 1, "yes"])
    def test_only_a_real_true_allows_unlisted_channels(self, tmp_path, truthy):
        """A truthy non-bool is a config mistake, not permission to write."""
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={"allow_unlisted_channels": truthy},
        )

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:IN:DB", 5.0)

        assert exc.value.violation_type == "UNLISTED_CHANNEL"
