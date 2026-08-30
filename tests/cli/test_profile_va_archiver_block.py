"""The ``va_archiver:`` block — the store's knobs, and what they must agree with.

The block declares where a simulated deployment keeps its history and every
number that store's shape depends on. Two properties are what make it worth a
schema rather than a free-form mapping: the numbers describe one grid, so pairs
that cannot coexist (a dense tier longer than retention, a tail cadence that
does not land on the dense one's timestamps) are refused; and the connector's
connection keys are DERIVED from it, so spelling them again under ``config:``
is refused too — one fact with two homes is free to disagree, and a stale
duplicate reads as an empty archive rather than as a broken one.

The honesty rule's build-time site is here too: declining to declare a store is
a legal choice for most deployments, but a profile that pairs a machine this
deployment stands up for itself — a virtual accelerator or the live stand-in —
with the mock archiver is refused outright, under the name of whichever of the
two it selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_archiver import (
    COMPRESSORS,
    CONNECTION_CONFIG_PREFIX,
    DEFAULT_ARCHIVER_HOST,
    FRESHNESS_CATEGORY,
    FRESHNESS_CHECK_NAME,
    FRESHNESS_CONFIG_KEY,
    FRESHNESS_FLOOR_SEC,
    FRESHNESS_SAFETY_MULTIPLE,
    KNOBS_CONFIG_PREFIX,
    VAArchiverConfig,
    parse_va_archiver_block,
    va_archiver_config_overrides,
    va_archiver_errors,
    va_mock_archiver_errors,
)
from osprey.cli.build_profile_load import _KNOWN_PROFILE_KEYS, load_profile
from osprey.cli.profile_cmd import profile
from osprey.errors import BuildProfileError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_profile(tmp_path: Path, va_archiver: Any, **profile_keys: Any) -> Path:
    body: dict[str, Any] = {"name": "Facility", "data_bundle": "hello_world"}
    body.update(profile_keys)
    if va_archiver is not None:
        body["va_archiver"] = va_archiver
    path = tmp_path / "profile.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _parse_refusal(block: Any) -> str:
    """The accumulated report for a block the parser cannot type."""
    with pytest.raises(BuildProfileError) as excinfo:
        parse_va_archiver_block({"name": "Facility", "va_archiver": block})
    return str(excinfo.value)


def _errors(config: Any = None, deploy_services: bool = True, **knobs: Any) -> list[str]:
    """Semantic errors for a block built from *knobs*."""
    return va_archiver_errors(VAArchiverConfig(**knobs), config or {}, deploy_services)


# ---------------------------------------------------------------------------
# The key is part of the schema, and the block is optional
# ---------------------------------------------------------------------------


def test_va_archiver_is_a_known_profile_key() -> None:
    assert "va_archiver" in _KNOWN_PROFILE_KEYS


def test_a_profile_without_the_block_parses_to_none(tmp_path: Path) -> None:
    """A stored archive is opt-in: without the block the deployment reads
    whatever archiver its config selects."""
    assert load_profile(_write_profile(tmp_path, None)).va_archiver is None


def test_an_empty_block_takes_every_default(tmp_path: Path) -> None:
    """Declaring the block with no keys is how a facility says "the shipped
    store, as shipped" — it must not need to restate the defaults to get them."""
    loaded = load_profile(_write_profile(tmp_path, {}))

    assert loaded.va_archiver == VAArchiverConfig()


def test_declared_knobs_reach_the_parsed_block(tmp_path: Path) -> None:
    loaded = load_profile(
        _write_profile(
            tmp_path,
            {"retention_days": 365, "hot_cadence_sec": 5, "tail_cadence_sec": 30},
        )
    )

    assert loaded.va_archiver is not None
    assert loaded.va_archiver.retention_days == 365
    assert loaded.va_archiver.hot_cadence_sec == 5
    assert loaded.va_archiver.tail_cadence_sec == 30


# ---------------------------------------------------------------------------
# Structure: the block is closed, and its knobs are typed
# ---------------------------------------------------------------------------


def test_a_non_mapping_block_is_refused() -> None:
    assert "must be a mapping" in _parse_refusal(["retention_days"])


def test_an_unknown_key_is_refused_with_the_closest_spelling() -> None:
    """Silently ignoring it would ship a store the facility did not ask for."""
    message = _parse_refusal({"retention_day": 7})

    assert "unknown key 'retention_day'" in message
    assert "did you mean 'retention_days'" in message


def test_a_mistyped_knob_is_refused() -> None:
    assert "'retention_days' must be an integer" in _parse_refusal({"retention_days": "30"})
    assert "'database' must be a string" in _parse_refusal({"database": 7})


def test_a_boolean_is_not_an_integer_knob() -> None:
    """YAML's `retention_days: yes` is a bool, and Python would take it as 1 —
    a month of history quietly becoming a day."""
    assert "'retention_days' must be an integer" in _parse_refusal({"retention_days": True})


def test_every_structural_problem_is_reported_at_once() -> None:
    """The block is a table being transcribed; one problem per rebuild is the
    slowest possible way to fill it in."""
    message = _parse_refusal({"retention_days": "30", "database": 7, "nonsense": 1})

    assert message.count("\n  - ") == 3


# ---------------------------------------------------------------------------
# Ranges and the grid the numbers describe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knob",
    [
        "retention_days",
        "hot_span_hours",
        "hot_cadence_sec",
        "tail_cadence_sec",
        "recorder_cadence_sec",
        "recorder_tail_cadence_sec",
        "recorder_poll_sec",
        "timeout_sec",
    ],
)
def test_counts_must_be_positive(knob: str) -> None:
    assert any(f"va_archiver.{knob} must be >= 1" in error for error in _errors(**{knob: 0}))


def test_port_must_be_a_port() -> None:
    assert any("port_host must be in 1..65535" in error for error in _errors(port_host=70000))


def test_a_dense_tier_longer_than_retention_is_refused() -> None:
    """It would describe dense history older than the oldest history there is."""
    errors = _errors(retention_days=1, hot_span_hours=48)

    assert any("exceeds retention_days" in error for error in errors)


def test_the_tail_cadence_must_land_on_the_dense_grid() -> None:
    """The tail is the sparse subset of the hot grid. A cadence that does not
    divide it puts the tiers on timestamps that never coincide — which surfaces
    as a seam in the data, not as an error."""
    errors = _errors(hot_cadence_sec=10, tail_cadence_sec=45)

    assert any("must be a whole multiple of" in error for error in errors)


def test_the_tail_cannot_sample_faster_than_the_dense_tier() -> None:
    errors = _errors(hot_cadence_sec=60, tail_cadence_sec=10)

    assert any("cannot sample faster than the dense one" in error for error in errors)


def test_the_recorder_cadences_answer_the_same_rule() -> None:
    errors = _errors(recorder_cadence_sec=10, recorder_tail_cadence_sec=45)

    assert any("recorder_tail_cadence_sec" in error for error in errors)


def test_matching_cadences_are_fine() -> None:
    """One tier at both cadences is a legitimate (if dense) configuration."""
    assert _errors(hot_cadence_sec=10, tail_cadence_sec=10) == []


def test_an_unknown_compressor_is_refused() -> None:
    """mongod would refuse it at container start, hours after the profile that
    named it was written."""
    errors = _errors(compression="brotli")

    assert any("compression must be one of" in error for error in errors)


@pytest.mark.parametrize("compressor", COMPRESSORS)
def test_every_supported_compressor_validates(compressor: str) -> None:
    assert _errors(compression=compressor) == []


def test_a_database_name_mongodb_forbids_is_refused() -> None:
    errors = _errors(database="my/archive")

    assert any("forbids in a database name" in error for error in errors)


def test_an_empty_name_is_refused() -> None:
    assert any("collection must be a non-empty" in error for error in _errors(collection="  "))


def test_password_env_must_name_a_variable_not_hold_one() -> None:
    errors = _errors(password_env="hunter 2")

    assert any("must be an environment variable NAME" in error for error in errors)
    assert any("never in the profile" in error for error in errors)


def test_the_defaults_validate() -> None:
    """The shipped store must be a legal store."""
    assert _errors() == []


# ---------------------------------------------------------------------------
# Agreement with the rest of the profile
# ---------------------------------------------------------------------------


def test_an_attached_project_must_name_the_host_it_reads() -> None:
    """It deploys no store of its own, so loopback points at nothing — a
    deploy-time mystery for a build-time omission."""
    errors = _errors(deploy_services=False)

    assert any("va_archiver.host is required when deploy_services is false" in e for e in errors)


def test_an_attached_project_with_a_host_validates() -> None:
    assert _errors(deploy_services=False, host="archive.example.org") == []


def test_a_self_deploying_project_needs_no_host() -> None:
    assert _errors(deploy_services=True) == []


def test_a_blank_host_is_refused() -> None:
    assert any("host must be a non-empty string" in error for error in _errors(host="  "))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({f"{CONNECTION_CONFIG_PREFIX}.host": "elsewhere"}, id="dotted-leaf"),
        pytest.param({CONNECTION_CONFIG_PREFIX: {"host": "elsewhere"}}, id="prefix-subtree"),
        pytest.param(
            {"archiver": {"mongodb_archiver": {"host": "elsewhere"}}}, id="nested-subtree"
        ),
    ],
)
def test_a_connection_key_spelled_in_config_is_refused(config: dict[str, Any]) -> None:
    """All three spellings reach the same rendered leaf, so all three are a
    second home for a fact this block already owns."""
    errors = _errors(config=config)

    assert any("one fact, two homes" in error for error in errors)
    assert any("The va_archiver block is the home" in error for error in errors)


def test_config_may_still_select_the_archiver_type() -> None:
    """Declaring where the archive lives and selecting it as the deployment's
    archiver are separate decisions — only the first belongs to this block."""
    assert _errors(config={"archiver.type": "mongodb_archiver"}) == []


def test_a_profile_without_the_block_is_free_to_write_connection_keys(tmp_path: Path) -> None:
    """The rejection exists because the block derives those keys. With no block
    there is nothing to collide with, and a hand-configured MongoDB archiver
    stays a supported configuration."""
    path = _write_profile(
        tmp_path, None, config={f"{CONNECTION_CONFIG_PREFIX}.host": "archive.example.org"}
    )

    assert load_profile(path).va_archiver is None


# ---------------------------------------------------------------------------
# What the block contributes to the rendered config
# ---------------------------------------------------------------------------


def test_no_block_contributes_nothing() -> None:
    assert va_archiver_config_overrides(None) == {}


def test_the_connector_gets_every_key_it_requires() -> None:
    """The connector raises on the first missing one, so a partial block would
    build cleanly and die at the first archiver call."""
    overrides = va_archiver_config_overrides(VAArchiverConfig())

    assert {
        f"{CONNECTION_CONFIG_PREFIX}.{leaf}"
        for leaf in (
            "host",
            "port",
            "name",
            "collection",
            "auth",
            "username",
            "password_env",
            "timeout",
        )
    } <= set(overrides)


def test_every_archive_knob_reaches_the_services_that_read_it() -> None:
    """FR7: retention and cadence are profile edits rather than code changes,
    which only holds if every one of them is in the rendered config."""
    overrides = va_archiver_config_overrides(VAArchiverConfig(retention_days=2))

    assert overrides[f"{KNOBS_CONFIG_PREFIX}.retention_days"] == 2
    for knob in (
        "hot_span_hours",
        "hot_cadence_sec",
        "tail_cadence_sec",
        "recorder_cadence_sec",
        "recorder_tail_cadence_sec",
        "recorder_poll_sec",
    ):
        assert f"{KNOBS_CONFIG_PREFIX}.{knob}" in overrides


def test_the_derived_keys_are_exactly_the_two_groups() -> None:
    """Nothing else is derived: a key here that no service reads would be a
    config entry the facility can neither find nor change."""
    overrides = va_archiver_config_overrides(VAArchiverConfig())

    assert all(
        key.startswith((f"{CONNECTION_CONFIG_PREFIX}.", f"{KNOBS_CONFIG_PREFIX}."))
        for key in overrides
    )


def test_the_derived_keys_carry_the_blocks_values() -> None:
    overrides = va_archiver_config_overrides(
        VAArchiverConfig(port_host=27100, database="facility", collection="history", timeout_sec=9)
    )

    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.port"] == 27100
    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.name"] == "facility"
    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.collection"] == "history"
    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.timeout"] == 9


def test_a_self_deployed_store_is_reached_on_loopback() -> None:
    """The service publishes its port on the deployment's bind address and the
    agent runs on the host."""
    overrides = va_archiver_config_overrides(VAArchiverConfig())

    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.host"] == DEFAULT_ARCHIVER_HOST


def test_an_attached_project_is_reached_where_it_says() -> None:
    overrides = va_archiver_config_overrides(VAArchiverConfig(host="archive.example.org"))

    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.host"] == "archive.example.org"


def test_the_password_is_named_never_written() -> None:
    overrides = va_archiver_config_overrides(VAArchiverConfig())

    assert overrides[f"{CONNECTION_CONFIG_PREFIX}.password_env"] == "MONGO_ROOT_PASSWORD"


def test_the_archiver_type_is_not_derived() -> None:
    """Flipping the connector out from under a facility's `archiver.type` would
    be making a decision the block was not given."""
    assert "archiver.type" not in va_archiver_config_overrides(VAArchiverConfig())


# ---------------------------------------------------------------------------
# `osprey profile validate` is where a facility meets all of this
# ---------------------------------------------------------------------------


def test_validate_accepts_a_profile_carrying_the_block(runner: CliRunner, tmp_path: Path) -> None:
    path = _write_profile(tmp_path, {"retention_days": 7, "hot_span_hours": 6})
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code == 0, result.output


def test_validate_exits_nonzero_and_names_the_problem(runner: CliRunner, tmp_path: Path) -> None:
    path = _write_profile(tmp_path, {"retention_days": 1, "hot_span_hours": 48})
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code != 0
    assert "hot_span_hours" in result.output


def test_validate_reports_the_blocks_problems_alongside_the_profiles(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Reported from BuildProfile.validate() rather than raised at parse time,
    so a facility meets every problem its profile has in one pass."""
    path = _write_profile(
        tmp_path, {"compression": "brotli"}, tier=2, default_panel="does-not-exist"
    )
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code != 0
    assert "compression" in result.output
    assert "tier" in result.output


# ---------------------------------------------------------------------------
# The honesty rule at build time: a simulated machine may not invent its past
# ---------------------------------------------------------------------------

_VA = "virtual_accelerator"


def _pairing_errors(config: dict[str, Any]) -> list[str]:
    return va_mock_archiver_errors(config)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"control_system.type": _VA, "archiver.type": "mock_archiver"}, id="flat"),
        pytest.param(
            {"control_system": {"type": _VA}, "archiver": {"type": "mock_archiver"}}, id="nested"
        ),
    ],
)
def test_a_virtual_accelerator_reading_the_mock_is_refused(config: dict[str, Any]) -> None:
    """Both spellings are legal YAML and both reach the same rendered config, so
    a refusal that saw one of them would be one a profile could be written
    around."""
    assert len(_pairing_errors(config)) == 1


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"control_system.type": _VA}, id="flat"),
        pytest.param({"control_system": {"type": _VA}}, id="nested"),
    ],
)
def test_saying_nothing_about_the_archiver_is_saying_mock(config: dict[str, Any]) -> None:
    """The connector factory falls back to the mock, so a profile that names no
    archiver has named one."""
    errors = _pairing_errors(config)
    assert len(errors) == 1
    assert "unset" in errors[0]
    assert "mock_archiver" in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"control_system.type": _VA, "archiver.type": "mongodb_archiver"}, id="store"),
        pytest.param(
            {"control_system.type": _VA, "archiver.type": "epics_archiver"}, id="facility"
        ),
        pytest.param({"control_system.type": "mock", "archiver.type": "mock_archiver"}, id="mock"),
        pytest.param(
            {"control_system.type": "epics", "archiver.type": "mock_archiver"}, id="epics"
        ),
        pytest.param({}, id="says-nothing"),
    ],
)
def test_every_other_pairing_is_left_alone(config: dict[str, Any]) -> None:
    """The rule is about a simulation inventing its own past — not about which
    store a facility runs, and not a general ban on the mock archiver."""
    assert _pairing_errors(config) == []


def test_the_refusal_names_both_ways_out() -> None:
    """This feature's standard: a refusal teaches rather than only blocks."""
    message = _pairing_errors({"control_system.type": _VA, "archiver.type": "mock_archiver"})[0]

    assert "va_archiver" in message
    assert "mongodb_archiver" in message
    assert "run yourself" in message


# ---------------------------------------------------------------------------
# The refusal names the machine the profile actually selected
#
# The rule covers every machine a deployment stands up for itself, so a profile
# baselined on the live stand-in is refused here too. A message naming the
# virtual accelerator would send its author to a `config:` key that says
# `live_standin`, and the fix it names would be for a machine they never asked
# for.
# ---------------------------------------------------------------------------

_STANDIN = "live_standin"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {"control_system.type": _STANDIN, "archiver.type": "mock_archiver"}, id="flat"
        ),
        pytest.param(
            {"control_system": {"type": _STANDIN}, "archiver": {"type": "mock_archiver"}},
            id="nested",
        ),
        pytest.param({"control_system.type": _STANDIN}, id="archiver-unset"),
    ],
)
def test_a_stand_in_with_an_invented_past_is_refused_under_its_own_name(
    config: dict[str, Any],
) -> None:
    errors = _pairing_errors(config)

    assert len(errors) == 1
    assert f"control_system.type to {_STANDIN!r}" in errors[0]
    # The shared reason sentence names both machines in prose; the *type* named
    # back must be the one this profile carries.
    assert repr(_VA) not in errors[0]


def test_the_virtual_accelerator_is_still_named_when_it_is_the_invented_machine() -> None:
    """The type is read from the profile rather than assumed, so the case that
    motivated the rule keeps the message it had."""
    errors = _pairing_errors({"control_system.type": _VA, "archiver.type": "mock_archiver"})

    assert f"control_system.type to {_VA!r}" in errors[0]
    assert repr(_STANDIN) not in errors[0]


def test_a_baseline_spelled_two_ways_is_named_by_the_invented_one() -> None:
    """Both spellings are live, and the pairing fails closed on either being a
    machine with no recorded past — so the message names the spelling that was
    refused, not whichever would win at render time."""
    errors = _pairing_errors({"control_system.type": "epics", "control_system": {"type": _STANDIN}})

    assert len(errors) == 1
    assert f"control_system.type to {_STANDIN!r}" in errors[0]


def test_the_pairing_is_reported_with_the_profiles_other_problems(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Accumulated into validate()'s one report, like every other profile rule."""
    path = _write_profile(tmp_path, None, config={"control_system.type": _VA}, tier=2)
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code != 0
    assert "archiver" in result.output
    assert "tier" in result.output


def test_declaring_the_block_alone_does_not_lift_the_refusal(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The block says where an archive lives; `archiver.type` says which archiver
    the deployment reads. A profile that declares the store and then leaves the
    connector on the mock deploys a store nothing reads."""
    path = _write_profile(tmp_path, {"retention_days": 2}, config={"control_system.type": _VA})
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code != 0
    assert "va_archiver" in result.output


def test_a_profile_that_pairs_honestly_validates(runner: CliRunner, tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        {"retention_days": 2, "hot_span_hours": 2},
        config={"control_system.type": _VA, "archiver.type": "mongodb_archiver"},
    )
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# What the REAL renderer does with two spellings, and why that means fail-closed
# ---------------------------------------------------------------------------


def _render_overrides(tmp_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply *overrides* through the build's own config-override machinery.

    Not a hand-built dict standing in for the renderer: this is
    ``_apply_config_overrides``, the function ``osprey build`` itself calls with
    ``{**profile.config, **derived}``, writing into a config.yml shaped like a
    rendered project's.
    """
    from osprey.cli.build_cmd import _apply_config_overrides

    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yml").write_text(
        "control_system:\n  type: mock\narchiver:\n  type: mock_archiver\n", encoding="utf-8"
    )
    _apply_config_overrides(project, overrides)
    return yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))


def test_both_spellings_reach_the_same_rendered_leaf(tmp_path: Path) -> None:
    """The premise the build-time rule rests on: a profile's dotted key and a
    nested mapping under `config:` are two ways of writing one rendered field."""
    dotted = _render_overrides(tmp_path / "a", {"archiver.type": "mongodb_archiver"})
    nested = _render_overrides(tmp_path / "b", {"archiver": {"type": "mongodb_archiver"}})

    assert dotted["archiver"]["type"] == "mongodb_archiver"
    assert nested["archiver"]["type"] == "mongodb_archiver"


def test_which_spelling_wins_depends_on_the_order_they_were_written_in(tmp_path: Path) -> None:
    """And the reason a profile spelling it BOTH ways is refused rather than
    resolved: the winner is whichever key comes later in the profile's `config:`
    block. Nothing in the profile says which that is, so the build has no
    honest way to pick — hence fail-closed."""
    dotted_last = _render_overrides(
        tmp_path / "c",
        {"archiver": {"type": "mock_archiver"}, "archiver.type": "mongodb_archiver"},
    )
    nested_last = _render_overrides(
        tmp_path / "d",
        {"archiver.type": "mongodb_archiver", "archiver": {"type": "mock_archiver"}},
    )

    assert dotted_last["archiver"]["type"] == "mongodb_archiver"
    assert nested_last["archiver"]["type"] == "mock_archiver"


def test_a_profile_spelling_the_archiver_both_ways_is_refused(tmp_path: Path) -> None:
    """Half of the orderings above render a virtual accelerator onto the mock.
    The build refuses the shape rather than gambling on key order."""
    errors = _pairing_errors(
        {
            "control_system.type": _VA,
            "archiver.type": "mongodb_archiver",
            "archiver": {"type": "mock_archiver"},
        }
    )

    assert len(errors) == 1
    assert "twice" in errors[0]


# ---------------------------------------------------------------------------
# The derived freshness check: one canary channel in, one health category out
# ---------------------------------------------------------------------------


def _checks(**knobs: Any) -> list[dict[str, Any]]:
    """The derived check list for a block built from *knobs*."""
    return va_archiver_config_overrides(VAArchiverConfig(**knobs)).get(FRESHNESS_CONFIG_KEY, [])


def test_a_block_naming_no_canary_derives_no_check() -> None:
    """Which channel is representative is a fact only the facility has. Without
    it the build derives nothing rather than guessing a channel that may not
    exist in this machine model."""
    assert _checks() == []
    assert FRESHNESS_CONFIG_KEY not in va_archiver_config_overrides(VAArchiverConfig())


def test_naming_a_canary_derives_the_whole_check() -> None:
    (check,) = _checks(freshness_channel="SR:DIAG:DCCT:01:CURRENT:RB")

    assert check["name"] == FRESHNESS_CHECK_NAME
    assert check["type"] == FRESHNESS_CHECK_NAME
    assert check["channel"] == "SR:DIAG:DCCT:01:CURRENT:RB"


def test_the_threshold_follows_the_recorder_not_a_second_number() -> None:
    """A facility that slows its recorder has changed how long a healthy archive
    can go without a sample; the check follows that edit on its own."""
    (slow,) = _checks(freshness_channel="X", recorder_cadence_sec=60)
    (slower,) = _checks(freshness_channel="X", recorder_cadence_sec=120)

    assert slow["max_age_s"] == 3 * 60
    assert slower["max_age_s"] == 3 * 120


def test_the_threshold_is_a_multiple_so_it_does_not_flap() -> None:
    """Equal to the cadence it would sit exactly on the sample age and alarm on
    ordinary jitter."""
    (check,) = _checks(freshness_channel="X", recorder_cadence_sec=100)

    assert check["max_age_s"] == FRESHNESS_SAFETY_MULTIPLE * 100
    assert check["max_age_s"] > 100


def test_a_fast_recorder_still_gets_a_survivable_floor() -> None:
    """3 s would be tripped by any container restart; nothing is learned from an
    alarm that fires faster than the deployment can recover."""
    (check,) = _checks(freshness_channel="X", recorder_cadence_sec=1)

    assert check["max_age_s"] == FRESHNESS_FLOOR_SEC


def test_the_tail_cadence_does_not_move_the_threshold() -> None:
    """Every recorder sample is written at the hot cadence; the tail grid only
    decides which of them survive ageing. So the newest sample is never a tail
    interval old, and the tail knob has no business in this threshold."""
    (dense,) = _checks(freshness_channel="X", recorder_tail_cadence_sec=60)
    (sparse,) = _checks(freshness_channel="X", recorder_tail_cadence_sec=3600)

    assert dense["max_age_s"] == sparse["max_age_s"]


def test_the_derived_check_is_what_the_health_framework_parses() -> None:
    """The end the derivation exists for: a config the health config loader
    accepts, carrying the probe's own parameter names."""
    from osprey.health.config import parse_health_config

    checks = _checks(freshness_channel="SR:DIAG:DCCT:01:CURRENT:RB", recorder_cadence_sec=30)
    parsed = parse_health_config({"categories": {FRESHNESS_CATEGORY: {"checks": checks}}})

    (spec,) = parsed.categories[FRESHNESS_CATEGORY].checks
    assert spec.type == FRESHNESS_CHECK_NAME
    assert spec.params["channel"] == "SR:DIAG:DCCT:01:CURRENT:RB"
    assert spec.params["max_age_s"] == 90


def test_the_derived_category_is_not_a_core_one() -> None:
    """A core category name rejects a `checks:` list outright, so deriving into
    one would render a config that fails to load."""
    from osprey.health.config import CORE_CATEGORY_NAMES

    assert FRESHNESS_CATEGORY not in CORE_CATEGORY_NAMES


def test_a_blank_canary_is_refused_rather_than_ignored() -> None:
    errors = _errors(freshness_channel="   ")

    assert any("freshness_channel" in error for error in errors)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({f"{FRESHNESS_CONFIG_KEY}": [{"name": "mine"}]}, id="dotted-leaf"),
        pytest.param({"health.categories.archiver": {"checks": []}}, id="dotted-category"),
        pytest.param({"health": {"categories": {"archiver": {"checks": []}}}}, id="nested"),
        pytest.param({"health.categories": {"archiver": {"checks": []}}}, id="mixed"),
    ],
)
def test_declaring_the_category_yourself_while_the_block_derives_it_is_refused(
    config: dict[str, Any],
) -> None:
    """One fact, two homes — and this one would be merged in whichever order the
    keys happen to be read, so the staleness threshold would depend on the order
    two unrelated lines were typed in."""
    errors = _errors(config=config, freshness_channel="X")

    assert any("health.categories.archiver" in error for error in errors), errors


def test_the_category_is_yours_again_once_the_block_names_no_canary() -> None:
    """Dropping `freshness_channel` is how a facility takes the category back."""
    config = {"health": {"categories": {"archiver": {"checks": []}}}}

    assert _errors(config=config) == []


def test_an_unrelated_health_category_is_left_alone() -> None:
    """The refusal is scoped to the one category the block derives."""
    config = {"health": {"categories": {"beamline": {"checks": []}}}}

    assert _errors(config=config, freshness_channel="X") == []


# ---------------------------------------------------------------------------
# The shipped preset's canary has to be a channel the shipped machine serves
# ---------------------------------------------------------------------------


def test_the_control_assistant_preset_names_a_canary() -> None:
    """The tutorial deployment gets the freshness check, not just the machinery
    to derive one. A preset that shipped the store and opted out of checking it
    would under-deliver the feature it exists to demonstrate."""
    from osprey.cli.build_profile import resolve_build_profile

    profile, _ = resolve_build_profile(None, "control-assistant")

    assert profile.va_archiver is not None
    assert profile.va_archiver.freshness_channel


def test_the_presets_canary_is_a_channel_the_virtual_accelerator_serves() -> None:
    """Same standard the L0 e2e holds its own canary to, applied where the
    choice is actually made.

    A canary the IOC does not serve is never recorded, so the check would report
    "no samples" forever — a permanently yellow health row that says nothing
    about the archive. Asserted against the manifest builder the VA and the
    recorder both read, so a machine-model change that drops this channel fails
    here rather than in a deployed operator's face.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.services.virtual_accelerator.manifest.build import build_manifest

    profile, _ = resolve_build_profile(None, "control-assistant")
    assert profile.va_archiver is not None
    canary = profile.va_archiver.freshness_channel

    served = {channel["address"] for channel in build_manifest()["channels"]}
    assert canary in served, (
        f"the control-assistant preset's freshness_channel {canary!r} is not served by the "
        f"shipped machine model, so the derived check could never see a sample. Pick a "
        f"channel the manifest carries."
    )
