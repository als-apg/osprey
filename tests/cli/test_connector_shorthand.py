"""Tests for the top-level ``connector`` profile shorthand.

``connector: epics`` is the short spelling of
``config: {control_system.type: epics}``. The shorthand is folded into the
literal dotted config key on every path a profile can arrive by — bundled
preset, ``-O`` file, ``--set`` pair, ``extends`` parent, or a hand-written
profile loaded directly — so it can never be accepted and then ignored. Its
value is validated against the same connector list the rest of the CLI offers,
so a misspelling fails the build instead of resolving to a control system the
facility never asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import _KNOWN_PROFILE_KEYS, _parse_profile, load_profile
from osprey.cli.build_profile_load import CONNECTOR_CONFIG_KEY, CONNECTOR_PROFILE_KEY
from osprey.cli.build_profile_resolve import (
    MODEL_SELECTION_OVERRIDE_KEYS,
    SHORTHAND_OVERRIDE_KEYS,
    explicit_model_override_keys,
    merge_cli_overrides,
    resolve_build_profile,
)
from osprey.cli.init_cmd import init
from osprey.connectors.types import CLI_CONTROL_SYSTEM_TYPES
from osprey.errors import BuildProfileError


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


# ── the shorthand is part of the schema ──────────────────────────────────────


def test_connector_is_a_known_profile_key() -> None:
    """A profile spelling ``connector:`` is not rejected as an unknown key."""
    assert CONNECTOR_PROFILE_KEY in _KNOWN_PROFILE_KEYS


def test_connector_joins_the_shorthand_override_keys() -> None:
    """The shorthand keys are the model-selection ones plus ``connector``."""
    assert SHORTHAND_OVERRIDE_KEYS == (*MODEL_SELECTION_OVERRIDE_KEYS, "connector")


# ── folding into the literal dotted config key ───────────────────────────────


def test_parse_folds_shorthand_into_dotted_config_key() -> None:
    """``connector:`` resolves to ``config['control_system.type']``."""
    profile = _parse_profile({"name": "x", "connector": "virtual_accelerator"})

    assert profile.config[CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_parse_consumes_the_shorthand_key() -> None:
    """The raw mapping is left with the literal spelling only."""
    raw = {"name": "x", "connector": "epics"}
    _parse_profile(raw)

    assert CONNECTOR_PROFILE_KEY not in raw
    assert raw["config"] == {CONNECTOR_CONFIG_KEY: "epics"}


def test_shorthand_overrides_an_existing_literal_key() -> None:
    """A profile naming both resolves to the shorthand's value, not the literal's."""
    profile = _parse_profile(
        {"name": "x", "connector": "doocs", "config": {CONNECTOR_CONFIG_KEY: "mock"}}
    )

    assert profile.config[CONNECTOR_CONFIG_KEY] == "doocs"


def test_parse_without_shorthand_leaves_config_untouched() -> None:
    """No shorthand, no injected config key — the fold is opt-in."""
    profile = _parse_profile({"name": "x", "config": {"control_system.type": "mock"}})

    assert profile.config == {"control_system.type": "mock"}


def test_config_must_be_a_mapping_to_carry_the_shorthand() -> None:
    """A scalar ``config:`` cannot hold the folded key, and says so."""
    with pytest.raises(BuildProfileError, match="must be a mapping to carry"):
        _parse_profile({"name": "x", "connector": "mock", "config": "not-a-mapping"})


# ── --set / -O layering ──────────────────────────────────────────────────────


def test_merge_cli_overrides_folds_set_shorthand() -> None:
    """``--set connector=…`` is baked as the literal dotted key, not the shorthand."""
    merged = merge_cli_overrides({}, (), ("connector=epics",))

    assert merged == {"config": {CONNECTOR_CONFIG_KEY: "epics"}}


def test_set_shorthand_wins_over_the_base_layer() -> None:
    """The base layer's connector is replaced, not merged alongside."""
    base = {"name": "x", "config": {CONNECTOR_CONFIG_KEY: "mock"}}
    merged = merge_cli_overrides(base, (), ("connector=virtual_accelerator",))

    assert merged["config"][CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_override_file_shorthand_is_folded(tmp_path: Path) -> None:
    """A ``-O`` file may spell the shorthand too."""
    override = tmp_path / "over.yml"
    override.write_text("connector: doocs\n", encoding="utf-8")

    merged = merge_cli_overrides({"name": "x"}, (override,), ())

    assert merged["config"][CONNECTOR_CONFIG_KEY] == "doocs"
    assert CONNECTOR_PROFILE_KEY not in merged


def test_last_layer_naming_the_connector_wins(tmp_path: Path) -> None:
    """``--set`` layers over a ``-O`` file that also names the connector."""
    override = tmp_path / "over.yml"
    override.write_text("connector: doocs\n", encoding="utf-8")

    merged = merge_cli_overrides({"name": "x"}, (override,), ("connector=epics",))

    assert merged["config"][CONNECTOR_CONFIG_KEY] == "epics"


def test_merge_without_shorthand_is_unchanged() -> None:
    """No ``connector`` anywhere means no ``config:`` block is invented."""
    merged = merge_cli_overrides({"name": "x"}, (), ("model=sonnet",))

    assert merged == {"name": "x", "model": "sonnet"}


# ── extends parents and plain file loads ─────────────────────────────────────


def test_shorthand_in_an_extends_parent_is_folded(tmp_path: Path) -> None:
    """A parent's shorthand reaches the child — extends resolution is not an escape."""
    parent = tmp_path / "parent.yml"
    parent.write_text("name: Parent\nconnector: virtual_accelerator\n", encoding="utf-8")
    child = tmp_path / "child.yml"
    child.write_text("name: Child\nextends: parent.yml\n", encoding="utf-8")

    profile = load_profile(child)

    assert profile.config[CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_shorthand_in_a_plain_profile_file_is_folded(tmp_path: Path) -> None:
    """``load_profile`` folds it too — not only the preset/override path."""
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text("name: Plain\nconnector: doocs\n", encoding="utf-8")

    profile = load_profile(profile_file)

    assert profile.config[CONNECTOR_CONFIG_KEY] == "doocs"


# ── value validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("connector", CLI_CONTROL_SYSTEM_TYPES)
def test_every_cli_connector_type_is_accepted(connector: str) -> None:
    """The shorthand accepts exactly the types the config CLI offers."""
    profile = _parse_profile({"name": "x", "connector": connector})

    assert profile.config[CONNECTOR_CONFIG_KEY] == connector


def test_misspelled_connector_suggests_the_nearest_type() -> None:
    """A typo names the intended type rather than failing later in the build."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "connector": "virtal_accelerator"})

    message = str(excinfo.value)
    assert "Unknown connector 'virtal_accelerator'" in message
    assert "did you mean 'virtual_accelerator'?" in message


def test_wrong_case_connector_suggests_the_exact_spelling() -> None:
    """Case is not silently normalized — the exact spelling is suggested."""
    with pytest.raises(BuildProfileError, match="did you mean 'epics'"):
        _parse_profile({"name": "x", "connector": "EPICS"})


def test_unrecognizable_connector_lists_every_valid_choice() -> None:
    """With nothing close enough to suggest, the full choice list still lands."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "connector": "tango"})

    message = str(excinfo.value)
    assert "Valid connectors are:" in message
    for choice in CLI_CONTROL_SYSTEM_TYPES:
        assert choice in message


def test_custom_connector_path_is_pointed_at_the_literal_key() -> None:
    """A dotted module path is not a shorthand value; the error says what is."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "connector": "mypackage.TangoConnector"})

    assert CONNECTOR_CONFIG_KEY in str(excinfo.value)


@pytest.mark.parametrize("value", [None, "", "   ", 3, True, ["epics"]])
def test_non_string_connector_values_are_rejected(value: object) -> None:
    """``--set connector=`` and friends fail rather than resolving to nothing."""
    with pytest.raises(BuildProfileError, match="must name a connector type"):
        _parse_profile({"name": "x", "connector": value})


def test_invalid_set_value_is_rejected_at_merge_time() -> None:
    """The CLI layering step validates too — before anything is written."""
    with pytest.raises(BuildProfileError, match="Unknown connector 'epcis'"):
        merge_cli_overrides({}, (), ("connector=epcis",))


# ── forwarding to persona renders ────────────────────────────────────────────


def test_explicit_set_connector_is_reported_for_forwarding() -> None:
    """A bare ``--set connector=`` counts as an explicit whole-stack override."""
    assert explicit_model_override_keys(("connector=epics",)) == ["connector"]


def test_dotted_config_override_is_not_reported() -> None:
    """The literal dotted key addresses the config directly and is not forwarded."""
    assert explicit_model_override_keys(("config.control_system.type=epics",)) == []


def test_reported_keys_keep_shorthand_order() -> None:
    """Model-selection keys still come first, in their declared order."""
    keys = explicit_model_override_keys(("connector=mock", "model=sonnet", "provider=anthropic"))

    assert keys == ["provider", "model", "connector"]


# ── preset resolution end to end ─────────────────────────────────────────────


def test_preset_resolution_applies_the_shorthand() -> None:
    """``--set connector=`` retints a bundled preset's control system."""
    profile, _profile_dir = resolve_build_profile(
        None, "hello-world", set_pairs=("connector=virtual_accelerator",)
    )

    assert profile.config[CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_preset_resolution_rejects_an_invalid_connector() -> None:
    """The invalid value stops the resolve rather than reaching the render."""
    with pytest.raises(BuildProfileError, match="Unknown connector"):
        resolve_build_profile(None, "hello-world", set_pairs=("connector=epcis",))


def test_init_exits_non_zero_on_an_invalid_connector(tmp_path: Path) -> None:
    """The CLI surfaces the misspelling as a usage error, materializing nothing."""
    target = tmp_path / "my-deployment"
    result = CliRunner().invoke(
        init,
        [
            str(target),
            "--preset",
            "hello-world",
            "--no-git",
            "--set",
            "connector=virtal_accelerator",
        ],
    )

    assert result.exit_code != 0
    assert "did you mean 'virtual_accelerator'?" in _flat(result.output)
    assert not (target / "profile.yml").exists()


def test_init_bakes_the_literal_key(tmp_path: Path) -> None:
    """The materialized profile states the connector at the key a reader edits."""
    target = tmp_path / "my-deployment"
    result = CliRunner().invoke(
        init,
        [str(target), "--preset", "hello-world", "--no-git", "--set", "connector=doocs"],
    )

    assert result.exit_code == 0, result.output
    # The repo IS the deployment: profile.yml sits at its root.
    baked = yaml.safe_load((target / "profile.yml").read_text(encoding="utf-8"))
    assert CONNECTOR_PROFILE_KEY not in baked
    assert baked["config"][CONNECTOR_CONFIG_KEY] == "doocs"
