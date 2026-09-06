"""Tests for the top-level ``connector`` profile shorthand.

``connector: epics`` is the short spelling of
``config: {control_system.type: epics}``. The shorthand is folded into the
literal dotted config key on every path a profile can arrive by — bundled
preset, ``--set`` edit, ``extends`` parent, or a hand-written profile loaded
directly — so it can never be accepted and then ignored. Its
value is validated against the settable connector types, so a misspelling
fails the build instead of resolving to a control system the facility never
asked for. That list is the initable one plus the live stand-in: a deployment
may be pointed at the soft IOC it already runs, while ``osprey init`` — which
has no stand-in to point a fresh project at — refuses the flag outright.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import _KNOWN_PROFILE_KEYS, _parse_profile, load_profile
from osprey.cli.build_profile_load import (
    CONNECTOR_CONFIG_KEY,
    CONNECTOR_PROFILE_KEY,
    PORT_BASE_PROFILE_KEY,
)
from osprey.cli.build_profile_resolve import (
    MODEL_SELECTION_OVERRIDE_KEYS,
    SHORTHAND_OVERRIDE_KEYS,
    apply_cli_edits,
    cli_edit_layer,
    explicit_model_override_keys,
    resolve_build_profile,
)
from osprey.cli.init_cmd import init
from osprey.cli.set_cmd import set as set_command
from osprey.connectors.types import (
    CLI_CONTROL_SYSTEM_TYPES,
    LIVE_STANDIN,
    SET_CONTROL_SYSTEM_TYPES,
)
from osprey.errors import BuildProfileError
from osprey.port_layout import PORT_BASE_CONFIG_KEY


@pytest.fixture(autouse=True)
def _facility_data_tree(tmp_path: Path) -> None:
    """The tree every profile's ``data:`` key names, beside the profile.

    ``data:`` is required of a repo profile and must resolve to a real
    directory, so without this each profile below would report one extra
    failure about a key none of these tests is about.
    """
    (tmp_path / "data").mkdir(exist_ok=True)


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


# ── the shorthand is part of the schema ──────────────────────────────────────


def test_connector_is_a_known_profile_key() -> None:
    """A profile spelling ``connector:`` is not rejected as an unknown key."""
    assert CONNECTOR_PROFILE_KEY in _KNOWN_PROFILE_KEYS


def test_connector_joins_the_shorthand_override_keys() -> None:
    """The shorthand keys are the model-selection ones plus ``connector`` and
    ``port_base``."""
    assert SHORTHAND_OVERRIDE_KEYS == (
        *MODEL_SELECTION_OVERRIDE_KEYS,
        "connector",
        "port_base",
    )


# ── folding into the literal dotted config key ───────────────────────────────


def test_parse_folds_shorthand_into_dotted_config_key() -> None:
    """``connector:`` resolves to ``config['control_system.type']``."""
    profile = _parse_profile({"name": "x", "data": "data", "connector": "virtual_accelerator"})

    assert profile.config[CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_parse_consumes_the_shorthand_key() -> None:
    """The raw mapping is left with the literal spelling only."""
    raw = {"name": "x", "data": "data", "connector": "epics"}
    _parse_profile(raw)

    assert CONNECTOR_PROFILE_KEY not in raw
    assert raw["config"] == {CONNECTOR_CONFIG_KEY: "epics"}


def test_shorthand_overrides_an_existing_literal_key() -> None:
    """A profile naming both resolves to the shorthand's value, not the literal's."""
    profile = _parse_profile(
        {
            "name": "x",
            "data": "data",
            "connector": "doocs",
            "config": {CONNECTOR_CONFIG_KEY: "mock"},
        }
    )

    assert profile.config[CONNECTOR_CONFIG_KEY] == "doocs"


def test_parse_without_shorthand_leaves_config_untouched() -> None:
    """No shorthand, no injected config key — the fold is opt-in."""
    profile = _parse_profile(
        {"name": "x", "data": "data", "config": {"control_system.type": "mock"}}
    )

    assert profile.config == {"control_system.type": "mock"}


def test_config_must_be_a_mapping_to_carry_the_shorthand() -> None:
    """A scalar ``config:`` cannot hold the folded key, and says so."""
    with pytest.raises(BuildProfileError, match="must be a mapping to carry"):
        _parse_profile(
            {"name": "x", "data": "data", "connector": "mock", "config": "not-a-mapping"}
        )


# ── --set edits ──────────────────────────────────────────────────────────────


def test_a_cli_edit_folds_the_set_shorthand() -> None:
    """``--set connector=…`` is baked as the literal dotted key, not the shorthand."""
    edit = cli_edit_layer(("connector=epics",))

    assert edit == {"config": {CONNECTOR_CONFIG_KEY: "epics"}}


def test_the_set_shorthand_replaces_the_documents_connector() -> None:
    """The resolved document's connector is replaced, not merged alongside."""
    base = {"name": "x", "data": "data", "config": {CONNECTOR_CONFIG_KEY: "mock"}}
    edited = apply_cli_edits(base, ("connector=virtual_accelerator",))

    assert edited["config"][CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_a_shorthand_already_in_the_document_is_folded() -> None:
    """A document that spells ``connector:`` itself is folded on the edit path too.

    The fold runs over the whole document, not only over the pairs, so no entry
    path — a preset, a hand-written profile — carries a shorthand past this
    point and has it silently ignored.
    """
    edited = apply_cli_edits({"name": "x", "data": "data", "connector": "doocs"}, ())

    assert edited["config"][CONNECTOR_CONFIG_KEY] == "doocs"
    assert CONNECTOR_PROFILE_KEY not in edited


def test_a_set_pair_outranks_the_documents_own_shorthand() -> None:
    """``--set connector=`` states the value, over a document naming another.

    Order is the whole of it: the document's own shorthand is folded into the
    literal config key BEFORE the edit lands, so the edit replaces a key that
    is already there. Folded afterwards, a hand-written ``connector:`` would
    quietly overwrite what the operator just typed — and an ``osprey set`` that
    writes the config key while leaving the shorthand beside it would be undone
    on the next read.
    """
    edited = apply_cli_edits(
        {"name": "x", "data": "data", "connector": "doocs"}, ("connector=epics",)
    )

    assert edited["config"][CONNECTOR_CONFIG_KEY] == "epics"
    assert CONNECTOR_PROFILE_KEY not in edited


def test_an_edit_without_the_shorthand_invents_no_config_block() -> None:
    """No ``connector`` anywhere means no ``config:`` block is invented."""
    edited = apply_cli_edits({"name": "x", "data": "data"}, ("model=sonnet",))

    assert edited == {"name": "x", "data": "data", "model": "sonnet"}


# ── extends parents and plain file loads ─────────────────────────────────────


def test_shorthand_in_an_extends_parent_is_folded(tmp_path: Path) -> None:
    """A parent's shorthand reaches the child — extends resolution is not an escape.

    The child supplies the archiver because a virtual accelerator may not read
    the mock one; that the two halves of the pairing can arrive from different
    layers and still be judged together is the point of checking the *merged*
    config rather than any single layer's.
    """
    (tmp_path / "data").mkdir(exist_ok=True)
    parent = tmp_path / "parent.yml"
    parent.write_text("name: Parent\nconnector: virtual_accelerator\n", encoding="utf-8")
    child = tmp_path / "child.yml"
    child.write_text(
        "name: Child\nextends: parent.yml\ndata: data\n"
        "config:\n  archiver.type: mongodb_archiver\n",
        encoding="utf-8",
    )

    profile = load_profile(child)

    assert profile.config[CONNECTOR_CONFIG_KEY] == "virtual_accelerator"


def test_shorthand_in_a_plain_profile_file_is_folded(tmp_path: Path) -> None:
    """``load_profile`` folds it too — not only the preset/edit path."""
    profile_file = tmp_path / "profile.yml"
    (tmp_path / "data").mkdir(exist_ok=True)
    profile_file.write_text("name: Plain\ndata: data\nconnector: doocs\n", encoding="utf-8")

    profile = load_profile(profile_file)

    assert profile.config[CONNECTOR_CONFIG_KEY] == "doocs"


# ── value validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("connector", CLI_CONTROL_SYSTEM_TYPES)
def test_every_cli_connector_type_is_accepted(connector: str) -> None:
    """The shorthand accepts exactly the types the config CLI offers."""
    profile = _parse_profile({"name": "x", "data": "data", "connector": connector})

    assert profile.config[CONNECTOR_CONFIG_KEY] == connector


def test_misspelled_connector_suggests_the_nearest_type() -> None:
    """A typo names the intended type rather than failing later in the build."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "data": "data", "connector": "virtal_accelerator"})

    message = str(excinfo.value)
    assert "Unknown connector 'virtal_accelerator'" in message
    assert "did you mean 'virtual_accelerator'?" in message


def test_wrong_case_connector_suggests_the_exact_spelling() -> None:
    """Case is not silently normalized — the exact spelling is suggested."""
    with pytest.raises(BuildProfileError, match="did you mean 'epics'"):
        _parse_profile({"name": "x", "data": "data", "connector": "EPICS"})


def test_unrecognizable_connector_lists_every_valid_choice() -> None:
    """With nothing close enough to suggest, the full choice list still lands."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "data": "data", "connector": "moat"})

    message = str(excinfo.value)
    assert "Valid connectors are:" in message
    for choice in CLI_CONTROL_SYSTEM_TYPES:
        assert choice in message


def test_custom_connector_path_is_pointed_at_the_literal_key() -> None:
    """A dotted module path is not a shorthand value; the error says what is."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "data": "data", "connector": "mypackage.MoatConnector"})

    assert CONNECTOR_CONFIG_KEY in str(excinfo.value)


@pytest.mark.parametrize("value", [None, "", "   ", 3, True, ["epics"]])
def test_non_string_connector_values_are_rejected(value: object) -> None:
    """``--set connector=`` and friends fail rather than resolving to nothing."""
    with pytest.raises(BuildProfileError, match="must name a connector type"):
        _parse_profile({"name": "x", "data": "data", "connector": value})


def test_invalid_set_value_is_rejected_when_the_edit_is_parsed() -> None:
    """The edit is validated too — before anything is written."""
    with pytest.raises(BuildProfileError, match="Unknown connector 'epcis'"):
        cli_edit_layer(("connector=epcis",))


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
    """``--set connector=`` retints a bundled preset's control system.

    Retinting a storeless preset to a virtual accelerator means declaring where
    that machine's history lives, hence the second pair: the mock archiver is
    refused for a simulated machine, and hello-world ships with no archive.
    """
    profile, _profile_dir = resolve_build_profile(
        None,
        "hello-world",
        set_pairs=("connector=virtual_accelerator", "config.archiver.type=mongodb_archiver"),
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


def test_osprey_set_retires_the_shorthands_it_rewrites(tmp_path: Path) -> None:
    """A file edit that writes a config key drops the shorthand beside it.

    Left in place, the parse-time fold would restore the shorthand's value
    over the key ``osprey set`` just wrote, on the very next read.
    """
    target = tmp_path / "my-deployment"
    runner = CliRunner()
    result = runner.invoke(init, [str(target), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    profile = target / "profile.yml"
    document = yaml.safe_load(profile.read_text(encoding="utf-8"))
    document[CONNECTOR_PROFILE_KEY] = "doocs"
    document[PORT_BASE_PROFILE_KEY] = 10000
    document["config"].pop(CONNECTOR_CONFIG_KEY, None)
    document["config"].pop(PORT_BASE_CONFIG_KEY, None)
    profile.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    written = runner.invoke(
        set_command, ["--repo", str(target), "connector=mock", "port_base=42000"]
    )
    assert written.exit_code == 0, written.output

    baked = yaml.safe_load(profile.read_text(encoding="utf-8"))
    assert CONNECTOR_PROFILE_KEY not in baked
    assert PORT_BASE_PROFILE_KEY not in baked
    assert baked["config"][CONNECTOR_CONFIG_KEY] == "mock"
    assert baked["config"][PORT_BASE_CONFIG_KEY] == 42000


# ── the stand-in is settable, not initable ───────────────────────────────────
#
# The live stand-in is a control target of its own, reached by pointing a
# deployment at the soft IOC it already runs. That makes it a legal value for
# the shorthand — `osprey set connector=live_standin` is the flip an operator
# performs — and an illegal one for `osprey init`, which materializes a fresh
# project that has no stand-in yet. Two lists, one shorthand: what the value is
# validated against is the wider one, and `init` is narrowed by refusing the
# flag outright rather than by a second spelling of the allowlist.


def test_set_list_is_the_cli_list_plus_the_stand_in() -> None:
    """The settable types are the initable ones and nothing but the stand-in."""
    assert SET_CONTROL_SYSTEM_TYPES == [*CLI_CONTROL_SYSTEM_TYPES, LIVE_STANDIN]


def test_the_stand_in_stays_out_of_the_init_list() -> None:
    """A fresh project is never materialized onto a stand-in it does not have."""
    assert LIVE_STANDIN not in CLI_CONTROL_SYSTEM_TYPES


def test_parse_accepts_the_stand_in() -> None:
    """``connector: live_standin`` folds to the literal key like any other type."""
    profile = _parse_profile({"name": "x", "data": "data", "connector": LIVE_STANDIN})

    assert profile.config[CONNECTOR_CONFIG_KEY] == LIVE_STANDIN


def test_a_cli_edit_accepts_the_stand_in() -> None:
    """``--set connector=live_standin`` survives the edit's own validation."""
    edit = cli_edit_layer((f"connector={LIVE_STANDIN}",))

    assert edit == {"config": {CONNECTOR_CONFIG_KEY: LIVE_STANDIN}}


def test_misspelled_stand_in_suggests_the_stand_in() -> None:
    """The suggestion draws from the settable list, so a stand-in typo lands."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "data": "data", "connector": "live_standn"})

    message = str(excinfo.value)
    assert "Unknown connector 'live_standn'" in message
    assert f"did you mean {LIVE_STANDIN!r}?" in message


def test_invalid_connector_lists_the_stand_in_among_the_choices() -> None:
    """The refusal names every settable type, the stand-in included."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "data": "data", "connector": "moat"})

    assert LIVE_STANDIN in str(excinfo.value)


def test_set_points_a_deployment_at_its_stand_in(lifecycle_repo: Path) -> None:
    """``osprey set connector=live_standin`` writes the profile's control key.

    The verb is the flip an operator performs on a deployment that already runs
    a stand-in; going back is the same command naming ``epics``.
    """
    repo = lifecycle_repo
    runner = CliRunner()

    result = runner.invoke(
        set_command, ["--repo", str(repo), f"connector={LIVE_STANDIN}"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    profile_text = (repo / "profile.yml").read_text(encoding="utf-8")
    assert f"control_system.type: {LIVE_STANDIN}" in profile_text

    back = runner.invoke(
        set_command, ["--repo", str(repo), "connector=epics"], catch_exceptions=False
    )

    assert back.exit_code == 0, back.output
    assert "control_system.type: epics" in (repo / "profile.yml").read_text(encoding="utf-8")


def test_init_still_refuses_the_stand_in_as_a_flag(tmp_path: Path) -> None:
    """``osprey init --connector live_standin`` materializes nothing.

    ``init`` carries no ``--connector`` option at all — the shorthands are
    registered as hidden, always-refusing flags that point at the ``--set``
    spelling — so widening the shorthand's allowlist cannot open a path to a
    project that begins on a stand-in it has not built yet.
    """
    target = tmp_path / "my-deployment"
    result = CliRunner().invoke(
        init,
        [str(target), "--preset", "hello-world", "--no-git", "--connector", LIVE_STANDIN],
    )

    assert result.exit_code != 0
    assert "There is no --connector option" in _flat(result.output)
    assert not (target / "profile.yml").exists()
