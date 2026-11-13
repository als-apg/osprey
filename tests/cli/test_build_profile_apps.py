"""The profile ``apps:`` block and the login's application keys.

``profile.yml`` lists the deployment's applications under ``apps:`` — one entry
per application it runs, closed against that application's ``app.yml``: a
free-form ``config`` and per-container ``services.<key>`` axes (``network``,
``env``). An application is on when it is listed and off when it is not, so
there is no ``enabled`` key, and its tool server is projected by the build, so
no tool-server key is written here. A login (``tiers/<name>.yml``) selects from
those applications: ``apps.<name>.shown`` and ``lists: own | add`` are its only
application keys, and its writes are armed by its own ``control_system.writes``
tier, never by a per-application key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import (
    _KNOWN_MCP_SERVER_KEYS,
    _parse_profile,
    load_profile_document,
    reject_unknown_kind_keys,
)
from osprey.cli.profile_root import ProfileKind
from osprey_connectors.errors import BuildProfileError

_MANIFEST = {"name": "knob", "containers": {"knob": {"runtime": "osprey"}}}


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root: dict[str, Any]) -> Path:
    """A deployment repo with one application, ``knob``, and the given root profile.

    The caller takes ``facility_data_tree`` for the ``data:`` directory.
    """
    _write_yaml(tmp_path / "apps" / "knob" / "app.yml", _MANIFEST)
    _write_yaml(tmp_path / "profile.yml", {"name": "site", "data": "data", **root})
    return tmp_path


def _login_refusal(body: dict[str, Any]) -> str:
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys({"name": "ro", **body}, ProfileKind.TIER, "tiers/ro.yml")
    return str(excinfo.value)


def _parse_refusal(apps: Any) -> str:
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "site", "apps": apps})
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# The profile's apps block
# ---------------------------------------------------------------------------


def test_the_apps_block_parses_config_and_service_axes() -> None:
    profile = _parse_profile(
        {
            "name": "site",
            "apps": {
                "knob": {
                    "config": {"step": 0.5, "labels": ["a"]},
                    "services": {"knob": {"network": "host", "env": ["HTTP_PROXY"]}},
                },
                "other": None,
            },
        }
    )

    knob = profile.apps["knob"]
    assert knob.config == {"step": 0.5, "labels": ["a"]}
    assert knob.services["knob"].network == "host"
    assert knob.services["knob"].env == ["HTTP_PROXY"]
    assert knob.shown is None
    assert profile.apps["other"].config == {}
    assert profile.lists == "add"


def test_a_profile_without_apps_has_none() -> None:
    profile = _parse_profile({"name": "site"})

    assert profile.apps == {}


def test_enabled_is_refused_naming_removal_of_the_entry() -> None:
    message = _parse_refusal({"knob": {"enabled": True}})

    assert "apps.knob.enabled" in message
    assert "remove the entry" in message


def test_arm_writes_is_refused_naming_the_tier() -> None:
    message = _parse_refusal({"knob": {"arm_writes": True}})

    assert "apps.knob.arm_writes" in message
    assert "control_system.writes" in message


@pytest.mark.parametrize("key", sorted(_KNOWN_MCP_SERVER_KEYS))
def test_a_tool_server_key_is_refused_naming_the_projection(key: str) -> None:
    message = _parse_refusal({"knob": {key: "x"}})

    assert f"apps.knob.{key}" in message
    assert "projects mcp_servers.knob" in message


def test_an_unknown_entry_key_is_refused_naming_the_valid_keys() -> None:
    message = _parse_refusal({"knob": {"confg": {}}})

    assert "apps.knob.confg" in message
    assert "config, panel_hidden, services, shown" in message


@pytest.mark.parametrize("key", ["image", "ports", "volumes", "http"])
def test_a_service_carries_only_axis_keys(key: str) -> None:
    message = _parse_refusal({"knob": {"services": {"knob": {key: "x"}}}})

    assert f"apps.knob.services.knob.{key}" in message
    assert "env, network" in message


@pytest.mark.parametrize(
    ("entry", "words"),
    [
        ({"config": ["a"]}, "apps.knob.config must be a mapping"),
        ({"services": ["knob"]}, "apps.knob.services must be a mapping"),
        ({"services": {"knob": "host"}}, "apps.knob.services.knob must be a mapping"),
        ({"shown": "yes"}, "apps.knob.shown must be true or false"),
    ],
)
def test_entry_shapes_are_checked(entry: dict[str, Any], words: str) -> None:
    assert words in _parse_refusal({"knob": entry})


def test_the_block_must_be_a_mapping() -> None:
    assert "apps must be a mapping" in _parse_refusal(["knob"])


def test_lists_takes_own_or_add() -> None:
    assert _parse_profile({"name": "s", "lists": "own"}).lists == "own"
    with pytest.raises(BuildProfileError, match="lists must be one of own, add"):
        _parse_profile({"name": "s", "lists": "all"})


# ---------------------------------------------------------------------------
# Closed against the manifest
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("facility_data_tree")
def test_a_listed_application_loads(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"apps": {"knob": {"services": {"knob": {"network": "host"}}}}})

    loaded = load_profile_document(repo / "profile.yml")

    assert loaded.profile.apps["knob"].services["knob"].network == "host"


@pytest.mark.usefixtures("facility_data_tree")
def test_an_unknown_application_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"apps": {"knobs": {}}})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(repo / "profile.yml")

    message = str(excinfo.value)
    assert "apps.knobs" in message
    assert "no application" in message
    assert "knob" in message.split("apps.knobs", 1)[1]


@pytest.mark.usefixtures("facility_data_tree")
def test_a_service_the_manifest_does_not_declare_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"apps": {"knob": {"services": {"worker": {"network": "host"}}}}})

    with pytest.raises(BuildProfileError, match=r"apps\.knob\.services\.worker"):
        load_profile_document(repo / "profile.yml")


@pytest.mark.usefixtures("facility_data_tree")
@pytest.mark.parametrize(
    ("axes", "words"),
    [
        ({"network": "lan"}, "apps.knob.services.knob.network must be one of bridge, host"),
        ({"env": "HTTP_PROXY"}, "apps.knob.services.knob.env must be a list"),
    ],
)
def test_service_axes_are_validated(tmp_path: Path, axes: dict[str, Any], words: str) -> None:
    repo = _repo(tmp_path, {"apps": {"knob": {"services": {"knob": axes}}}})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(repo / "profile.yml")

    assert words in str(excinfo.value)


# ---------------------------------------------------------------------------
# The login's application keys
# ---------------------------------------------------------------------------


def test_a_login_selects_by_shown_and_lists() -> None:
    reject_unknown_kind_keys(
        {"name": "ro", "lists": "own", "apps": {"knob": {"shown": False}, "other": None}},
        ProfileKind.TIER,
        "tiers/ro.yml",
    )


@pytest.mark.parametrize("key", ["config", "services"])
def test_a_login_may_not_configure_an_application(key: str) -> None:
    message = _login_refusal({"apps": {"knob": {key: {}}}})

    assert "tiers/ro.yml" in message
    assert f"apps.knob.{key}" in message
    assert "profile.yml" in message


def test_a_login_may_not_spell_enabled() -> None:
    message = _login_refusal({"apps": {"knob": {"enabled": False}}})

    assert "tiers/ro.yml" in message
    assert "apps.knob.enabled" in message
    assert "remove the entry" in message
    assert "profile.yml" in message


def test_a_login_may_not_arm_an_application() -> None:
    message = _login_refusal({"apps": {"knob": {"arm_writes": True}}})

    assert "apps.knob.arm_writes" in message
    assert "control_system.writes" in message


def test_a_login_may_not_spell_a_tool_server_key() -> None:
    message = _login_refusal({"apps": {"knob": {"writes": ["set"]}}})

    assert "projects mcp_servers.knob" in message


@pytest.mark.parametrize("key", ["exclude", "extends"])
def test_a_login_never_subtracts_or_inherits(key: str) -> None:
    message = _login_refusal({key: {"apps": ["knob"]}})

    assert f"{key!r} (legal only in the deployment's profile.yml)" in message


def test_a_login_lists_value_is_checked() -> None:
    assert "lists must be one of own, add" in _login_refusal({"lists": "none"})


def test_a_login_panel_hidden_is_a_boolean() -> None:
    assert "apps.knob.panel_hidden must be true or false" in _login_refusal(
        {"apps": {"knob": {"shown": True, "panel_hidden": "yes"}}}
    )


def test_a_login_shown_is_a_boolean() -> None:
    assert "apps.knob.shown must be true or false" in _login_refusal(
        {"apps": {"knob": {"shown": "no"}}}
    )


@pytest.mark.usefixtures("facility_data_tree")
def test_validate_refuses_a_login_file_as_the_loader_does(tmp_path: Path, runner) -> None:
    from osprey.cli.validate_cmd import validate

    repo = _repo(tmp_path, {})
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"knob": {"config": {}}}})

    result = runner.invoke(validate, [str(layer)])
    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    assert result.exit_code == 2, result.output
    assert "apps.knob.config" in result.output
    assert "apps.knob.config" in str(excinfo.value)


@pytest.mark.usefixtures("facility_data_tree")
def test_a_manifest_refusal_reaches_validate_as_its_violations(tmp_path: Path, runner) -> None:
    from osprey.apps.errors import AppsRefusedError
    from osprey.cli.validate_cmd import validate

    repo = _repo(tmp_path, {"apps": {"knob": {}}})
    _write_yaml(repo / "apps" / "knob" / "app.yml", {**_MANIFEST, "colour": "blue"})

    result = runner.invoke(validate, [str(repo / "profile.yml")])
    with pytest.raises(AppsRefusedError) as excinfo:
        load_profile_document(repo / "profile.yml")

    assert [v.rule for v in excinfo.value.violations] == ["extra_forbidden"]
    assert result.exit_code == 2, result.output
    assert "apps/knob/app.yml: colour: " in result.output
    assert "[extra_forbidden]" in result.output
    assert "1 violation(s)." in result.output


@pytest.mark.usefixtures("facility_data_tree")
def test_a_manifest_refusal_is_a_profile_error_to_every_other_reader(tmp_path: Path) -> None:
    # A command that only asks whether the profile resolves (a profile card, the
    # scaffold, a tolerant loop over logins) catches BuildProfileError or
    # ConfigurationError; the refusal reaches it as one, never as a traceback.
    from osprey_connectors.errors import ConfigurationError

    repo = _repo(tmp_path, {"apps": {"knob": {}}})
    _write_yaml(repo / "apps" / "knob" / "app.yml", {**_MANIFEST, "colour": "blue"})

    with pytest.raises(ConfigurationError) as excinfo:
        load_profile_document(repo / "profile.yml")

    assert isinstance(excinfo.value, BuildProfileError)
    assert "apps/knob/app.yml: colour: " in str(excinfo.value)
    assert "[extra_forbidden]" in str(excinfo.value)


@pytest.mark.usefixtures("facility_data_tree")
def test_a_manifest_refusal_is_reported_beside_the_other_profile_errors(tmp_path: Path) -> None:
    from osprey.apps.errors import AppsRefusedError

    repo = _repo(tmp_path, {"apps": {"knob": {}}, "channel_finder_mode": "bogus"})
    _write_yaml(repo / "apps" / "knob" / "app.yml", {**_MANIFEST, "colour": "blue"})

    with pytest.raises(AppsRefusedError) as excinfo:
        load_profile_document(repo / "profile.yml")

    assert [v.rule for v in excinfo.value.violations] == ["extra_forbidden"]
    assert any("channel_finder_mode" in error for error in excinfo.value.errors)
    assert "channel_finder_mode" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A login render: the root merged with the login
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("facility_data_tree")
def test_a_merged_login_render_on_a_roster_deployment_loads(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        {
            "apps": {"knob": {"config": {"step": 1}}},
            "config": {
                "modules.web_terminals.users": [{"name": "alice", "index": 0, "tier": "ro"}],
            },
        },
    )
    layer = _write_yaml(
        repo / "tiers" / "ro.yml", {"name": "ro", "lists": "own", "apps": {"knob": {"shown": True}}}
    )

    profile = load_profile_document(layer).profile

    assert profile.lists == "own"
    assert profile.apps["knob"].config == {"step": 1}
    assert profile.apps["knob"].shown is True
    assert profile.app_shown("knob") is True


@pytest.mark.parametrize(
    ("lists", "shown", "expected"),
    [
        ("add", None, True),
        ("own", None, False),
        ("add", False, False),
        ("own", True, True),
    ],
)
def test_shown_defaults_follow_lists(lists: str, shown: bool | None, expected: bool) -> None:
    entry = {} if shown is None else {"shown": shown}
    profile = _parse_profile({"name": "s", "lists": lists, "apps": {"knob": entry}})

    assert profile.app_shown("knob") is expected


def test_an_unlisted_application_is_never_shown() -> None:
    assert _parse_profile({"name": "s"}).app_shown("knob") is False


@pytest.mark.usefixtures("facility_data_tree")
def test_a_login_selects_only_applications_the_profile_lists(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"apps": {"knob": {}}})
    _write_yaml(repo / "apps" / "other" / "app.yml", {**_MANIFEST, "name": "other"})
    layer = _write_yaml(
        repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"other": {"shown": True}}}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    message = str(excinfo.value)
    assert "tiers/ro.yml" in message
    assert "apps.other" in message
    assert "profile.yml" in message


@pytest.mark.usefixtures("facility_data_tree")
def test_the_profile_may_not_spell_lists(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"lists": "own", "apps": {"knob": {}}})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(repo / "profile.yml")

    message = str(excinfo.value)
    assert "lists" in message
    assert "tiers/" in message


@pytest.mark.usefixtures("facility_data_tree")
def test_a_login_render_refuses_a_root_lists(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"lists": "own", "apps": {"knob": {}}})
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro"})

    with pytest.raises(BuildProfileError, match="lists"):
        load_profile_document(layer)


def test_a_login_reports_unknown_keys_and_application_problems_at_once() -> None:
    message = _login_refusal({"port_base": 9000, "apps": {"knob": {"enabled": True}}})

    assert "'port_base'" in message
    assert "apps.knob.enabled" in message


# ---------------------------------------------------------------------------
# A login's default panel
# ---------------------------------------------------------------------------

_PANEL_MANIFEST = {
    **_MANIFEST,
    "panel": {"id": "knob-board", "label": "Knob", "container": "knob"},
}


def _panel_login(tmp_path: Path, apps: dict[str, Any], **keys: Any) -> Path:
    """A login naming the ``knob`` application's panel, with ``apps`` as its selection.

    It opens on the panel unless ``keys`` name the panel some other way.
    """
    repo = _repo(tmp_path, {"apps": {"knob": {}}})
    _write_yaml(repo / "apps" / "knob" / "app.yml", _PANEL_MANIFEST)
    return _write_yaml(
        repo / "tiers" / "ro.yml",
        {"name": "ro", "lists": "own", "apps": apps, **(keys or {"default_panel": "knob-board"})},
    )


@pytest.mark.usefixtures("facility_data_tree")
def test_a_login_may_open_on_the_panel_of_an_application_it_shows(tmp_path: Path) -> None:
    """The manifest's panel id, not the application's name: the build adds it to the tabs."""
    layer = _panel_login(tmp_path, {"knob": {"shown": True}})

    profile = load_profile_document(layer).profile

    assert profile.default_panel == "knob-board"


@pytest.mark.usefixtures("facility_data_tree")
def test_a_login_may_keep_a_shown_applications_panel_off_the_sidebar(tmp_path: Path) -> None:
    layer = _panel_login(tmp_path, {"knob": {"shown": True, "panel_hidden": True}})

    profile = load_profile_document(layer).profile

    assert profile.app_shown("knob") is True
    assert profile.app_panel_hidden("knob") is True
    assert profile.app_panel_hidden("unlisted") is False


@pytest.mark.usefixtures("facility_data_tree")
def test_a_login_may_not_open_on_the_panel_of_an_application_it_hides(tmp_path: Path) -> None:
    layer = _panel_login(tmp_path, {"knob": {"shown": False}})

    with pytest.raises(BuildProfileError, match="Unknown default_panel 'knob-board'"):
        load_profile_document(layer)


_PANEL_PRESET = {"panel_presets": {"Knob": {"tabs": ["knob-board"]}}}


@pytest.mark.usefixtures("facility_data_tree")
def test_a_preset_may_lay_out_the_panel_of_an_application_the_login_shows(tmp_path: Path) -> None:
    layer = _panel_login(tmp_path, {"knob": {"shown": True}}, **_PANEL_PRESET)

    profile = load_profile_document(layer).profile

    assert profile.panel_presets == _PANEL_PRESET["panel_presets"]


@pytest.mark.usefixtures("facility_data_tree")
def test_a_preset_may_not_lay_out_the_panel_of_an_application_the_login_hides(
    tmp_path: Path,
) -> None:
    layer = _panel_login(tmp_path, {"knob": {"shown": False}}, **_PANEL_PRESET)

    with pytest.raises(BuildProfileError) as refusal:
        load_profile_document(layer)

    message = str(refusal.value)
    assert "Unknown panel_presets['Knob'] member 'knob-board'" in message
    assert "not the panel of an application this login shows" in message
