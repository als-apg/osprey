"""The ``watchdogs:`` profile block: its parser, its rules, and its protection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile import _parse_profile, load_profile
from osprey.cli.build_profile_emit import _COMMENTED_TEMPLATES
from osprey.cli.build_profile_load import _KNOWN_PROFILE_KEYS, ROOT_ONLY_KEYS
from osprey.cli.profile_conventions import KNOWN_ROOT_ENTRIES, is_protected_key
from osprey.mcp_server.python_executor.executor import (
    PROFILE_SOURCE_ENTRIES,
    resolve_protected_roots,
)
from osprey.registry.mcp import framework_write_tools
from osprey.services.watchdog.config import (
    VerdictOption,
    WatchdogDefinition,
    parse_watchdogs_block,
    watchdog_errors,
)
from osprey_connectors.errors import BuildProfileError
from tests.services.python_executor.test_runtime_guard import _Project, project  # noqa: F401

_OPTIONS = [
    {"id": "risky", "text": "Pushing past a guard", "flag": True},
    {"id": "routine", "text": "Routine work", "flag": False},
]


def _decision(**keys: Any) -> dict[str, Any]:
    return {"engine": "decision", "options": _OPTIONS, **keys}


def _agent(**keys: Any) -> dict[str, Any]:
    return {"engine": "agent", "tools": ["mcp__controls__channel_read"], **keys}


def _files(base: Path, *names: str) -> Path:
    for name in names:
        directory = base / "watchdogs" / name
        directory.mkdir(parents=True)
        (directory / "instructions.md").write_text(f"Judge as {name}.\n", encoding="utf-8")
    return base


def _errors(block: Any, base: Path, *, hooks: Any = ("turn-state",), writes: Any = ()) -> list[str]:
    return watchdog_errors(block, base_dir=base, write_tools=writes, hook_selection=hooks)


def _validation_problems(profile: Any, base: Path) -> list[str]:
    try:
        profile.validate(base)
    except BuildProfileError as exc:
        return str(exc).splitlines()
    return []


def _write_profile(base: Path, block: dict[str, Any], hooks: list[str]) -> Path:
    (base / "data").mkdir()
    path = base / "profile.yml"
    path.write_text(
        yaml.safe_dump(
            {"name": "w", "data": "data", "hooks": hooks, "watchdogs": block}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_defaults(tmp_path: Path) -> None:
    _files(tmp_path, "a", "d")
    agent, decision = parse_watchdogs_block(
        {"a": {"engine": "agent"}, "d": _decision()}, base_dir=tmp_path
    )
    assert agent == WatchdogDefinition(name="a", engine="agent", instructions="Judge as a.\n")
    assert (agent.enabled, agent.timeout_s, agent.max_cost_usd) == (False, 90.0, 0.25)
    assert (agent.threshold, agent.model, agent.tools) == (None, None, ())
    assert decision.threshold == 0.5
    assert decision.enabled is False
    assert decision.options == (
        VerdictOption("risky", "Pushing past a guard", True),
        VerdictOption("routine", "Routine work", False),
    )


def test_stated_values_are_kept_and_the_key_is_the_name(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    [definition] = parse_watchdogs_block(
        {"a": _agent(enabled=True, model="m", timeout_s=5, max_cost_usd=1)}, base_dir=tmp_path
    )
    assert definition.name == "a"
    assert definition.enabled is True
    assert definition.model == "m"
    assert (definition.timeout_s, definition.max_cost_usd) == (5.0, 1.0)
    assert definition.tools == ("mcp__controls__channel_read",)


def test_references_are_derived_from_the_watchdog_directory(tmp_path: Path) -> None:
    _files(tmp_path, "a", "b")
    references = tmp_path / "watchdogs" / "a" / "references"
    (references / "deep").mkdir(parents=True)
    (references / "z.md").write_text("z")
    (references / "deep" / "y.md").write_text("y")
    a, b = parse_watchdogs_block({"a": _agent(), "b": _agent()}, base_dir=tmp_path)
    assert a.references == (references / "deep" / "y.md", references / "z.md")
    assert b.references == ()


@pytest.mark.parametrize("empty", [None, {}])
def test_no_watchdogs_parse_to_nothing(tmp_path: Path, empty: Any) -> None:
    assert parse_watchdogs_block(empty, base_dir=tmp_path) == ()
    assert _errors(empty, tmp_path) == []


def test_parser_refuses_what_the_errors_list(tmp_path: Path) -> None:
    with pytest.raises(BuildProfileError, match="engine must be one of"):
        parse_watchdogs_block({"a": {"engine": "oracle"}}, base_dir=tmp_path)
    with pytest.raises(BuildProfileError, match="must be a mapping"):
        parse_watchdogs_block(["a"], base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_unknown_engine(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    [error] = _errors({"a": {"engine": "oracle"}}, tmp_path)
    assert "watchdogs.a.engine" in error and "decision" in error and "agent" in error


def test_missing_instructions_names_the_remedy_a_child_can_use(tmp_path: Path) -> None:
    [error] = _errors({"a": _agent(enabled=True)}, tmp_path)
    assert "instructions.md" in error
    assert "add the file" in error
    assert "extends" in error
    assert "osprey init --force" not in error


@pytest.mark.parametrize("entry", [_agent(), _agent(enabled=False), _decision(enabled=False)])
def test_a_disabled_entry_needs_no_files_until_it_is_parsed(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    assert _errors({"a": entry}, tmp_path) == []
    with pytest.raises(BuildProfileError, match="instructions.md"):
        parse_watchdogs_block({"a": entry}, base_dir=tmp_path)


def test_a_malformed_enabled_is_still_held_to_the_files(tmp_path: Path) -> None:
    errors = _errors({"a": _agent(enabled="yes")}, tmp_path)
    assert any("instructions.md" in error for error in errors)


def test_empty_instructions_are_refused(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    (tmp_path / "watchdogs" / "a" / "instructions.md").write_text("  \n")
    [error] = _errors({"a": _agent(enabled=True)}, tmp_path)
    assert "empty" in error


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ([{"id": "r", "text": "t", "flag": False}], "needs a flag option"),
        ([{"id": "r", "text": "t", "flag": True}], "needs a non-flag option"),
        ([{"id": "r", "text": "t"}], "needs a flag option"),
        ([], "needs a flag option"),
    ],
)
def test_decision_needs_both_kinds_of_option(
    tmp_path: Path, options: list[Any], fragment: str
) -> None:
    _files(tmp_path, "a")
    assert any(fragment in e for e in _errors({"a": _decision(options=options)}, tmp_path))


def test_decision_without_options_is_refused(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    [error] = _errors({"a": {"engine": "decision"}}, tmp_path)
    assert "watchdogs.a.options" in error


@pytest.mark.parametrize(
    "bad_option",
    [
        "risky",
        {"id": "", "text": "t", "flag": True},
        {"id": "x", "flag": True},
        {"id": "x", "text": "t", "flag": "yes"},
        {"id": "x", "text": "t", "flag": True, "weight": 2},
        {"id": "routine", "text": "t", "flag": True},
    ],
)
def test_malformed_options_are_refused(tmp_path: Path, bad_option: Any) -> None:
    _files(tmp_path, "a")
    assert _errors({"a": _decision(options=[*_OPTIONS, bad_option])}, tmp_path)


@pytest.mark.parametrize("threshold", [0, 1, 0.0, 1.0, -0.1, 1.5, True, "0.5", None, float("nan")])
def test_threshold_outside_the_open_interval(tmp_path: Path, threshold: Any) -> None:
    _files(tmp_path, "a")
    [error] = _errors({"a": _decision(threshold=threshold)}, tmp_path)
    assert "threshold" in error


@pytest.mark.parametrize("threshold", [0.01, 0.5, 0.99])
def test_threshold_inside_the_open_interval(tmp_path: Path, threshold: float) -> None:
    _files(tmp_path, "a")
    assert _errors({"a": _decision(threshold=threshold)}, tmp_path) == []


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__controls__channel_write",
        "mcp__controls",
        "mcp__controls__*",
        "mcp__controls__.*",
        "mcp__*",
        "*",
    ],
)
def test_agent_tools_may_not_reach_a_write_tool(tmp_path: Path, tool: str) -> None:
    _files(tmp_path, "a")
    [error] = _errors(
        {"a": _agent(tools=[tool])}, tmp_path, writes=["mcp__controls__channel_write"]
    )
    assert "reaches write tools" in error and tool in error


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__controls__.+",
        "mcp__controls__channel_(read|write)",
        "mcp__controls__channel_[rw]rite",
        "mcp__si*__push",
        "mcp__*__read",
        "mcp__si.*__push",
        "**",
    ],
)
def test_a_pattern_the_check_cannot_read_is_refused(tmp_path: Path, tool: str) -> None:
    _files(tmp_path, "a")
    block = {"a": _agent(tools=[tool])}
    assert any("cannot read" in error for error in _errors(block, tmp_path))
    with pytest.raises(BuildProfileError, match="cannot read"):
        parse_watchdogs_block(block, base_dir=tmp_path)


@pytest.mark.parametrize("classified", ["mcp__site__push_*", "mcp__site__push_.*"])
def test_a_pattern_in_the_classification_is_read_as_a_pattern(
    tmp_path: Path, classified: str
) -> None:
    _files(tmp_path, "a")
    assert _errors({"a": _agent(tools=["mcp__site__push_x"])}, tmp_path, writes=[classified])
    assert _errors({"a": _agent(tools=["mcp__site__read"])}, tmp_path, writes=[classified]) == []


@pytest.mark.parametrize("key", ["timeout_s", "max_cost_usd"])
def test_a_number_too_large_for_a_float_is_an_error(tmp_path: Path, key: str) -> None:
    _files(tmp_path, "a")
    assert any(key in error for error in _errors({"a": _agent(**{key: 10**400})}, tmp_path))


def test_a_threshold_too_large_for_a_float_is_an_error(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    assert any(
        "threshold" in error for error in _errors({"a": _decision(threshold=10**400)}, tmp_path)
    )


@pytest.mark.parametrize("whole_server", ["mcp__site", "mcp__site__.*"])
def test_a_whole_server_write_classification_covers_its_tools(
    tmp_path: Path, whole_server: str
) -> None:
    _files(tmp_path, "a")
    block = {"a": _agent(tools=["mcp__site__anything"])}
    assert _errors(block, tmp_path, writes=[whole_server])
    assert _errors({"a": _agent(tools=["mcp__site2__x"])}, tmp_path, writes=[whole_server]) == []


def test_agent_read_tools_pass_the_classification(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    assert _errors({"a": _agent()}, tmp_path, writes=framework_write_tools()) == []


@pytest.mark.parametrize(
    "name", ["../escape", "a/b", "A", ".hidden", "-dash", "", "a b", "a.b", 7, "a\n"]
)
def test_unsafe_name(tmp_path: Path, name: Any) -> None:
    [error] = _errors({name: _agent()}, tmp_path)
    assert "unsafe" in error


def test_enabled_watchdog_needs_the_turn_state_hook(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    [error] = _errors({"a": _agent(enabled=True)}, tmp_path, hooks=["approval"])
    assert "turn-state" in error and "watchdogs.a" in error


def _turn_state_problems(base: Path, hooks: list[str]) -> list[str]:
    _files(base, "a")
    own = base / "hooks" / "osprey_turn_state.py"
    own.parent.mkdir()
    own.write_text("# the deployment's turn-state hook\n", encoding="utf-8")
    profile = _parse_profile(
        {"name": "w", "hooks": hooks, "watchdogs": {"a": _agent(enabled=True)}}
    )
    return [problem for problem in _validation_problems(profile, base) if "watchdogs.a" in problem]


@pytest.mark.parametrize("hooks", [["turn-state"], ["approval", "hooks/osprey_turn_state.py"]])
def test_enabled_watchdog_with_the_hook_passes(tmp_path: Path, hooks: list[str]) -> None:
    assert _turn_state_problems(tmp_path, hooks) == []


@pytest.mark.parametrize("hooks", [["hooks/turn-state"], ["approval"]])
def test_a_spelling_that_selects_no_hook_is_refused(tmp_path: Path, hooks: list[str]) -> None:
    [problem] = _turn_state_problems(tmp_path, hooks)
    assert "turn-state" in problem


def test_disabled_watchdog_does_not_need_the_hook(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    assert _errors({"a": _agent()}, tmp_path, hooks=[]) == []


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        (_decision(tools=["x"]), "does not read: tools"),
        (_decision(timeout_s=5), "does not read: timeout_s"),
        (_agent(threshold=0.5), "does not read: threshold"),
        (_agent(options=_OPTIONS), "does not read: options"),
        (_agent(colour="red"), "unknown keys: colour"),
        (_agent(enabled="yes"), "enabled must be true or false"),
        (_agent(tools="mcp__x__y"), "tools must be a list"),
        (_agent(model=3), "model must be"),
        (_agent(timeout_s=0), "timeout_s must be a positive"),
        (_agent(max_cost_usd=True), "max_cost_usd must be a positive"),
    ],
)
def test_keys_of_the_wrong_shape(tmp_path: Path, entry: dict[str, Any], fragment: str) -> None:
    _files(tmp_path, "a")
    assert any(fragment in e for e in _errors({"a": entry}, tmp_path))


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    errors = _errors(
        {"a": {"engine": "oracle", "enabled": True}, "../b": _agent(), "c": "nope"}, tmp_path
    )
    assert any("watchdogs.a.engine" in e for e in errors)
    assert any("instructions.md" in e for e in errors)
    assert any("unsafe" in e for e in errors)
    assert any("watchdogs.c must be a mapping" in e for e in errors)


def test_a_block_that_is_not_a_map_is_an_error(tmp_path: Path) -> None:
    [error] = _errors(["a"], tmp_path)
    assert "must be a mapping" in error


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


def test_profile_with_two_of_each_engine_builds(tmp_path: Path) -> None:
    """R1: any number of watchdogs, any mix of engines, several of one engine."""
    names = ("guard", "scope", "lattice", "budget")
    _files(tmp_path, *names)
    block = {
        "guard": _decision(enabled=True),
        "scope": _decision(threshold=0.7),
        "lattice": _agent(enabled=True),
        "budget": _agent(),
    }
    profile = load_profile(_write_profile(tmp_path, block, ["turn-state"]))
    assert profile.watchdogs == block
    definitions = parse_watchdogs_block(profile.watchdogs, base_dir=tmp_path)
    assert [d.name for d in definitions] == list(names)
    assert [d.engine for d in definitions] == ["decision", "decision", "agent", "agent"]
    assert [d.enabled for d in definitions] == [True, False, True, False]


def test_a_profile_with_no_block_has_no_watchdogs() -> None:
    assert _parse_profile({"name": "w"}).watchdogs == {}


def test_a_present_but_empty_block_reads_as_none() -> None:
    assert _parse_profile({"name": "w", "watchdogs": None}).watchdogs == {}


@pytest.mark.parametrize(
    ("block", "hooks", "fragment"),
    [
        ({"a": _agent(enabled=True)}, ["approval"], "turn-state"),
        ({"a": {"engine": "oracle"}}, ["turn-state"], "engine must be one of"),
        (
            {"a": _agent(tools=["mcp__python__execute"])},
            ["turn-state"],
            "reaches write tools",
        ),
    ],
)
def test_profile_load_reports_the_block(
    tmp_path: Path, block: dict[str, Any], hooks: list[str], fragment: str
) -> None:
    _files(tmp_path, "a")
    with pytest.raises(BuildProfileError, match=fragment):
        load_profile(_write_profile(tmp_path, block, hooks))


def test_profile_classification_takes_in_every_framework_write_tool(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    profile = _parse_profile({"name": "w", "hooks": ["turn-state"], "watchdogs": {"a": _agent()}})
    assert set(framework_write_tools()) <= set(profile._rendered_write_tools())


@pytest.mark.parametrize(
    ("server", "tool"),
    [
        ({"command": "x", "writes": ["push"]}, "mcp__site__push"),
        ({"command": "x", "permissions": {"ask": ["push"]}}, "mcp__site__push"),
        ({"command": "x", "hooks": {"pre_tool_use": ["writes_check"]}}, "mcp__site__read"),
    ],
)
def test_profile_classification_takes_in_the_facility_servers(
    tmp_path: Path, server: dict[str, Any], tool: str
) -> None:
    _files(tmp_path, "a")
    profile = _parse_profile(
        {
            "name": "w",
            "mcp_servers": {"site": server},
            "watchdogs": {"a": _agent(tools=[tool])},
        }
    )
    with pytest.raises(BuildProfileError, match="reaches write tools"):
        profile.validate(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {"control_system.write_tools": ["mcp__site__push"]},
        {"control_system": {"write_tools": ["mcp__site__push"]}},
    ],
)
def test_profile_classification_takes_in_the_control_system_extras(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    _files(tmp_path, "a")
    profile = _parse_profile(
        {"name": "w", "config": config, "watchdogs": {"a": _agent(tools=["mcp__site__push"])}}
    )
    with pytest.raises(BuildProfileError, match="reaches write tools"):
        profile.validate(tmp_path)


@pytest.mark.parametrize(
    ("config", "servers", "tool"),
    [
        pytest.param(
            {"claude_code.servers.phoebus2": {"extends": "phoebus"}},
            {},
            "mcp__phoebus2__phoebus_drive",
            id="extends-clone",
        ),
        pytest.param(
            {"claude_code": {"servers": {"site": {"command": "x", "writes": ["push"]}}}},
            {},
            "mcp__site__push",
            id="config-spelled-writes",
        ),
        pytest.param(
            {
                "claude_code": {
                    "servers": {
                        "site": {"command": "x", "hooks": {"pre_tool_use": ["writes_check"]}}
                    }
                }
            },
            {},
            "mcp__site__push",
            id="config-spelled-writes-check",
        ),
        pytest.param(
            {"claude_code": {"servers": {"site": {"enabled": True}}}},
            {},
            "mcp__site__anything",
            id="server-the-registry-would-skip",
        ),
        pytest.param(
            {},
            {"site": {"command": "x", "writes": [], "permissions": {"allow": ["delete_all"]}}},
            "mcp__site__delete_all",
            id="destructive-allow-on-a-declaring-server",
        ),
        pytest.param(
            {},
            {"site": {"command": "x", "writes": [], "permissions": {"ask": ["mcp__site__push"]}}},
            "mcp__site__push",
            id="prefixed-ask",
        ),
        pytest.param(
            {},
            {"site": {"command": "x", "permissions": {"ask": ["mcp__site__push"]}}},
            "mcp__site__push",
            id="prefixed-ask-without-a-list",
        ),
    ],
)
def test_profile_classification_follows_the_registry(
    tmp_path: Path, config: dict[str, Any], servers: dict[str, Any], tool: str
) -> None:
    _files(tmp_path, "a")
    profile = _parse_profile(
        {
            "name": "w",
            "config": config,
            "mcp_servers": servers,
            "watchdogs": {"a": _agent(tools=[tool])},
        }
    )
    with pytest.raises(BuildProfileError, match="reaches write tools"):
        profile.validate(tmp_path)


@pytest.mark.parametrize(
    ("tool", "classified", "reaches"),
    [
        ("mcp__si*", "mcp__site__.*", True),
        ("mcp__site__*", "mcp__site__push_*", True),
        ("mcp__site__push_*", "mcp__site__.*", True),
        ("mcp__site__p*", "mcp__site__(push|pull)", True),
        ("mcp__other*", "mcp__site__.*", False),
        ("mcp__site__read*", "mcp__site__push_.*", False),
    ],
)
def test_a_wildcard_entry_is_compared_with_a_pattern_classification(
    tmp_path: Path, tool: str, classified: str, reaches: bool
) -> None:
    _files(tmp_path, "a")
    errors = _errors({"a": _agent(tools=[tool])}, tmp_path, writes=[classified])
    assert bool(errors) is reaches
    assert all("reaches write tools" in error for error in errors)


def test_a_null_permission_list_reads_as_absent(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    server = {"command": "x", "writes": ["push"], "permissions": {"ask": None}}
    profile = _parse_profile(
        {
            "name": "w",
            "mcp_servers": {"site": server},
            "watchdogs": {"a": _agent(tools=["mcp__site__read"])},
        }
    )
    assert not [
        problem for problem in _validation_problems(profile, tmp_path) if "watchdogs" in problem
    ]


def test_a_server_no_render_could_resolve_is_reported_not_raised(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    spec = {"command": "x", "writes": "push"}
    profile = _parse_profile(
        {
            "name": "w",
            "config": {"claude_code": {"servers": {"site": spec}}},
            "watchdogs": {"a": _agent()},
        }
    )
    with pytest.raises(BuildProfileError, match="cannot be checked against the write tools"):
        profile.validate(tmp_path)


@pytest.mark.parametrize("tool", ["Bash", "Write", "Edit"])
def test_profile_classification_takes_in_the_denied_builtins(tmp_path: Path, tool: str) -> None:
    _files(tmp_path, "a")
    profile = _parse_profile({"name": "w", "watchdogs": {"a": _agent(tools=[tool])}})
    with pytest.raises(BuildProfileError, match="reaches write tools"):
        profile.validate(tmp_path)


def test_a_clean_agent_watchdog_validates_against_the_full_classification(tmp_path: Path) -> None:
    _files(tmp_path, "a")
    (tmp_path / "data").mkdir()
    profile = _parse_profile(
        {
            "name": "w",
            "data": "data",
            "config": {"claude_code.servers.phoebus2": {"extends": "phoebus"}},
            "watchdogs": {"a": _agent(tools=["Read", "mcp__controls__channel_read"])},
        }
    )
    assert not [
        problem for problem in _validation_problems(profile, tmp_path) if "watchdogs" in problem
    ]


def test_block_is_a_known_root_only_key_with_a_commented_template() -> None:
    assert "watchdogs" in _KNOWN_PROFILE_KEYS
    assert "watchdogs" in ROOT_ONLY_KEYS
    template = _COMMENTED_TEMPLATES["watchdogs"]
    assert "# watchdogs:" in template
    assert "max_cost_usd: 0.25" in template and "timeout_s: 90" in template


# ---------------------------------------------------------------------------
# Protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "watchdogs",
        "watchdogs.guard",
        "watchdogs.guard.enabled",
        "api.jev",
        "api.jev.base_url",
        "api.jev.api_key",
    ],
)
def test_watchdogs_and_jev_config_keys_are_protected(key: str) -> None:
    assert is_protected_key("config.yml", key) is True


def test_watchdogs_directory_is_a_source_entry() -> None:
    assert "watchdogs" in KNOWN_ROOT_ENTRIES
    assert "watchdogs" in PROFILE_SOURCE_ENTRIES


def test_watchdogs_directory_is_a_protected_root(tmp_path: Path) -> None:
    protected = resolve_protected_roots(tmp_path)
    assert (tmp_path / "watchdogs").resolve() in protected


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
def test_an_executed_write_to_the_watchdogs_directory_is_refused(
    project: _Project,  # noqa: F811
    mode: str,
) -> None:
    (project.root / "watchdogs" / "guard").mkdir(parents=True)
    out = project.probe("open('watchdogs/guard/instructions.md', 'w').close()", execution_mode=mode)
    assert out.startswith("REFUSED:"), out
