"""An application's subagents: declared in app.yml, one checked file each, projected per login."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.agents import AppAgent, delegated_tools_phrase, parse_agent
from osprey.apps.errors import AppManifestError
from osprey.apps.loader import _check_agents, check_deployment
from osprey.apps.manifest import NO_HOOKS_MESSAGE, AppManifest, parse_manifest
from osprey.apps.projection import login_agents, project_login
from osprey.apps.rules import AppSelection, LoginSelection
from tests.fixtures.lifecycle_repo import (  # noqa: F401
    lifecycle_repo,
    lifecycle_repo_factory,
)

FRONTMATTER: dict[str, Any] = {
    "name": "knob-helper",
    "description": "Reads the knob and explains it.",
    "tools": ["mcp__knob__knob_read", "Read"],
    "delegates": ["knob_read"],
    "delegation": "For knob questions, delegate to knob-helper; do not call {delegated_tools} yourself.",
}


def _text(frontmatter: dict[str, Any], body: str = "You read the knob.\n") -> str:
    return f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{body}"


def _parse(**changes: Any) -> AppAgent:
    return parse_agent(_text({**FRONTMATTER, **changes}), app="knob", name="knob-helper")


def _refused(**changes: Any) -> list[tuple[str, str, str]]:
    with pytest.raises(AppManifestError) as exc:
        _parse(**changes)
    return [(v.field, v.rule, v.message) for v in exc.value.violations]


def _manifest(name: str = "knob", agents: list[str] | None = None, **extra: Any) -> AppManifest:
    data: dict[str, Any] = {
        "name": name,
        "tools": {"module": f"{name.replace('-', '_')}_app.tools", "read": [f"{name}_read"]},
        "agents": ["knob-helper"] if agents is None else agents,
        **extra,
    }
    return parse_manifest(data, app=name)


def _login(**apps: bool) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={n: AppSelection(shown=s) for n, s in apps.items()},
        tier="asked",
        shared_cards=(),
    )


# --- the manifest key ----------------------------------------------------------


def test_manifest_lists_agents() -> None:
    assert _manifest().agents == ["knob-helper"]


def test_agents_default_to_none() -> None:
    assert _manifest(agents=[]).agents == []


def test_an_agent_name_is_a_slug() -> None:
    with pytest.raises(AppManifestError):
        _manifest(agents=["Knob Helper"])


# --- the agent file ------------------------------------------------------------


def test_the_frontmatter_is_read() -> None:
    agent = _parse(summary="Reads the knob", disallowedTools=["Bash"], maxTurns=10)
    assert agent.name == "knob-helper"
    assert agent.tools == ["mcp__knob__knob_read", "Read"]
    assert agent.delegates == ["knob_read"]
    assert agent.disallowed_tools == ["Bash"]
    assert agent.max_turns == 10
    assert agent.summary == "Reads the knob"


def test_tools_may_be_one_comma_separated_line() -> None:
    assert _parse(tools="mcp__knob__knob_read, Read").tools == ["mcp__knob__knob_read", "Read"]


@pytest.mark.parametrize("key", ["description", "tools", "delegation"])
def test_a_required_key_is_required(key: str) -> None:
    frontmatter = {k: v for k, v in FRONTMATTER.items() if k != key}
    with pytest.raises(AppManifestError) as exc:
        parse_agent(_text(frontmatter), app="knob", name="knob-helper")
    assert [v.field for v in exc.value.violations] == [f"agents.knob-helper.{key}"]


def test_the_name_is_the_file_name() -> None:
    [(field, rule, message)] = _refused(name="other")
    assert (field, rule) == ("agents.knob-helper.name", "agent_name")
    assert "knob-helper" in message


def test_an_unknown_key_is_refused() -> None:
    [(field, rule, _message)] = _refused(color="blue")
    assert (field, rule) == ("agents.knob-helper.color", "extra_forbidden")


def test_a_file_without_frontmatter_is_refused() -> None:
    with pytest.raises(AppManifestError) as exc:
        parse_agent("You read the knob.\n", app="knob", name="knob-helper")
    assert exc.value.rule == "agent_frontmatter"


@pytest.mark.parametrize(
    ("key", "value", "rule", "message"),
    [
        ("model", "opus", "agent_key_refused", "the build sets the model from"),
        ("hooks", {"PreToolUse": []}, "no_hooks", NO_HOOKS_MESSAGE),
        ("permissionMode", "acceptEdits", "agent_key_refused", "tier and the write switch"),
        ("mcpServers", ["x"], "agent_key_refused", "an app's tools are its `tools:` server"),
    ],
)
def test_a_refused_key_names_its_replacement(key: str, value: Any, rule: str, message: str) -> None:
    [(field, got_rule, got_message)] = _refused(**{key: value})
    assert (field, got_rule) == (f"agents.knob-helper.{key}", rule)
    assert message in got_message


def test_every_refused_key_is_reported() -> None:
    refused = _refused(model="opus", hooks={}, permissionMode="x", mcpServers=[])
    assert sorted(field for field, _rule, _message in refused) == [
        "agents.knob-helper.hooks",
        "agents.knob-helper.mcpServers",
        "agents.knob-helper.model",
        "agents.knob-helper.permissionMode",
    ]


# --- the brief note ------------------------------------------------------------


@pytest.mark.parametrize(
    ("tools", "phrase"),
    [
        (["a"], "`a`"),
        (["a", "b"], "`a` or `b`"),
        (
            ["list_concepts", "read_concept", "search"],
            "`list_concepts`, `read_concept`, or `search`",
        ),
    ],
)
def test_delegated_tools_render_in_house_form(tools: list[str], phrase: str) -> None:
    assert delegated_tools_phrase(tools) == phrase


def test_the_note_names_the_delegated_tools() -> None:
    agent = _parse()
    assert agent.brief_note() == (
        "For knob questions, delegate to knob-helper; do not call `knob_read` yourself."
    )


# --- the loader rules, as `osprey app check` reports them ----------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _app(repo: Path, name: str = "knob", **extra: Any) -> Path:
    """List one application with a read tool and the agents ``extra`` names."""
    app_dir = repo / "apps" / name
    data = {
        "name": name,
        "tools": {"module": f"{name}_app.tools", "read": [f"{name}_read"]},
        **extra,
    }
    _write(app_dir / "app.yml", yaml.safe_dump(data, sort_keys=False))
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"] = {**(profile.get("apps") or {}), name: {}}
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return app_dir


def _agent_file(app_dir: Path, **changes: Any) -> None:
    frontmatter = {**FRONTMATTER, **changes}
    _write(app_dir / "agents" / f"{frontmatter['name']}.md", _text(frontmatter))


def _rules(repo: Path) -> list[tuple[str, str]]:
    with pytest.raises(AppManifestError) as exc:
        check_deployment(repo)
    return [(v.field, v.rule) for v in exc.value.violations]


def test_a_listed_agent_with_its_file_passes(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]))
    found = check_deployment(lifecycle_repo)
    [agent] = found.agents["knob"]
    assert agent.name == "knob-helper"


def test_a_listed_agent_needs_its_file(lifecycle_repo: Path) -> None:  # noqa: F811
    _app(lifecycle_repo, agents=["knob-helper"])
    assert _rules(lifecycle_repo) == [("agents.knob-helper", "agent_missing")]


def test_two_applications_cannot_carry_one_agent_name(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]))
    _agent_file(
        _app(lifecycle_repo, name="dial", agents=["knob-helper"]),
        tools=["Read"],
        delegates=[],
        delegation="Delegate dial questions.",
    )
    assert ("agents.knob-helper", "agent_collision") in _rules(lifecycle_repo)


@pytest.mark.parametrize("name", ["data-visualizer", "channel-finder"])
def test_a_framework_agent_name_is_reserved(
    lifecycle_repo: Path,  # noqa: F811
    name: str,
) -> None:
    _agent_file(
        _app(lifecycle_repo, agents=[name]),
        name=name,
        tools=["Read"],
        delegates=[],
        delegation="Delegate.",
    )
    assert _rules(lifecycle_repo) == [(f"agents.{name}", "agent_reserved")]


def test_an_agent_may_share_a_name_with_its_apps_skill(lifecycle_repo: Path) -> None:  # noqa: F811
    app_dir = _app(lifecycle_repo, agents=["knob-helper"], skills=["knob-helper"])
    _agent_file(app_dir)
    _write(
        app_dir / "skills" / "knob-helper" / "SKILL.md",
        "---\nname: knob-helper\ndescription: d\n---\n",
    )
    check_deployment(lifecycle_repo)


def test_a_tool_of_the_app_must_be_declared(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(
        _app(lifecycle_repo, agents=["knob-helper"]),
        tools=["mcp__knob__knob_read", "mcp__knob__knob_spin"],
    )
    assert _rules(lifecycle_repo) == [("agents.knob-helper.tools", "agent_tool_unknown")]


def test_a_delegated_tool_must_be_held(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]), tools=["Read"])
    assert _rules(lifecycle_repo) == [("agents.knob-helper.delegates", "delegate_not_held")]


def test_a_delegated_tool_must_be_a_tool_of_the_app(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(
        _app(lifecycle_repo, agents=["knob-helper"]),
        tools=["mcp__knob__knob_read"],
        delegates=["knob_spin"],
    )
    assert _rules(lifecycle_repo) == [("agents.knob-helper.delegates", "agent_tool_unknown")]


def test_delegated_tools_need_the_placeholder(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]), delegation="Delegate knob work.")
    assert _rules(lifecycle_repo) == [("agents.knob-helper.delegation", "delegates_unused")]


def test_the_placeholder_needs_delegated_tools(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]), delegates=[])
    assert _rules(lifecycle_repo) == [("agents.knob-helper.delegates", "delegates_missing")]


def test_hooks_in_an_agents_frontmatter_are_refused(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]), hooks={"PreToolUse": []})
    assert _rules(lifecycle_repo) == [("agents.knob-helper.hooks", "no_hooks")]


def test_an_approval_tool_is_a_tool_of_the_app(tmp_path: Path) -> None:
    manifest = parse_manifest(
        {
            "name": "knob",
            "containers": {"knob": {"runtime": "osprey", "drives_target": True}},
            "tools": {
                "module": "knob_app.tools",
                "approval": [{"tool": "knob_write", "prompt": "Write the knob"}],
            },
            "agents": ["knob-helper"],
        },
        app="knob",
    )
    _agent_file(
        tmp_path / "apps" / "knob",
        tools=["mcp__knob__knob_write"],
        delegates=["knob_write"],
    )
    [agent] = _check_agents(tmp_path, (manifest,))["knob"]
    assert agent.delegates == ["knob_write"]


def test_an_override_of_an_app_agent_is_refused(lifecycle_repo: Path) -> None:  # noqa: F811
    _agent_file(_app(lifecycle_repo, agents=["knob-helper"]))
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"]["claude_code.agents.knob-helper.enabled"] = False
    (lifecycle_repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)
    [violation] = exc.value.violations
    assert violation.rule == "app_agent_override"
    assert "apps.knob.shown: false" in violation.message


# --- the projection ------------------------------------------------------------


def test_login_agents_follow_shown() -> None:
    agents = {"knob": (_parse(),)}
    manifests = [_manifest(), _manifest("dial", agents=[])]
    assert login_agents(manifests, agents, _login(knob=True)) == [("knob", agents["knob"][0])]
    assert login_agents(manifests, agents, _login(knob=False)) == []
    assert login_agents(manifests, agents, _login()) == []


def test_a_shown_app_projects_its_agents_note() -> None:
    projected = project_login([_manifest()], {}, _login(knob=True), agents={"knob": (_parse(),)})
    assert projected["apps"]["knob"]["agents"] == {
        "knob-helper": {
            "description": "Reads the knob and explains it.",
            "delegation": (
                "For knob questions, delegate to knob-helper; do not call `knob_read` yourself."
            ),
        }
    }


def test_an_app_of_agents_only_is_projected() -> None:
    manifest = parse_manifest({"name": "guide", "agents": ["knob-helper"]}, app="guide")
    agent = _parse(tools=["Read"], delegates=[], delegation="Delegate guide questions.")
    projected = project_login([manifest], {}, _login(guide=True), agents={"guide": (agent,)})
    assert projected["apps"] == {
        "guide": {
            "agents": {
                "knob-helper": {
                    "description": "Reads the knob and explains it.",
                    "delegation": "Delegate guide questions.",
                }
            }
        }
    }
    assert projected["mcp_servers"] == {}


def test_an_app_without_agents_projects_no_agents_key() -> None:
    projected = project_login([_manifest(agents=[])], {}, _login(knob=True))
    assert "agents" not in projected["apps"]["knob"]


# --- the block closes where the session closes it: on a line that is exactly --- --


def _hidden(line: str, key: str) -> str:
    """A complete agent frontmatter, then ``line``, then a refused ``key``, then the close."""
    head = yaml.safe_dump(FRONTMATTER, sort_keys=False)
    return f"---\n{head}{line}\n{key}: opus\n---\n\nYou read the knob.\n"


@pytest.mark.parametrize("key", ["hooks", "model"])
@pytest.mark.parametrize(
    "line", ["---x: 1", "--- x: 1", "----"], ids=["dashes-then-text", "dashes-space-text", "four"]
)
def test_a_line_starting_with_dashes_never_hides_a_refused_key(line: str, key: str) -> None:
    with pytest.raises(AppManifestError) as exc:
        parse_agent(_hidden(line, key), app="knob", name="knob-helper")
    assert [(v.field, v.rule) for v in exc.value.violations] == [
        ("agents.knob-helper", "agent_frontmatter")
    ]


def test_an_unclosed_frontmatter_is_refused() -> None:
    text = _text(FRONTMATTER).replace("---\n\n", "\n", 1)
    with pytest.raises(AppManifestError) as exc:
        parse_agent(text, app="knob", name="knob-helper")
    assert exc.value.rule == "agent_frontmatter"


def test_a_horizontal_rule_in_the_body_is_body() -> None:
    text = _text(FRONTMATTER, body="You read the knob.\n\n---\n\nmodel: opus\n---\n")
    assert parse_agent(text, app="knob", name="knob-helper").name == "knob-helper"


def test_trailing_blanks_on_the_delimiters_are_allowed() -> None:
    text = _text(FRONTMATTER).replace("---\n", "--- \t\n")
    assert parse_agent(text, app="knob", name="knob-helper").name == "knob-helper"


def _agent_bytes(app_dir: Path, raw: bytes) -> None:
    path = app_dir / "agents" / "knob-helper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


_ENCODINGS = [
    lambda text: text.replace("\n", "\r\n").encode("utf-8"),
    lambda text: b"\xef\xbb\xbf" + text.encode("utf-8"),
]


@pytest.mark.parametrize("encode", _ENCODINGS, ids=["crlf", "byte-order-mark"])
def test_an_agent_file_reads_whatever_the_line_endings_or_bom(
    lifecycle_repo: Path,  # noqa: F811
    encode: Any,
) -> None:
    _agent_bytes(_app(lifecycle_repo, agents=["knob-helper"]), encode(_text(FRONTMATTER)))
    [agent] = check_deployment(lifecycle_repo).agents["knob"]
    assert agent.name == "knob-helper"


@pytest.mark.parametrize("encode", _ENCODINGS, ids=["crlf", "byte-order-mark"])
def test_dashes_never_hide_a_refused_key_whatever_the_line_endings_or_bom(
    lifecycle_repo: Path,  # noqa: F811
    encode: Any,
) -> None:
    _agent_bytes(_app(lifecycle_repo, agents=["knob-helper"]), encode(_hidden("---x: 1", "hooks")))
    assert _rules(lifecycle_repo) == [("agents.knob-helper", "agent_frontmatter")]


def test_an_opening_a_looser_reader_takes_never_hides_a_refused_key() -> None:
    text = _hidden("name: knob-helper", "hooks").replace("---\n", "---\u00a0\n", 1)
    with pytest.raises(AppManifestError) as exc:
        parse_agent(text, app="knob", name="knob-helper")
    assert exc.value.rule == "agent_frontmatter"
