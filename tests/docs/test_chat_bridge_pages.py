"""The chat-bridge how-tos show profiles the build takes and variables the bridge reads.

Each page under ``docs/source/how-to/agent-interfaces/chat-bridges/`` (Google
Chat, Nextcloud Talk, Microsoft Teams) has two things a deployer copies:

* a profile fragment (``<bridge>_bridge:``, ``dispatch:``, ``env.required``),
  which is a file under ``docs/source/how-to/agent-interfaces/examples/`` the
  page includes. It is loaded by :func:`~osprey.cli.build_profile.load_profile`,
  which also runs the build's bridge checks, beside a stub triggers file that
  declares the fragment's trigger. Its ``env.required`` is the set of
  variables the bridge refuses to start without, less the three a build fills
  in (``DISPATCH_TRIGGER`` and the two dispatch tokens);
* the environment variables named in the page's tables and ``.env`` blocks.

An environment variable **is read by a bridge** when it is one of:

* a name the bridge's ``from_env`` or :meth:`CoreConfig.from_env
  <osprey.bridges.core.config.CoreConfig.from_env>` reads: an upper-case
  string literal passed as the first argument of a call in that method's body
  (``e.get("NAME", ...)``, ``_f("NAME", ...)``);
* a ``${NAME}``, ``${NAME:-...}`` or ``${NAME:?...}`` interpolation in the
  value of an ``environment:`` entry (a line ``  SOME_VAR: ...``) of the
  bridge's compose template: compose fills it from the deploy environment and
  hands it to the bridge process. An interpolation in a comment, or in a
  service key such as ``image:``, is not one: the bridge never sees it.

Both directions are checked: every variable a page names is read by that
bridge (the only exception, :data:`SET_BY_COMPOSE`, is listed with its reason),
and every variable the bridge reads is named on its page. The second catches a
setting a deployer can change but cannot find.

The Teams page also configures the Azure relay. The settings its
``appsettings set`` command writes are held equal to the ones the relay names:
:data:`QUEUE_NAME_SETTING` and :data:`CONNECTION_SETTING` in
``relay/function_app.py``, and the names that file reads with
``os.environ.get``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from osprey.bridges.core.config import CoreConfig
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.nextcloud_talk.config import NextcloudBridgeConfig
from osprey.bridges.teams.config import CLOUDS, TeamsBridgeConfig
from osprey.cli.build_profile import load_profile
from osprey_connectors.errors import BuildProfileError
from tests.docs._rst_blocks import code_blocks

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES = REPO_ROOT / "docs/source/how-to/agent-interfaces/chat-bridges"
EXAMPLES = REPO_ROOT / "docs/source/how-to/agent-interfaces/examples"
SRC = REPO_ROOT / "src/osprey"
TEMPLATES = SRC / "templates/services"
RELAY = TEMPLATES / "teams_bridge/relay/function_app.py"

#: The required variables a build supplies, so a profile does not list them.
FILLED_BY_BUILD = frozenset({"DISPATCH_TRIGGER", "EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"})

#: Variables a page names that no ``from_env`` reads and no ``${...}`` passes in,
#: each with the reason. Each must still be a key the compose template sets.
SET_BY_COMPOSE: dict[str, str] = {
    "TZ": "rendered from system.timezone; read by the C library and Python's time module",
}


@dataclass(frozen=True)
class Bridge:
    page: str
    example: str
    profile_key: str
    config: type
    module: str
    template: str


BRIDGES = [
    Bridge(
        "google-chat.rst",
        "profile-gchat-bridge.yml",
        "gchat_bridge",
        GoogleChatBridgeConfig,
        "google_chat",
        "gchat_bridge",
    ),
    Bridge(
        "nextcloud-talk.rst",
        "profile-nextcloud-bridge.yml",
        "nextcloud_bridge",
        NextcloudBridgeConfig,
        "nextcloud_talk",
        "nextcloud_bridge",
    ),
    Bridge(
        "microsoft-teams.rst",
        "profile-teams-bridge.yml",
        "teams_bridge",
        TeamsBridgeConfig,
        "teams",
        "teams_bridge",
    ),
]

_bridges = pytest.mark.parametrize("bridge", BRIDGES, ids=lambda b: b.profile_key)

_ENV_NAME = r"[A-Z][A-Z0-9_]*"
#: A list-table row whose first cell names one or more variables.
_TABLE_ROW = re.compile(rf"^\s*\* - ((?:``{_ENV_NAME}``(?:, )?)+)\s*$")
#: An ``environment:`` entry of a compose template: an upper-case key, indented.
_COMPOSE_ENV_ENTRY = re.compile(rf"^\s+{_ENV_NAME}:\s")
#: A line of a ``.env`` block: ``NAME=value``, or the same commented out.
_ENV_LINE = re.compile(rf"^(?:#\s*)?({_ENV_NAME})=")


def _env_blocks(text: str) -> list[str]:
    """The ``bash`` blocks that are ``.env`` fragments: every line ``NAME=...``."""
    return [
        block
        for block in code_blocks(text, "bash")
        if block and all(_ENV_LINE.match(line) for line in block.splitlines() if line.strip())
    ]


def _page_env_names(bridge: Bridge) -> set[str]:
    """Every variable the page names in a table row, a ``.env`` block or the profile."""
    text = (PAGES / bridge.page).read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        if match := _TABLE_ROW.match(line):
            names.update(re.findall(rf"``({_ENV_NAME})``", match.group(1)))
    for block in _env_blocks(text):
        names.update(m.group(1) for line in block.splitlines() if (m := _ENV_LINE.match(line)))
    names.update(_example(bridge).get("env", {}).get("required", []))
    return names


def _from_env_reads(path: Path, class_name: str) -> set[str]:
    """The names *class_name*'s ``from_env`` in *path* reads (see the module docstring)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "from_env":
                return {
                    call.args[0].value
                    for call in ast.walk(item)
                    if isinstance(call, ast.Call)
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                    and re.fullmatch(_ENV_NAME, call.args[0].value)
                }
    raise AssertionError(f"no {class_name}.from_env in {path}")


def _compose_text(bridge: Bridge) -> str:
    return (TEMPLATES / bridge.template / "docker-compose.yml.j2").read_text(encoding="utf-8")


def _bridge_reads(bridge: Bridge) -> set[str]:
    """Every variable *bridge* reads, as the module docstring defines it."""
    core = _from_env_reads(SRC / "bridges/core/config.py", CoreConfig.__name__)
    own = _from_env_reads(SRC / "bridges" / bridge.module / "config.py", bridge.config.__name__)
    compose = {
        name
        for line in _compose_text(bridge).splitlines()
        if _COMPOSE_ENV_ENTRY.match(line)
        for name in re.findall(rf"\$\{{({_ENV_NAME})", line)
    }
    return core | own | compose


def _example(bridge: Bridge) -> dict:
    return yaml.safe_load((EXAMPLES / bridge.example).read_text(encoding="utf-8"))


def _required_at_startup(bridge: Bridge) -> list[str]:
    """The variables the bridge refuses to start without, in its own order."""
    with pytest.raises(ValueError, match="missing required config: ") as excinfo:
        bridge.config.from_env({}).require_startup()
    return str(excinfo.value).split(": ", 1)[1].split(", ")


def _write_profile(bridge: Bridge, directory: Path, declared_trigger: str | None = None) -> Path:
    """The page's fragment as a whole profile in *directory*, with a stub triggers file.

    The fragment is wrapped with the two keys every profile carries (``name``,
    ``data``). A fragment without a ``dispatch:`` block gets the one the page's
    prerequisites ask for.
    """
    fragment = _example(bridge)
    profile = {"name": "docs-example", "data": "data", **fragment}
    profile.setdefault("dispatch", {"triggers": "triggers.yml"})
    trigger = declared_trigger or fragment[bridge.profile_key]["trigger"]
    (directory / "data").mkdir(exist_ok=True)
    (directory / profile["dispatch"]["triggers"]).write_text(
        f"triggers:\n  - name: {trigger}\n    source: webhook\n    action:\n      prompt: Answer.\n",
        encoding="utf-8",
    )
    path = directory / "profile.yml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


# ── profile fragments ────────────────────────────────────────────────────────


@_bridges
def test_the_page_includes_its_profile_example(bridge: Bridge):
    text = (PAGES / bridge.page).read_text(encoding="utf-8")
    assert f".. literalinclude:: ../examples/{bridge.example}" in text
    inline = [b for b in code_blocks(text, "yaml") if f"{bridge.profile_key}:" in b]
    assert inline == [], f"{bridge.page} shows a {bridge.profile_key}: block inline"


@_bridges
def test_the_profile_example_loads(bridge: Bridge, tmp_path: Path):
    profile = load_profile(_write_profile(bridge, tmp_path))
    block = getattr(profile, bridge.profile_key)
    default = type(block)()
    # The fragment marks both settings "# default".
    assert (block.trigger, block.mentions) == (default.trigger, default.mentions)


@_bridges
def test_the_stub_triggers_file_is_what_lets_the_example_load(bridge: Bridge, tmp_path: Path):
    """The loader does check the trigger, so the passing load above means something."""
    with pytest.raises(BuildProfileError, match="is not declared in"):
        load_profile(_write_profile(bridge, tmp_path, declared_trigger="some-other-trigger"))


@_bridges
def test_the_profile_requires_what_the_bridge_cannot_start_without(bridge: Bridge):
    required = _example(bridge)["env"]["required"]
    expected = [name for name in _required_at_startup(bridge) if name not in FILLED_BY_BUILD]
    assert required == expected


# ── environment variables ────────────────────────────────────────────────────


@_bridges
def test_every_variable_the_page_names_is_read_by_the_bridge(bridge: Bridge):
    unread = _page_env_names(bridge) - _bridge_reads(bridge) - set(SET_BY_COMPOSE)
    assert unread == set(), (
        f"{bridge.page} names variable(s) the bridge never reads: {sorted(unread)}"
    )


@_bridges
def test_every_variable_the_bridge_reads_is_on_the_page(bridge: Bridge):
    missing = _bridge_reads(bridge) - _page_env_names(bridge)
    assert missing == set(), (
        f"{bridge.page} does not name variable(s) the bridge reads: {sorted(missing)}"
    )


@_bridges
def test_an_exempt_variable_is_one_the_compose_template_sets(bridge: Bridge):
    for name in set(SET_BY_COMPOSE) & _page_env_names(bridge):
        assert re.search(rf"^\s+{name}:", _compose_text(bridge), re.M), (
            f"{name} is exempt as set by compose, but {bridge.template}'s template does not set it"
        )


def test_the_page_names_each_teams_cloud():
    """The ``TEAMS_CLOUD`` row names exactly the clouds the bridge accepts."""
    text = (PAGES / "microsoft-teams.rst").read_text(encoding="utf-8")
    row = re.search(r"^(\s*)\* - ``TEAMS_CLOUD``\n(.*?)(?=^\1\* - )", text, re.M | re.S)
    assert row, "no TEAMS_CLOUD row on the Teams page"
    named = {literal for literal in re.findall(r"``([^`]+)``", row.group(2)) if literal in CLOUDS}
    assert named == set(CLOUDS)


# ── Teams relay app settings ─────────────────────────────────────────────────


def _relay_settings() -> set[str]:
    """The app settings the relay names or reads (see the module docstring)."""
    tree = ast.parse(RELAY.read_text(encoding="utf-8"))
    settings: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & {"QUEUE_NAME_SETTING", "CONNECTION_SETTING"}:
                settings.add(str(node.value.value).strip("%"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            settings.add(node.args[0].value)
    return settings


def test_the_relay_settings_on_the_page_are_the_relays():
    """The ``appsettings set`` command, plus the GCC High addition, sets every relay setting."""
    text = (PAGES / "microsoft-teams.rst").read_text(encoding="utf-8")
    (command,) = [b for b in code_blocks(text, "bash") if "functionapp config appsettings set" in b]
    on_page = set(re.findall(rf"\b({_ENV_NAME})=", command))
    on_page |= set(re.findall(rf"add ``({_ENV_NAME})=\w+`` to that same command", text))
    assert len(_relay_settings()) >= 3
    assert on_page == _relay_settings()


def test_the_teams_env_blocks_are_split_from_commands():
    """A shell command and ``.env`` lines never share one block on the Teams page."""
    text = (PAGES / "microsoft-teams.rst").read_text(encoding="utf-8")
    for block in code_blocks(text, "bash"):
        lines = [line for line in block.splitlines() if line.strip()]
        env_lines = [line for line in lines if _ENV_LINE.match(line) and "$(" not in line]
        assert not env_lines or len(env_lines) == len(lines), f"mixed block:\n{block}"
