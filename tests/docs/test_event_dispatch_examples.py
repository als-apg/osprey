"""The event-dispatch how-to's trigger files and lists are what the dispatcher reads.

``docs/source/how-to/agent-interfaces/event-dispatch.rst`` shows three
``triggers.yml`` examples. Each is a file under
``docs/source/how-to/agent-interfaces/examples/`` that the page includes, and
each is checked here the way the dispatcher reads it:

* :func:`~osprey.dispatch.trigger_config.load_triggers` takes the file;
* every ``cron`` trigger has a schedule the cron source runs: a clock schedule
  :func:`~osprey.dispatch.clock_schedule.parse_clock_schedule` reads, or a
  positive ``interval_sec``;
* every ``epics_ca`` trigger is armed by
  :meth:`~osprey.dispatch.sources.epics_ca.EpicsCaSource.start` (with the
  connector build replaced by a fake, so no Channel Access is touched); the
  source skips a trigger with no ``pv``, an unknown ``edge`` or an unusable
  number, with only a log line, so a trigger it skips is one the page teaches
  wrongly.

The page's hand-written worker denylist is held equal to
:data:`~osprey.agent_runner.tool_names.DISPATCH_DENIED_TOOLS`, and every
environment variable its run commands set is one the started process, or the
compose file ``osprey up`` renders, reads. "Reads" means exactly one of:

* a ``os.environ.get/pop/[...]`` or ``os.getenv`` call in the named module
  whose first argument is the name, either as a literal or as a module-level
  string constant;
* a ``${NAME}`` (or ``${NAME:-...}``/``${NAME:?...}``) interpolation in the
  named compose template, which compose fills from the deploy environment;
* for ``FASTMCP_*``, a field of :class:`fastmcp.settings.Settings`, which reads
  its fields from the environment under that prefix.

The check runs in one direction: the processes read more variables than a
reader sets by hand (the compose files set the rest), so a variable the code
reads and the page does not name is not an error.
"""

from __future__ import annotations

import ast
import asyncio
import re
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS
from osprey.cli.build_profile_schema import DispatchConfig
from osprey.dispatch.clock_schedule import parse_clock_schedule
from osprey.dispatch.sources import epics_ca
from osprey.dispatch.trigger_config import (
    DEFAULT_MAX_CONCURRENT_RUNS,
    DEFAULT_MAX_QUEUE_DEPTH,
    load_triggers,
)
from osprey.utils.seconds import positive_seconds

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs/source/how-to/agent-interfaces/event-dispatch.rst"
EXAMPLES = REPO_ROOT / "docs/source/how-to/agent-interfaces/examples"
SRC = REPO_ROOT / "src/osprey"
TEMPLATES = SRC / "templates/services"

#: The trigger files the page includes, by file name.
TRIGGER_EXAMPLES = sorted(EXAMPLES.glob("triggers-*.yml"))

#: Each environment variable a run command on the page sets, and where it is read.
#: ``("py", module)`` is a module under ``src/osprey``; ``("compose", template)``
#: a compose template under ``src/osprey/templates/services``; ``("fastmcp",)``
#: the fastmcp settings. See the module docstring for what counts as a read.
ENV_READERS: dict[str, tuple[str, ...]] = {
    "OSPREY_DISPATCH_IMAGE": ("compose", "event_dispatcher/docker-compose.yml.j2"),
    "OSPREY_WORKER_IMAGE": ("compose", "dispatch_worker/docker-compose.yml.j2"),
    "OSPREY_PROJECT_DIR": ("py", "agent_runner/artifact_resolve.py"),
    "DISPATCH_WORKER_TOKEN": ("py", "mcp_server/dispatch_worker/credentials.py"),
    "DISPATCH_WORKER_PORT": ("py", "mcp_server/dispatch_worker/__main__.py"),
    "TRIGGERS_YML": ("py", "dispatch/server.py"),
    "EVENT_DISPATCHER_TOKEN": ("py", "dispatch/server.py"),
    "FASTMCP_TRANSPORT": ("fastmcp",),
    "FASTMCP_HOST": ("fastmcp",),
    "FASTMCP_PORT": ("fastmcp",),
}

#: A variable a shell command on the page sets: ``NAME=`` at the start of a
#: command, or after the whitespace that separates it from the one before.
_ENV_ASSIGNMENT = re.compile(r"(?:^|\s)([A-Z][A-Z0-9_]*)=")

#: The page's denylist paragraph, from its bold lead-in to the blank line.
_DENYLIST_START = "**Tool denylist (defence in depth).**"


def _code_blocks(text: str, language: str) -> list[str]:
    """The bodies of every ``.. code-block:: <language>`` on the page, dedented."""
    lines = text.splitlines()
    blocks: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() != f".. code-block:: {language}":
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for following in lines[i + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


def _env_names_set_on_page() -> set[str]:
    """Every variable a ``bash`` block on the page assigns, outside ``$(...)``."""
    names: set[str] = set()
    for block in _code_blocks(PAGE.read_text(encoding="utf-8"), "bash"):
        for line in block.splitlines():
            if line.lstrip().startswith("export "):
                continue  # the token-generation step; its names are checked as set below
            names.update(_ENV_ASSIGNMENT.findall(line))
    return names


def _python_env_reads(path: Path) -> set[str]:
    """The names *path* reads from the environment (see the module docstring)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def _name(arg: ast.expr) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return constants.get(arg.id)
        return None

    def _is_environ(node: ast.expr) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "environ"

    reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute):
            func = node.func
            if (func.attr in {"get", "pop"} and _is_environ(func.value)) or func.attr == "getenv":
                if (name := _name(node.args[0])) is not None:
                    reads.add(name)
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            if (name := _name(node.slice)) is not None:
                reads.add(name)
    return reads


def _compose_env_reads(path: Path) -> set[str]:
    """The ``${NAME...}`` interpolations in compose template *path*."""
    return set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", path.read_text(encoding="utf-8")))


def _is_read(name: str, reader: tuple[str, ...]) -> bool:
    kind = reader[0]
    if kind == "py":
        return name in _python_env_reads(SRC / reader[1])
    if kind == "compose":
        return name in _compose_env_reads(TEMPLATES / reader[1])
    from fastmcp.settings import Settings

    prefix = Settings.model_config.get("env_prefix", "")
    return name.startswith(prefix) and name[len(prefix) :].lower() in Settings.model_fields


# ── trigger files ────────────────────────────────────────────────────────────


def test_the_page_includes_every_trigger_example():
    """Each ``triggers-*.yml`` is on the page, and the page has no YAML left inline."""
    text = PAGE.read_text(encoding="utf-8")
    assert TRIGGER_EXAMPLES, f"no triggers-*.yml under {EXAMPLES}"
    included = set(re.findall(r"^\s*\.\. literalinclude:: examples/(\S+)$", text, re.M))
    assert {path.name for path in TRIGGER_EXAMPLES} <= included
    trigger_blocks = [
        b for b in _code_blocks(text, "yaml") if re.search(r"^(triggers|dispatcher):", b, re.M)
    ]
    assert trigger_blocks == [], "a triggers.yml example is inline; move it under examples/"


@pytest.mark.parametrize("path", TRIGGER_EXAMPLES, ids=lambda p: p.name)
def test_a_trigger_example_loads(path: Path):
    _dispatcher, triggers = load_triggers(str(path))
    for trigger in triggers:
        assert trigger.action.get("prompt"), f"{path.name}: {trigger.name} has no prompt"


def test_the_webhook_example_shows_the_defaults_it_says_it_shows():
    """The page says the pool limits and ``max_turns`` are shown at their defaults."""
    dispatcher, (trigger,) = load_triggers(str(EXAMPLES / "triggers-webhook.yml"))
    assert dispatcher.max_concurrent_runs == DEFAULT_MAX_CONCURRENT_RUNS
    assert dispatcher.max_queue_depth == DEFAULT_MAX_QUEUE_DEPTH
    assert trigger.max_turns == DispatchConfig(triggers="triggers.yml").max_turns


def _cron_triggers():
    return [
        pytest.param(trigger, id=f"{path.name}:{trigger.name}")
        for path in TRIGGER_EXAMPLES
        for trigger in load_triggers(str(path))[1]
        if trigger.source == "cron"
    ]


@pytest.mark.parametrize("trigger", _cron_triggers())
def test_a_cron_example_has_a_schedule_the_cron_source_runs(trigger):
    schedule = parse_clock_schedule(trigger.name, trigger.source_config)
    if schedule is None:
        interval = positive_seconds(trigger.source_config.get("interval_sec"))
        assert interval is not None, f"{trigger.name} has neither 'at' nor a positive interval_sec"
    else:
        assert schedule == trigger.schedule


def test_the_examples_show_both_kinds_of_cron_schedule():
    triggers = [t for param in _cron_triggers() for t in param.values]
    assert any(t.schedule is not None for t in triggers), "no clock-time cron example"
    assert any(t.schedule is None for t in triggers), "no interval cron example"


class _FakeConnector:
    """Records the channels the source subscribes to; never touches CA."""

    def __init__(self) -> None:
        self.channels: list[str] = []

    async def subscribe(self, channel: str, callback) -> str:  # noqa: ARG002 - connector signature
        self.channels.append(channel)
        return f"sub-{len(self.channels)}"

    async def unsubscribe(self, sub_id: str) -> None:  # noqa: ARG002 - connector signature
        return None

    async def disconnect(self) -> None:
        return None


def _epics_ca_examples() -> list[Path]:
    return [
        path
        for path in TRIGGER_EXAMPLES
        if any(t.source == "epics_ca" for t in load_triggers(str(path))[1])
    ]


def test_the_examples_show_an_epics_ca_trigger():
    assert _epics_ca_examples(), "no epics_ca trigger in the examples"


@pytest.mark.parametrize("path", _epics_ca_examples(), ids=lambda p: p.name)
def test_every_epics_ca_example_is_armed(path: Path):
    triggers = [t for t in load_triggers(str(path))[1] if t.source == "epics_ca"]
    connector = _FakeConnector()

    async def _arm() -> None:
        source = epics_ca.EpicsCaSource()
        with patch.object(epics_ca, "_connect", AsyncMock(return_value=connector)):
            await source.start(triggers, MagicMock())
        await source.stop()

    asyncio.run(_arm())
    assert connector.channels == [t.source_config["pv"] for t in triggers]


# ── worker denylist ──────────────────────────────────────────────────────────


def _denylist_paragraph() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.find(_DENYLIST_START)
    assert start != -1, f"no {_DENYLIST_START!r} paragraph in {PAGE}"
    end = text.find("\n\n", start)
    return text[start:end]


def test_the_page_lists_the_worker_denylist():
    """The paragraph names every entry of the worker's denylist, and nothing else.

    The page writes the dispatcher's own tool by its short name, ``manual_fire``;
    every other entry is spelled as the constant spells it.
    """
    named = set(re.findall(r"``([^`]+)``", _denylist_paragraph()))
    short = {name.removeprefix("mcp__event_dispatcher__"): name for name in DISPATCH_DENIED_TOOLS}
    found = {short.get(name, name) for name in named}
    assert found == set(DISPATCH_DENIED_TOOLS), (
        f"missing from the page: {sorted(set(DISPATCH_DENIED_TOOLS) - found)}; "
        f"on the page but not in the code: {sorted(found - set(DISPATCH_DENIED_TOOLS))}"
    )


# ── environment variables ────────────────────────────────────────────────────


def test_every_variable_the_page_sets_is_one_the_table_covers():
    """The run commands set exactly the variables :data:`ENV_READERS` lists."""
    assert _env_names_set_on_page() == set(ENV_READERS)


@pytest.mark.parametrize("name", sorted(ENV_READERS))
def test_a_variable_the_page_sets_is_read(name: str):
    reader = ENV_READERS[name]
    assert _is_read(name, reader), f"{name} is not read by {'/'.join(reader)}"


def test_the_exported_tokens_are_the_ones_the_processes_read():
    """The token step exports the two bearers the worker and dispatcher read."""
    exported = {
        match
        for block in _code_blocks(PAGE.read_text(encoding="utf-8"), "bash")
        for match in re.findall(r"^export ([A-Z][A-Z0-9_]*)=", block, re.M)
    }
    assert exported == {"EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"}
    worker = _python_env_reads(SRC / "mcp_server/dispatch_worker/credentials.py")
    dispatcher = _python_env_reads(SRC / "dispatch/server.py")
    assert exported <= worker
    assert exported <= dispatcher


def test_the_env_reader_finds_a_constant_and_a_literal(tmp_path: Path):
    module = tmp_path / "m.py"
    module.write_text(
        "import os\n"
        'TOKEN_ENV = "A_TOKEN"\n'
        "os.environ.pop(TOKEN_ENV)\n"
        'os.environ.get("B_NAME")\n'
        'os.environ["C_NAME"]\n'
        'os.getenv("D_NAME")\n'
        'other.get("E_NAME")\n',
        encoding="utf-8",
    )
    assert _python_env_reads(module) == {"A_TOKEN", "B_NAME", "C_NAME", "D_NAME"}
