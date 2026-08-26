"""The fastmcp contract the MCP audit layer is built on, pinned in-repo.

``fastmcp`` is declared as a floor with no upper bound (``fastmcp>=3.4.4``),
which is the right dependency shape and also the reason this file exists: a
minor release is free to move anything the audit layer leans on, and three of
those things are load-bearing safety properties rather than conveniences.

1. **The middleware API.** ``Middleware.on_call_tool`` / ``add_middleware`` /
   ``ToolError`` / ``CallToolRequestParams.name`` are what
   :class:`~osprey.mcp_server.audit_middleware.AuditMiddleware` is written
   against. If any of them moves, every tool call stops being recorded and the
   readonly clamp stops applying — silently, because a middleware that is never
   invoked raises nothing.

2. **Settings are an import-time snapshot.** ``fastmcp.settings`` is built when
   ``fastmcp`` is first imported, and it is that snapshot — never the current
   environment — that ``FastMCP.run_async`` consults to pick a transport. The
   whole design of :func:`osprey.mcp_server.startup.fastmcp_transport` rests on
   that being true, and on the first ``fastmcp`` import in ``run_mcp_server``
   landing *after* ``load_dotenv_from_project()``. Both are pinned here in a
   real subprocess, against a project rendered by ``TemplateManager``, because
   an in-process test cannot observe first-import order: by the time pytest has
   collected this file, ``fastmcp`` is long imported.

3. **The skip predicate and the served transport are the same value.** The
   audit middleware is installed only for stdio, so "skipped" must mean "this
   process is visibly not speaking stdio" and never "this process speaks stdio
   without an audit layer". The subprocess cases below assert the predicate,
   the installed middleware chain, and the transport ``server.run()`` actually
   selects, all from one run.

Beyond the library contract, two invariants of our own that no other test
covers end to end, both against a project this file really renders:

* an ``extends`` clone is clamped by the render's own rewritten matchers, on the
  *verified* path — including a clone that pins ``OSPREY_SERVER_NAME`` (which it
  may) and one that tries to pin ``OSPREY_MCP_TOOL_PREFIX`` at a write-free
  server's name (which it may not);
* every server on ``registry.mcp.FRAMEWORK_SERVERS`` reaches its
  ``create_server()`` through ``startup.run_mcp_server``, so the single install
  site really does mean "every framework server is audited".
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import Middleware
from mcp.types import CallToolRequestParams

import osprey
from osprey.audit import writer
from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.templates.manager import TemplateManager
from osprey.mcp_server import audit_middleware as am
from osprey.mcp_server import startup
from osprey.registry.mcp import FRAMEWORK_SERVERS, TOOL_PREFIX_ENV, resolve_servers
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

pytestmark = pytest.mark.unit

#: The declared floor. Bumping the pin in ``pyproject.toml`` should bump this in
#: the same commit — that is the point of asserting it twice.
FASTMCP_FLOOR = (3, 4, 4)

#: The source tree the AST-only roster walk reads. Derived from the imported
#: package rather than from ``__file__`` relative paths, so it follows an
#: editable install wherever it points.
SRC_ROOT = Path(osprey.__file__).resolve().parent.parent

#: The clones the rendered fixture project declares. ``controls_pinned`` pins
#: ``OSPREY_SERVER_NAME`` — the one identity of the drift triad a facility is
#: allowed to pin — precisely so this file can show the tool prefix, and
#: therefore the clamp, does NOT follow it.
CLONE = "controls_ring"
PINNED_CLONE = "controls_pinned"
MIXED_CLONE = "python2"

#: A clone whose spec tries to pin the identity that is NOT pinnable, at the
#: name of a write-free server — the shape of "walk out of my own clamp set".
FORGED_CLONE = "controls_forged"
FORGED_PREFIX_TARGET = "osprey_workspace"


# ---------------------------------------------------------------------------
# A real rendered project
# ---------------------------------------------------------------------------


def _render_project(root: Path, *, servers: dict | None = None) -> Path:
    """Render a project the way ``osprey build`` lays one out, and return ``build/``.

    Same two-zone shape as ``tests/audit/test_resolver_equality.py``'s
    ``_render_two_zone_project``: ``create_project`` renders into
    ``<repo_root>/build`` and the ``project_root`` context override stamps the
    repo root into the render, so the env chain (``<repo_root>/.env``) and the
    hook config (``<repo_root>/build/.claude/hooks/``) sit in the two different
    zones a deployment really has.

    When *servers* is given, the render is re-run through
    ``regenerate_claude_code`` with those ``claude_code.servers`` entries — the
    same second render path ``tests/cli/test_hook_config_render.py`` uses, and
    the only one that accepts a server spec after the fact.
    """
    manager = TemplateManager()
    build = manager.create_project(
        project_name="build",
        output_dir=root,
        data_bundle="control_assistant",
        context={"project_root": str(root), "channel_finder_mode": "hierarchical"},
    )
    if servers is not None:
        config_file = build / "config.yml"
        config = yaml.safe_load(config_file.read_text())
        config.setdefault("claude_code", {})["servers"] = servers
        config_file.write_text(yaml.dump(config))
        manager.regenerate_claude_code(build)
    return build


@pytest.fixture(scope="module")
def cloned_render(tmp_path_factory) -> dict:
    """One rendered project carrying three ``extends`` clones.

    Module-scoped because rendering is the expensive part and nothing below
    mutates the render except the missing-hook-config case, which restores it.
    Returns the paths plus the two files the audit layer reads as data:
    ``hook_config.json`` and ``.mcp.json``.
    """
    root = tmp_path_factory.mktemp("cloned") / "deployment"
    build = _render_project(
        root,
        servers={
            CLONE: {"extends": "controls", "enabled": True},
            PINNED_CLONE: {
                "extends": "controls",
                "enabled": True,
                "env": {"OSPREY_SERVER_NAME": "controls"},
            },
            FORGED_CLONE: {
                "extends": "controls",
                "enabled": True,
                "env": {
                    "OSPREY_SERVER_NAME": "controls",
                    TOOL_PREFIX_ENV: FORGED_PREFIX_TARGET,
                },
            },
            MIXED_CLONE: {"extends": "python", "enabled": True},
        },
    )
    hook_config = build / ".claude" / "hooks" / "hook_config.json"
    return {
        "root": root,
        "build": build,
        "config": build / "config.yml",
        "hook_config": hook_config,
        "hook": json.loads(hook_config.read_text()),
        "mcp_json": json.loads((build / ".mcp.json").read_text()),
    }


@pytest.fixture
def audited(cloned_render, tmp_path, monkeypatch):
    """Point the audit middleware at the rendered project, writing to *tmp_path*.

    Clears every marker the middleware and the identity ladder read, so a test
    that does not set one is really exercising the absent case rather than
    inheriting a neighbour's.
    """
    for marker in (
        TERMINAL_USER_ENV,
        AUDIT_IDENTITY_ENV,
        writer.AUDIT_WRITER_ENV,
        am.POSTURE_ENV,
        am.POSTURE_SOURCE_ENV,
        am.POSTURE_SESSION_ENV,
        am.TOOL_PREFIX_ENV,
    ):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setenv(am.CONFIG_ENV, str(cloned_render["config"]))

    audit_root = tmp_path / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: audit_root)
    am.reset_audit_state()
    yield audit_root
    am.reset_audit_state()


def _sandbox(monkeypatch) -> None:
    monkeypatch.setenv(am.POSTURE_ENV, am.SANDBOX_MODE)


def _as_server(monkeypatch, prefix: str) -> None:
    """Run the middleware as the server the render calls *prefix*."""
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, prefix)


def _context(tool: str):
    from fastmcp.server.middleware.middleware import MiddlewareContext

    return MiddlewareContext(
        message=CallToolRequestParams(name=tool, arguments={}),
        method="tools/call",
    )


async def _call(tool: str) -> list[str]:
    """Drive one ``tools/call`` through the middleware; returns the hops taken."""
    seen: list[str] = []

    async def call_next(context):
        seen.append(context.message.name)
        return f"{context.message.name}-result"

    await am.AuditMiddleware().on_call_tool(_context(tool), call_next)
    return seen


async def _refused(tool: str) -> ToolError:
    """Assert *tool* is refused before it runs; returns the raised error."""
    seen: list[str] = []

    async def call_next(context):  # pragma: no cover - must never run
        seen.append(context.message.name)
        return "ran"

    with pytest.raises(ToolError) as excinfo:
        await am.AuditMiddleware().on_call_tool(_context(tool), call_next)
    assert seen == [], f"{tool} was refused but the tool still ran"
    return excinfo.value


def _records(audit_root: Path, surface: str) -> list[dict]:
    from osprey.utils.identity import acting_identity

    path = audit_root / acting_identity() / f"{surface}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _base_ctx() -> dict:
    """Minimal render context for ``resolve_servers`` (matches tests/registry)."""
    return {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
    }


# ---------------------------------------------------------------------------
# 1. The library contract: floor pin and middleware API
# ---------------------------------------------------------------------------


class TestFastmcpFloorDependency:
    """``fastmcp>=3.4.4``, no ceiling — and the API that floor promises."""

    def test_pyproject_declares_a_fastmcp_floor_with_no_upper_bound(self):
        """A ceiling here would be the wrong fix for a break this file catches.

        The dependency is deliberately open-ended, so the protection against a
        breaking minor is this test file rather than a pin. A ``<`` or ``==``
        appearing in the requirement means someone chose the other strategy and
        should say so out loud.
        """
        pyproject = SRC_ROOT.parent / "pyproject.toml"
        text = pyproject.read_text()
        requirements = [
            line.strip().strip(",").strip('"')
            for line in text.splitlines()
            if line.strip().strip(",").strip('"').startswith("fastmcp")
        ]
        assert requirements == [f"fastmcp>={'.'.join(str(part) for part in FASTMCP_FLOOR)}"], (
            f"unexpected fastmcp requirement(s) in {pyproject}: {requirements}"
        )

    def test_the_installed_fastmcp_satisfies_the_declared_floor(self):
        import fastmcp

        installed = tuple(int(part) for part in fastmcp.__version__.split(".")[:3])
        assert installed >= FASTMCP_FLOOR, (
            f"fastmcp {fastmcp.__version__} is below the declared floor "
            f"{'.'.join(str(part) for part in FASTMCP_FLOOR)}"
        )

    def test_the_audit_middleware_subclasses_the_hook_fastmcp_dispatches(self):
        """``on_call_tool`` must be a hook fastmcp knows about, not a private name.

        A rename on fastmcp's side turns ``AuditMiddleware.on_call_tool`` into a
        method nothing ever calls: the class still installs, every call is still
        served, and neither the ledger nor the clamp does anything at all.
        """
        assert issubclass(am.AuditMiddleware, Middleware)
        assert hasattr(Middleware, "on_call_tool")
        assert am.AuditMiddleware.on_call_tool is not Middleware.on_call_tool

    def test_add_middleware_appends_to_the_servers_own_chain(self):
        """The one call the install site makes, and where its effect shows up."""
        server = FastMCP("contract-check")
        before = list(server.middleware)
        server.add_middleware(am.AuditMiddleware())
        assert [type(entry) for entry in server.middleware] == [
            *[type(entry) for entry in before],
            am.AuditMiddleware,
        ]

    def test_a_tool_call_request_carries_the_bare_tool_name(self):
        """``context.message.name`` is the only input the middleware reads."""
        assert _context("channel_write").message.name == "channel_write"

    def test_tool_error_is_the_refusal_type_on_both_sides(self):
        """``make_error`` raises what fastmcp turns into ``isError=True``.

        The middleware both raises this type (its own refusal) and catches it
        (a tool's own refusal, recorded as ``tool_error``), so the two halves
        must stay the same class.
        """
        from osprey.mcp_server.errors import make_error

        with pytest.raises(ToolError):
            make_error("safety_error", "refused", ["do the other thing"])

    def test_the_transport_setting_is_snapshotted_when_fastmcp_is_imported(self, tmp_path):
        """The premise the whole predicate design rests on, checked directly.

        Two subprocesses: one where ``FASTMCP_TRANSPORT`` is in the environment
        before the import, one where it is written to ``os.environ`` after it.
        The first must be visible in ``fastmcp.settings``; the second must not.
        Both run from an empty directory, since fastmcp also reads a ``.env``
        from the working directory.
        """
        script = (
            "import os, sys; import fastmcp;"
            " os.environ['FASTMCP_TRANSPORT'] = 'sse';"
            " print(fastmcp.settings.transport)"
        )
        env = dict(os.environ)
        env.pop("FASTMCP_TRANSPORT", None)

        after = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert after.returncode == 0, after.stderr
        assert after.stdout.strip() == startup.STDIO_TRANSPORT

        env["FASTMCP_TRANSPORT"] = "http"
        before = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert before.returncode == 0, before.stderr
        assert before.stdout.strip() == "http"


# ---------------------------------------------------------------------------
# 2. Case (a): an extends clone is clamped by the render's rewritten matchers
# ---------------------------------------------------------------------------


class TestExtendsCloneClamps:
    """A clone's own name, not its template's, is what the clamp is spelled in.

    ``build_extended_server`` rewrites the template's hook matchers
    ``mcp__controls__… → mcp__<clone>__…``; the hook-config template turns every
    writes-check matcher into a ``write_tools`` entry; and the registry assigns
    ``OSPREY_MCP_TOOL_PREFIX = <clone>`` post-merge, so the running process
    qualifies its tool names the same way. Three separate mechanisms have to
    agree for a clone to be refused at all, and no other test drives all three
    against one real render.
    """

    @pytest.mark.parametrize("clone", [CLONE, PINNED_CLONE, FORGED_CLONE])
    def test_the_matcher_rewrite_reaches_the_rendered_write_tools(self, cloned_render, clone):
        hook = cloned_render["hook"]
        assert f"mcp__{clone}__channel_write" in hook["write_tools"]
        assert f"mcp__{clone}__" in hook["server_prefixes"]

    def test_the_mixed_clone_is_excluded_from_the_clamp_by_the_render(self, cloned_render):
        """``execute`` is read/write; a clone of it inherits that, not a clamp."""
        hook = cloned_render["hook"]
        assert f"mcp__{MIXED_CLONE}__execute" in hook["write_tools"]
        assert f"mcp__{MIXED_CLONE}__execute" in hook["mixed_read_write_tools"]

    @pytest.mark.parametrize("clone", [CLONE, PINNED_CLONE, FORGED_CLONE, MIXED_CLONE])
    def test_the_render_launches_each_clone_under_its_own_tool_prefix(self, cloned_render, clone):
        """Read off the rendered ``.mcp.json`` — the file that really launches it."""
        env = cloned_render["mcp_json"]["mcpServers"][clone]["env"]
        assert env[TOOL_PREFIX_ENV] == clone

    def test_a_pinned_server_name_does_not_move_the_tool_prefix(self, cloned_render):
        """The drift triad, on the clone that pins the pinnable member.

        ``OSPREY_SERVER_NAME`` is deliberately pinnable — a facility points a
        clone at an existing web-terminal panel — and it looks enough like the
        tool-prefix identity that the two are worth separating in a render, not
        just in a unit test.
        """
        env = cloned_render["mcp_json"]["mcpServers"][PINNED_CLONE]["env"]
        assert env["OSPREY_SERVER_NAME"] == "controls"
        assert env[TOOL_PREFIX_ENV] == PINNED_CLONE

    def test_a_spec_cannot_pin_the_tool_prefix_out_of_its_own_clamp_set(self, cloned_render):
        """The other member of the triad, which is NOT pinnable, end to end.

        A spec that could pin ``OSPREY_MCP_TOOL_PREFIX`` could launch a
        write-capable clone of ``controls`` advertising a write-free server's
        prefix: the render would list that prefix (so the middleware would call
        itself verified), the loaded clamp set holds no ``mcp__osprey_workspace__``
        write tool, and the clone would write freely under the sandbox posture.
        The assignment is therefore unconditional and post-merge, and it has to
        survive the whole render — which is what is checked here, on the file
        that really launches the server rather than on the resolver alone.
        """
        env = cloned_render["mcp_json"]["mcpServers"][FORGED_CLONE]["env"]
        assert env[TOOL_PREFIX_ENV] == FORGED_CLONE
        assert env[TOOL_PREFIX_ENV] != FORGED_PREFIX_TARGET

        resolved = resolve_servers(
            {
                "servers": {
                    FORGED_CLONE: {
                        "extends": "controls",
                        "env": {TOOL_PREFIX_ENV: FORGED_PREFIX_TARGET},
                    }
                }
            },
            _base_ctx(),
        )
        clone = next(server for server in resolved if server["name"] == FORGED_CLONE)
        assert clone["env"][TOOL_PREFIX_ENV] == FORGED_CLONE

    @pytest.mark.parametrize("clone", [CLONE, PINNED_CLONE, FORGED_CLONE])
    async def test_a_clone_write_tool_is_refused_under_the_sandbox_posture(
        self, audited, monkeypatch, clone
    ):
        _sandbox(monkeypatch)
        _as_server(monkeypatch, clone)

        error = await _refused("channel_write")
        assert json.loads(str(error))["error_type"] == "safety_error"

        records = _records(audited, clone)
        assert [r["subject"] for r in records] == [f"mcp__{clone}__channel_write"]
        assert records[0]["decision"] == "refused"
        assert records[0]["reason"] == am.REASON_POSTURE

    @pytest.mark.parametrize("clone", [CLONE, PINNED_CLONE, FORGED_CLONE])
    async def test_the_clone_refusal_comes_from_the_render_not_the_floor(
        self, audited, monkeypatch, clone
    ):
        """The verified path, which is the whole point of the clone case.

        Nothing in the framework floor is spelled ``mcp__controls_ring__``, so a
        refusal reported as ``clamp=hook_config`` proves the render's rewritten
        matcher did the refusing — and that the prefix was matched against the
        render's ``server_prefixes`` rather than falling back to a bare-name
        match, which would refuse the same call for the wrong reason.
        """
        _sandbox(monkeypatch)
        _as_server(monkeypatch, clone)

        await _refused("channel_write")

        assert _records(audited, clone)[0]["detail"] == am.CLAMP_SOURCE_LOADED

    async def test_a_clone_read_tool_still_runs_under_the_sandbox_posture(
        self, audited, monkeypatch
    ):
        _sandbox(monkeypatch)
        _as_server(monkeypatch, CLONE)

        assert await _call("channel_read") == ["channel_read"]
        assert _records(audited, CLONE)[0]["decision"] == "allowed"

    async def test_a_mixed_clone_tool_still_runs_under_the_sandbox_posture(
        self, audited, monkeypatch
    ):
        """A readonly ``execute`` is exactly what a sandboxed session is for."""
        _sandbox(monkeypatch)
        _as_server(monkeypatch, MIXED_CLONE)

        assert await _call("execute") == ["execute"]

    async def test_a_clone_write_tool_runs_under_the_writes_posture(self, audited, monkeypatch):
        """The clamp is the posture's, not the clone's: writes posture, writes."""
        monkeypatch.setenv(am.POSTURE_ENV, "readwrite")
        _as_server(monkeypatch, CLONE)

        assert await _call("channel_write") == ["channel_write"]


# ---------------------------------------------------------------------------
# 3. Case (b): the render is missing — floor behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def no_hook_config(cloned_render, audited, monkeypatch):
    """The same rendered project with ``hook_config.json`` taken away.

    Restored on teardown so the module-scoped render stays reusable. A missing
    file is the honest shape of the case the floor exists for: a deployment
    built before this feature, or a render that never completed.
    """
    path: Path = cloned_render["hook_config"]
    saved = path.read_text()
    path.unlink()
    am.reset_audit_state()
    yield audited
    path.write_text(saved)
    am.reset_audit_state()


class TestMissingHookConfigFloor:
    """No render to read: the framework floor, matched by bare tool name.

    The floor is spelled ``mcp__controls__channel_write`` — a name no clone
    carries — so this is also the case that would silently do nothing if the
    unverified branch matched fully-qualified. Both directions are pinned: what
    is still refused, and what keeps working.
    """

    async def test_a_clone_write_tool_is_still_refused(self, no_hook_config, monkeypatch):
        _sandbox(monkeypatch)
        _as_server(monkeypatch, CLONE)

        error = await _refused("channel_write")
        assert json.loads(str(error))["error_type"] == "safety_error"

    async def test_the_refusal_says_it_came_from_the_floor(self, no_hook_config, monkeypatch):
        _sandbox(monkeypatch)
        _as_server(monkeypatch, CLONE)

        await _refused("channel_write")

        assert _records(no_hook_config, CLONE)[0]["detail"] == am.CLAMP_SOURCE_FLOOR

    async def test_the_mixed_tool_survives_the_floor(self, no_hook_config, monkeypatch):
        """``_FALLBACK_MIXED_TOOLS`` is subtracted from the floor, not from a loaded list."""
        _sandbox(monkeypatch)
        _as_server(monkeypatch, MIXED_CLONE)

        assert await _call("execute") == ["execute"]

    async def test_a_read_tool_survives_the_floor(self, no_hook_config, monkeypatch):
        _sandbox(monkeypatch)
        _as_server(monkeypatch, CLONE)

        assert await _call("channel_read") == ["channel_read"]

    async def test_the_writes_posture_is_untouched_by_a_missing_render(
        self, no_hook_config, monkeypatch
    ):
        """The floor narrows the sandbox posture; it does not invent a new one."""
        monkeypatch.setenv(am.POSTURE_ENV, "readwrite")
        _as_server(monkeypatch, CLONE)

        assert await _call("channel_write") == ["channel_write"]

    async def test_the_missing_render_is_named_once_with_its_remedy(
        self, no_hook_config, monkeypatch, caplog
    ):
        """A degraded clamp must not be a silent one — nor a per-call one."""
        _sandbox(monkeypatch)
        _as_server(monkeypatch, CLONE)

        with caplog.at_level("WARNING", logger=am.logger.name):
            await _refused("channel_write")
            await _refused("channel_write")

        # Filtered to the middleware's own logger: the writer logs every refusal
        # at WARNING too, and counting those would make this a test of how many
        # times the fixture called a refused tool.
        warnings = [
            record
            for record in caplog.records
            if record.name == am.logger.name and record.levelname == "WARNING"
        ]
        assert len(warnings) == 1
        assert "osprey build" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# 4. Cases (c) and (d): the project .env, the predicate, and the real transport
# ---------------------------------------------------------------------------

#: A stand-in server package. Its module body imports fastmcp exactly the way a
#: real server module does (``server.py`` holds a module-scope ``FastMCP(...)``),
#: which is what makes the FIRST fastmcp import in the process land at
#: ``import_module(server_module)`` — after the dotenv load — rather than
#: earlier. It then replaces the two coroutines ``run_async`` dispatches to, so
#: ``server.run()`` resolves a transport for real and records which one it
#: reached instead of serving on it forever.
_PROBE_SERVER = '''\
"""A minimal stand-in for a framework server module."""

import fastmcp
from fastmcp import FastMCP

RECORD = {"transport_at_server_module_import": fastmcp.settings.transport}


async def _stdio(self, **kwargs):
    RECORD["served_transport"] = "stdio"


async def _http(self, transport=None, **kwargs):
    RECORD["served_transport"] = transport


FastMCP.run_stdio_async = _stdio
FastMCP.run_http_async = _http


def create_server():
    server = FastMCP("contract-probe")
    RECORD["server"] = server
    return server
'''

#: The driver: wraps the dotenv load to observe the import closure at exactly
#: the moment it runs, then hands the whole startup sequence to the real
#: ``run_mcp_server`` and reports what it did. Nothing about the ordering is
#: reimplemented here — that is the thing under test.
_PROBE_DRIVER = '''\
"""Run the real ``run_mcp_server`` and report what it observed."""

import json
import os
import sys

probe = {}
out_path = sys.argv[1]

import osprey.mcp_env as mcp_env

_real_dotenv = mcp_env.load_dotenv_from_project


def _observed_dotenv():
    probe["fastmcp_imported_at_dotenv"] = "fastmcp" in sys.modules
    probe["env_transport_before_dotenv"] = os.environ.get("FASTMCP_TRANSPORT")
    _real_dotenv()
    probe["env_transport_after_dotenv"] = os.environ.get("FASTMCP_TRANSPORT")


mcp_env.load_dotenv_from_project = _observed_dotenv

from osprey.mcp_server import startup

probe["fastmcp_imported_at_entry"] = "fastmcp" in sys.modules

try:
    startup.run_mcp_server("contract_probe.server")
finally:
    from contract_probe import server as probe_server

    record = dict(probe_server.RECORD)
    built = record.pop("server", None)
    if built is not None:
        record["middleware"] = [type(entry).__name__ for entry in built.middleware]
    probe.update(record)
    probe["predicate"] = startup.fastmcp_transport()
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(probe, handle)
'''


def _run_transport_probe(root: Path, env_text: str) -> dict:
    """Render a project with *env_text* as its ``.env`` and drive it in a subprocess.

    ``FASTMCP_TRANSPORT`` is removed from the subprocess environment, so the
    only way it can reach ``fastmcp.settings`` is through the project's own env
    chain — which is the case the import-order fix exists for.

    **The subprocess is deliberately started in an EMPTY directory.** ``fastmcp``
    configures its settings with pydantic-settings' ``env_file=".env"``, so it
    reads a ``.env`` from the *working directory* on its own, with no help from
    us (see :meth:`TestProjectEnvTransportIsHonored.\
test_fastmcp_reads_a_dotenv_from_the_working_directory_by_itself`). Running the
    probe from the repo root would hand the transport to fastmcp by that second
    route, and every assertion here would hold whether or not
    ``load_dotenv_from_project()`` ran first — which is exactly the thing under
    test. A neutral cwd leaves the deployment's env chain as the only path in,
    and it is also the honest shape: a container's cwd is the render zone, one
    level below the repo root the chain lives at.
    """
    build = _render_project(root)
    (root / ".env").write_text(env_text)

    probe_dir = root / "probe"
    (probe_dir / "contract_probe").mkdir(parents=True)
    (probe_dir / "contract_probe" / "__init__.py").write_text("")
    (probe_dir / "contract_probe" / "server.py").write_text(_PROBE_SERVER)
    driver = probe_dir / "driver.py"
    driver.write_text(_PROBE_DRIVER)

    cwd = root / "elsewhere"
    cwd.mkdir()
    assert not (cwd / ".env").exists()

    env = dict(os.environ)
    env.pop("FASTMCP_TRANSPORT", None)
    env["OSPREY_CONFIG"] = str(build / "config.yml")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), str(probe_dir)])

    out_path = root / "probe.json"
    result = subprocess.run(
        [sys.executable, str(driver), str(out_path)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(out_path.read_text())


@pytest.fixture(scope="module")
def transport_probes(tmp_path_factory) -> dict:
    """Both subprocess runs, once: a project ``.env`` asking for http, and one not.

    The second run is not decoration. Without it every assertion below could be
    satisfied by a predicate that answered ``http`` unconditionally, and the
    install site would look correct while never installing anything.
    """
    base = tmp_path_factory.mktemp("transport")
    return {
        "http": _run_transport_probe(base / "http", "FASTMCP_TRANSPORT=http\n"),
        "stdio": _run_transport_probe(base / "stdio", "OSPREY_CONTRACT_PROBE=1\n"),
    }


class TestProjectEnvTransportIsHonored:
    """A project ``.env`` transport reaches fastmcp's settings, and everyone agrees.

    The regression this exists for is an ordering one, and it is invisible in
    process: reading the transport anywhere before ``load_dotenv_from_project()``
    imports fastmcp against the pre-dotenv environment, and its settings then
    snapshot ``stdio`` while nothing else in the deployment changes. The
    predicate would report stdio, the middleware would install, and
    ``server.run()`` — reading the same frozen snapshot — would serve stdio
    against a deployment that asked for http. Each assertion below fails
    independently under that mutation.
    """

    def test_fastmcp_is_not_yet_imported_when_the_dotenv_load_runs(self, transport_probes):
        """The invariant itself: the ``.env`` lands before the snapshot is taken."""
        for name, probe in transport_probes.items():
            assert probe["fastmcp_imported_at_entry"] is False, name
            assert probe["fastmcp_imported_at_dotenv"] is False, name

    def test_the_transport_reaches_the_environment_only_via_the_project_env(self, transport_probes):
        http = transport_probes["http"]
        assert http["env_transport_before_dotenv"] is None
        assert http["env_transport_after_dotenv"] == "http"

        stdio = transport_probes["stdio"]
        assert stdio["env_transport_before_dotenv"] is None
        assert stdio["env_transport_after_dotenv"] is None

    def test_the_predicate_reports_the_transport_the_project_env_asked_for(self, transport_probes):
        assert transport_probes["http"]["predicate"] == "http"
        assert transport_probes["stdio"]["predicate"] == startup.STDIO_TRANSPORT

    def test_the_settings_snapshot_taken_at_server_import_already_has_it(self, transport_probes):
        """Not a duplicate of the predicate: this is the value fastmcp froze.

        ``fastmcp.settings.transport`` read inside the server module's own body
        is the earliest moment the snapshot exists. Asserting it here pins that
        the ``.env`` was loaded before that import, not merely before the
        predicate call at the install site.
        """
        assert transport_probes["http"]["transport_at_server_module_import"] == "http"
        assert (
            transport_probes["stdio"]["transport_at_server_module_import"]
            == startup.STDIO_TRANSPORT
        )

    def test_the_transport_server_run_actually_selects_is_the_same_value(self, transport_probes):
        """``server.run()`` with no argument resolves the same settings object.

        This is what makes the skip honest: a deployment whose env skips the
        audit middleware is visibly not speaking stdio. If fastmcp ever stopped
        defaulting ``run()``'s transport from ``settings``, the predicate and the
        served transport would part company here and nowhere else.
        """
        assert transport_probes["http"]["served_transport"] == "http"
        assert transport_probes["stdio"]["served_transport"] == startup.STDIO_TRANSPORT

    def test_the_middleware_is_installed_for_stdio_and_skipped_for_http(self, transport_probes):
        assert "AuditMiddleware" in transport_probes["stdio"]["middleware"]
        assert "AuditMiddleware" not in transport_probes["http"]["middleware"]

    def test_fastmcp_reads_a_dotenv_from_the_working_directory_by_itself(self, tmp_path):
        """A second, cwd-dependent route into the same snapshot — pinned, not relied on.

        ``fastmcp.Settings`` is configured with pydantic-settings'
        ``env_file=".env"``, so a ``.env`` in the process's working directory
        reaches ``fastmcp.settings`` with nothing of ours involved. Two
        consequences, and both are reasons the deployment's env chain must go
        through ``load_dotenv_from_project()`` rather than be left to fastmcp:

        * it is *not* a substitute — the chain lives at the repo root while a
          container's cwd is the render zone one level down, so fastmcp alone
          would never see it;
        * it is what would make the probe fixture above pass vacuously, which is
          why that fixture runs from an empty directory.

        If fastmcp ever drops this, nothing of ours breaks. The test is here so
        that the fixture's neutral cwd reads as a decision rather than an
        accident.
        """
        (tmp_path / ".env").write_text("FASTMCP_TRANSPORT=sse\n")
        env = dict(os.environ)
        env.pop("FASTMCP_TRANSPORT", None)

        result = subprocess.run(
            [sys.executable, "-c", "import fastmcp; print(fastmcp.settings.transport)"],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "sse"


# ---------------------------------------------------------------------------
# 5. The roster invariant: one install site means every server
# ---------------------------------------------------------------------------


def _module_path(module: str) -> Path:
    return SRC_ROOT.joinpath(*module.split("."))


def _expanded_roster() -> dict[str, str]:
    """Every framework server's launch package, channel-finder variants expanded.

    ``FRAMEWORK_SERVERS["channel-finder"].module`` carries the render-time
    placeholder ``{channel_finder_pipeline}``; each valid mode is a real package
    with its own ``__main__``, and all four must be covered rather than skipped
    for being unresolvable at import time.
    """
    roster: dict[str, str] = {}
    for name, definition in FRAMEWORK_SERVERS.items():
        module = definition.module
        if "{channel_finder_pipeline}" in module:
            for mode in VALID_CHANNEL_FINDER_MODES:
                roster[f"{name}[{mode}]"] = module.replace("{channel_finder_pipeline}", mode)
        else:
            roster[name] = module
    # The event dispatcher is not on the registry roster (it is a service, not a
    # `.mcp.json` server) but it is a FastMCP server all the same, and the
    # transport predicate is the only thing that keeps it off the audit path.
    roster["dispatch"] = "osprey.dispatch"
    return roster


#: The two names an entry point may reach. ``run_cf_main`` is a three-line
#: delegate to ``run_mcp_server``; a test below pins that so this list cannot
#: quietly grow a second startup sequence.
_AUDITED_ENTRY_POINTS = ("run_mcp_server", "run_cf_main")


def _called_names(tree: ast.AST) -> set[str]:
    """Every plain function name called anywhere in *tree*."""
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _fastmcp_constructing_packages() -> set[Path]:
    """Directories under ``src/osprey`` holding a module that builds a ``FastMCP``.

    AST only — nothing is imported, so this stays cheap and cannot be defeated
    by an import guard.
    """
    packages: set[Path] = set()
    for path in SRC_ROOT.joinpath("osprey").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "FastMCP":
                    packages.add(path.parent)
                    break
    return packages


class TestEveryFrameworkServerLaunchesThroughRunMcpServer:
    """The invariant that makes ONE install site mean "every server is audited".

    ``install_audit_middleware`` is called from exactly one place. That is only
    a safety property while every server actually goes through it — an invariant
    maintained by convention across fourteen entry-point files, whose failure
    mode is a silently unaudited server rather than a broken one. Checked by AST
    against the real source, so a new server package that calls ``mcp.run()``
    itself fails here instead of shipping without an audit layer.
    """

    def test_the_roster_walk_covers_every_entry_point_on_disk(self):
        """Guard against the walk silently covering nothing.

        Every ``__main__.py`` under ``osprey/mcp_server`` that belongs to a
        FastMCP-building package must appear in the expanded roster; the
        dispatch worker is a uvicorn app, not a FastMCP server, and is excluded
        by that rule rather than by name.
        """
        rostered = {_module_path(module) for module in _expanded_roster().values()}
        on_disk = {
            path.parent
            for path in SRC_ROOT.joinpath("osprey").rglob("__main__.py")
            if path.parent in _fastmcp_constructing_packages()
        }
        assert on_disk, "found no FastMCP entry points at all — the walk is broken"
        assert on_disk == rostered

    @pytest.mark.parametrize("name,module", sorted(_expanded_roster().items()))
    def test_every_framework_server_reaches_run_mcp_server(self, name, module):
        main = _module_path(module) / "__main__.py"
        assert main.is_file(), f"{name}: no entry point at {main}"

        tree = ast.parse(main.read_text(encoding="utf-8"), filename=str(main))
        called = _called_names(tree)
        assert called & set(_AUDITED_ENTRY_POINTS), (
            f"{name}: {main} calls {sorted(called)}, none of which is an audited "
            f"entry point ({', '.join(_AUDITED_ENTRY_POINTS)}) — this server would "
            f"start without the audit middleware"
        )

    @pytest.mark.parametrize("name,module", sorted(_expanded_roster().items()))
    def test_no_entry_point_serves_a_server_itself(self, name, module):
        """A ``mcp.run()`` in an entry point is the exact bypass this guards.

        It would start a perfectly working server that no install site ever
        touched. Nothing but the audited delegate may be called here.
        """
        main = _module_path(module) / "__main__.py"
        tree = ast.parse(main.read_text(encoding="utf-8"), filename=str(main))
        served = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"run", "run_async"}
        ]
        assert served == [], f"{name}: {main} serves the server itself via {served}"

    def test_run_cf_main_is_a_delegate_and_not_a_second_startup_sequence(self):
        """The one indirection the roster check allows, held to three lines.

        ``run_cf_main`` exists so the four channel-finder entry points keep
        their historical name. If it ever grew its own dotenv/logging/run
        sequence again it would need its own install site, and the two would
        drift — so the delegation is pinned rather than trusted.
        """
        source = SRC_ROOT / "osprey" / "mcp_server" / "channel_finder_common.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_cf_main"
        )
        assert "run_mcp_server" in _called_names(func)

    def test_run_mcp_server_holds_the_only_create_server_call_site(self):
        """The single funnel, stated as a property of the whole source tree.

        Every framework server exposes a ``create_server()`` factory; exactly
        one place in ``src/osprey`` calls one. A second call site is how a
        server gets built — and served — outside the sequence that installs the
        middleware.
        """
        call_sites: list[str] = []
        for path in SRC_ROOT.joinpath("osprey").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                named = (isinstance(func, ast.Name) and func.id == "create_server") or (
                    isinstance(func, ast.Attribute) and func.attr == "create_server"
                )
                if named:
                    call_sites.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")

        assert len(call_sites) == 1, f"expected one create_server() call site, got {call_sites}"
        assert call_sites[0].startswith("osprey/mcp_server/startup.py:")
