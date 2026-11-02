"""Tests for the MCP audit middleware — every tools/call recorded, writes clamped.

The middleware is the server-side half of the posture clamp: the PreToolUse hook
fails open by design, so these tests care most about the paths where something is
missing or stale. Each degrade case is pinned in both directions — what is still
refused, and what keeps working — because a clamp that quietly widens is as much a
bug as one that quietly disappears.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext
from mcp.types import CallToolRequestParams, RequestParams

from osprey.agent_record import writer as record_writer
from osprey.audit.dedup import mark_recorded
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    POSTURE_SOURCE_PROCESS,
    POSTURE_SOURCE_SPAWN,
)
from osprey.mcp_server import audit_middleware as am
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import write_control_context

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

CONTROLS_WRITE = "mcp__controls__channel_write"
PYTHON_EXECUTE = "mcp__python__execute"
SITECTL_WRITE = "mcp__sitectl__do_thing"

#: The shape a real `extends` clone is launched with: the registry assigns
#: OSPREY_MCP_TOOL_PREFIX = the clone's own name, so nothing in the framework
#: floor is spelled with it. Every fail-closed test uses this rather than
#: `controls`, whose subject coincides with a floor entry and would pass whether
#: or not the rule does anything.
CLONE_PREFIX = "controls_ring"


#: A ``control_system`` section whose tier allows writes and whose live
#: machine is real, so ``live`` resolves.
TIER_ALLOWED_SECTION = {"type": "epics", "writes": "allowed"}

#: The same deployment with a tier that allows nothing.
TIER_NEVER_SECTION = {"type": "epics", "writes": "never"}


#: Passed as ``mixed=`` to omit the key entirely — a pre-feature render. A bare
#: ``None`` would be indistinguishable from "the default healthy render", which
#: is exactly the mistake that makes every other test in the file weaker.
NO_MIXED_KEY = object()


def _hook_config(
    *,
    server_prefixes: list[str] | None = None,
    write_tools: list[str] | None = None,
    mixed: list[str] | object | None = None,
) -> dict:
    """A healthy hook_config.json body; ``mixed=NO_MIXED_KEY`` drops the key."""
    body: dict = {
        "server_prefixes": (
            ["mcp__controls__", "mcp__python__"] if server_prefixes is None else server_prefixes
        ),
        "approval_prefixes": [],
        "write_tools": [CONTROLS_WRITE, PYTHON_EXECUTE] if write_tools is None else write_tools,
    }
    if mixed is not NO_MIXED_KEY:
        body["mixed_read_write_tools"] = [PYTHON_EXECUTE] if mixed is None else mixed
    return body


@pytest.fixture
def project(tmp_path, monkeypatch, recorded_entries):
    """A render zone with ``build/config.yml`` and a hook config beside it.

    Returns a small handle: ``.write(**kwargs)`` re-renders the hook config,
    ``.remove()`` deletes it, ``.corrupt()`` leaves unparseable bytes, and
    ``.entries()`` reads the audit entries recorded so far.
    """
    for marker in (
        TERMINAL_USER_ENV,
        AUDIT_IDENTITY_ENV,
        am.POSTURE_ENV,
        am.POSTURE_SOURCE_ENV,
        am.POSTURE_SESSION_ENV,
    ):
        monkeypatch.delenv(marker, raising=False)

    build = tmp_path / "build"
    (build / ".claude" / "hooks").mkdir(parents=True)
    config = build / "config.yml"
    config.write_text("control_system: {}\n")
    monkeypatch.setenv(am.CONFIG_ENV, str(config))
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, "controls")

    am.reset_audit_state()

    # The backstop's section, through the audit package's one reader. The
    # config loader is a process-wide singleton that does not follow
    # OSPREY_CONFIG, so the seam is patched rather than the file relied on.
    from osprey.audit import posture as posture_module

    monkeypatch.setattr(posture_module, "control_system_section", lambda: _Project.section)

    class _Project:
        hook_config = build / ".claude" / "hooks" / "hook_config.json"
        root = tmp_path

        #: A tier that allows writes, on a deployment whose live machine is
        #: real — the state in which only a read-only run is refused here.
        section: object = dict(TIER_ALLOWED_SECTION)

        @staticmethod
        def entries() -> list:
            return recorded_entries("audit")

        @staticmethod
        def write(**kwargs) -> None:
            _Project.hook_config.write_text(json.dumps(_hook_config(**kwargs)))
            _touch_forward(_Project.hook_config)

        @staticmethod
        def raw(text: str) -> None:
            _Project.hook_config.write_text(text)
            _touch_forward(_Project.hook_config)

        @staticmethod
        def remove() -> None:
            _Project.hook_config.unlink()

    _Project.write()
    yield _Project
    am.reset_audit_state()


_MTIME_STEP = [0]


def _touch_forward(path: Path) -> None:
    """Move *path*'s mtime strictly forward, so a re-render is always visible."""
    _MTIME_STEP[0] += 1
    stamp = path.stat().st_mtime + _MTIME_STEP[0]
    os.utime(path, (stamp, stamp))


def _context(
    tool: str, meta: dict | None = None, *, message_meta: dict | None = None
) -> MiddlewareContext:
    """One tools/call context; *meta* rides where FastMCP leaves the client's ``_meta``.

    FastMCP rebuilds ``context.message`` without the client's meta and keeps it
    on ``fastmcp_context.request_context.meta``; *message_meta* puts it on the
    message instead, the fallback path.
    """
    params: dict = {"name": tool, "arguments": {}}
    if message_meta is not None:
        params["_meta"] = message_meta
    fastmcp_context = None
    if meta is not None:
        fastmcp_context = SimpleNamespace(
            request_context=SimpleNamespace(meta=RequestParams.Meta(**meta))
        )
    return MiddlewareContext(
        message=CallToolRequestParams.model_validate(params),
        fastmcp_context=fastmcp_context,  # type: ignore[arg-type]
        method="tools/call",
    )


async def _call(
    mw,
    tool: str,
    *,
    raises: BaseException | None = None,
    meta: dict | None = None,
    message_meta: dict | None = None,
    result: object = None,
):
    """Drive one tools/call; returns ``(result, seen)`` where *seen* counts hops."""
    seen: list[str] = []

    async def call_next(context):
        seen.append(context.message.name)
        if raises is not None:
            raise raises
        return f"{context.message.name}-result" if result is None else result

    returned = await mw.on_call_tool(_context(tool, meta, message_meta=message_meta), call_next)
    return returned, seen


async def _refused(mw, tool: str, *, meta: dict | None = None) -> ToolError:
    seen: list[str] = []

    async def call_next(context):  # pragma: no cover - must never run
        seen.append(context.message.name)
        return "ran"

    with pytest.raises(ToolError) as excinfo:
        await mw.on_call_tool(_context(tool, meta), call_next)
    assert seen == [], f"{tool} was refused but the tool still ran"
    return excinfo.value


def _records(project, surface: str = "controls") -> list[dict]:
    """The bodies of the audit entries recorded on *surface*, oldest first."""
    entries = sorted(project.entries(), key=lambda entry: entry.ts)
    return [dict(entry.body) for entry in entries if entry.body.get("surface") == surface]


def _sandbox(monkeypatch) -> None:
    monkeypatch.setenv(am.POSTURE_ENV, "readonly")


# --------------------------------------------------------------------------
# The clamp: what the loaded lists mean
# --------------------------------------------------------------------------


class TestPostureClamp:
    @pytest.mark.usefixtures("project")
    async def test_a_write_tool_is_refused_under_the_sandbox_posture(self, monkeypatch):
        _sandbox(monkeypatch)
        error = await _refused(am.AuditMiddleware(), "channel_write")
        envelope = json.loads(str(error))
        assert envelope["error_type"] == "safety_error"

    @pytest.mark.usefixtures("project")
    async def test_a_write_tool_runs_under_the_writes_posture(self, monkeypatch):
        monkeypatch.setenv(am.POSTURE_ENV, "readwrite")
        result, seen = await _call(am.AuditMiddleware(), "channel_write")
        assert seen == ["channel_write"]
        assert result == "channel_write-result"

    @pytest.mark.usefixtures("project")
    async def test_an_absent_posture_marker_is_not_a_sandbox(self, monkeypatch):
        monkeypatch.delenv(am.POSTURE_ENV, raising=False)
        _, seen = await _call(am.AuditMiddleware(), "channel_write")
        assert seen == ["channel_write"]

    @pytest.mark.usefixtures("project")
    async def test_a_read_tool_runs_under_the_sandbox_posture(self, monkeypatch):
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "channel_read")
        assert seen == ["channel_read"]

    @pytest.mark.usefixtures("project")
    async def test_a_mixed_tool_runs_under_the_sandbox_posture(self, monkeypatch):
        """``execute`` is read/write; its in-tool clamp owns the posture, not this."""
        _sandbox(monkeypatch)
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python")
        _, seen = await _call(am.AuditMiddleware(), "execute")
        assert seen == ["execute"]

    async def test_the_clamp_is_the_loaded_write_list_not_the_floor(self, project, monkeypatch):
        """A facility write tool the floor has never heard of is still clamped."""
        project.write(
            server_prefixes=["mcp__sitectl__"],
            write_tools=[SITECTL_WRITE],
            mixed=[PYTHON_EXECUTE],
        )
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "do_thing")

    async def test_the_loaded_mixed_list_is_subtracted_from_the_loaded_write_list(
        self, project, monkeypatch
    ):
        """A tool named by BOTH loaded lists is mixed, so it runs."""
        project.write(
            server_prefixes=["mcp__controls__"],
            write_tools=[CONTROLS_WRITE],
            mixed=[CONTROLS_WRITE],
        )
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "channel_write")
        assert seen == ["channel_write"]

    async def test_a_write_tool_of_another_server_is_not_clamped_here(self, project, monkeypatch):
        """Membership is on the fully-qualified name, so prefixes cannot collide."""
        project.write(
            server_prefixes=["mcp__controls__", "mcp__sitectl__"],
            write_tools=[SITECTL_WRITE],
            mixed=[],
        )
        _sandbox(monkeypatch)  # this server is `controls`
        _, seen = await _call(am.AuditMiddleware(), "do_thing")
        assert seen == ["do_thing"]


class TestADeclaredFacilityWriteIsClamped:
    """A server that declares ``writes:`` renders each declared tool exactly.

    The render puts ``mcp__<srv>__<tool>`` for each declared tool in
    ``write_tools`` and nothing for the server's other tools, so the clamp
    refuses the declared ones and lets the rest through — under the same
    backstop as a framework write. The rendered-fixture half of this is
    ``tests/registry/test_tool_classification_drift.py``.
    """

    DECLARED = ("mcp__plc__set_value", "mcp__plc__ramp_value")

    @pytest.fixture
    def plc(self, project, monkeypatch):
        project.write(
            server_prefixes=["mcp__controls__", "mcp__plc__"],
            write_tools=[CONTROLS_WRITE, *self.DECLARED],
            mixed=[],
        )
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "plc")
        project.section = dict(TIER_NEVER_SECTION)
        return project

    @pytest.mark.parametrize("tool", ["set_value", "ramp_value"])
    async def test_each_declared_tool_is_refused_by_a_tier_that_allows_nothing(self, plc, tool):
        error = await _refused(am.AuditMiddleware(), tool)

        assert refusal_phrase("tier_never") in json.loads(str(error))["error_message"]
        record = _records(plc, surface="plc")[-1]
        assert record["subject"] == f"mcp__plc__{tool}"
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == am.REASON_POSTURE

    @pytest.mark.parametrize("tool", ["set_value", "ramp_value"])
    async def test_each_declared_tool_is_refused_by_a_readonly_run(self, plc, monkeypatch, tool):
        plc.section = dict(TIER_ALLOWED_SECTION)
        _sandbox(monkeypatch)

        error = await _refused(am.AuditMiddleware(), tool)

        assert refusal_phrase("readonly_run") in json.loads(str(error))["error_message"]

    async def test_the_declaring_servers_other_tools_run(self, plc):
        _, seen = await _call(am.AuditMiddleware(), "get_value")

        assert seen == ["get_value"]
        assert _records(plc, surface="plc")[-1]["decision"] == DECISION_ALLOWED

    async def test_a_declared_tool_runs_under_a_tier_that_allows_writes(self, plc):
        plc.section = dict(TIER_ALLOWED_SECTION)

        _, seen = await _call(am.AuditMiddleware(), "set_value")

        assert seen == ["set_value"]


# --------------------------------------------------------------------------
# Degraded reads: both lists fall together
# --------------------------------------------------------------------------


class TestDegradedHookConfig:
    async def test_a_missing_hook_config_falls_to_the_floor(self, project, monkeypatch):
        """A clone the floor cannot spell: the floor is matched by BARE name.

        `controls_ring` is what an ``extends`` clone of the controls server is
        actually launched with, and ``mcp__controls_ring__channel_write`` is in
        no floor. Matching fully-qualified here would make the whole degrade
        path a no-op for every clone in the deployment.
        """
        project.remove()
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, CLONE_PREFIX)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_missing_hook_config_falls_to_the_floor_for_a_framework_name(
        self, project, monkeypatch
    ):
        """The ordinary case: the subject IS a floor entry."""
        project.remove()
        _sandbox(monkeypatch)  # this server is `controls`
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_clone_of_the_mixed_server_still_runs_execute_when_degraded(
        self, project, monkeypatch
    ):
        """Bare-name matching must not swallow the mixed exclusion.

        The floor has `execute` subtracted from it, so matching by bare name
        cannot re-clamp it — for the framework server or for a clone of it.
        """
        project.remove()
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python_ring")
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "execute")
        assert seen == ["execute"]

    async def test_a_missing_hook_config_still_lets_the_mixed_floor_run(self, project, monkeypatch):
        project.remove()
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python")
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "execute")
        assert seen == ["execute"]

    async def test_a_malformed_hook_config_falls_to_the_floor(self, project, monkeypatch):
        project.raw("{ not json")
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_config_without_the_mixed_key_degrades_BOTH_lists(self, project, monkeypatch):
        """The stated consequence of a pre-feature render: floor only.

        A loaded write list against a floor exclusion would clamp
        ``mcp__python__execute`` as a pure write tool, so the two lists fall
        together — the facility write tool goes back to hook-only coverage.
        """
        project.write(
            server_prefixes=["mcp__controls__", "mcp__sitectl__"],
            write_tools=[CONTROLS_WRITE, SITECTL_WRITE],
            mixed=NO_MIXED_KEY,
        )
        _sandbox(monkeypatch)

        # A facility tool, deliberately: every framework write tool is in the
        # floor, so only a name the floor has never heard of can show that the
        # LOADED list is what fell away.
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        _, seen = await _call(am.AuditMiddleware(), "do_thing")
        assert seen == ["do_thing"], "the loaded write list must not survive the degrade"

        am.reset_audit_state()
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "controls")
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_config_without_the_mixed_key_does_not_clamp_the_mixed_floor(
        self, project, monkeypatch
    ):
        project.write(write_tools=[CONTROLS_WRITE, PYTHON_EXECUTE], mixed=NO_MIXED_KEY)
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python")
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "execute")
        assert seen == ["execute"]

    async def test_a_keyless_config_names_osprey_build_once(self, project, monkeypatch, caplog):
        project.write(mixed=NO_MIXED_KEY)
        _sandbox(monkeypatch)
        mw = am.AuditMiddleware()
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _call(mw, "channel_read")
            await _call(mw, "channel_read")
        warnings = [r for r in caplog.records if "osprey build" in r.getMessage()]
        assert len(warnings) == 1, [r.getMessage() for r in caplog.records]

    async def test_a_non_list_write_tools_value_degrades(self, project, monkeypatch):
        project.raw(json.dumps({"write_tools": "everything", "mixed_read_write_tools": []}))
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_write_tools_list_of_junk_degrades_rather_than_emptying(
        self, project, monkeypatch, caplog
    ):
        """`[null, 123]` is as strong evidence as `"everything"`, and must be as loud.

        Filtering it instead would leave an EMPTY clamp with degraded=False: the
        widest possible answer, reached silently. The prefix here is listed, so
        without the degrade the call would be verified against that empty set.
        """
        project.raw(
            json.dumps(
                {
                    "server_prefixes": ["mcp__controls__"],
                    "write_tools": [None, 123],
                    "mixed_read_write_tools": [],
                }
            )
        )
        _sandbox(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _refused(am.AuditMiddleware(), "channel_write")
        assert any("osprey build" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(json.dumps([CONTROLS_WRITE]), id="array"),
            pytest.param(json.dumps("everything"), id="string"),
            pytest.param(json.dumps(42), id="number"),
        ],
    )
    async def test_a_hook_config_that_is_not_an_object_degrades_rather_than_raising(
        self, project, monkeypatch, payload
    ):
        """A half-written render whose TOP LEVEL is not a JSON object.

        Without ``_read_hook_config``'s ``isinstance(parsed, dict)`` guard,
        ``(parsed or {}).get("write_tools")`` raises ``AttributeError`` out of
        ``_resolve_state`` — which ``on_call_tool`` deliberately does not wrap
        — so every tools/call on that server raises instead of clamping the
        floor. Fails closed either way; the floor is the documented way.
        """
        project.raw(payload)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        assert _records(project)[-1]["detail"] == f"{am.CLAMP_SOURCE_FLOOR} reason=readonly_run"

    async def test_a_missing_hook_config_still_floors_the_bluesky_arming_pair(
        self, project, monkeypatch
    ):
        """Opt-in is exactly the population the floor is for.

        A deployment that enabled bluesky is the only one whose sandboxed
        session can reach ``queue_add``/``queue_start`` — arming and starting a
        plan queue, both control-system writes — and a degraded render is when
        nothing else is left to refuse them. The client-side hook falls back to
        the same literal for the same case, so both layers open together.
        """
        project.remove()
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "bluesky")
        _sandbox(monkeypatch)
        for tool in ("queue_add", "queue_start"):
            await _refused(am.AuditMiddleware(), tool)
            detail = _records(project, surface="bluesky")[-1]["detail"]
            assert detail == f"{am.CLAMP_SOURCE_FLOOR} reason=readonly_run"

    def test_a_partly_unusable_list_is_rejected_whole(self):
        assert am._string_list(["a", "b"]) == ["a", "b"]
        assert am._string_list([]) == [], "an explicitly empty list is usable"
        assert am._string_list(["a", 123]) is None
        assert am._string_list(["a", ""]) is None
        assert am._string_list(["a", None]) is None
        assert am._string_list("everything") is None


class TestConfigAnchor:
    @pytest.mark.usefixtures("project")
    async def test_an_unset_osprey_config_degrades_loudly(self, monkeypatch, caplog):
        monkeypatch.delenv(am.CONFIG_ENV, raising=False)
        _sandbox(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _refused(am.AuditMiddleware(), "channel_write")
        assert any(am.CONFIG_ENV in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]

    @pytest.mark.usefixtures("project")
    async def test_an_unset_osprey_config_warns_once(self, monkeypatch, caplog):
        monkeypatch.delenv(am.CONFIG_ENV, raising=False)
        mw = am.AuditMiddleware()
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _call(mw, "channel_read")
            await _call(mw, "channel_read")
        assert len([r for r in caplog.records if am.CONFIG_ENV in r.getMessage()]) == 1

    @pytest.mark.usefixtures("project")
    async def test_the_repo_root_is_never_read_when_osprey_config_is_unset(
        self, monkeypatch, tmp_path
    ):
        """A repo-root hook config must not be picked up as a consolation prize."""
        repo_hooks = tmp_path / ".claude" / "hooks"
        repo_hooks.mkdir(parents=True)
        (repo_hooks / "hook_config.json").write_text(
            json.dumps(
                _hook_config(
                    server_prefixes=["mcp__sitectl__"],
                    write_tools=[SITECTL_WRITE],
                    mixed=[],
                )
            )
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(am.CONFIG_ENV, raising=False)
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "do_thing")
        assert seen == ["do_thing"], "the repo-root hook config was read"

    def test_the_path_is_anchored_on_the_config_directory(self, project):
        assert am.hook_config_path() == project.hook_config

    @pytest.mark.usefixtures("project")
    def test_the_path_is_none_without_the_env_var(self, monkeypatch):
        monkeypatch.delenv(am.CONFIG_ENV, raising=False)
        assert am.hook_config_path() is None

    def test_a_padded_env_value_is_stripped_before_the_path_is_built(self, project, monkeypatch):
        """Whitespace must not survive into the path, where it only fails to open."""
        monkeypatch.setenv(am.CONFIG_ENV, f"  {project.hook_config.parents[2] / 'config.yml'}  ")
        assert am.hook_config_path() == project.hook_config

    @pytest.mark.usefixtures("project")
    def test_a_relative_env_value_is_refused_not_guessed(self, monkeypatch, caplog):
        """A relative OSPREY_CONFIG resolves against the cwd — the one guess forbidden."""
        monkeypatch.setenv(am.CONFIG_ENV, "build/config.yml")
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            assert am.hook_config_path() is None
        assert any(am.CONFIG_ENV in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]

    @pytest.mark.usefixtures("project")
    async def test_a_relative_env_value_degrades_without_claiming_it_is_unset(
        self, monkeypatch, caplog
    ):
        """One warning, and the accurate one: the env var is set, just unusable."""
        monkeypatch.setenv(am.CONFIG_ENV, "build/config.yml")
        _sandbox(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _refused(am.AuditMiddleware(), "channel_write")
        # The audit writer logs the refusal on its own logger; this is about ours.
        messages = [r.getMessage() for r in caplog.records if r.name == am.logger.name]
        assert len(messages) == 1, messages
        assert "not an absolute path" in messages[0]
        assert "is unset" not in messages[0]

    @pytest.mark.usefixtures("project")
    async def test_a_relative_env_value_does_not_read_the_working_directory(
        self, monkeypatch, tmp_path
    ):
        """The cwd's own render must not be picked up as a consolation prize."""
        stray = tmp_path / "stray"
        (stray / "build" / ".claude" / "hooks").mkdir(parents=True)
        (stray / "build" / ".claude" / "hooks" / "hook_config.json").write_text(
            json.dumps(
                _hook_config(server_prefixes=["mcp__sitectl__"], write_tools=[SITECTL_WRITE])
            )
        )
        monkeypatch.chdir(stray)
        monkeypatch.setenv(am.CONFIG_ENV, "build/config.yml")
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "do_thing")
        assert seen == ["do_thing"], "the working directory's hook config was read"


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------


class TestFreshness:
    async def test_a_re_render_is_seen_without_a_restart(self, project, monkeypatch):
        _sandbox(monkeypatch)
        mw = am.AuditMiddleware()
        _, seen = await _call(mw, "shutter_open")
        assert seen == ["shutter_open"]

        project.write(
            server_prefixes=["mcp__controls__"],
            write_tools=["mcp__controls__shutter_open"],
            mixed=[],
        )
        await _refused(mw, "shutter_open")

    @pytest.mark.usefixtures("project")
    async def test_an_unchanged_file_is_parsed_once(self, monkeypatch):
        parses: list[Path] = []
        original = am._read_hook_config

        def counting(path):
            parses.append(path)
            return original(path)

        monkeypatch.setattr(am, "_read_hook_config", counting)
        mw = am.AuditMiddleware()
        await _call(mw, "channel_read")
        await _call(mw, "channel_read")
        await _call(mw, "channel_read")
        assert len(parses) == 1

    async def test_a_replacement_with_the_same_mtime_and_size_is_still_seen(
        self, project, monkeypatch
    ):
        """The stat key is more than mtime and size.

        A replacement can preserve both — a restore with ``cp -p``/``rsync
        -t``/``tar -p``, an image-layer or bind-mount swap, an in-place editor
        on a coarse-mtime filesystem. The direction that matters is a re-render
        that ADDS a write tool: the running server would keep clamping the old,
        narrower set for the life of the process. This probe forges exactly the
        old key — same size (the two tool names are the same length) and the
        original mtime put back with ``os.utime`` — so only ``st_ino`` /
        ``st_ctime_ns`` can catch it.
        """
        _sandbox(monkeypatch)
        project.write(
            server_prefixes=["mcp__controls__"],
            write_tools=["mcp__controls__aaaaaaaaaaaa"],
            mixed=[],
        )
        before = project.hook_config.stat()

        mw = am.AuditMiddleware()
        _, seen = await _call(mw, "shutter_open")
        assert seen == ["shutter_open"], "the first render must not clamp shutter_open"

        project.hook_config.write_text(
            json.dumps(
                _hook_config(
                    server_prefixes=["mcp__controls__"],
                    write_tools=["mcp__controls__shutter_open"],
                    mixed=[],
                )
            )
        )
        os.utime(project.hook_config, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = project.hook_config.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size), (
            "the probe must forge exactly the pair the old stat key compared"
        )

        await _refused(mw, "shutter_open")

    async def test_a_failed_re_parse_keeps_the_last_good_set_plus_the_floor(
        self, project, monkeypatch
    ):
        project.write(
            server_prefixes=["mcp__controls__", "mcp__sitectl__"],
            write_tools=[SITECTL_WRITE],
            mixed=[],
        )
        _sandbox(monkeypatch)
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        mw = am.AuditMiddleware()
        await _refused(mw, "do_thing")

        project.raw("{ truncated")

        await _refused(mw, "do_thing")  # last good survives

        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "controls")
        await _refused(mw, "channel_write")  # ...and the floor is added


# --------------------------------------------------------------------------
# The tool prefix and the fail-closed rule
# --------------------------------------------------------------------------


class TestToolPrefix:
    async def test_the_subject_is_fully_qualified(self, project, monkeypatch):
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        assert _records(project)[-1]["subject"] == CONTROLS_WRITE

    async def test_an_unset_prefix_records_the_bare_tool_name(self, project, monkeypatch):
        monkeypatch.delenv(am.TOOL_PREFIX_ENV, raising=False)
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project, surface=am.SURFACE_UNPREFIXED)[-1]["subject"] == "channel_read"

    @pytest.mark.usefixtures("project")
    async def test_an_unset_prefix_warns_once(self, monkeypatch, caplog):
        monkeypatch.delenv(am.TOOL_PREFIX_ENV, raising=False)
        mw = am.AuditMiddleware()
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _call(mw, "channel_read")
            await _call(mw, "channel_read")
        hits = [r for r in caplog.records if am.TOOL_PREFIX_ENV in r.getMessage()]
        assert len(hits) == 1

    @pytest.mark.usefixtures("project")
    async def test_an_unset_prefix_still_clamps_the_floor_by_bare_name(self, monkeypatch):
        """Fail closed: an unstamped server is exactly the case not to trust."""
        monkeypatch.delenv(am.TOOL_PREFIX_ENV, raising=False)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_an_unset_prefix_still_clamps_the_RENDERED_write_tools(
        self, project, monkeypatch
    ):
        """A healthy render's own write tools are clamped too, not just the floor.

        A facility whose write tools are all deployment-specific would otherwise
        get no server-side clamp at all in this state — the floor's only bare
        name is `channel_write`.
        """
        project.write(
            server_prefixes=["mcp__sitectl__"],
            write_tools=["mcp__sitectl__set_value"],
            mixed=[],
        )
        monkeypatch.delenv(am.TOOL_PREFIX_ENV, raising=False)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "set_value")

    async def test_a_prefix_absent_from_server_prefixes_fails_closed(self, project, monkeypatch):
        """A stale clone the render does not know clamps the floor, by bare name.

        The prefix is clone-shaped on purpose: `mcp__controls_ring__channel_write`
        is in no clamp set anywhere, so this passes only if an unverified prefix
        drops to bare-name matching.
        """
        project.write(
            server_prefixes=["mcp__sitectl__"],
            write_tools=[SITECTL_WRITE],
            mixed=[],
        )
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, CLONE_PREFIX)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_a_framework_named_prefix_absent_from_server_prefixes_fails_closed(
        self, project, monkeypatch
    ):
        """The same rule where the subject happens to be a floor entry."""
        project.write(
            server_prefixes=["mcp__sitectl__"],
            write_tools=[SITECTL_WRITE],
            mixed=[],
        )
        _sandbox(monkeypatch)  # this server is `controls`, absent from the list
        await _refused(am.AuditMiddleware(), "channel_write")

    async def test_an_unlisted_prefix_still_clamps_the_rendered_write_set(
        self, project, monkeypatch
    ):
        """Fail closed must not be WEAKER than the healthy path.

        The render still names every write tool the deployment has; discarding
        it at the moment the render says it has never heard of this server
        would clamp `set_value` for `sitectl` and not for the stranger.
        """
        project.write(
            server_prefixes=["mcp__sitectl__"],
            write_tools=["mcp__sitectl__set_value"],
            mixed=[],
        )
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, CLONE_PREFIX)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "set_value")

    async def test_a_verified_prefix_is_still_matched_fully_qualified(self, project, monkeypatch):
        """Bare-name matching is the unverified path ONLY.

        A verified server sharing a bare tool name with another server's write
        tool keeps running — that is the whole reason the subject is qualified.
        """
        project.write(
            server_prefixes=["mcp__controls__", "mcp__sitectl__"],
            write_tools=["mcp__sitectl__channel_write"],
            mixed=[],
        )
        _sandbox(monkeypatch)  # this server is `controls`, and the render lists it
        _, seen = await _call(am.AuditMiddleware(), "channel_write")
        assert seen == ["channel_write"]

    async def test_a_prefix_absent_from_server_prefixes_warns_once(self, project, caplog):
        project.write(server_prefixes=["mcp__sitectl__"], write_tools=[], mixed=[])
        mw = am.AuditMiddleware()
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            await _call(mw, "channel_read")
            await _call(mw, "channel_read")
        hits = [r for r in caplog.records if "server_prefixes" in r.getMessage()]
        assert len(hits) == 1

    async def test_a_write_free_server_passes_silently(self, project, monkeypatch, caplog):
        """No write tools is not a mismatch: it runs, and it says nothing."""
        project.write(
            server_prefixes=["mcp__controls__", "mcp__phoebus__"],
            write_tools=[CONTROLS_WRITE],
            mixed=[],
        )
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "phoebus")
        _sandbox(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=am.logger.name):
            _, seen = await _call(am.AuditMiddleware(), "open_display")
        assert seen == ["open_display"]
        assert caplog.records == []

    async def test_the_floor_clamp_of_a_mismatched_prefix_excludes_mixed(
        self, project, monkeypatch
    ):
        project.write(server_prefixes=["mcp__controls__"], write_tools=[], mixed=[])
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python")  # absent from server_prefixes
        _sandbox(monkeypatch)
        _, seen = await _call(am.AuditMiddleware(), "execute")
        assert seen == ["execute"]


# --------------------------------------------------------------------------
# What the record says
# --------------------------------------------------------------------------


class TestAuditRecord:
    async def test_an_allowed_call_is_recorded(self, project):
        await _call(am.AuditMiddleware(), "channel_read")
        record = _records(project)[-1]
        assert record["decision"] == DECISION_ALLOWED
        assert record["reason"] == am.REASON_TOOL_CALL
        assert record["subject"] == "mcp__controls__channel_read"

    async def test_a_clamped_call_is_recorded_as_a_refusal(self, project, monkeypatch):
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        record = _records(project)[-1]
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == am.REASON_POSTURE

    async def test_the_surface_is_the_server_name(self, project, monkeypatch):
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, "sitectl")
        await _call(am.AuditMiddleware(), "read_thing")
        assert _records(project, surface="sitectl")[-1]["surface"] == "sitectl"

    async def test_the_actor_comes_from_the_identity_ladder(self, project, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "carol")
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project)[-1]["actor"] == "carol"

    async def test_the_posture_is_spelled_sandbox_not_readonly(self, project, monkeypatch):
        _sandbox(monkeypatch)
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project)[-1]["posture"] == am.POSTURE_SANDBOX

    async def test_the_posture_is_spelled_writes_not_readwrite(self, project, monkeypatch, session):
        """``writes`` is the verdict allowing; ``readwrite`` is the env's word for a run."""
        monkeypatch.setenv(am.POSTURE_ENV, "readwrite")
        session.on("live")
        session.switch("live")
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project)[-1]["posture"] == am.POSTURE_WRITES

    async def test_the_marker_pair_is_carried_verbatim(self, project, monkeypatch):
        monkeypatch.setenv(am.POSTURE_SOURCE_ENV, POSTURE_SOURCE_SPAWN)
        monkeypatch.setenv(am.POSTURE_SESSION_ENV, "sess-7")
        await _call(am.AuditMiddleware(), "channel_read")
        record = _records(project)[-1]
        assert record["posture_source"] == POSTURE_SOURCE_SPAWN
        assert record["session"] == "sess-7"

    async def test_an_absent_marker_records_process_not_a_guess(self, project, monkeypatch):
        _sandbox(monkeypatch)
        monkeypatch.delenv(am.POSTURE_SOURCE_ENV, raising=False)
        await _call(am.AuditMiddleware(), "channel_read")
        record = _records(project)[-1]
        assert record["posture_source"] == POSTURE_SOURCE_PROCESS
        assert record["session"] is None

    @pytest.mark.parametrize("blank", ["", "   "])
    async def test_a_blank_session_marker_records_null_not_an_empty_string(
        self, project, monkeypatch, blank
    ):
        """A rendered-but-blank marker is the unset case.

        ``session`` is the key toggle events and tool records join on, and the
        envelope documents ``null`` as "no audit session id exists". An empty
        string joins nothing, and the envelope does not reject it — ``session``
        is not in its non-empty required set — so this guard is only as good as
        this test.
        """
        monkeypatch.setenv(am.POSTURE_SESSION_ENV, blank)
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project)[-1]["session"] is None

    async def test_an_unrecognised_marker_records_process(self, project, monkeypatch):
        monkeypatch.setenv(am.POSTURE_SOURCE_ENV, "vibes")
        await _call(am.AuditMiddleware(), "channel_read")
        assert _records(project)[-1]["posture_source"] == POSTURE_SOURCE_PROCESS

    async def test_the_record_says_which_clamp_set_refused(self, project, monkeypatch):
        project.remove()
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        assert _records(project)[-1]["detail"] == f"{am.CLAMP_SOURCE_FLOOR} reason=readonly_run"

    async def test_a_loaded_clamp_says_so(self, project, monkeypatch):
        _sandbox(monkeypatch)
        error = await _refused(am.AuditMiddleware(), "channel_write")
        assert _records(project)[-1]["detail"] == f"{am.CLAMP_SOURCE_LOADED} reason=readonly_run"
        # A verified refusal is the posture doing its job; it names no remedy
        # beyond the posture switch.
        assert "degraded" not in str(error)

    async def test_a_degraded_refusal_names_its_cause_to_the_agent(self, project, monkeypatch):
        """Off the verified path a read tool sharing a bare name can land here, so
        the error says the clamp is degraded and what fixes it."""
        project.remove()
        _sandbox(monkeypatch)
        error = await _refused(am.AuditMiddleware(), "channel_write")
        assert "degraded" in str(error)
        assert "osprey build" in str(error)

    async def test_an_unverified_prefix_is_neither_loaded_nor_the_floor(self, project, monkeypatch):
        """A render that parsed but does not know this server gets its own spelling."""
        monkeypatch.setenv(am.TOOL_PREFIX_ENV, CLONE_PREFIX)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        detail = _records(project, surface=CLONE_PREFIX)[-1]["detail"]
        assert detail == f"{am.CLAMP_SOURCE_UNVERIFIED} reason=readonly_run"


_TOOL_USE_META = {"claudecode/toolUseId": "toolu_01abc"}


class TestToolUseId:
    """Every record a call files names the harness's id for that call."""

    async def test_the_allowed_record_carries_the_id_from_request_meta(self, project):
        await _call(am.AuditMiddleware(), "channel_read", meta=_TOOL_USE_META)
        assert _records(project)[-1]["tool_use_id"] == "toolu_01abc"

    async def test_the_posture_refusal_carries_it(self, project, monkeypatch):
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        record = _records(project)[-1]
        assert record["reason"] == am.REASON_POSTURE
        assert record["tool_use_id"] == "toolu_01abc"

    async def test_the_tool_error_record_carries_it(self, project):
        with pytest.raises(ToolError):
            await _call(
                am.AuditMiddleware(), "channel_write", raises=ToolError("x"), meta=_TOOL_USE_META
            )
        record = _records(project)[-1]
        assert record["reason"] == am.REASON_TOOL_ERROR
        assert record["tool_use_id"] == "toolu_01abc"

    async def test_message_meta_is_the_fallback(self, project):
        await _call(am.AuditMiddleware(), "channel_read", message_meta=_TOOL_USE_META)
        assert _records(project)[-1]["tool_use_id"] == "toolu_01abc"

    async def test_no_meta_no_key(self, project):
        await _call(am.AuditMiddleware(), "channel_read")
        assert "tool_use_id" not in _records(project)[-1]

    async def test_a_malformed_id_is_dropped(self, project):
        await _call(
            am.AuditMiddleware(), "channel_read", meta={"claudecode/toolUseId": "../etc/passwd"}
        )
        assert "tool_use_id" not in _records(project)[-1]

    async def test_the_id_survives_a_real_fastmcp_round_trip(self, project):
        """Guards the FastMCP fact the reader rests on: a real client's ``_meta``
        reaches the middleware, wherever the server keeps it."""
        from fastmcp import Client, FastMCP

        server = FastMCP("roundtrip")
        server.add_middleware(am.AuditMiddleware())

        @server.tool
        def channel_read() -> str:
            return "ok"

        async with Client(server) as client:
            await client.call_tool("channel_read", {}, meta={"claudecode/toolUseId": "toolu_x"})

        record = _records(project)[-1]
        assert record["subject"] == "mcp__controls__channel_read"
        assert record["tool_use_id"] == "toolu_x"


@pytest.fixture
def ask_stamps(tmp_path, monkeypatch):
    """Where the approval hook leaves its ask stamps, as this server resolves it."""
    from osprey_connectors_tests._harness import pin_identity, state_dir_under

    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))

    def stamp(
        tool_use_id: str,
        approver: str = "human",
        mode: str = "default",
        position: str | None = None,
    ) -> Path:
        path = state / f"{am.APPROVAL_ASK_PREFIX}{tool_use_id}{am.APPROVAL_ASK_SUFFIX}"
        payload = {"tool_use_id": tool_use_id, "approver": approver, "permission_mode": mode}
        if position is not None:
            payload["position"] = position
        path.write_text(json.dumps(payload))
        return path

    return stamp


class TestApprovalOutcome:
    """A call an approval prompt let through says so in its own record's detail."""

    async def test_a_stamped_call_records_approval_in_detail(self, project, ask_stamps):
        ask_stamps("toolu_01abc")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        record = _records(project)[-1]
        assert record["decision"] == DECISION_ALLOWED
        assert record["detail"] == "approval=approved approver=human"

    async def test_an_unstamped_call_has_no_approval_detail(self, project, ask_stamps):
        ask_stamps("toolu_someone_else")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        assert "detail" not in _records(project)[-1]

    @pytest.mark.usefixtures("project")
    async def test_the_stamp_is_not_consumed_here(self, ask_stamps):
        path = ask_stamps("toolu_01abc")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        assert path.exists()

    async def test_a_refused_call_after_approval_names_both(self, project, ask_stamps):
        ask_stamps("toolu_01abc")
        with pytest.raises(ToolError):
            await _call(
                am.AuditMiddleware(), "channel_write", raises=ToolError("x"), meta=_TOOL_USE_META
            )
        record = _records(project)[-1]
        assert record["reason"] == am.REASON_TOOL_ERROR
        assert record["detail"] == "approval=approved approver=human"

    async def test_a_posture_refusal_keeps_its_clamp_source_first(
        self, project, ask_stamps, monkeypatch
    ):
        ask_stamps("toolu_01abc")
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        record = _records(project)[-1]
        assert record["detail"] == (
            f"{am.CLAMP_SOURCE_LOADED} reason=readonly_run approval=approved approver=human"
        )

    async def test_noted_switch_endpoints_ride_the_detail(self, project):
        from osprey.audit.call import note

        async def call_next(_ctx):
            note(from_target="live", to_target="va", limits={"verdict": "passed"})
            return "switched"

        await am.AuditMiddleware().on_call_tool(_context("control_target_set"), call_next)
        assert _records(project)[-1]["detail"] == "from_target=live to_target=va"


class TestToolErrors:
    async def test_a_tool_raised_error_is_recorded_as_a_refusal(self, project):
        with pytest.raises(ToolError):
            await _call(am.AuditMiddleware(), "channel_write", raises=ToolError("nope"))
        record = _records(project)[-1]
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == am.REASON_TOOL_ERROR

    async def test_the_tool_error_message_never_reaches_the_record(self, project):
        secret = "PV=SR:BEND:SETPOINT value=hunter2"
        with pytest.raises(ToolError):
            await _call(am.AuditMiddleware(), "channel_write", raises=ToolError(secret))
        blob = json.dumps(_records(project)[-1])
        assert "hunter2" not in blob and "SETPOINT" not in blob

    @pytest.mark.usefixtures("project")
    async def test_a_tool_raised_error_is_re_raised_unchanged(self):
        raised = ToolError("original")
        with pytest.raises(ToolError) as excinfo:
            await _call(am.AuditMiddleware(), "channel_write", raises=raised)
        assert excinfo.value is raised

    async def test_an_ordinary_exception_is_not_recorded_as_a_refusal(self, project):
        with pytest.raises(ValueError):
            await _call(am.AuditMiddleware(), "channel_read", raises=ValueError("boom"))
        assert [r for r in _records(project) if r["reason"] == am.REASON_TOOL_ERROR] == []

    @pytest.mark.usefixtures("project")
    async def test_an_ordinary_exception_still_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            await _call(am.AuditMiddleware(), "channel_read", raises=ValueError("boom"))


class TestAnUnstoredInnerRecordIsNotSubstituted:
    """The one place the two exit paths are deliberately asymmetric.

    Both face the same situation: an inner layer decided and marked the call as
    its own, and its write did not land (``stored=False``). On the ``ToolError``
    path this layer files its own ``refused``/``tool_error`` — the call raising
    is a first-person observation it made and can sign. On the success path it
    files nothing: the call succeeded, so ``allowed`` is false; ``refused`` is a
    decision this layer did not take; and copying the inner marker's decision
    and reason would file another layer's finding under THIS surface, without
    the ``source`` the executor surface carries. An operator grepping the
    executor surface would still find nothing and would now also find a lookalike
    filed where the refusal never happened.
    """

    async def test_a_tool_error_over_an_unstored_marker_files_the_middlewares_own_record(
        self, project
    ):
        async def call_next(_context):
            mark_recorded(DECISION_REFUSED, "runtime_guard", stored=False)
            raise ToolError("refused, and the inner write never landed")

        with pytest.raises(ToolError):
            await am.AuditMiddleware().on_call_tool(_context("channel_read"), call_next)

        assert [(r["decision"], r["reason"]) for r in _records(project)] == [
            (DECISION_REFUSED, am.REASON_TOOL_ERROR)
        ]

    async def test_a_successful_call_over_an_unstored_marker_files_no_substitute_record(
        self, project
    ):
        async def call_next(_context):
            mark_recorded(DECISION_REFUSED, "runtime_guard", stored=False)
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("channel_read"), call_next)

        assert _records(project) == []


class TestTheClampDecisionFailsClosed:
    """The decision may cost the call; it may never silently allow it.

    `_record` swallows every exception because a lost record is not a lost
    decision. The inverse must stay true one line higher: wrapping the clamp
    decision in the same defensive try/except would convert every internal
    error into an allowed write under the sandbox posture, and nothing else in
    this file would notice.
    """

    @pytest.mark.usefixtures("project")
    async def test_a_raising_state_resolve_never_reaches_the_tool(self, monkeypatch):
        def boom():
            raise RuntimeError("the hook config exploded")

        monkeypatch.setattr(am, "_resolve_state", boom)
        _sandbox(monkeypatch)
        seen: list[str] = []

        async def call_next(context):  # pragma: no cover - must never run
            seen.append(context.message.name)
            return "ran"

        with pytest.raises(RuntimeError, match="exploded"):
            await am.AuditMiddleware().on_call_tool(_context("channel_write"), call_next)
        assert seen == [], "an error in the clamp decision let the write through"

    @pytest.mark.usefixtures("project")
    async def test_a_raising_membership_test_never_reaches_the_tool(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("the clamp exploded")

        monkeypatch.setattr(am, "_is_clamped", boom)
        _sandbox(monkeypatch)
        seen: list[str] = []

        async def call_next(context):  # pragma: no cover - must never run
            seen.append(context.message.name)
            return "ran"

        with pytest.raises(RuntimeError, match="exploded"):
            await am.AuditMiddleware().on_call_tool(_context("channel_read"), call_next)
        assert seen == [], "an error in the clamp decision let the call through"


class TestAuditNeverCostsTheOperation:
    """The failure is injected at the record writer's spool append, below the
    never-raises boundary the middleware relies on, so what is pinned is the
    real path an unwritable spool takes and not a seam the middleware happens
    to call."""

    @pytest.mark.usefixtures("project")
    async def test_an_unwritable_spool_does_not_break_an_allowed_call(self, monkeypatch):
        reached: list[Path] = []

        def explode(directory, _data):
            reached.append(directory)
            raise OSError("read-only file system")

        monkeypatch.setattr(record_writer, "_append", explode)
        _, seen = await _call(am.AuditMiddleware(), "channel_read")
        assert seen == ["channel_read"]
        assert reached, "the writer's append boundary was never reached"

    @pytest.mark.usefixtures("project")
    async def test_an_unwritable_spool_does_not_lift_the_clamp(self, monkeypatch):
        reached: list[Path] = []

        def explode(directory, _data):
            reached.append(directory)
            raise OSError("read-only file system")

        monkeypatch.setattr(record_writer, "_append", explode)
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        assert reached, "the writer's append boundary was never reached"


# --------------------------------------------------------------------------
# Contracts other modules and tasks depend on
# --------------------------------------------------------------------------


class TestFastMCPContract:
    def test_the_middleware_is_a_fastmcp_middleware(self):
        assert issubclass(am.AuditMiddleware, Middleware)

    def test_on_call_tool_matches_the_base_signature(self):
        import inspect

        ours = inspect.signature(am.AuditMiddleware.on_call_tool)
        theirs = inspect.signature(Middleware.on_call_tool)
        assert list(ours.parameters) == list(theirs.parameters)

    def test_the_middleware_takes_no_required_arguments(self):
        assert am.AuditMiddleware() is not None


class TestSpellings:
    """Names re-spelled here rather than imported; pinned so they cannot drift."""

    def test_the_tool_prefix_env_matches_the_registry(self):
        from osprey.registry import mcp as registry

        assert am.TOOL_PREFIX_ENV == registry.TOOL_PREFIX_ENV

    def test_the_posture_marker_envs_match_the_registry(self):
        from osprey.registry import mcp as registry

        assert am.POSTURE_SOURCE_ENV == registry.POSTURE_SOURCE_ENV
        assert am.POSTURE_SESSION_ENV == registry.POSTURE_SESSION_ENV

    def test_the_mixed_floor_is_the_registry_floor(self):
        from osprey.registry.mcp import framework_mixed_read_write_tools

        assert am._FALLBACK_MIXED_TOOLS == framework_mixed_read_write_tools()

    def test_the_mixed_floor_is_pinned_literally(self):
        """Task 1.8's drift guard imports this exact list."""
        assert am._FALLBACK_MIXED_TOOLS == [
            "mcp__python__execute",
            "mcp__python__execute_file",
        ]

    def test_the_write_floor_is_the_registry_floor(self):
        """Every ``_WRITES_CHECK``-gated tool the framework ships, not the ones
        a default render enables: registry growth must not strand the floor."""
        from osprey.registry.mcp import framework_write_tools

        assert am._FALLBACK_WRITE_TOOLS == framework_write_tools()

    def test_the_write_floor_matches_the_hook(self):
        """Same fallback floor as the hooks' shared ``osprey_hook_log.py``, by
        construction — the one ``osprey_writes_check.py`` and
        ``osprey_approval.py`` both read through ``write_tools()``."""
        hook = (
            Path(__file__).resolve().parents[2]
            / "src/osprey/templates/claude_code/claude/hooks/osprey_hook_log.py"
        )
        tree = ast.parse(hook.read_text())
        literal = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "FALLBACK_WRITE_TOOLS" for t in node.targets
            )
        )
        assert am._FALLBACK_WRITE_TOOLS == literal

    def test_the_mixed_floor_is_a_subset_of_the_write_floor(self):
        assert set(am._FALLBACK_MIXED_TOOLS) <= set(am._FALLBACK_WRITE_TOOLS)

    def test_the_three_clamp_sources_are_distinct(self):
        """Loaded, unverified and floor are three findings, not two."""
        sources = {am.CLAMP_SOURCE_LOADED, am.CLAMP_SOURCE_UNVERIFIED, am.CLAMP_SOURCE_FLOOR}
        assert len(sources) == 3

    def test_the_posture_spellings_match_the_web_terminal(self):
        assert am.POSTURE_SANDBOX == "sandbox"
        assert am.POSTURE_WRITES == "writes"

    def test_the_sandbox_mode_is_matched_by_value(self):
        assert am.SANDBOX_MODE == "readonly"


class TestResetSeam:
    @pytest.mark.usefixtures("project")
    async def test_reset_clears_the_parsed_state(self, monkeypatch):
        parses: list[Path] = []
        original = am._read_hook_config
        monkeypatch.setattr(am, "_read_hook_config", lambda p: (parses.append(p), original(p))[1])
        mw = am.AuditMiddleware()
        await _call(mw, "channel_read")
        am.reset_audit_state()
        await _call(mw, "channel_read")
        assert len(parses) == 2


def test_importing_the_module_does_not_pull_in_the_registry():
    """The running server must not import the render/launch side to learn its name.

    This is the other half of the re-spelled constants above: they are copies
    precisely so that a server process stays free of ``osprey.registry``, whose
    import re-enters ``fastmcp`` and whose job is over by the time a server is
    running. A drive-by ``from osprey.registry.mcp import TOOL_PREFIX_ENV``
    would look like a tidy-up and would quietly undo that.
    """
    probe = (
        "import sys; import osprey.mcp_server.audit_middleware as m; "
        "print(any(n.startswith('osprey.registry') for n in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", completed.stderr


def test_importing_the_module_reads_nothing(project):
    """Import is side-effect free; the first call is what resolves state.

    A subprocess, like its neighbour above, and for the same reason: asserting
    ``_STATE is None`` in-process right after ``reset_audit_state()`` re-reads
    what the line before it just wrote and holds no matter what import does.
    Here the module is imported fresh with ``OSPREY_CONFIG`` pointing at a real
    render, so an import-time stat/parse would show up as resolved state.
    """
    probe = (
        "import osprey.mcp_server.audit_middleware as m; "
        "print(m._STATE, len(m._LAST_GOOD_CLAMP), len(m._WARNED))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, am.CONFIG_ENV: str(project.root / "build" / "config.yml")},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "None 0 0", completed.stderr


# --------------------------------------------------------------------------
# The backstop: a read-only run and the tier, never the switch
# --------------------------------------------------------------------------


SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def session(project, monkeypatch):
    """This server as a child of a session whose record carries a switch.

    The person turns writes on or off for one machine from the chip; nothing
    respawns this server and nothing sets ``OSPREY_EXECUTION_MODE``. The entry
    posture reads that record through :func:`osprey.audit.posture.posture`, so
    the fixture stamps the session key and the agent-data root, clears every
    variable that moves a verdict behind a test's back, and writes the record.

    ``on``, ``switch`` and ``off`` write the SAME file: the deployment's
    target and its switch are fields of one record. Until one of them is
    called there is no record at all, which is every switch off. A per-target
    sandbox narrowing in that record is ignored by the verdict, so turning
    writes off is the switch's job alone.
    """
    from osprey.audit import posture as posture_module
    from osprey_connectors_tests._harness import VERDICT_ENV_VARS, reset_verdict_state

    root = project.root / "agent_data"
    for name in VERDICT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(posture_module.OSPREY_AGENT_DATA_ROOT, str(root))
    monkeypatch.setenv(am.POSTURE_SESSION_ENV, SESSION_KEY)
    monkeypatch.delenv(posture_module.CONTROL_TARGET_ENV_VAR, raising=False)

    class _Session:
        key = SESSION_KEY
        target = "live"

        @classmethod
        def _write(cls, switch: dict[str, str] | None = None) -> None:
            write_control_context(root, target=cls.target, posture={}, switch=switch)
            reset_verdict_state()

        @classmethod
        def on(cls, target: str) -> None:
            """Point the deployment's record at *target*; the switch is kept."""
            cls.target = target
            cls._write()

        @classmethod
        def switch(cls, *targets: str) -> None:
            """Make the record's switch exactly *targets*, all ``on``."""
            cls._write(dict.fromkeys(targets, "on"))

        @classmethod
        def off(cls) -> None:
            """Turn every machine's switch off in that record; the target is kept."""
            cls._write({})

    reset_verdict_state()
    yield _Session
    reset_verdict_state()


async def _refused_in_the_tool(mw, tool: str) -> tuple[ToolError, list[str]]:
    """Drive a call whose tool asks the write verdict itself and refuses on it.

    The stand-in for a real write tool's own gate: it asks
    :func:`~osprey_connectors.posture_store.write_verdict` about the recorded
    target, as the in-tool gates do, and raises the verdict's phrase. Returns
    the error and the hops, so a test can tell "refused before the tool" from
    "refused in it".
    """
    from osprey.audit import posture as posture_module
    from osprey_connectors.posture_store import write_verdict
    from osprey_connectors.refusals import refusal_phrase

    seen: list[str] = []

    async def call_next(context):
        seen.append(context.message.name)
        verdict = write_verdict(
            posture_module.control_system_section(), posture_module.recorded_control_target()
        )
        if not verdict.allowed:
            raise ToolError(refusal_phrase(verdict.reason, verdict.kind))
        return "written"

    with pytest.raises(ToolError) as excinfo:
        await mw.on_call_tool(_context(tool), call_next)
    return excinfo.value, seen


class TestTheBackstopRefusesOnlyReadonlyAndTier:
    """The two reasons the backstop has, each in the verdict's own words."""

    async def test_a_readonly_run_is_refused_in_its_phrase(self, project, monkeypatch):
        _sandbox(monkeypatch)

        error = await _refused(am.AuditMiddleware(), "channel_write")

        envelope = json.loads(str(error))
        assert envelope["error_type"] == "safety_error"
        assert refusal_phrase("readonly_run") in envelope["error_message"]
        assert _records(project)[-1]["reason"] == am.REASON_POSTURE

    async def test_a_tier_that_allows_nothing_is_refused_in_its_phrase(self, project):
        project.section = dict(TIER_NEVER_SECTION)

        error = await _refused(am.AuditMiddleware(), "channel_write")

        assert refusal_phrase("tier_never") in json.loads(str(error))["error_message"]
        record = _records(project)[-1]
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == am.REASON_POSTURE
        assert "reason=tier_never" in record["detail"].split()

    async def test_a_readonly_run_wins_over_the_tier(self, project, monkeypatch):
        """The same order as the verdict's first two terms."""
        _sandbox(monkeypatch)
        project.section = dict(TIER_NEVER_SECTION)

        error = await _refused(am.AuditMiddleware(), "channel_write")

        assert refusal_phrase("readonly_run") in json.loads(str(error))["error_message"]

    async def test_an_unreadable_config_is_a_tier_that_allows_nothing(self, project):
        """The section reader answers ``None`` when the config cannot be read."""
        project.section = None

        error = await _refused(am.AuditMiddleware(), "channel_write")

        assert refusal_phrase("tier_never") in json.loads(str(error))["error_message"]

    async def test_the_detail_names_the_reason_and_the_recorded_target(self, project, session):
        session.on("live")
        project.section = dict(TIER_NEVER_SECTION)

        await _refused(am.AuditMiddleware(), "channel_write")

        assert _records(project)[-1]["detail"] == (
            f"{am.CLAMP_SOURCE_LOADED} reason=tier_never target=live"
        )

    @pytest.mark.usefixtures("project")
    async def test_the_refusal_names_no_key_variable_or_file(self, monkeypatch):
        """Each phrase says who lifts it; none names what an admin edits."""
        _sandbox(monkeypatch)
        error = await _refused(am.AuditMiddleware(), "channel_write")

        envelope = json.loads(str(error))
        text = " ".join([envelope["error_message"], *envelope["suggestions"]])
        for word in ("OSPREY_EXECUTION_MODE", "config.yml", "writes_enabled", "control_context"):
            assert word not in text, text

    async def test_a_read_tool_runs_under_a_refusing_backstop(self, project):
        project.section = dict(TIER_NEVER_SECTION)
        _, seen = await _call(am.AuditMiddleware(), "channel_read")
        assert seen == ["channel_read"]


class TestTheSwitchIsNeverABackstopReason:
    """``switch_off`` belongs to the tool's verdict and the monitor, never here.

    The backstop names no target and opens no record, so whatever the switch
    says, a write tool reaches its own gate — which asks the verdict about the
    machine the write actually lands on.
    """

    @pytest.mark.usefixtures("project", "session")
    async def test_no_record_passes_the_clamp(self):
        """No record is every switch off, and the backstop still lets it through."""
        _, seen = await _call(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"]

    async def test_a_switch_that_is_off_passes_the_clamp_and_reads_sandbox(self, project, session):
        session.on("live")

        _, seen = await _call(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"]
        record = _records(project)[-1]
        assert record["decision"] == DECISION_ALLOWED
        assert record["posture"] == am.POSTURE_SANDBOX

    async def test_a_switch_turned_off_with_the_tier_allowed_passes_the_clamp_and_is_refused_in_the_tool(
        self, project, session
    ):
        session.on("live")
        session.switch("live")
        session.off()

        error, seen = await _refused_in_the_tool(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"], "the backstop refused a switched-off target"
        assert refusal_phrase("switch_off", "real") in str(error)
        record = _records(project)[-1]
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == am.REASON_TOOL_ERROR
        assert record["posture"] == am.POSTURE_SANDBOX

    @pytest.mark.usefixtures("project")
    async def test_a_switch_on_for_another_machine_still_passes(self, session):
        session.on("live")
        session.switch("va")

        error, seen = await _refused_in_the_tool(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"]
        assert refusal_phrase("switch_off", "real") in str(error)

    async def test_a_switch_on_for_this_machine_reaches_the_write(self, project, session):
        session.on("live")
        session.switch("live")

        _, seen = await _call(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"]
        record = _records(project)[-1]
        assert record["decision"] == DECISION_ALLOWED
        assert record["posture"] == am.POSTURE_WRITES
        assert record["session"] == SESSION_KEY

    @pytest.mark.usefixtures("project")
    async def test_a_turn_off_lands_without_a_respawn_and_still_never_clamps(self, session):
        """One process: the entry posture moves, the backstop's answer does not."""
        session.on("live")
        session.switch("live")
        middleware = am.AuditMiddleware()
        _, seen = await _call(middleware, "channel_write")
        assert seen == ["channel_write"]

        session.off()

        _, seen = await _refused_in_the_tool(middleware, "channel_write")
        assert seen == ["channel_write"]

    @pytest.mark.usefixtures("project")
    async def test_a_server_carrying_no_session_is_answered_the_same(self, session, monkeypatch):
        session.on("live")
        monkeypatch.delenv(am.POSTURE_SESSION_ENV, raising=False)

        _, seen = await _refused_in_the_tool(am.AuditMiddleware(), "channel_write")

        assert seen == ["channel_write"]


class TestTheEntryPostureIsComputedOncePerCall:
    async def test_both_entries_of_one_call_share_one_posture_read(
        self, project, session, monkeypatch
    ):
        """The verdict is read once on entry and reused for both audit entries."""
        from osprey.audit import posture as posture_module

        session.on("live")
        session.switch("live")
        reads: list[str] = []
        real = posture_module.posture

        def _counted() -> str:
            reads.append("posture")
            return real()

        monkeypatch.setattr(posture_module, "posture", _counted)

        await am.AuditMiddleware().on_call_tool(_context("channel_write"), _a_write())

        assert reads == ["posture"]
        assert _records(project)[-1]["posture"] == am.POSTURE_WRITES
        assert _tool_calls(project)[-1]["posture"] == am.POSTURE_WRITES

    @pytest.mark.usefixtures("project")
    async def test_outside_a_call_it_is_asked_afresh(self, monkeypatch):
        from osprey.audit import posture as posture_module

        monkeypatch.setattr(posture_module, "posture", lambda: "writes")
        assert am._entry_posture() == "writes"


# --------------------------------------------------------------------------
# The tool_call record: what a call ran against
# --------------------------------------------------------------------------


@pytest.fixture
def bound(monkeypatch):
    """Put the deployment on control target ``va`` at generation 7."""
    from osprey_connectors import control_context

    monkeypatch.setattr(
        control_context, "read_record", lambda: SimpleNamespace(target="va", generation=7)
    )


def _tool_calls(project) -> list[dict]:
    return _records(project, surface="tool_call")


def _a_write(result: object = None):
    """A ``call_next`` that notes what channel_write notes, then returns."""
    from osprey.audit.call import note

    async def call_next(_ctx):
        note(ca_user="op", ca_host="ioc-host")
        note(limits={"verdict": "passed"})
        note(old_values={"A:1": {"value": 1.0, "timestamp": None}})
        return {"written": True} if result is None else result

    return call_next


class TestToolCallRecord:
    @pytest.mark.usefixtures("bound")
    async def test_with_the_key_absent_a_write_files_one_trimmed_entry(self, project):
        """No ``audit.tool_call`` setting, one channel_write: one entry, no payload."""
        assert "audit" not in Path(os.environ[am.CONFIG_ENV]).read_text()
        context = _context("channel_write", _TOOL_USE_META)
        context.message.arguments = {"operations": [{"channel": "A:1", "value": 2}]}

        await am.AuditMiddleware().on_call_tool(context, _a_write())

        (record,) = _tool_calls(project)
        assert record["surface"] == "tool_call"
        assert record["subject"] == "mcp__controls__channel_write"
        assert record["server"] == "controls"
        assert record["tool_use_id"] == "toolu_01abc"
        assert (record["target"], record["generation"]) == ("va", 7)
        assert record["facts"]["limits"] == {"verdict": "passed"}
        assert record["facts"]["old_values"] == {"A:1": {"value": 1.0, "timestamp": None}}
        assert record["duration_ms"] >= 0
        for dropped in (
            "arguments",
            "arguments_ref",
            "result",
            "result_ref",
            "decision",
            "reason",
            "error",
            "is_error",
        ):
            assert dropped not in record, dropped
        assert "written" not in json.dumps(record)
        assert '"A:1", "value": 2' not in json.dumps(record)

    async def test_the_entry_joins_the_default_record_by_tool_use_id(self, project, monkeypatch):
        monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "conv-9")
        await am.AuditMiddleware().on_call_tool(
            _context("channel_write", _TOOL_USE_META), _a_write()
        )
        (record,) = _tool_calls(project)
        default = _records(project)[-1]
        assert record["tool_use_id"] == default["tool_use_id"] == "toolu_01abc"
        assert record["session_id"] == "conv-9"
        assert default["decision"] == DECISION_ALLOWED

    async def test_a_call_with_no_facts_files_no_entry(self, project):
        await _call(am.AuditMiddleware(), "channel_read", meta=_TOOL_USE_META)
        assert _tool_calls(project) == []
        assert _records(project)[-1]["decision"] == DECISION_ALLOWED

    @pytest.mark.usefixtures("bound")
    async def test_a_read_on_a_target_is_recorded_too(self, project):
        await _call(am.AuditMiddleware(), "channel_read", result={"value": 1.5})
        (record,) = _tool_calls(project)
        assert record["subject"] == "mcp__controls__channel_read"
        assert (record["target"], record["generation"]) == ("va", 7)
        assert "result" not in record

    async def test_an_approval_alone_is_enough(self, project, ask_stamps):
        ask_stamps("toolu_01abc", approver="human", mode="default")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        (record,) = _tool_calls(project)
        assert record["approval"] == {
            "outcome": "approved",
            "approver": "human",
            "permission_mode": "default",
            "position": None,
        }
        assert record["target"] is None

    @pytest.mark.usefixtures("bound")
    async def test_a_posture_refusal_files_the_entry_once(self, project, monkeypatch):
        _sandbox(monkeypatch)
        await _refused(am.AuditMiddleware(), "channel_write")
        (record,) = _tool_calls(project)
        assert (record["target"], record["generation"]) == ("va", 7)
        assert _records(project)[-1]["decision"] == DECISION_REFUSED

    @pytest.mark.usefixtures("bound")
    async def test_a_tool_error_keeps_its_message_out_of_the_entry(self, project):
        with pytest.raises(ToolError):
            await _call(am.AuditMiddleware(), "channel_write", raises=ToolError("limit hit"))
        (record,) = _tool_calls(project)
        assert "limit hit" not in json.dumps(record)
        assert _records(project)[-1]["reason"] == am.REASON_TOOL_ERROR

    @pytest.mark.usefixtures("bound")
    async def test_a_failing_tool_call_write_never_costs_the_call(self, project, monkeypatch):
        def broken(*_args, **_kwargs):
            raise RuntimeError("record gone")

        monkeypatch.setattr(am.tool_call, "build_record", broken)
        result, _ = await _call(am.AuditMiddleware(), "channel_read")
        assert result == "channel_read-result"
        assert _tool_calls(project) == []
        assert _records(project)[-1]["decision"] == DECISION_ALLOWED

    async def test_the_default_record_is_the_same_with_or_without_the_entry(
        self, project, monkeypatch
    ):
        await _call(am.AuditMiddleware(), "channel_read", meta=_TOOL_USE_META)
        without = _records(project)[-1]
        assert _tool_calls(project) == []

        from osprey_connectors import control_context

        monkeypatch.setattr(
            control_context, "read_record", lambda: SimpleNamespace(target="va", generation=7)
        )
        await _call(am.AuditMiddleware(), "channel_read", meta=_TOOL_USE_META)
        with_entry = _records(project)[-1]
        assert len(_tool_calls(project)) == 1

        without.pop("ts")
        with_entry.pop("ts")
        assert with_entry == without

    def test_the_settings_reader_is_gone(self):
        assert not hasattr(am, "_tool_call_settings")
        assert not hasattr(am.tool_call, "settings")


class TestToolCallApproval:
    """``tool_call.approval`` names who answered and the switch position it ran under."""

    async def test_a_switch_approved_call_names_write_switch_at_allowed(self, project, ask_stamps):
        ask_stamps("toolu_01abc", approver="write_switch", position="allowed")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        (record,) = _tool_calls(project)
        assert record["approval"] == {
            "outcome": "approved",
            "approver": "write_switch",
            "permission_mode": "default",
            "position": "allowed",
        }
        # The position never rides in the decision: the envelope keeps its words.
        assert _records(project)[-1]["decision"] == DECISION_ALLOWED

    async def test_a_human_approved_call_carries_asked(self, project, ask_stamps):
        ask_stamps("toolu_01abc", approver="human", mode="acceptEdits", position="asked")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        (record,) = _tool_calls(project)
        assert record["approval"] == {
            "outcome": "approved",
            "approver": "human",
            "permission_mode": "acceptEdits",
            "position": "asked",
        }
        assert _records(project)[-1]["decision"] == DECISION_ALLOWED

    async def test_a_dispatch_run_stays_dispatch_policy_at_allowed(self, project, ask_stamps):
        ask_stamps("toolu_01abc", approver="dispatch_policy", position="allowed")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        (record,) = _tool_calls(project)
        assert record["approval"]["approver"] == "dispatch_policy"
        assert record["approval"]["position"] == "allowed"
        assert _records(project)[-1]["decision"] == DECISION_ALLOWED

    async def test_a_non_governed_ask_has_no_position(self, project, ask_stamps):
        ask_stamps("toolu_01abc")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        (record,) = _tool_calls(project)
        assert record["approval"]["position"] is None
        assert record["approval"]["approver"] == "human"

    async def test_the_detail_names_the_switch_approver(self, project, ask_stamps):
        ask_stamps("toolu_01abc", approver="write_switch", position="allowed")
        await _call(am.AuditMiddleware(), "channel_write", meta=_TOOL_USE_META)
        record = _records(project)[-1]
        assert record["detail"] == "approval=approved approver=write_switch"
        assert record["decision"] == DECISION_ALLOWED

    @pytest.mark.usefixtures("project")
    def test_the_answer_reads_approver_mode_and_position(self, ask_stamps):
        ask_stamps("toolu_01abc", approver="write_switch", position="allowed")
        assert am._approval_answered("toolu_01abc") == ("write_switch", "default", "allowed")
        assert am._approval_answered("toolu_absent") is None

    @pytest.mark.usefixtures("project")
    def test_a_malformed_position_is_absent(self, ask_stamps):
        path = ask_stamps("toolu_01abc")
        path.write_text(json.dumps({"approver": "human", "position": 3}))
        assert am._approval_answered("toolu_01abc") == ("human", None, None)
