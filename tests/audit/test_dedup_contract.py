"""Contract tests for the dedup invariant: the innermost recorder owns the decision.

Two safety layers can see the same call — the MCP audit middleware wraps every
``tools/call``, and the tool's own gates run inside it. Without a marker the
outer layer either double-records the decision or, worse, stamps ``allowed`` on
a call an inner guard refused while still returning a successful result. The
marker in :mod:`osprey.audit.dedup` carries the inner *decision*, not merely a
"handled" flag, so the outer layer defers to a specific answer and can say so
when the two disagree.

The tests below pin the properties the invariant rests on: the record is
written once, it is written by the layer that decided, the deferral carries
outward so a stacked layer defers to the same answer, the marker never survives
the call that set it, and a marker inherited by another process is not
believed. One test pins the mechanism's *limitation* instead — a recorder that
runs off the awaiting task is not seen — because that is the shape a future
inner recorder is most likely to be written in.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import MiddlewareContext
from mcp.types import CallToolRequestParams

from osprey.audit import dedup, writer
from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED, SURFACE_EXECUTOR
from osprey.mcp_server import audit_middleware as am
from osprey.mcp_server.python_executor.tools import _execution_gates as gates
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV, acting_identity

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

PYTHON_EXECUTE = "mcp__python__execute"
CONTROLS_WRITE = "mcp__controls__channel_write"


@pytest.fixture(autouse=True)
def _no_marker_between_tests():
    """Start and end every test with an empty marker.

    The carrier is a ``ContextVar``, and pytest runs a module's tests in one
    context — so a marker one test leaves behind is visible to the next. On the
    running path that cannot happen: the outer layer wraps each call in
    :func:`~osprey.audit.dedup.decision_scope`, which is what this fixture
    stands in for.
    """
    dedup.clear_recorded()
    yield
    dedup.clear_recorded()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A healthy render zone with the audit zone redirected under ``tmp_path``.

    Prefix ``python`` throughout: the dedup pairs that exist today are the
    middleware and the executor's own gates, and a prefix the render lists is
    what puts the middleware on its verified path — the one where a missing
    defer would be visible as a second record rather than masked by a
    fail-closed refusal.
    """
    for marker in (
        TERMINAL_USER_ENV,
        AUDIT_IDENTITY_ENV,
        writer.AUDIT_WRITER_ENV,
        am.POSTURE_ENV,
        am.POSTURE_SOURCE_ENV,
        am.POSTURE_SESSION_ENV,
    ):
        monkeypatch.delenv(marker, raising=False)

    build = tmp_path / "build"
    (build / ".claude" / "hooks").mkdir(parents=True)
    config = build / "config.yml"
    config.write_text("control_system: {}\n")
    (build / ".claude" / "hooks" / "hook_config.json").write_text(
        json.dumps(
            {
                "server_prefixes": ["mcp__controls__", "mcp__python__"],
                "approval_prefixes": [],
                "write_tools": [CONTROLS_WRITE, PYTHON_EXECUTE],
                "write_servers": [],
                "mixed_read_write_tools": [PYTHON_EXECUTE],
            }
        )
    )
    monkeypatch.setenv(am.CONFIG_ENV, str(config))
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, "python")

    audit_root = tmp_path / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: audit_root)
    am.reset_audit_state()
    dedup.clear_recorded()

    yield audit_root

    am.reset_audit_state()
    dedup.clear_recorded()


def _records(audit_root, surface: str) -> list[dict]:
    path = audit_root / acting_identity() / f"{surface}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_records(audit_root) -> list[dict]:
    """Every record in the zone, whatever surface it landed on."""
    if not audit_root.exists():
        return []
    out: list[dict] = []
    for path in sorted(audit_root.rglob("*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def _context(tool: str) -> MiddlewareContext:
    return MiddlewareContext(
        message=CallToolRequestParams(name=tool, arguments={}),
        method="tools/call",
    )


def _inner_refusal(audit_root, *, reason: str = "runtime_guard"):
    """A ``call_next`` whose tool refuses internally yet answers successfully.

    The shape the invariant exists for: the runtime guard fires inside the
    subprocess, the tool reports it and returns whatever the script produced,
    so nothing the middleware can observe about the *result* says a write was
    refused.
    """

    async def call_next(context):
        dedup.record_and_mark(
            decision=DECISION_REFUSED,
            reason=reason,
            surface=SURFACE_EXECUTOR,
            posture=am.POSTURE_SANDBOX,
            posture_source="process",
            session=None,
            subject=f"mcp__python__{context.message.name}",
        )
        return f"{context.message.name}-result"

    return call_next


# --------------------------------------------------------------------------
# The marker itself
# --------------------------------------------------------------------------


class TestMarkerPrimitives:
    def test_no_marker_is_the_default(self):
        assert dedup.recorded_decision() is None

    def test_a_marker_carries_the_decision_and_the_reason(self):
        dedup.mark_recorded(DECISION_REFUSED, "posture")
        marked = dedup.recorded_decision()
        assert marked is not None
        assert (marked.decision, marked.reason) == (DECISION_REFUSED, "posture")

    def test_clearing_forgets_the_marker(self):
        dedup.mark_recorded(DECISION_REFUSED, "posture")
        dedup.clear_recorded()
        assert dedup.recorded_decision() is None

    def test_a_scope_hides_a_marker_set_before_it(self):
        dedup.mark_recorded(DECISION_REFUSED, "stale")
        with dedup.decision_scope():
            assert dedup.recorded_decision() is None

    def test_a_scope_forgets_the_marker_set_inside_it(self):
        with dedup.decision_scope():
            dedup.mark_recorded(DECISION_REFUSED, "posture")
        assert dedup.recorded_decision() is None

    def test_a_scope_forgets_the_marker_even_when_the_body_raises(self):
        with pytest.raises(RuntimeError), dedup.decision_scope():
            dedup.mark_recorded(DECISION_REFUSED, "posture")
            raise RuntimeError("boom")
        assert dedup.recorded_decision() is None

    def test_a_marker_inherited_by_another_process_is_not_observed(self, monkeypatch):
        """A fork inherits the context; the marker must not survive the pid change."""
        dedup.mark_recorded(DECISION_REFUSED, "posture")
        monkeypatch.setattr(dedup, "_current_pid", lambda: 424242)
        assert dedup.recorded_decision() is None

    def test_an_empty_decision_is_not_marked(self):
        assert dedup.mark_recorded("", "posture") is None
        assert dedup.recorded_decision() is None

    def test_an_empty_reason_is_not_marked(self):
        assert dedup.mark_recorded(DECISION_REFUSED, "") is None
        assert dedup.recorded_decision() is None


# --------------------------------------------------------------------------
# The middleware defers
# --------------------------------------------------------------------------


class TestTheMiddlewareDefersToTheInnerLayer:
    async def test_a_runtime_guard_refusal_that_returns_a_result_is_recorded_once(
        self, project, monkeypatch
    ):
        """The headline case: one record, ``refused``, from the inner layer.

        Remove the defer in ``AuditMiddleware.on_call_tool`` and this fails
        with two records, the second of them ``allowed`` — the middleware
        contradicting a refusal that really happened.
        """
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")
        result = await am.AuditMiddleware().on_call_tool(
            _context("execute"), _inner_refusal(project)
        )

        assert result == "execute-result"
        records = _all_records(project)
        assert len(records) == 1
        assert records[0]["decision"] == DECISION_REFUSED
        assert records[0]["surface"] == SURFACE_EXECUTOR
        assert _records(project, "python") == []

    async def test_the_production_runtime_reporter_is_the_layer_the_middleware_defers_to(
        self, project, monkeypatch
    ):
        """The headline case again, with the *shipped* recorder and no stand-in.

        ``_inner_refusal`` calls :func:`~osprey.audit.dedup.record_and_mark`
        itself, so it pins that the middleware defers — not that the executor's
        own recorder asks it to. Swap ``record_and_mark`` for ``writer.record``
        inside ``_record_write_refusal`` (identical kwargs, one import line)
        and every other test in the repository still passes while a refused
        control-system write starts being stamped ``allowed`` by the layer
        above it. This is the one test that notices.
        """
        from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

        monkeypatch.setenv(am.POSTURE_ENV, "readonly")
        alerts: list[str] = []

        async def _alert(tool, kind, *, detail):
            alerts.append(detail)

        monkeypatch.setattr(gates, "notify_agent_activity_async", _alert)

        async def call_next(context):
            # The runtime guard fires inside the subprocess: all that reaches
            # the tool is a marked line on stderr, and the tool still answers
            # with whatever the script produced before the refusal.
            reported = await gates.report_runtime_refusal(
                tool="execute",
                stderr=f"RuntimeError: {READONLY_REFUSAL_MARKER}: refused a write to SR:MAG:1",
                code="caput('SR:MAG:1', 1)",
                description="nudge a magnet",
                execution_mode="readonly",
            )
            assert reported, "the marked stderr must be recognised as a refusal"
            return "execute-result"

        result = await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert result == "execute-result"
        records = _all_records(project)
        assert len(records) == 1, records
        assert (records[0]["surface"], records[0]["decision"], records[0]["reason"]) == (
            SURFACE_EXECUTOR,
            DECISION_REFUSED,
            gates.LAYER_RUNTIME_GUARD,
        )
        assert _records(project, "python") == [], "the middleware filed a record of its own"
        assert alerts, "the operator alert is the other half of the report"

    async def test_the_runtime_report_records_the_mode_the_run_actually_asked_for(
        self, project, monkeypatch
    ):
        """``readwrite`` in the run is ``readwrite`` in the record and the alert.

        Only the readonly marker reaches this reporter today, but that is a
        property of another module's message strings rather than of this one:
        a guard that emits the marker mid-``readwrite`` must not have the
        ledger call the run readonly and the alert tell an operator approved
        for writes that they were sandboxed.
        """
        from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

        alerts: list[str] = []

        async def _alert(tool, kind, *, detail):
            alerts.append(detail)

        monkeypatch.setattr(gates, "notify_agent_activity_async", _alert)

        await gates.report_runtime_refusal(
            tool="execute",
            stderr=f"RuntimeError: {READONLY_REFUSAL_MARKER}: refused a write to SR:MAG:1",
            code="caput('SR:MAG:1', 1)",
            description=None,
            execution_mode="readwrite",
        )

        (record,) = _records(project, SURFACE_EXECUTOR)
        assert "mode=readwrite" in record["detail"], record["detail"]
        assert alerts == ["BLOCKED a control-system write in readwrite mode (runtime_guard)"]

    async def test_the_middleware_records_the_call_when_no_inner_layer_did(self, project):
        async def call_next(context):
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        records = _records(project, "python")
        assert [(r["decision"], r["reason"]) for r in records] == [
            (DECISION_ALLOWED, am.REASON_TOOL_CALL)
        ]

    async def test_a_tool_error_the_inner_layer_recorded_is_not_recorded_twice(self, project):
        async def call_next(context):
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason=am.REASON_POSTURE,
                surface=SURFACE_EXECUTOR,
                posture=am.POSTURE_SANDBOX,
                posture_source="process",
                session=None,
                subject=PYTHON_EXECUTE,
            )
            raise ToolError("refused in the tool")

        with pytest.raises(ToolError):
            await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert len(_all_records(project)) == 1
        assert _records(project, "python") == []
        assert _records(project, SURFACE_EXECUTOR)[0]["reason"] == am.REASON_POSTURE

    async def test_a_tool_error_no_inner_layer_recorded_is_still_recorded(self, project):
        async def call_next(context):
            raise ToolError("refused with no record")

        with pytest.raises(ToolError):
            await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        records = _records(project, "python")
        assert [(r["decision"], r["reason"]) for r in records] == [
            (DECISION_REFUSED, am.REASON_TOOL_ERROR)
        ]

    async def test_the_marker_does_not_leak_into_the_next_call(self, project, monkeypatch):
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")
        mw = am.AuditMiddleware()

        await mw.on_call_tool(_context("execute"), _inner_refusal(project))

        async def clean(context):
            return "ran"

        await mw.on_call_tool(_context("execute"), clean)

        assert [r["decision"] for r in _records(project, "python")] == [DECISION_ALLOWED]
        assert len(_all_records(project)) == 2

    async def test_a_marker_left_by_an_earlier_call_does_not_suppress_this_one(self, project):
        """A marker set outside any call must not silence the next one."""
        dedup.mark_recorded(DECISION_REFUSED, "stale")

        async def call_next(context):
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert [r["decision"] for r in _records(project, "python")] == [DECISION_ALLOWED]

    async def test_a_marker_from_another_process_does_not_suppress_the_outer_record(
        self, project, monkeypatch
    ):
        """A forked child inherits the context but is not the layer that decided."""

        async def call_next(context):
            dedup.mark_recorded(DECISION_REFUSED, "runtime_guard")
            monkeypatch.setattr(dedup, "_current_pid", lambda: 424242)
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert [r["decision"] for r in _records(project, "python")] == [DECISION_ALLOWED]


# --------------------------------------------------------------------------
# The in-tool posture clamp is an inner recorder
# --------------------------------------------------------------------------


class TestThePostureClampRecordsAndMarks:
    def test_the_clamp_files_one_executor_record_and_marks_it(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)

        with pytest.raises(ToolError):
            gates.enforce_posture_clamp("readwrite", tool="execute")

        records = _records(project, SURFACE_EXECUTOR)
        assert len(records) == 1
        assert records[0]["decision"] == DECISION_REFUSED
        assert records[0]["reason"] == gates.REASON_POSTURE
        assert records[0]["subject"] == PYTHON_EXECUTE
        assert records[0]["posture"] == gates.POSTURE_SANDBOX
        assert records[0].get("source") is None

        marked = dedup.recorded_decision()
        assert marked is not None
        assert (marked.decision, marked.reason) == (DECISION_REFUSED, gates.REASON_POSTURE)

    def test_the_clamp_names_the_tool_it_was_called_for(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)

        with pytest.raises(ToolError):
            gates.enforce_posture_clamp("readwrite", tool="execute_file")

        assert _records(project, SURFACE_EXECUTOR)[0]["subject"] == "mcp__python__execute_file"

    def test_the_clamp_records_the_stamped_posture_source_and_session(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)
        monkeypatch.setenv(am.POSTURE_SOURCE_ENV, "spawn")
        monkeypatch.setenv(am.POSTURE_SESSION_ENV, "operator-7")

        with pytest.raises(ToolError):
            gates.enforce_posture_clamp("readwrite", tool="execute")

        record = _records(project, SURFACE_EXECUTOR)[0]
        assert record["posture_source"] == "spawn"
        assert record["session"] == "operator-7"

    def test_an_unrecognised_posture_source_degrades_to_process(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)
        monkeypatch.setenv(am.POSTURE_SOURCE_ENV, "made-up")

        with pytest.raises(ToolError):
            gates.enforce_posture_clamp("readwrite", tool="execute")

        assert _records(project, SURFACE_EXECUTOR)[0]["posture_source"] == "process"

    def test_a_readonly_run_is_neither_recorded_nor_marked(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)
        gates.enforce_posture_clamp("readonly", tool="execute")
        assert _all_records(project) == []
        assert dedup.recorded_decision() is None

    def test_the_writes_posture_is_neither_recorded_nor_marked(self, project, monkeypatch):
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, "readwrite")
        gates.enforce_posture_clamp("readwrite", tool="execute")
        assert _all_records(project) == []
        assert dedup.recorded_decision() is None

    def test_an_unwritable_audit_zone_still_refuses_and_still_marks(self, project, monkeypatch):
        """Recording never costs the refusal — and never costs the defer either."""
        monkeypatch.setenv(gates.POSTURE_ENV_VAR, gates.SANDBOX_POSTURE)

        def boom():
            raise OSError("no audit zone")

        monkeypatch.setattr(writer, "audit_dir", boom)

        with pytest.raises(ToolError):
            gates.enforce_posture_clamp("readwrite", tool="execute")

        marked = dedup.recorded_decision()
        assert marked is not None and marked.decision == DECISION_REFUSED

    async def test_the_middleware_defers_to_the_clamp(self, project, monkeypatch):
        """End to end: one record for a call the in-tool clamp refused."""
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")

        async def call_next(context):
            gates.enforce_posture_clamp("readwrite", tool=context.message.name)
            return "ran"

        with pytest.raises(ToolError):
            await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        records = _all_records(project)
        assert len(records) == 1
        assert records[0]["surface"] == SURFACE_EXECUTOR
        assert records[0]["reason"] == gates.REASON_POSTURE
        assert _records(project, "python") == []


# --------------------------------------------------------------------------
# Spellings the two layers must agree on
# --------------------------------------------------------------------------


class TestPinnedSpellings:
    """The gates and the middleware address the same markers under their own names.

    The posture markers are :mod:`osprey.audit.posture`'s, re-exported by both;
    the tool-prefix marker is re-spelled, because an inner recorder importing
    the outer layer would invert the dependency and drag fastmcp's middleware
    machinery into every executor tool. Pinned either way, so a recorder that
    went back to its own spelling fails here.
    """

    def test_the_posture_env_matches(self):
        assert gates.POSTURE_ENV_VAR == am.POSTURE_ENV

    def test_the_posture_source_env_matches(self):
        assert gates.POSTURE_SOURCE_ENV_VAR == am.POSTURE_SOURCE_ENV

    def test_the_posture_session_env_matches(self):
        assert gates.POSTURE_SESSION_ENV_VAR == am.POSTURE_SESSION_ENV

    def test_the_tool_prefix_env_matches(self):
        assert gates.TOOL_PREFIX_ENV_VAR == am.TOOL_PREFIX_ENV

    def test_the_sandbox_mode_value_matches(self):
        assert gates.SANDBOX_POSTURE == am.SANDBOX_MODE

    def test_the_ledger_posture_spelling_matches(self):
        assert gates.POSTURE_SANDBOX == am.POSTURE_SANDBOX

    def test_the_clamp_reason_matches(self):
        assert gates.REASON_POSTURE == am.REASON_POSTURE


# --------------------------------------------------------------------------
# The deferral carries outward, and only along the awaiting task
# --------------------------------------------------------------------------


def _tool_fn(tool):
    """The raw function behind a FastMCP ``FunctionTool`` (or the function itself)."""
    return getattr(tool, "fn", tool)


class TestTheDeferralIsTransitive:
    """A layer that defers re-asserts the marker, so the layer above defers too.

    ``startup.run_mcp_server`` installs one middleware today. The transitivity
    is not for that case: it is so that the *next* layer to wrap this one —
    the HTTP pairing around an in-process call, a second install during a
    migration — inherits the invariant instead of quietly stamping ``allowed``
    over a refusal the inner instance honoured.
    """

    async def test_two_stacked_middlewares_record_a_refusal_once(self, project, monkeypatch):
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")
        inner_mw = am.AuditMiddleware()
        outer_mw = am.AuditMiddleware()

        async def through_the_inner_middleware(context):
            return await inner_mw.on_call_tool(context, _inner_refusal(project))

        result = await outer_mw.on_call_tool(_context("execute"), through_the_inner_middleware)

        assert result == "execute-result"
        records = _all_records(project)
        assert len(records) == 1
        assert records[0]["decision"] == DECISION_REFUSED
        assert records[0]["surface"] == SURFACE_EXECUTOR
        assert _records(project, "python") == []

    async def test_two_stacked_middlewares_record_a_tool_error_once(self, project):
        inner_mw = am.AuditMiddleware()
        outer_mw = am.AuditMiddleware()

        async def call_next(context):
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason=am.REASON_POSTURE,
                surface=SURFACE_EXECUTOR,
                posture=am.POSTURE_SANDBOX,
                posture_source="process",
                session=None,
                subject=PYTHON_EXECUTE,
            )
            raise ToolError("refused in the tool")

        async def through_the_inner_middleware(context):
            return await inner_mw.on_call_tool(context, call_next)

        with pytest.raises(ToolError):
            await outer_mw.on_call_tool(_context("execute"), through_the_inner_middleware)

        assert len(_all_records(project)) == 1
        assert _records(project, "python") == []

    async def test_a_stacked_middleware_still_records_a_call_nobody_claimed(self, project):
        """Transitivity must not turn the *outer* layer off for an unclaimed call."""
        inner_mw = am.AuditMiddleware()
        outer_mw = am.AuditMiddleware()

        async def call_next(context):
            return "ran"

        async def through_the_inner_middleware(context):
            return await inner_mw.on_call_tool(context, call_next)

        await outer_mw.on_call_tool(_context("execute"), through_the_inner_middleware)

        # Both layers record: nothing claimed the decision, and this is the
        # double-record that stacking has always produced. What must not happen
        # is a *split* decision, which the refusal tests above pin.
        assert [r["decision"] for r in _records(project, "python")] == [
            DECISION_ALLOWED,
            DECISION_ALLOWED,
        ]


class TestTheInnerRecorderMustBeOnTheAwaitingTask:
    """The mechanism's boundary, pinned rather than papered over.

    A ``ContextVar`` is *copied* into a new task, never shared back. A recorder
    reached through ``create_task``/``gather``, ``to_thread``, or a synchronous
    tool body the server runs on a worker thread therefore leaves its mark
    where no outer layer can see it, and the outer layer files ``allowed`` on
    top of a refusal. Nothing raises when that happens, so the property is
    asserted here in the direction it actually behaves.
    """

    async def test_a_recorder_on_a_child_task_is_not_seen(self, project, monkeypatch):
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")

        async def refuse_on_a_child_task():
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason="runtime_guard",
                surface=SURFACE_EXECUTOR,
                posture=am.POSTURE_SANDBOX,
                posture_source="process",
                session=None,
                subject=PYTHON_EXECUTE,
            )

        async def call_next(context):
            await asyncio.gather(asyncio.create_task(refuse_on_a_child_task()))
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        # Two records, and the middleware's says `allowed` over a refusal that
        # really happened. This is the documented limitation of the carrier,
        # not a behaviour to rely on: an inner recorder has to run inline on
        # the task the middleware awaits.
        assert [r["decision"] for r in _records(project, "python")] == [DECISION_ALLOWED]
        assert len(_all_records(project)) == 2

    async def test_a_recorder_awaited_inline_is_seen(self, project, monkeypatch):
        """The same body, awaited on the middleware's own task: one record."""
        monkeypatch.setenv(am.POSTURE_ENV, "readonly")

        async def refuse():
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason="runtime_guard",
                surface=SURFACE_EXECUTOR,
                posture=am.POSTURE_SANDBOX,
                posture_source="process",
                session=None,
                subject=PYTHON_EXECUTE,
            )

        async def call_next(context):
            await refuse()
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert len(_all_records(project)) == 1
        assert _records(project, "python") == []

    def test_the_executor_tools_are_coroutine_functions(self):
        """The live inner recorders sit inside ``async def`` tools.

        FastMCP hands a synchronous tool body to ``anyio.to_thread.run_sync``,
        which copies the context — so were either of these to become a plain
        ``def``, every refusal the executor's own gates record would be
        followed by an ``allowed`` from the middleware.
        """
        from osprey.mcp_server.python_executor.tools.python_execute import execute
        from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

        assert inspect.iscoroutinefunction(_tool_fn(execute))
        assert inspect.iscoroutinefunction(_tool_fn(execute_file))


class TestAnInnerRecordThatDidNotLand:
    """``stored=False``: the decision is owned, but no line reached the ledger.

    Deferring blindly then costs the ledger the record entirely when the tool
    also raises. The rule: the outer layer may file its own record only where
    that record cannot contradict the inner decision — the ``ToolError``
    branch, never the success branch.
    """

    async def test_a_tool_error_over_an_unstored_marker_is_recorded_by_the_middleware(
        self, project
    ):
        async def call_next(context):
            dedup.mark_recorded(DECISION_REFUSED, "runtime_guard", stored=False)
            raise ToolError("refused, and the inner write never landed")

        with pytest.raises(ToolError):
            await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert [(r["decision"], r["reason"]) for r in _records(project, "python")] == [
            (DECISION_REFUSED, am.REASON_TOOL_ERROR)
        ]

    async def test_a_successful_call_over_an_unstored_marker_is_still_deferred(self, project):
        """Never ``allowed`` over a refusal, even when the refusal is unrecorded."""

        async def call_next(context):
            dedup.mark_recorded(DECISION_REFUSED, "runtime_guard", stored=False)
            return "ran"

        await am.AuditMiddleware().on_call_tool(_context("execute"), call_next)

        assert _all_records(project) == []

    def test_a_failed_write_marks_the_decision_as_unstored(self, project, monkeypatch):
        def boom():
            raise OSError("no audit zone")

        monkeypatch.setattr(writer, "audit_dir", boom)
        dedup.record_and_mark(
            decision=DECISION_REFUSED,
            reason="runtime_guard",
            surface=SURFACE_EXECUTOR,
            posture=am.POSTURE_SANDBOX,
            posture_source="process",
            session=None,
            subject=PYTHON_EXECUTE,
        )

        marked = dedup.recorded_decision()
        assert marked is not None
        assert marked.stored is False

    def test_a_landed_write_marks_the_decision_as_stored(self, project):
        dedup.record_and_mark(
            decision=DECISION_REFUSED,
            reason="runtime_guard",
            surface=SURFACE_EXECUTOR,
            posture=am.POSTURE_SANDBOX,
            posture_source="process",
            session=None,
            subject=PYTHON_EXECUTE,
        )

        marked = dedup.recorded_decision()
        assert marked is not None
        assert marked.stored is True
