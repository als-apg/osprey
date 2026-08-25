"""Pre-migration capture of every ``--json`` payload shape the CLI emits.

Five verbs have a ``--json`` path: ``audit``, ``health``, ``query``,
``ariel status`` and ``ariel search``. The output-hierarchy work reroutes each
verb's *human* output through a single renderer; none of it is allowed to move
a key in the *machine* output. These tests record what each payload looks like
today so that a later migration which quietly reshapes one is caught here
rather than in a downstream consumer.

**What is captured.** Payload *values* are non-deterministic by construction
(timestamps, live probes, model text), so only the key set is pinned. Each
golden in ``tests/cli/data/json_keysets/`` is a sorted JSON array of key paths:

* a bare name (``"summary"``) is a top-level key;
* ``"parent.child"`` is a key of a nested object;
* ``"parent[].child"`` is a key of every element of a nested array.

**Why only some nesting.** One level deeper is pinned only where the nested
shape is a literal dict written out in the source — it cannot vary at runtime,
so equality is a real contract rather than a fixture artifact:

* ``audit`` → ``findings[]`` (the ``AuditFinding`` pydantic model);
* ``query`` → ``tool_traces[]`` (a literal dict comprehension in
  ``_build_json_output``);
* ``ariel status`` → ``database``, ``embedding_tables[]``,
  ``enhancement_modules``, ``search_modules`` (all literal dicts in
  ``get_status``);
* ``ariel search`` → ``entries[]`` (``_entry_summary``'s fixed projection).

Deliberately **not** nested: ``health``'s ``results[]`` rows, whose ``value`` /
``latency_ms`` / ``details`` keys are emitted only when truthy, so the row key
set varies per check and per run — equality against one golden would pin
whichever shapes the fixture happened to produce. That row contract is pinned
instead by :func:`test_health_row_keys_are_required_plus_optional`, which drives
a report carrying every shape and asserts the four required keys are always
present and the three optional ones are the only permitted additions. Also not
nested: ``query``'s ``mcp_servers``, whose keys are server *names* from the
deployment rather than a schema.

**Capture seams.** Every verb is driven end-to-end through ``CliRunner`` with
only its outermost I/O boundary faked, so the payload is assembled by the real
production code path:

* ``audit`` — ``_SDK_AVAILABLE``/``asyncio`` patched (as in
  ``test_audit_cmd.py``); the report is still parsed and re-serialised by
  ``_display_report``.
* ``health`` — ``osprey.cli.health_cmd._run_suite`` patched to return a
  ``CheckReport``; ``render_json`` does the serialising.
* ``query`` — ``run_query`` / ``read_only_disallowed_tools`` /
  ``expected_mcp_servers`` patched (as in ``test_query_cmd.py``) over a stub
  build.
* ``ariel status`` / ``ariel search`` — ``osprey.cli.ariel.get_config_value``
  supplies a config section and
  ``osprey.services.ariel_search.create_ariel_service`` is routed to a stub
  service (as in ``tests/services/ariel_search/test_cli_operations.py``), so
  the real ``get_status`` / ``run_search`` assemble the dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.agent_runner import SDKWorkflowResult, ToolTrace
from osprey.cli.ariel import ariel_group
from osprey.cli.audit_prompts import AuditFinding, AuditReport
from osprey.cli.health_cmd import health
from osprey.cli.query_cmd import query
from osprey.health.models import CheckReport, CheckResult, Status
from tests.cli._lifecycle_build import stub_build

# --------------------------------------------------------------------------- #
# Golden handling
# --------------------------------------------------------------------------- #

_GOLDEN_DIR = Path(__file__).parent / "data" / "json_keysets"

# Nested paths pinned per verb. ``name[]`` expands the element shape of a list;
# ``name`` expands the keys of an object. See the module docstring for why each
# one is safe to pin and why the omitted ones are not.
_NESTED: dict[str, tuple[str, ...]] = {
    "audit": ("findings[]",),
    "health": (),
    "query": ("tool_traces[]",),
    "ariel_status": (
        "database",
        "embedding_tables[]",
        "enhancement_modules",
        "search_modules",
    ),
    "ariel_search": ("entries[]",),
}


def _golden(verb: str) -> list[str]:
    """Return the recorded key paths for *verb*."""
    return json.loads((_GOLDEN_DIR / f"{verb}.json").read_text())


def _keyset(payload: dict[str, Any], verb: str) -> list[str]:
    """Project *payload* onto the sorted key-path list recorded for *verb*.

    Args:
        payload: The parsed ``--json`` document.
        verb: Golden name, which selects the nested paths to expand.

    Returns:
        Sorted key paths: top-level keys plus one expanded level for every
        path in ``_NESTED[verb]``.
    """
    paths = list(payload)
    for spec in _NESTED[verb]:
        is_list = spec.endswith("[]")
        key = spec[:-2] if is_list else spec
        node = payload[key]
        if is_list:
            assert node, f"{verb}: {spec} must be non-empty to capture its shape"
            shapes = {tuple(sorted(item)) for item in node}
            assert len(shapes) == 1, f"{verb}: {spec} elements disagree on keys"
            node = node[0]
        paths.extend(f"{spec}.{k}" for k in node)
    return sorted(paths)


def _assert_matches_golden(payload: dict[str, Any], verb: str) -> None:
    """Fail unless *payload* still has exactly the recorded key paths."""
    assert _keyset(payload, verb) == _golden(verb)


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def runner() -> CliRunner:
    """A Click runner that captures stdout and stderr separately."""
    return CliRunner()


# --------------------------------------------------------------------------- #
# osprey audit --json
# --------------------------------------------------------------------------- #


@pytest.fixture
def audit_project(tmp_path: Path) -> Path:
    """A minimal directory that ``audit`` classifies as a project."""
    (tmp_path / "config.yml").write_text("name: test\n")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre_write.sh").write_text("#!/bin/bash\n")
    return tmp_path


@patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
@patch("osprey.cli.audit_cmd.asyncio")
def test_audit_json_keyset(mock_asyncio: MagicMock, runner: CliRunner, audit_project: Path) -> None:
    """``audit --json`` emits the recorded ``AuditReport`` key set."""
    report = AuditReport(
        summary="Capture run",
        overall_risk="low",
        findings=[
            AuditFinding(
                category="permissions",
                severity="warning",
                title="Open permission",
                explanation="A permission is too broad",
                file_path="settings.json",
                recommendation="Restrict the permission scope",
            ),
        ],
    )
    mock_asyncio.run.return_value = (report.model_dump_json(), 0.01, 5)

    result = runner.invoke(_audit_cmd(), [str(audit_project), "--json"])

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "audit")


def _audit_cmd():
    """Import the ``audit`` command lazily, as ``test_audit_cmd.py`` does."""
    from osprey.cli.audit_cmd import audit

    return audit


# --------------------------------------------------------------------------- #
# osprey health --json
# --------------------------------------------------------------------------- #

# Row keys of ``results[]``. ``CheckResult.to_dict`` always writes the first
# four and adds each of the second three only when populated (``latency_ms``
# when greater than zero), as documented in
# ``docs/source/reference/contracts/health-json.rst``.
_HEALTH_ROW_REQUIRED = frozenset({"name", "category", "status", "message"})
_HEALTH_ROW_OPTIONAL = frozenset({"value", "latency_ms", "details"})


@pytest.fixture
def health_project(tmp_path: Path) -> Path:
    """A project whose ``config.yml`` parses and touches no network."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yml").write_text(
        "project_name: keyset_capture\n"
        "models:\n"
        "  python_code_generator:\n"
        "    provider: mock\n"
        "    model_id: mock-model\n"
    )
    return project


def test_health_json_keyset(runner: CliRunner, health_project: Path) -> None:
    """``health --json`` emits the recorded ``CheckReport`` wire shape."""
    report = CheckReport(
        results=[
            CheckResult(
                name="capture",
                category="demo",
                status=Status.OK,
                message="ok",
            )
        ],
        elapsed_ms=1.5,
        deadline_hit=False,
    )
    with patch("osprey.cli.health_cmd._run_suite", AsyncMock(return_value=report)):
        result = runner.invoke(health, ["--project", str(health_project), "--json"])

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "health")


def test_health_row_keys_are_required_plus_optional(
    runner: CliRunner, health_project: Path
) -> None:
    """Every ``results[]`` row carries the required keys and only the optional ones.

    The golden pins the envelope; the row shape varies per check, so it is
    pinned here instead over a report that carries all three shapes: a row with
    nothing measured, one carrying ``value`` and ``latency_ms``, and one
    carrying ``details``. A row shows the optional keys are genuinely omitted
    when unpopulated, the other two show each is still emitted when it is.
    """
    report = CheckReport(
        results=[
            CheckResult(
                name="bare",
                category="demo",
                status=Status.OK,
                message="nothing measured",
            ),
            CheckResult(
                name="measured",
                category="demo",
                status=Status.OK,
                message="probe returned",
                value="401.2 mA",
                latency_ms=12.34,
            ),
            CheckResult(
                name="diagnosed",
                category="demo",
                status=Status.WARNING,
                message="probe degraded",
                details="partial frame from the archiver",
            ),
        ],
        elapsed_ms=2.5,
        deadline_hit=False,
    )
    with patch("osprey.cli.health_cmd._run_suite", AsyncMock(return_value=report)):
        result = runner.invoke(health, ["--project", str(health_project), "--json"])

    assert result.exit_code == 1, result.output
    rows = json.loads(result.stdout)["results"]
    for row in rows:
        keys = set(row)
        assert _HEALTH_ROW_REQUIRED <= keys, f"{row['name']}: missing a required row key"
        assert keys <= _HEALTH_ROW_REQUIRED | _HEALTH_ROW_OPTIONAL, (
            f"{row['name']}: emits a row key outside the documented contract"
        )

    shapes = {row["name"]: set(row) for row in rows}
    assert shapes["bare"] == _HEALTH_ROW_REQUIRED
    assert shapes["measured"] == _HEALTH_ROW_REQUIRED | {"value", "latency_ms"}
    assert shapes["diagnosed"] == _HEALTH_ROW_REQUIRED | {"details"}


# --------------------------------------------------------------------------- #
# osprey query --json
# --------------------------------------------------------------------------- #


@pytest.fixture
def query_deployment(lifecycle_repo: Path) -> Path:
    """A repo with a render that declares a ``controls`` MCP server."""
    build = stub_build(lifecycle_repo, config="api:\n  providers: {}\n")
    (build / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"controls": {"command": "osprey-controls"}}})
    )
    return lifecycle_repo


def test_query_json_keyset(runner: CliRunner, query_deployment: Path) -> None:
    """``query --json`` emits the recorded ``_build_json_output`` shape."""
    result_msg = MagicMock()
    result_msg.is_error = False
    result_msg.subtype = "end_turn"
    workflow = SDKWorkflowResult(
        text_blocks=["Beam current is 42 mA."],
        tool_traces=[
            ToolTrace(
                name="mcp__controls__read_pv",
                input={"pv": "BEAM:CURRENT"},
                result="42 mA",
                is_error=False,
            )
        ],
        mcp_servers=[{"name": "controls", "status": "connected", "tools": []}],
    )
    workflow.result = result_msg

    with (
        patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=workflow)),
        patch(
            "osprey.cli.query_cmd.read_only_disallowed_tools",
            return_value=["mcp__controls__channel_write"],
        ),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value={"controls"}),
    ):
        result = runner.invoke(query, ["--repo", str(query_deployment), "--json", "beam current"])

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "query")


# --------------------------------------------------------------------------- #
# osprey ariel status --json / osprey ariel search --json
# --------------------------------------------------------------------------- #

# A config section that ``ARIELConfig.from_dict`` accepts without a project on
# disk; no connection is ever opened because the service boundary is stubbed.
_ARIEL_CONFIG = {"database": {"uri": "postgresql://localhost/test"}}


class _StubService:
    """Async-context-manager stub standing in for the ARIEL service."""

    def __init__(self, *, repository: Any = None, search_result: Any = None) -> None:
        self.repository = repository or MagicMock()
        self._search_result = search_result

    async def __aenter__(self) -> _StubService:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def health_check(self) -> tuple[bool, str]:
        return True, "OK"

    async def search(self, **kwargs: Any) -> Any:
        return self._search_result


@pytest.fixture
def ariel_service(monkeypatch: pytest.MonkeyPatch):
    """Route ``create_ariel_service`` to a caller-supplied stub service."""
    import osprey.services.ariel_search as ariel_pkg

    def _install(service: _StubService) -> None:
        async def _fake_create(config: Any) -> _StubService:
            return service

        monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)

    return _install


def test_ariel_status_json_keyset(
    runner: CliRunner, ariel_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ariel status --json`` emits the recorded ``get_status`` shape."""
    repository = MagicMock()
    repository.get_enhancement_stats = AsyncMock(return_value={"total_entries": 7})
    repository.get_embedding_tables = AsyncMock(
        return_value=[
            SimpleNamespace(
                table_name="text_embeddings_nomic",
                entry_count=7,
                dimension=768,
                is_active=True,
            )
        ]
    )
    ariel_service(_StubService(repository=repository))
    monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda *a, **kw: dict(_ARIEL_CONFIG))

    result = runner.invoke(ariel_group, ["status", "--json"])

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "ariel_status")


def test_ariel_search_json_keyset(
    runner: CliRunner, ariel_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ariel search --json`` emits the recorded ``run_search`` shape."""
    search_result = SimpleNamespace(
        answer="Two coupler trips overnight.",
        sources=["E1"],
        search_modes_used=["keyword"],
        reasoning="keyword match on 'coupler'",
        entries=[{"entry_id": "E1", "author": "op", "raw_text": "coupler trip", "_score": 0.9}],
    )
    ariel_service(_StubService(search_result=search_result))
    monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda *a, **kw: dict(_ARIEL_CONFIG))

    result = runner.invoke(ariel_group, ["search", "coupler", "--json"])

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "ariel_search")


# --------------------------------------------------------------------------- #
# Golden-file hygiene
# --------------------------------------------------------------------------- #


def test_goldens_are_exactly_the_five_json_verbs() -> None:
    """The golden directory covers every ``--json`` verb and nothing else."""
    assert {p.stem for p in _GOLDEN_DIR.glob("*.json")} == set(_NESTED)


@pytest.mark.parametrize("verb", sorted(_NESTED))
def test_golden_is_sorted_and_newline_terminated(verb: str) -> None:
    """Goldens stay stable-sorted so a formatter never rewrites them."""
    text = (_GOLDEN_DIR / f"{verb}.json").read_text()
    assert text.endswith("\n")
    paths = json.loads(text)
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
