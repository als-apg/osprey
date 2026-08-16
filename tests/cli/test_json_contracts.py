"""Standing structural contracts for every ``--json`` path the CLI exposes.

Five verbs have a ``--json`` flag: ``audit``, ``health``, ``query``,
``ariel status`` and ``ariel search``. Whatever the output hierarchy does to a
verb's *human* output, its *machine* output has one job -- a caller pipes stdout
into ``json.load`` and gets the document it expected. Three things have to hold
for that, and each is pinned here for all five paths:

1. **stdout is exactly one JSON document.** It parses from the first byte (no
   banner, no progress line ahead of it) and nothing follows it (no trailing
   prose, no second document). This is what a naive ``osprey ... --json | jq``
   depends on.
2. **The top-level key set is the pre-migration one.** The goldens in
   ``tests/cli/data/json_keysets/`` were captured before any verb moved onto the
   renderer; a bare entry in one of those arrays is a top-level key (entries
   holding a ``.`` pin nested shapes and belong to
   ``test_json_keyset_capture.py``, which owns the deep comparison). Those files
   are read-only here: a failure means the payload moved, not that the golden is
   stale.
3. **Human lines land on stderr.** Each verb's body is driven with a renderer
   call injected into it, from the same depth a real warning would come from,
   and the line has to come out on stderr with stdout still holding one clean
   document. That is the ``osprey.cli.output.machine_mode`` contract, tested
   through the verb rather than against the context manager directly. Two
   probes, because they can fail apart: ``output.report`` is redirected *only*
   by an open machine-mode block, so it is what proves the block is really
   around the body; ``output.warn`` prints to stderr under every mode, so it
   pins the trouble stream independently of how the verb opts into machine
   mode.

**Known limit -- error payloads are not golden-pinned.** ``ariel status`` and
``ariel search`` also emit failure documents (``{"status": ..., "message": ...}``
and ``{"error": ...}``), whose shapes have no golden and are deliberately not
pinned here: the capture suite recorded success payloads only, and inventing a
golden for the error forms now would pin a shape nobody captured before the
migration rather than protect one. The three contracts above are asserted for
the success path of each verb; a caller relying on the error shape of an ARIEL
verb has no test standing behind it.

All five verbs now hold contract 3 the same way, by opening a machine-mode block
over the whole verb body. ``audit`` was the last one to do so: it used to rely on
``if not json_output:`` guards, which cover the lines audit itself prints and
nothing else, so a renderer line from deeper in the stack landed on stdout ahead
of the document. That was not hypothetical -- under ``-v`` the reviewer's own
transcript is echoed from inside the agent loop, where no guard reaches, and it
broke the document outright. Both halves are pinned below: the shared contract-3
test for the guard-independent case, and
``test_audit_verbose_keeps_the_reviewer_transcript_off_the_document`` for the
``-v`` path specifically.

**Capture seams.** Every verb runs end-to-end through ``CliRunner`` with only
its outermost I/O boundary faked, mirroring ``test_json_keyset_capture.py`` so
that both files exercise the same production path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from osprey.agent_runner import SDKWorkflowResult, ToolTrace
from osprey.cli import output
from osprey.cli.ariel import ariel_group
from osprey.cli.audit_prompts import AuditFinding, AuditReport
from osprey.cli.health_cmd import health
from osprey.cli.query_cmd import query
from osprey.health.models import CheckReport, CheckResult, Status
from tests.cli._lifecycle_build import stub_build

if TYPE_CHECKING:
    from collections.abc import Callable

# --------------------------------------------------------------------------- #
# Contract helpers
# --------------------------------------------------------------------------- #

_GOLDEN_DIR = Path(__file__).parent / "data" / "json_keysets"

#: The line injected into a verb's body to prove where human output goes. No
#: spaces and no punctuation a console might re-wrap or re-style, so a failure
#: means the stream moved rather than that the text was reshaped.
_PROBE = "json-contract-probe-line"


def _sole_document(stdout: str) -> dict[str, Any]:
    """Parse *stdout* as one JSON object, failing if anything else is there.

    Args:
        stdout: The verb's stdout, verbatim.

    Returns:
        The parsed document.
    """
    assert stdout, "the verb wrote no document to stdout"
    assert stdout[0] == "{", f"stdout does not open with the document: {stdout[:120]!r}"
    payload, end = json.JSONDecoder().raw_decode(stdout)
    assert isinstance(payload, dict), f"the document is a {type(payload).__name__}, not an object"
    assert not stdout[end:].strip(), (
        f"stdout carries more than the document: {stdout[end:][:120]!r}"
    )
    return payload


def _golden_top_level(verb: str) -> list[str]:
    """Return the recorded top-level keys for *verb*.

    Args:
        verb: Golden name, i.e. the stem of a file in the golden directory.

    Returns:
        The sorted bare (dot-free) entries of the golden key-path array.
    """
    paths = json.loads((_GOLDEN_DIR / f"{verb}.json").read_text())
    return sorted(path for path in paths if "." not in path)


# --------------------------------------------------------------------------- #
# Per-verb invocation
# --------------------------------------------------------------------------- #
#
# Each invoker drives one verb's ``--json`` path and calls ``hook`` once from
# inside the verb body, at the point where the verb is waiting on its own
# outermost boundary -- the agent run, the check suite, the ARIEL service. That
# is the depth a warning from a shared helper would print from, which is the
# thing ``machine_mode`` exists to redirect.


def _audit_project(tmp_path: Path) -> Path:
    """Write the minimal directory ``audit`` classifies as a project."""
    (tmp_path / "config.yml").write_text("name: test\n")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre_write.sh").write_text("#!/bin/bash\n")
    return tmp_path


def _audit_report() -> AuditReport:
    """The report the faked reviewer returns."""
    return AuditReport(
        summary="Contract run",
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


def _invoke_audit(
    runner: CliRunner, tmp_path: Path, repo: Path, hook: Callable[[], None]
) -> Result:
    """Run ``audit --json`` over a minimal project with the agent faked."""
    project = _audit_project(tmp_path)
    report = _audit_report()

    def _run(coro: Any) -> tuple[str, float, int]:
        coro.close()
        hook()
        return report.model_dump_json(), 0.01, 5

    from osprey.cli.audit_cmd import audit

    with (
        patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True),
        patch("osprey.cli.audit_cmd.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.side_effect = _run
        return runner.invoke(audit, [str(project), "--json"])


def _invoke_health(
    runner: CliRunner, tmp_path: Path, repo: Path, hook: Callable[[], None]
) -> Result:
    """Run ``health --json`` with the check suite faked."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yml").write_text(
        "project_name: json_contracts\n"
        "models:\n"
        "  python_code_generator:\n"
        "    provider: mock\n"
        "    model_id: mock-model\n"
    )
    report = CheckReport(
        results=[CheckResult(name="contract", category="demo", status=Status.OK, message="ok")],
        elapsed_ms=1.5,
        deadline_hit=False,
    )

    def _suite(*args: Any, **kwargs: Any) -> CheckReport:
        hook()
        return report

    with patch("osprey.cli.health_cmd._run_suite", AsyncMock(side_effect=_suite)):
        return runner.invoke(health, ["--project", str(project), "--json"])


def _invoke_query(
    runner: CliRunner, tmp_path: Path, repo: Path, hook: Callable[[], None]
) -> Result:
    """Run ``query --json`` over a stub build with the agent faked."""
    build = stub_build(repo, config="api:\n  providers: {}\n")
    (build / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"controls": {"command": "osprey-controls"}}})
    )

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

    def _run_query(*args: Any, **kwargs: Any) -> SDKWorkflowResult:
        hook()
        return workflow

    with (
        patch("osprey.cli.query_cmd.run_query", new=AsyncMock(side_effect=_run_query)),
        patch(
            "osprey.cli.query_cmd.read_only_disallowed_tools",
            return_value=["mcp__controls__channel_write"],
        ),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value={"controls"}),
    ):
        return runner.invoke(query, ["--repo", str(repo), "--json", "beam current"])


# A config section ``ARIELConfig.from_dict`` accepts without a project on disk;
# no connection is ever opened because the service boundary is stubbed.
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


def _install_ariel_service(service: _StubService) -> Any:
    """Return a patch context routing ``create_ariel_service`` to *service*."""

    async def _create(config: Any) -> _StubService:
        return service

    return patch("osprey.services.ariel_search.create_ariel_service", new=_create)


def _invoke_ariel_status(
    runner: CliRunner, tmp_path: Path, repo: Path, hook: Callable[[], None]
) -> Result:
    """Run ``ariel status --json`` against a stub repository."""

    def _stats() -> dict[str, int]:
        hook()
        return {"total_entries": 7}

    repository = MagicMock()
    repository.get_enhancement_stats = AsyncMock(side_effect=_stats)
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

    with (
        _install_ariel_service(_StubService(repository=repository)),
        patch("osprey.cli.ariel.get_config_value", new=lambda *a, **kw: dict(_ARIEL_CONFIG)),
    ):
        return runner.invoke(ariel_group, ["status", "--json"])


def _invoke_ariel_search(
    runner: CliRunner, tmp_path: Path, repo: Path, hook: Callable[[], None]
) -> Result:
    """Run ``ariel search --json`` against a stub service."""
    search_result = SimpleNamespace(
        answer="Two coupler trips overnight.",
        sources=["E1"],
        search_modes_used=["keyword"],
        reasoning="keyword match on 'coupler'",
        entries=[{"entry_id": "E1", "author": "op", "raw_text": "coupler trip", "_score": 0.9}],
    )

    class _HookingService(_StubService):
        async def search(self, **kwargs: Any) -> Any:
            hook()
            return await super().search(**kwargs)

    with (
        _install_ariel_service(_HookingService(search_result=search_result)),
        patch("osprey.cli.ariel.get_config_value", new=lambda *a, **kw: dict(_ARIEL_CONFIG)),
    ):
        return runner.invoke(ariel_group, ["search", "coupler", "--json"])


#: Golden name -> the invoker that drives that verb's ``--json`` path.
_INVOKERS: dict[str, Any] = {
    "ariel_search": _invoke_ariel_search,
    "ariel_status": _invoke_ariel_status,
    "audit": _invoke_audit,
    "health": _invoke_health,
    "query": _invoke_query,
}


@pytest.fixture
def json_run(runner: CliRunner, tmp_path: Path, lifecycle_repo: Path):
    """Return ``run(verb, hook=None)``, driving one verb's ``--json`` path."""

    def _run(verb: str, hook: Callable[[], None] | None = None) -> Result:
        return _INVOKERS[verb](runner, tmp_path, lifecycle_repo, hook or (lambda: None))

    return _run


@pytest.fixture
def runner() -> CliRunner:
    """A Click runner that keeps stdout and stderr apart."""
    return CliRunner()


# --------------------------------------------------------------------------- #
# The contracts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("verb", sorted(_INVOKERS))
def test_stdout_carries_exactly_one_json_document(verb: str, json_run) -> None:
    """stdout opens with the document, and nothing follows it."""
    result = json_run(verb)

    assert result.exit_code == 0, result.output
    _sole_document(result.stdout)


@pytest.mark.parametrize("verb", sorted(_INVOKERS))
def test_top_level_keys_are_the_pre_migration_golden(verb: str, json_run) -> None:
    """The document still has exactly the top-level keys captured before."""
    payload = _sole_document(json_run(verb).stdout)

    assert sorted(payload) == _golden_top_level(verb)


@pytest.mark.parametrize("verb", sorted(_INVOKERS))
def test_human_lines_from_inside_the_verb_land_on_stderr(verb: str, json_run) -> None:
    """A renderer line printed mid-run reaches stderr, not the document."""
    result = json_run(verb, hook=lambda: output.report(_PROBE))

    assert _PROBE in result.stderr, "the human line did not reach stderr"
    assert _PROBE not in result.stdout, "the human line landed in the document's stream"
    _sole_document(result.stdout)


@pytest.mark.parametrize("verb", sorted(_INVOKERS))
def test_trouble_from_inside_the_verb_lands_on_stderr(verb: str, json_run) -> None:
    """A warning printed mid-run reaches stderr, not the document."""
    result = json_run(verb, hook=lambda: output.warn(_PROBE))

    assert _PROBE in result.stderr, "the warning did not reach stderr"
    assert _PROBE not in result.stdout, "the warning landed in the document's stream"
    _sole_document(result.stdout)


def test_audit_verbose_keeps_the_reviewer_transcript_off_the_document(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``audit --json -v`` prints the transcript to stderr, document intact.

    The one path that used to break the document rather than merely risk it:
    ``-v`` makes ``_run_audit`` echo every block the reviewer says, from inside
    the agent loop where no ``if not json_output:`` guard reaches. The SDK
    boundary is faked at ``query`` -- one assistant turn, then a result -- so
    the real ``_run_audit`` does the echoing.
    """
    # Imported here, not at module scope: the SDK is optional to ``audit_cmd``,
    # and a missing one should cost this test alone rather than the whole file.
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    project = _audit_project(tmp_path)
    transcript = "reading the permission block"
    answer = f"{transcript}\n{_audit_report().model_dump_json()}"

    async def _query(**kwargs: Any):
        yield AssistantMessage(content=[TextBlock(text=answer)], model="stub-model")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="stub-session",
            total_cost_usd=0.01,
        )

    from osprey.cli.audit_cmd import audit

    with (
        patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True),
        patch("osprey.cli.audit_cmd.query", new=_query),
    ):
        result = runner.invoke(audit, [str(project), "--json", "-v"])

    assert result.exit_code == 0, result.output
    assert transcript in result.stderr, "the reviewer transcript did not reach stderr"
    assert transcript not in result.stdout, "the transcript landed in the document's stream"
    assert sorted(_sole_document(result.stdout)) == _golden_top_level("audit")


def test_every_json_verb_has_a_golden_and_an_invoker() -> None:
    """The contracts cover every ``--json`` verb the goldens record."""
    assert {path.stem for path in _GOLDEN_DIR.glob("*.json")} == set(_INVOKERS)
