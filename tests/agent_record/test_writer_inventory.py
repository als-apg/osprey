"""No production writer files anything outside the agent record's spool.

Six file kinds used to be written beside the record: the ``var/audit`` ledger,
the dispatch run-record files, a bridge's ``history.json``, the web terminal's
feedback store, ``hook_debug.jsonl`` and the API-call ``.txt`` files. Every one
of those facts is now an entry the writer puts in a spool segment, and nothing
else. This module pins that two ways:

* **Symbol check.** An AST scan of every production module (``src/osprey`` and
  ``packages/*/src``) finds no definition of, call to, or import of the retired
  writers and their path constants, and no string naming one of the retired
  files. Any hit at all fails.
* **Runtime check.** Every writer is driven through its production entry point
  inside a temporary project — once with the spool directory a launcher names,
  once with the writer's own ``<project>/var/spool/<identity>`` fallback — and
  every file that appears afterwards is a spool file. None is a ledger file, a
  run record, a ``history.json``, a feedback record, a ``hook_debug.jsonl`` or an
  API-call ``.txt``.
"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey.agent_record import contract
from osprey.agent_record import writer as record_writer

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The production trees the symbol check reads.
PRODUCTION_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "src" / "osprey",
    *sorted((REPO_ROOT / "packages").glob("*/src")),
)

HOOKS_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"

# ---------------------------------------------------------------------------
# What is retired
# ---------------------------------------------------------------------------

#: Names no production module may define, call, import or read, and what each was.
RETIRED_SYMBOLS: dict[str, str] = {
    "append_envelope": "the audit ledger write path",
    "LEDGER_SUFFIX": "the audit ledger file constants",
    "LEDGER_FILE_MODE": "the audit ledger file constants",
    "LEDGER_DIR_MODE": "the audit ledger file constants",
    "writer_context": "the audit ledger path resolver",
    "ledger_name": "the audit ledger path resolver",
    "ensure_audit_dir": "the audit ledger directory provisioning",
    "_container_audit_dir": "the audit ledger directory provisioning",
    "AUDIT_DIR_RELPATH": "the var/audit path constant",
    "_persist_run": "the dispatch run-record file writer",
    "_load_persisted_runs": "the dispatch run-record file reader",
    "load_run_record": "the dispatch run-record file reader",
    "clear_worker_history": "the dispatch run-record file pruner",
    "HistoryStore": "the bridge history.json store",
    "prune_store": "the feedback store pruner",
}

#: The retired web-terminal feedback store module, by its dotted name.
RETIRED_MODULES: dict[str, str] = {
    "osprey.interfaces.web_terminal.feedback_store": "the web-terminal feedback store",
}

#: Fragments no production string (docstrings aside) may hold, and what each names.
RETIRED_LITERALS: dict[str, str] = {
    "dispatch/*.json": "the dispatch run-record files",
    "history.json": "the bridge history file",
    "HISTORY_PATH": "the bridge history file setting",
    "fb-*.json": "the feedback store's record files",
    "ctx-*.json": "the feedback store's context files",
    "hook_debug": "the hook debug log file",
    "api_calls_dir": "the API-call .txt directory",
    "_latest.txt": "the API-call .txt files",
}


# ---------------------------------------------------------------------------
# Symbol check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One place a production module names a retired token."""

    path: str
    line: int
    token: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.token} ({self.what})"


def _production_modules() -> Iterator[Path]:
    for root in PRODUCTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            yield path


def _module_name(path: Path) -> str | None:
    """The dotted module name of *path*, for modules under ``src/osprey``."""
    src = REPO_ROOT / "src"
    try:
        rel = path.relative_to(src)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import(module_name: str | None, path: Path, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` names (relative imports resolved)."""
    if not node.level or module_name is None:
        return node.module or ""
    package = module_name.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    if node.level > 1:
        package = package[: len(package) - (node.level - 1)]
    return ".".join([*package, node.module] if node.module else package)


def _bare_strings(tree: ast.AST) -> set[int]:
    """``id()`` of every string constant that is a statement of its own (a docstring)."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _hits_in(path: Path) -> list[Hit]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_name = _module_name(path)
    docstrings = _bare_strings(tree)
    hits: list[Hit] = []

    def symbol(token: str, line: int) -> None:
        if token in RETIRED_SYMBOLS:
            hits.append(Hit(rel, line, token, RETIRED_SYMBOLS[token]))

    def module(name: str, line: int) -> None:
        for retired, what in RETIRED_MODULES.items():
            if name == retired or name.startswith(retired + "."):
                hits.append(Hit(rel, line, retired, what))

    if module_name in RETIRED_MODULES:
        hits.append(Hit(rel, 1, module_name, RETIRED_MODULES[module_name]))

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 1)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            symbol(node.name, line)
        elif isinstance(node, ast.Name):
            symbol(node.id, line)
        elif isinstance(node, ast.Attribute):
            symbol(node.attr, line)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module(alias.name, line)
        elif isinstance(node, ast.ImportFrom):
            source = _resolve_import(module_name, path, node)
            module(source, line)
            for alias in node.names:
                symbol(alias.name, line)
                module(f"{source}.{alias.name}", line)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            for fragment, what in RETIRED_LITERALS.items():
                if fragment in node.value:
                    hits.append(Hit(rel, line, fragment, what))
    return hits


@pytest.fixture(scope="module")
def production_hits() -> list[Hit]:
    hits: list[Hit] = []
    for path in _production_modules():
        hits.extend(_hits_in(path))
    return hits


def test_the_scan_reads_the_production_trees():
    modules = list(_production_modules())
    assert any(p.name == "writer.py" and p.parent.name == "agent_record" for p in modules)
    assert any("osprey_connectors" in p.parts for p in modules)
    assert not any("tests" in p.relative_to(REPO_ROOT).parts[2:] for p in modules)


def test_no_production_module_names_a_retired_writer(production_hits):
    """Every hit is a retired writer still in use."""
    assert not production_hits, "production code still names a retired writer:\n  " + "\n  ".join(
        str(hit) for hit in production_hits
    )


def test_the_web_terminal_feedback_store_module_is_gone():
    store = REPO_ROOT / "src" / "osprey" / "interfaces" / "web_terminal" / "feedback_store.py"
    assert not store.exists()


@pytest.mark.parametrize(
    ("source", "token"),
    [
        ("from osprey.audit.writer import append_envelope\n", "append_envelope"),
        ("from osprey.bridges.core import history\nhistory.HistoryStore('/x')\n", "HistoryStore"),
        ("def _persist_run(run_id):\n    pass\n", "_persist_run"),
        ("PATH = '.claude/hooks/hook_debug.jsonl'\n", "hook_debug"),
        ("name = f'{base}_latest.txt'\n", "_latest.txt"),
        ("GLOB = 'fb-*.json'\n", "fb-*.json"),
        (
            "import osprey.interfaces.web_terminal.feedback_store\n",
            "osprey.interfaces.web_terminal.feedback_store",
        ),
    ],
)
def test_the_scan_catches_a_retired_writer(tmp_path, monkeypatch, source, token):
    """The scan is not vacuous: each shape of a retired writer is a hit."""
    fake = tmp_path / "src" / "osprey" / "fake.py"
    fake.parent.mkdir(parents=True)
    fake.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    assert token in {hit.token for hit in _hits_in(fake)}


def test_a_docstring_naming_a_retired_file_is_not_a_hit(tmp_path, monkeypatch):
    fake = tmp_path / "src" / "osprey" / "fake.py"
    fake.parent.mkdir(parents=True)
    fake.write_text('"""Once wrote var/audit and history.json."""\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    assert _hits_in(fake) == []


# ---------------------------------------------------------------------------
# Runtime check
# ---------------------------------------------------------------------------

#: The hook subprocess starts from these variables only, so a developer's own
#: ``OSPREY_*`` or ``CLAUDE_*`` never leaks into what it writes.
_BASE_VARS = ("PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "LANG")

#: Files a retired writer would leave, as ``fnmatch`` patterns over the
#: tmp-relative path, and what each one is.
RETIRED_FILES: dict[str, str] = {
    "*var/audit/*": "an audit ledger file",
    "*/dispatch/*.json": "a dispatch run-record file",
    "*history.json": "a bridge history file",
    "*fb-*.json": "a feedback store record",
    "*ctx-*.json": "a feedback store context file",
    "*hook_debug.jsonl": "the hook debug log",
    "*.txt": "an API-call text file",
}

#: The kinds the writers below put in the spool, so a writer that silently
#: stopped writing cannot make the "nothing else appeared" check pass.
EXPECTED_KINDS: frozenset[str] = frozenset(
    {
        contract.KIND_AUDIT,
        contract.KIND_RUN,
        contract.KIND_BRIDGE_TURN,
        contract.KIND_FEEDBACK,
        contract.KIND_DEBUG,
        contract.KIND_API_CALL,
    }
)

SIDECAR_IDENTITY = "sidecar"


@dataclass
class Project:
    """A temporary project the writers resolve, and the tree they may write in."""

    root: Path
    tmp: Path
    zone: Any
    use_zone: bool
    patch: pytest.MonkeyPatch

    @property
    def agent_data(self) -> Path:
        return self.root / "var" / "agent_data"

    def write_as(self, identity: str) -> Path:
        """Write as *identity* from now on; the spool directory it lands in."""
        record_writer.seal_all()
        if self.use_zone:
            return self.zone.spool_for(identity)
        self.patch.delenv(contract.SPOOL_DIR_ENV, raising=False)
        return self.root / "var" / "spool" / identity

    def spool_env(self, identity: str) -> dict[str, str]:
        """The spool variable a launcher hands a subprocess writing as *identity*."""
        if self.use_zone:
            return {contract.SPOOL_DIR_ENV: str(self.zone.root / identity)}
        return {}


@pytest.fixture(params=["named-spool", "project-fallback"])
def project(request, tmp_path, monkeypatch, spool_zone) -> Iterator[Project]:
    """A project at ``<tmp>/project``; the spool named by a launcher, or not."""
    import osprey_connectors.workspace as workspace

    root = tmp_path / "project"
    root.mkdir()
    (root / "profile.yml").write_text("name: writer-inventory\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(workspace, "resolve_project_root", lambda config=None: root)
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(root))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(root))
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root / "var" / "agent_data"))
    monkeypatch.delenv(contract.RECORD_OFF_ENV, raising=False)
    record_writer.seal_all()
    project = Project(root, tmp_path, spool_zone, request.param == "named-spool", monkeypatch)
    yield project
    record_writer.seal_all()


def _files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()} if root.exists() else set()


def _spool_roots(project: Project) -> tuple[Path, ...]:
    return (project.zone.root, project.root / "var" / "spool")


def _spool_kinds(project: Project) -> set[str]:
    """The kind of every line in every segment under the spool roots."""
    record_writer.seal_all()
    kinds: set[str] = set()
    for root in _spool_roots(project):
        for path in _files(root):
            if contract.parse_segment_name(path.name) is None:
                continue
            for raw in path.read_bytes().splitlines():
                if raw.strip():
                    kinds.add(contract.parse_spool_line(raw).kind)
    return kinds


# --- the writers, each through its production entry point ---------------------


def _audit(project: Project) -> None:
    from osprey.audit import tool_call
    from osprey.audit import writer as audit_writer
    from osprey.audit.envelope import DECISION_ALLOWED, POSTURE_SOURCE_PROCESS

    project.write_as("alice")
    assert audit_writer.record(
        surface="mcp_python",
        posture="sandbox",
        posture_source=POSTURE_SOURCE_PROCESS,
        session=None,
        subject="mcp__python__execute",
        decision=DECISION_ALLOWED,
        reason="allowed",
    )
    assert audit_writer.append_record(
        tool_call.SURFACE_TOOL_CALL,
        {"ts": "2026-10-25T00:00:00Z", "actor": "alice", "subject": "mcp__x__y"},
    )


def _hook(project: Project) -> None:
    """A write the readonly posture refuses: one audit line and one debug line."""
    project.write_as("alice")
    env = {name: os.environ[name] for name in _BASE_VARS if name in os.environ}
    env.update(
        {
            "CLAUDE_PROJECT_DIR": str(project.root),
            "OSPREY_TERMINAL_USER": "alice",
            "OSPREY_AGENT_DATA_ROOT": str(project.agent_data),
            "OSPREY_EXECUTION_MODE": "readonly",
            "OSPREY_HOOK_DEBUG": "1",
            **project.spool_env("alice"),
        }
    )
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "osprey_writes_check.py")],
        input=json.dumps({"tool_name": "mcp__controls__channel_write", "tool_input": {}}),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project.root),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert '"deny"' in result.stdout


def _sidecar(project: Project, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from osprey.services.auth_sidecar.app import create_app
    from osprey.services.auth_sidecar.passwords import hash_password
    from osprey.services.auth_sidecar.revocation import STATE_DIR_ENV
    from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

    state = project.root / "var" / "state" / SIDECAR_IDENTITY
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(STATE_DIR_ENV, str(state))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR_IDENTITY)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    project.write_as(SIDECAR_IDENTITY)
    origin = "https://terminals.example.org"
    env = {
        "OSPREY_AUTH_METHOD": "password",
        "OSPREY_AUTH_SESSION_SECRET": "session-secret-value",
        "OSPREY_AUTH_SESSION_LIFETIME": "3600",
        "OSPREY_AUTH_USERS": "alice,bob",
        "OSPREY_AUTH_PW_HASH_ALICE": hash_password("alice-password"),
        "OSPREY_AUTH_EXTERNAL_ORIGIN": origin,
        "OSPREY_AUTH_TLS_ENABLED": "true",
    }
    with TestClient(create_app(env), base_url="https://testserver") as client:
        response = client.post(
            "/auth/login",
            data={"user": "alice", "next": "", "password": "alice-password"},
            headers={"Origin": origin},
            follow_redirects=False,
        )
    assert response.status_code == 303
    monkeypatch.delenv(AUDIT_IDENTITY_ENV)
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")


def _dispatch(project: Project, monkeypatch) -> None:
    from osprey.mcp_server.dispatch_worker import dispatch_api, sdk_runner
    from tests.agent_record.fakes import FakeRecordClient

    project.write_as(project.zone.dispatch)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])
    monkeypatch.setattr(dispatch_api, "describe_run_input_artifacts", lambda run_id: [])
    unavailable = FakeRecordClient(unavailable="record stopped")
    monkeypatch.setattr(dispatch_api, "record_client", lambda **kw: unavailable)

    async def _agent(**kw):
        return {
            "status": "completed",
            "text_output": "the answer",
            "tool_calls": [],
            "error": None,
            "duration_sec": 0.25,
            "cost_usd": 0.03,
            "num_turns": 2,
        }

    monkeypatch.setattr(sdk_runner, "run_dispatch", _agent)

    async def _run() -> None:
        request = dispatch_api.DispatchRequest(prompt="why?", allowed_tools=[], owner="alice")
        response = await dispatch_api.dispatch(request)
        task = dispatch_api._tasks.get(response.run_id)
        if task is not None:
            await task

    asyncio.run(_run())


def _bridge(project: Project) -> None:
    from osprey.bridges.core.history import ConversationHistory
    from tests.agent_record.fakes import FakeRecordClient

    project.write_as(project.zone.bridge)
    history = ConversationHistory(client=FakeRecordClient(unavailable="record stopped"))
    assert history.append("chat-1", "what is the lifetime?", run_id="run-1")
    assert history.append_failed("chat-1", "and the current?", run_id=None)


def _feedback(project: Project, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from osprey.interfaces.web_terminal.routes import feedback as module
    from osprey.interfaces.web_terminal.routes.feedback import router
    from tests.agent_record.fakes import FakeRecordClient

    project.write_as("alice")
    store = MagicMock()
    store.list_entries.return_value = []
    record = FakeRecordClient(unavailable="record stopped")
    monkeypatch.setattr(module, "record_client", lambda: record)
    monkeypatch.setattr("osprey.stores.artifact_store.ArtifactStore", lambda *a, **kw: store)
    app = FastAPI()
    app.include_router(router)
    app.state.project_cwd = str(project.root)
    app.state.workspace_dir = project.root / "_agent_data"
    app.state.terminal_user = "alice"
    app.state.app_name = "OSPREY Test Rig"
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    with TestClient(app) as client:
        response = client.post(
            "/api/feedback",
            json={
                "text": "the terminal froze",
                "channel": "local",
                "include_metadata": True,
                "include_context": True,
                "session_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                "scrollback": "$ osprey up",
            },
        )
    assert response.status_code == 200, response.text


def _api_call(project: Project, monkeypatch) -> None:
    from osprey.models import logging as model_logging

    project.write_as("alice")
    monkeypatch.setattr(
        model_logging,
        "get_config_value",
        MagicMock(return_value={"api_calls": {"save_all": True, "latest_only": True}}),
    )
    model_logging.log_api_call(
        message="test input message",
        result="test output",
        provider="anthropic",
        model_id="claude-3-sonnet",
        max_tokens=1024,
        temperature=0.7,
    )


def test_every_writer_leaves_only_spool_files(project, monkeypatch):
    before = _files(project.tmp)

    _audit(project)
    _hook(project)
    _sidecar(project, monkeypatch)
    _dispatch(project, monkeypatch)
    _bridge(project)
    _feedback(project, monkeypatch)
    _api_call(project, monkeypatch)
    record_writer.seal_all()

    assert _spool_kinds(project) >= EXPECTED_KINDS

    appeared = sorted(_files(project.tmp) - before)
    retired = [
        f"{path.relative_to(project.tmp)}: {what}"
        for path in appeared
        for pattern, what in RETIRED_FILES.items()
        if fnmatch.fnmatch(path.relative_to(project.tmp).as_posix(), pattern)
    ]
    assert not retired, "a retired writer filed:\n  " + "\n  ".join(retired)

    spool_roots = _spool_roots(project)
    elsewhere = [
        path.relative_to(project.tmp).as_posix()
        for path in appeared
        if not any(path.is_relative_to(root) for root in spool_roots)
    ]
    assert not elsewhere, "a writer filed outside the spool:\n  " + "\n  ".join(elsewhere)
