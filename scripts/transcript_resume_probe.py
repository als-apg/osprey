#!/usr/bin/env python3
"""Prove the pinned Claude Code build resumes a transcript OSPREY authored.

OSPREY lays down past agent conversations (the example facility's
``demo-history`` scenario) as CLI transcripts written by
``osprey.agent_runner.transcript_author``, in the format of the build it pins
(``_DEFAULT_CLAUDE_CLI_VERSION``). This probe checks, without a model, that the
build still reads such a file as a conversation it recorded.

It authors one ``demo-history`` session part into a fresh config directory,
starts a localhost stub as ``ANTHROPIC_BASE_URL`` that records the first
``/v1/messages`` request offering tools and answers every request ``500``, and
runs the build with ``-p --resume <session> <prompt>``. It passes when that
request's ``messages`` hold the authored prompts and the new one, in order and
count, and every authored ``tool_use`` id is sent once with a ``tool_result``
carrying it. It fails when no request arrives, when the build reports no
conversation for the id, or when the request body is not JSON.

``--cli`` probes a given binary; by default the pinned build is installed from
npm into a temporary prefix, as ``cli_tool_inventory.py`` does.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = REPO_ROOT / "src/osprey/templates/facilities/example/scenarios/demo-history.yaml"
#: The session authored: three turns, two reads and an approved write, so the
#: transcript carries the approval hook's attachment line too.
SESSION_SLUG = "sector-3-corrector-nudge"
#: Any fixed aware anchor; the probe checks the format, not the instants.
ANCHOR = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
SESSION_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "osprey/transcript-resume-probe"))
PROMPT = "Resume probe: what did we just do?"
RUN_TIMEOUT_S = 60
NO_CONVERSATION = "no conversation found"

REMEDY = (
    "make osprey.agent_runner.transcript_author write the pinned build's transcript "
    "format (see scripts/README.md, transcript_resume_probe.py)"
)


class ProbeError(RuntimeError):
    """The pinned build could not be installed or started."""


# ---------------------------------------------------------------------------
# Verdict (pure; unit-tested with recorded bodies and outputs)
# ---------------------------------------------------------------------------


def authored_expectations(lines: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return the prompts and the ``tool_use`` ids of authored transcript lines, in order."""
    prompts: list[str] = []
    tool_use_ids: list[str] = []
    for line in lines:
        message = line.get("message") or {}
        content = message.get("content")
        if line.get("type") == "user" and isinstance(content, str):
            prompts.append(content)
        elif line.get("type") == "assistant" and isinstance(content, list):
            tool_use_ids.extend(
                b["id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
            )
    return prompts, tool_use_ids


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def verdict(
    raw_body: bytes | None,
    output: str,
    prompts: Sequence[str],
    tool_use_ids: Sequence[str],
) -> list[str]:
    """Return why the resumed request does not carry the authored conversation.

    Args:
        raw_body: The first ``/v1/messages`` request body the stub recorded, or
            ``None`` when none arrived.
        output: The build's stdout and stderr, for its own refusal.
        prompts: The prompts the request must hold, in order: the authored ones,
            then the new one.
        tool_use_ids: The authored ``tool_use`` ids, in order.

    Returns:
        One reason per failure; empty when the request is the authored conversation.
    """
    if NO_CONVERSATION in output.lower():
        return [f"the build found no conversation to resume: {output.strip()[:300]!r}"]
    if raw_body is None:
        detail = f"; it printed {output.strip()[:300]!r}" if output.strip() else ""
        return [f"the build sent no /v1/messages request{detail}"]
    try:
        body = json.loads(raw_body)
    except ValueError as exc:
        return [f"the /v1/messages request body is not JSON: {exc}"]
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return ["the /v1/messages request carries no messages list"]

    failures: list[str] = []
    # The build may merge consecutive same-role messages and add context blocks of
    # its own, so prompts are compared as user text blocks, not by message index.
    wanted = set(prompts)
    sent_prompts = [
        b["text"]
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        for b in _blocks(m)
        if b.get("type") == "text" and b.get("text") in wanted
    ]
    if sent_prompts != list(prompts):
        failures.append(f"prompts sent {sent_prompts!r}, expected {list(prompts)!r}")

    uses: list[str] = []
    results: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        for b in _blocks(m):
            if m.get("role") == "assistant" and b.get("type") == "tool_use":
                uses.append(b.get("id"))
            elif m.get("role") == "user" and b.get("type") == "tool_result":
                results.append(b.get("tool_use_id"))
    if uses != list(tool_use_ids):
        failures.append(f"tool_use ids sent {uses!r}, expected {list(tool_use_ids)!r}")
    unanswered = [i for i in tool_use_ids if results.count(i) != 1]
    if unanswered:
        failures.append(f"tool_use ids without exactly one tool_result: {unanswered!r}")
    orphans = sorted({str(i) for i in results} - set(tool_use_ids))
    if orphans:
        failures.append(f"tool_result ids no authored tool_use carries: {orphans!r}")
    return failures


# ---------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------


def author(config_dir: Path, project: Path, cli_version: str) -> list[dict[str, Any]]:
    """Write the probe's transcript where the build looks for ``project``'s sessions."""
    import yaml

    from osprey.agent_runner.project_paths import encode_claude_project_path
    from osprey.agent_runner.transcript_author import author_transcript
    from osprey.simulation.sessions import parse_sessions, resolve

    raw = yaml.safe_load(SCENARIO.read_text(encoding="utf-8")) or {}
    sessions = resolve(parse_sessions(raw.get("sessions"), f"{SCENARIO.name}: sessions"), ANCHOR)
    by_slug = {s.session.slug: s for s in sessions}
    if SESSION_SLUG not in by_slug:
        raise ValueError(f"{SCENARIO}: no session {SESSION_SLUG!r}")
    cwd = str(project.resolve())
    lines = author_transcript(
        by_slug[SESSION_SLUG].parts[0], session_id=SESSION_ID, cwd=cwd, cli_version=cli_version
    )
    target = config_dir / "projects" / encode_claude_project_path(cwd) / f"{SESSION_ID}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return lines


# ---------------------------------------------------------------------------
# Stub and run
# ---------------------------------------------------------------------------


class _CapturingStub(BaseHTTPRequestHandler):
    """Records the first agent-turn ``/v1/messages`` body; answers everything 500."""

    protocol_version = "HTTP/1.1"
    captured: list[bytes] = []
    arrived = threading.Event()

    def log_message(self, *_args: Any) -> None:
        pass

    def _answer(self, status: int) -> None:
        payload = json.dumps(
            {"type": "error", "error": {"type": "api_error", "message": "resume probe"}}
        ).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._answer(404)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        if self.path.split("?")[0].endswith("/v1/messages") and not self.captured:
            try:
                side = not json.loads(raw).get("tools")
            except (ValueError, AttributeError):
                side = False  # an unparseable body is the verdict's to report
            # A request without tools is a side request (a title, a summary).
            if not side:
                type(self).captured.append(raw)
                type(self).arrived.set()
        self._answer(500)


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # the killed build drops its connections mid-request


def run(cli: Path, home: Path, project: Path) -> tuple[bytes | None, str]:
    """Resume the probe's session under ``cli``; return the recorded body and the output."""
    _CapturingStub.captured = []
    _CapturingStub.arrived = threading.Event()
    server = _QuietServer(("127.0.0.1", 0), _CapturingStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "ANTHROPIC_API_KEY": "resume-probe",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        argv = [
            str(cli),
            "-p",
            "--resume",
            SESSION_ID,
            "--setting-sources",
            "project",
            "--max-turns",
            "1",
            PROMPT,
        ]
        with tempfile.TemporaryFile() as out:
            with subprocess.Popen(
                argv, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out
            ) as proc:
                # The build retries the 500 with backoff, so it is stopped as soon
                # as the request is in; the deadline bounds a build that never asks.
                deadline = time.monotonic() + RUN_TIMEOUT_S
                while proc.poll() is None and time.monotonic() < deadline:
                    if _CapturingStub.arrived.wait(0.2):
                        break
                proc.kill()
            out.seek(0)
            output = out.read().decode("utf-8", "replace")
        captured = _CapturingStub.captured[0] if _CapturingStub.captured else None
        return captured, output
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


def _npm_build(version: str, prefix: Path) -> Path:
    """Install the pinned build the way ``cli_tool_inventory.py`` does."""
    spec = importlib.util.spec_from_file_location(
        "cli_tool_inventory", Path(__file__).with_name("cli_tool_inventory.py")
    )
    assert spec is not None and spec.loader is not None
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    try:
        return inventory._npm_build(version, prefix)
    except inventory.InventoryError as exc:
        raise ProbeError(str(exc)) from None


def probe(cli: Path, cli_version: str) -> int:
    """Author, resume and judge; print ``ok`` or each failure and return the exit code."""
    with tempfile.TemporaryDirectory(prefix="resume-probe-") as tmp:
        home = Path(tmp) / "home"
        project = Path(tmp) / "project"
        project.mkdir(parents=True)
        lines = author(home / ".claude", project, cli_version)
        prompts, tool_use_ids = authored_expectations(lines)
        raw, output = run(cli, home, project)
    failures = verdict(raw, output, [*prompts, PROMPT], tool_use_ids)
    for failure in failures:
        print(f"FAIL {cli_version}: {failure}; remedy: {REMEDY}")
    if not failures:
        print(
            f"ok: {cli_version} resumed {SESSION_SLUG} "
            f"({len(prompts)} prompts, {len(tool_use_ids)} tool calls)"
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cli", type=Path, help="probe this binary instead of installing the pinned build"
    )
    args = parser.parse_args(argv)
    from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

    version = _DEFAULT_CLAUDE_CLI_VERSION
    with tempfile.TemporaryDirectory(prefix="resume-probe-npm-") as npm_prefix:
        try:
            cli = args.cli or _npm_build(version, Path(npm_prefix))
            return probe(cli, version)
        except (ProbeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
