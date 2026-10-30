"""Thinking reaches the transcripts on every launch route, live.

``claude_code.thinking.display`` reaches Claude Code three ways: an interactive
session reads ``showThinkingSummaries`` from the rendered project settings, an
``osprey chat --print`` run passes ``--thinking-display`` on its argv, and an
Agent SDK run passes it through ``build_agent_options``. This module runs each
route once against the CLI version the deployment pins, with a prompt that
makes the model think and spawn one subagent, and reads the transcripts the
CLI wrote: the main transcript and at least one subagent transcript of every
``summarized`` session hold a thinking block with text. A control run of the
SDK route on a build that says ``omitted`` holds no thinking text at all, so
the setting moves the outcome both ways.

Every process this module starts, and this process itself, sees a
``CLAUDE_CONFIG_DIR`` of its own: the host's ``~/.claude`` is never read or
written. The first thinking line of each transcript is printed for the
owner's acceptance.

Reaches a model, so it carries no ``model_free`` marker and runs in the
``e2e-tests`` lane under ``full-ci``.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_runner import launcher
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION
from tests.e2e.profile_edits import set_pairs
from tests.e2e.provider import build_model, build_provider, e2e_provider

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.requires_e2e_provider,
]

CLI_PACKAGE = "@anthropic-ai/claude-code"
SESSION_TIMEOUT_S = 420
#: Two small problems that each need a few steps of reasoning, the second one
#: handed to a subagent, so both the main agent and the subagent think.
PROMPT = (
    "First, think it through: how many positive integers below 100 are divisible "
    "by 3 or by 5 but not by both? Then use the Agent tool (the Task tool) to spawn "
    "exactly one general-purpose subagent, and ask it to think carefully about how "
    "many primes lie strictly between 50 and 100 and to report that count. "
    "Do not use any other tool. Reply with both numbers."
)
#: Settings layers and tools the SDK legs run with: no project settings layer,
#: so the display can only come from the launch argument.
SDK_ALLOWED_TOOLS = ("Agent", "Task")
SDK_DISALLOWED_TOOLS = ("Bash", "Edit", "Write", "WebFetch", "WebSearch")
TERMINAL_ROWS, TERMINAL_COLUMNS = 50, 200
#: Tools through which a subagent hands its final report back to the main agent.
HANDBACK_TOOLS = ("SubagentHandback",)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _is_session_var(key: str) -> bool:
    """Whether *key* belongs to a Claude Code session the test may run inside."""
    return key == "CLAUDECODE" or key.startswith(("OTEL_", "CLAUDE_CODE_"))


def _child_env(config_dir: Path, cli_bin: Path) -> dict[str, str]:
    """A complete child environment: own config dir, pinned CLI first on PATH."""
    env = {k: v for k, v in os.environ.items() if not _is_session_var(k)}
    env[CLAUDE_CONFIG_DIR_ENV] = str(config_dir)
    env["PATH"] = os.pathsep.join([str(cli_bin), env.get("PATH", "")])
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


@pytest.fixture(scope="module")
def module_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The config dir the module-scoped setup (install, init, build) runs with."""
    return tmp_path_factory.mktemp("claude-config-setup")


@pytest.fixture(scope="module")
def pinned_cli(tmp_path_factory: pytest.TempPathFactory, module_config_dir: Path) -> Path:
    """The CLI at the deployment's pinned version, installed into a prefix of its own."""
    npm = shutil.which("npm")
    if npm is None:
        pytest.fail("npm is required to install the pinned Claude Code CLI")
    prefix = tmp_path_factory.mktemp("pinned_cli")
    bin_dir = prefix / "node_modules" / ".bin"
    env = _child_env(module_config_dir, bin_dir)
    install = subprocess.run(
        [
            npm,
            "install",
            "--prefix",
            str(prefix),
            "--no-save",
            "--no-audit",
            "--no-fund",
            f"{CLI_PACKAGE}@{_DEFAULT_CLAUDE_CLI_VERSION}",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        check=False,
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    cli = bin_dir / "claude"
    version = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, timeout=60, env=env, check=False
    )
    assert version.returncode == 0, version.stderr
    reported = version.stdout.split()[0]
    assert reported == _DEFAULT_CLAUDE_CLI_VERSION, (
        f"the installed CLI reports {reported!r}, the pin is {_DEFAULT_CLAUDE_CLI_VERSION!r}"
    )
    return cli


def _build(base: Path, name: str, display: str | None, config_dir: Path, cli: Path) -> Path:
    """A built hello-world repo pinned to the default CLI, with *display* when given."""
    osprey = _osprey_bin()
    repo = base / name
    config: dict[str, Any] = {
        "claude_code.cli_version": _DEFAULT_CLAUDE_CLI_VERSION,
        # No gateway runs here; a session should not export to one.
    }
    if display is not None:
        config[launcher.THINKING_DISPLAY_KEY] = display
    edits = {
        "provider": build_provider(e2e_provider()),
        "model": build_model(None),
        "config": config,
    }
    env = _child_env(config_dir, cli.parent)
    for args in (
        ["init", str(repo), "--preset", "hello-world", "--no-git", *set_pairs(edits)],
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
    ):
        result = subprocess.run(
            [str(osprey), *args],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"osprey {args[0]} failed:\n{result.stdout}\n{result.stderr}")
    cc = yaml.safe_load((repo / "build" / "config.yml").read_text())["claude_code"]
    assert cc["cli_version"] == _DEFAULT_CLAUDE_CLI_VERSION, cc.get("cli_version")
    settings = json.loads((repo / "build" / ".claude" / "settings.json").read_text())
    assert settings["showThinkingSummaries"] is (display in (None, "summarized")), settings
    return repo


@pytest.fixture(scope="module")
def summarized_project(
    tmp_path_factory: pytest.TempPathFactory, module_config_dir: Path, pinned_cli: Path
) -> Path:
    """A build that states no display, so it gets the default, ``summarized``."""
    return _build(
        tmp_path_factory.mktemp("thinking_summarized"),
        "osprey-e2e-thinking-summarized",
        None,
        module_config_dir,
        pinned_cli,
    )


@pytest.fixture(scope="module")
def omitted_project(
    tmp_path_factory: pytest.TempPathFactory, module_config_dir: Path, pinned_cli: Path
) -> Path:
    """A build that says ``claude_code.thinking.display: omitted``."""
    return _build(
        tmp_path_factory.mktemp("thinking_omitted"),
        "osprey-e2e-thinking-omitted",
        "omitted",
        module_config_dir,
        pinned_cli,
    )


@pytest.fixture
def isolated(tmp_path: Path, pinned_cli: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """This process and its children on a config dir of their own and the pinned CLI.

    Yields the config dir. The launcher's per-process CLI resolution is cleared
    after the patch and again at teardown, so it resolves against this PATH and
    no other test inherits the answer.
    """
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    for key in [k for k in os.environ if _is_session_var(k)]:
        monkeypatch.delenv(key)
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv("PATH", os.pathsep.join([str(pinned_cli.parent), os.environ["PATH"]]))
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "1")
    launcher._project_agent_cli.cache_clear()
    try:
        yield config_dir
    finally:
        launcher._project_agent_cli.cache_clear()


def _provider_env(repo: Path, config_dir: Path, cli: Path) -> dict[str, str]:
    """The CLI's environment with the build's provider routing, as a launcher writes it."""
    from osprey.agent_runner.provider_env import load_provider_spec

    spec = load_provider_spec(repo / "build", env_dir=repo)
    assert spec is not None, "the project names no provider"
    if spec.needs_proxy:
        pytest.fail(f"provider {spec.provider!r} needs the translation proxy; name a native one")
    env = {**_child_env(config_dir, cli.parent), **spec.env_block}
    if spec.auth_env_var != spec.auth_secret_env:
        secret = os.environ.get(spec.auth_secret_env)
        assert secret, f"{spec.auth_secret_env} is not set"
        env[spec.auth_env_var] = secret
    return env


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transcript:
    path: Path
    entries: list[dict[str, Any]]

    def thinking_texts(self) -> list[str]:
        """The text of every thinking block, in order; empty strings included."""
        texts: list[str] = []
        for entry in self.entries:
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "thinking",
                    "redacted_thinking",
                ):
                    texts.append(str(block.get("thinking") or ""))
        return texts

    def versions(self) -> set[str]:
        return {str(e["version"]) for e in self.entries if e.get("version")}

    def ran_a_subagent(self) -> bool:
        """Whether the main agent called the Agent (Task) tool and got its result back."""
        calls: set[str] = set()
        returned = False
        for entry in self.entries:
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") in ("Agent", "Task"):
                    calls.add(str(block.get("id")))
                if block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                    returned = True
        return returned

    def replied(self) -> bool:
        """Whether the agent of this transcript gave its answer.

        Either as reply text or, for a subagent, through the hand-back tool
        that delivers its final report.
        """
        for entry in self.entries:
            if entry.get("type") != "assistant":
                continue
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and str(block.get("text") or "").strip():
                    return True
                if block.get("type") == "tool_use" and block.get("name") in HANDBACK_TOOLS:
                    return True
        return False


def _read(path: Path) -> Transcript:
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return Transcript(path, entries)


def _main_transcripts(config_dir: Path) -> list[Path]:
    return sorted((config_dir / "projects").glob("*/*.jsonl"))


def _session_transcripts(config_dir: Path) -> tuple[Transcript, list[Transcript]]:
    """The one session's main transcript and its subagent transcripts."""
    mains = _main_transcripts(config_dir)
    assert len(mains) == 1, f"expected one session transcript under {config_dir}: {mains}"
    main = mains[0]
    subagents = sorted((main.parent / main.stem / "subagents").glob("*.jsonl"))
    return _read(main), [_read(p) for p in subagents]


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _assert_thinking_recorded(
    leg: str, config_dir: Path, capsys: pytest.CaptureFixture[str], context: str = ""
) -> None:
    """The main and at least one subagent transcript hold thinking text; print each first line."""
    main, subagents = _session_transcripts(config_dir)
    assert main.versions() == {_DEFAULT_CLAUDE_CLI_VERSION}, (
        f"{leg}: the session ran CLI {main.versions()}, the pin is {_DEFAULT_CLAUDE_CLI_VERSION}"
    )
    assert main.ran_a_subagent(), (
        f"{leg}: the model did not spawn a subagent; the transcript proves nothing. {context}"
    )
    assert subagents, f"{leg}: no subagent transcript under {main.path.parent / main.path.stem}"
    main_texts = [t for t in main.thinking_texts() if t.strip()]
    assert main_texts, (
        f"{leg}: the main transcript holds no thinking text "
        f"({len(main.thinking_texts())} thinking blocks, all empty). {context}"
    )
    thinking_subagents = [(s, [t for t in s.thinking_texts() if t.strip()]) for s in subagents]
    thinking_subagents = [(s, texts) for s, texts in thinking_subagents if texts]
    assert thinking_subagents, (
        f"{leg}: no subagent transcript holds thinking text "
        f"({[(s.path.name, len(s.thinking_texts())) for s in subagents]}). {context}"
    )
    subagent, sub_texts = thinking_subagents[0]
    with capsys.disabled():
        print(f"\n[thinking {leg}] main     : {_first_line(main_texts[0])}")
        print(f"[thinking {leg}] subagent : {_first_line(sub_texts[0])}  ({subagent.path.name})")


# --------------------------------------------------------------------------- #
# The SDK route
# --------------------------------------------------------------------------- #


def _run_sdk(repo: Path, pinned_cli: Path, expected_display: str) -> str:
    """One SDK session through ``build_agent_options``; returns what it streamed, for messages."""
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage

    from osprey.agent_runner.primitives import build_agent_options

    render = repo / "build"
    options = build_agent_options(
        render,
        disallowed_tools=list(SDK_DISALLOWED_TOOLS),
        allowed_tools=list(SDK_ALLOWED_TOOLS),
        setting_sources=[],
        max_turns=10,
        max_budget_usd=1.0,
    )
    assert Path(options.cli_path).resolve() == pinned_cli.resolve(), (
        f"the SDK run resolved {options.cli_path}, not the pinned CLI at {pinned_cli}"
    )
    assert options.extra_args == {"thinking-display": expected_display}, options.extra_args
    assert CLAUDE_CONFIG_DIR_ENV not in (options.env or {}) or (
        options.env[CLAUDE_CONFIG_DIR_ENV] == os.environ[CLAUDE_CONFIG_DIR_ENV]
    ), "the SDK run would use a different config dir"

    log: list[str] = []

    async def session() -> None:
        async with ClaudeSDKClient(options) as client:
            await client.query(PROMPT)
            async for message in client.receive_response():
                log.append(f"{type(message).__name__}: {str(message)[:300]}")
                if isinstance(message, ResultMessage):
                    assert not message.is_error, f"the SDK session failed: {message}"

    asyncio.run(asyncio.wait_for(session(), SESSION_TIMEOUT_S))
    return "\n".join(log[-20:])


def test_sdk_route_records_thinking_in_main_and_subagent_transcripts(
    summarized_project: Path,
    pinned_cli: Path,
    isolated: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    streamed = _run_sdk(summarized_project, pinned_cli, "summarized")
    _assert_thinking_recorded("sdk", isolated, capsys, streamed)


def test_sdk_route_with_display_omitted_records_no_thinking_text(
    omitted_project: Path,
    pinned_cli: Path,
    isolated: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    streamed = _run_sdk(omitted_project, pinned_cli, "omitted")
    main, subagents = _session_transcripts(isolated)
    assert main.versions() == {_DEFAULT_CLAUDE_CLI_VERSION}, main.versions()
    assert main.ran_a_subagent(), (
        f"control: the model did not spawn a subagent; the transcript proves nothing.\n{streamed}"
    )
    for transcript in (main, *subagents):
        texts = [t for t in transcript.thinking_texts() if t.strip()]
        assert not texts, (
            f"control: {transcript.path.name} holds thinking text although the display "
            f"is omitted: {_first_line(texts[0])!r}"
        )
    with capsys.disabled():
        blocks = [len(t.thinking_texts()) for t in (main, *subagents)]
        print(f"\n[thinking control] no thinking text; thinking blocks per transcript: {blocks}")


# --------------------------------------------------------------------------- #
# The --print route
# --------------------------------------------------------------------------- #


def test_chat_print_route_records_thinking_in_main_and_subagent_transcripts(
    summarized_project: Path,
    pinned_cli: Path,
    isolated: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _child_env(isolated, pinned_cli.parent)
    turn = subprocess.run(
        [str(_osprey_bin()), "chat", "--repo", str(summarized_project), "--print", PROMPT],
        cwd=str(summarized_project),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=SESSION_TIMEOUT_S,
        check=False,
    )
    output = f"stdout:\n{turn.stdout[-4000:]}\nstderr:\n{turn.stderr[-4000:]}"
    assert turn.returncode == 0, f"osprey chat --print failed:\n{output}"
    _assert_thinking_recorded("print", isolated, capsys, output)


# --------------------------------------------------------------------------- #
# The interactive route
# --------------------------------------------------------------------------- #


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")


class Terminal:
    """A real CLI on a pseudo-terminal, read continuously so it never blocks on output."""

    def __init__(self, argv: list[str], cwd: Path, env: Mapping[str, str]) -> None:
        self._master, slave = pty.openpty()
        fcntl.ioctl(
            slave, termios.TIOCSWINSZ, struct.pack("HHHH", TERMINAL_ROWS, TERMINAL_COLUMNS, 0, 0)
        )
        self._buffer = bytearray()
        self.process = subprocess.Popen(  # argv is built by the test, not by input
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)

    def pump(self, timeout: float) -> None:
        try:
            ready, _, _ = select.select([self._master], [], [], timeout)
        except OSError:
            return
        if not ready:
            return
        try:
            chunk = os.read(self._master, 65536)
        except OSError:
            return
        self._buffer.extend(chunk)

    def text(self) -> str:
        return _ANSI.sub("", self._buffer.decode("utf-8", "replace"))

    def close(self) -> None:
        """End the CLI and everything it started, and release the terminal."""
        with contextlib.suppress(OSError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        os.close(self._master)


def test_interactive_route_records_thinking_in_main_and_subagent_transcripts(
    summarized_project: Path,
    pinned_cli: Path,
    isolated: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from osprey.agent_runner.claude_state import seed_claude_state

    render = summarized_project / "build"
    env = _provider_env(summarized_project, isolated, pinned_cli)
    # Onboarding, workspace trust and the provider key, seeded through the
    # temp config dir only, so the session opens straight onto the prompt.
    seed_claude_state(render, env=env)
    assert (isolated / ".claude.json").is_file(), "the first-run state was not seeded"

    cc_config = launcher.project_claude_code_config(render)
    cli = launcher.agent_cli_path(cc_config)
    assert Path(cli).resolve() == pinned_cli.resolve(), f"resolved {cli}, not {pinned_cli}"
    # The launch prefix the interactive launcher builds, run by the pinned binary.
    prefix = launcher.build_claude_launch_argv(cc_config, no_pin=True)
    assert prefix[0] == launcher.CLI_NAME, prefix
    argv = [cli, *prefix[1:], PROMPT]
    assert not any(arg.startswith("--thinking-display") for arg in argv), argv

    terminal = Terminal(argv, render, env)
    deadline = time.monotonic() + SESSION_TIMEOUT_S
    done = False
    try:
        while time.monotonic() < deadline:
            terminal.pump(0.5)
            if terminal.process.poll() is not None:
                break
            # An interactive subagent may run in the background, so the main
            # agent can answer first: the session is complete once a subagent
            # has written its own reply, not once the main agent has.
            mains = _main_transcripts(isolated)
            if len(mains) == 1 and _read(mains[0]).ran_a_subagent():
                main = mains[0]
                subagents = (main.parent / main.stem / "subagents").glob("*.jsonl")
                if any(_read(path).replied() for path in subagents):
                    done = True
                    break
        screen = terminal.text()[-6000:]
    finally:
        terminal.close()
    assert done, (
        "the interactive session did not spawn a subagent that replied within "
        f"{SESSION_TIMEOUT_S}s (exit {terminal.process.returncode}).\n--- screen ---\n{screen}"
    )
    _assert_thinking_recorded("interactive", isolated, capsys, f"--- screen ---\n{screen}")
