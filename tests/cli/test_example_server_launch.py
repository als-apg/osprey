"""Packaging checks for the hello-world preset's seeded example MCP server.

Four checks live here. The git-visibility one asserts the three packaged files
are not swallowed by any ignore rule, or the seed would silently fail to reach
a deployment repository. The drift guard asserts that the preset's live
``mcp_servers:`` block still teaches what the emitter's commented appendix
teaches: hello-world is the one preset whose block is active, so its readers
never see that appendix, and its comments are the only copy of that lesson they
get. The launch tests -- named to contain "spawns" -- materialize and build a
fresh hello-world repo, then actually spawn the rendered ``example_server``
entry over stdio and confirm ``tools/list`` advertises ``example_status`` and
that the build wires the ``mcp__example_server__.*`` guidance hook: the only
way to know the seeded server is not just present on disk but actually
launchable through the entry OSPREY itself renders.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_emit import _MCP_SERVERS_APPENDIX
from osprey.cli.init_cmd import init
from tests.integration._mcp_handshake import _STARTUP_TIMING, list_mcp_tools

#: How long the spawned entry gets to say it is serving. It is a bound on a
#: failure, not a wait: the read returns the moment the line arrives. The
#: stream is drained on its own thread because a server that announces nothing
#: never closes stderr either -- a blocking read would hang the test instead of
#: failing it.
_ANNOUNCE_TIMEOUT = 60.0
_ANNOUNCE_POLL = 0.1

PACKAGE_DIR = Path("src/osprey/templates/apps/hello_world/mcp_servers/example_server")
PACKAGED_FILES = ("__init__.py", "server.py", "__main__.py")

# The only comment lines the hello-world block may carry that the appendix does
# not: two facts that are true of the seeded example and of nothing else. Both
# are pinned in both directions below -- an appendix line may not impersonate
# one, and dropping one from the preset fails here rather than quietly costing
# a reader the warning.
_HELLO_WORLD_ONLY_COMMENTS: tuple[str, ...] = (
    "# Delete this block and mcp_servers/example_server/ together: one without",
    "# the other costs every session a 20 s wait for a server that cannot start.",
    "# Editing either is a source change, not a build tweak: rebuild afterwards,",
    "# since the directory enters the image context and the drift fingerprint.",
)


def _repo_root() -> Path:
    """Return the git checkout root, skipping the test when there is none."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return Path(completed.stdout.strip())


def test_packaged_seed_is_not_git_ignored() -> None:
    """Every file of the seeded example server is visible to git."""
    root = _repo_root()
    for filename in PACKAGED_FILES:
        path = root / PACKAGE_DIR / filename
        assert path.is_file(), f"packaged seed file is missing: {path}"
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=root,
            check=False,
        )
        assert result.returncode == 1, (
            f"{path} is git-ignored (git check-ignore exit {result.returncode}); "
            "the seeded example server would never reach a deployment repo"
        )


def _hello_world_preset_text() -> str:
    """Read the packaged hello-world preset, the file the emitter copies from."""
    presets = importlib.resources.files("osprey.profiles.presets")
    return presets.joinpath("hello-world.yml").read_text(encoding="utf-8")


def _mcp_servers_comment_lines() -> list[str]:
    """Return the stripped comment lines inside the preset's mcp_servers block.

    The block runs from the top-level ``mcp_servers:`` key to the next line
    that starts in column 0 -- the next key or section divider. Comments above
    the key introduce the section rather than living in it, so they are out of
    scope; everything indented under it is the block.
    """
    lines = _hello_world_preset_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("mcp_servers:")]
    assert len(starts) == 1, "hello-world must carry exactly one live mcp_servers block"

    comments: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line and not line[0].isspace():
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped)
    return comments


def test_block_comments_are_appendix_lines_or_pinned_extras() -> None:
    """Nothing in the block is invented prose.

    Every other preset gets the ``mcp_servers`` lesson from the emitter's
    commented appendix. hello-world spends that slot on a working entry, so the
    lesson has to be re-taught inside the block -- and re-teaching it in fresh
    words is how the two drift apart, one gaining a caveat the other never
    hears about. Requiring each line to be an appendix line verbatim makes the
    appendix the single source, and makes an appendix edit fail here until the
    preset is brought along.
    """
    appendix = {line.strip() for line in _MCP_SERVERS_APPENDIX.splitlines()}
    allowed = appendix | set(_HELLO_WORLD_ONLY_COMMENTS)

    unknown = [line for line in _mcp_servers_comment_lines() if line not in allowed]
    assert not unknown, (
        "hello-world's mcp_servers block carries comment lines that are neither a "
        f"line of _MCP_SERVERS_APPENDIX nor a pinned hello-world-only fact: {unknown}"
    )


def test_hello_world_only_comments_are_all_present() -> None:
    """The two hello-world-only facts stay in the block.

    They are the half of the block the appendix cannot supply: the seeded
    package and the profile entry are a pair, and editing either is a source
    change. Losing them costs a reader a 20 s startup wait or a stale build
    with no error to explain it, which is exactly the kind of loss a comment
    edit makes silently.
    """
    present = set(_mcp_servers_comment_lines())

    missing = [line for line in _HELLO_WORLD_ONLY_COMMENTS if line not in present]
    assert not missing, (
        f"hello-world's mcp_servers block no longer carries these pinned lines: {missing}"
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_rendered_entry_spawns_the_seeded_server(runner: CliRunner, tmp_path: Path) -> None:
    """The rendered ``example_server`` entry is not just present -- it launches.

    Every other test in this module reasons about text: files on disk, comment
    lines in a preset. This one is the proof that the text is true -- that
    ``osprey init`` then ``osprey build`` produce an ``.mcp.json`` entry a real
    client can spawn over stdio, that it answers the MCP handshake, and that
    the build wires the ``mcp__example_server__.*`` guidance hook the registry
    attaches whenever a server declares ``permissions``. A drift in the seeded
    package, the emitted command, or the ``PYTHONPATH`` would leave every
    other check in this module green while every session that reaches this
    server waits 20 s for a launch that was never going to succeed.
    """
    target = tmp_path / "my-facility"
    init_result = runner.invoke(init, [str(target), "--preset", "hello-world", "--no-git"])
    assert init_result.exit_code == 0, init_result.output

    build_result = runner.invoke(build, ["--repo", str(target), "--skip-deps", "--skip-lifecycle"])
    assert build_result.exit_code == 0, build_result.output

    server_py = target / "build" / "_mcp_servers" / "example_server" / "server.py"
    assert server_py.is_file(), f"build did not copy the seeded server to {server_py}"

    mcp_json = target / "build" / ".mcp.json"
    servers = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"]
    entry = servers["example_server"]

    command = Path(entry["command"])
    assert command.is_absolute(), f"rendered command is not absolute: {entry['command']}"

    pythonpath = Path(entry["env"]["PYTHONPATH"])
    assert pythonpath.is_dir(), f"rendered PYTHONPATH does not exist: {pythonpath}"

    tools = list_mcp_tools(entry["command"], list(entry["args"]), entry.get("env"), timeout=30.0)
    assert "example_status" in tools, (
        f"example_server did not advertise example_status; advertised: {tools}"
    )

    settings_json = target / "build" / ".claude" / "settings.json"
    settings = json.loads(settings_json.read_text(encoding="utf-8"))
    post_rules = settings["hooks"]["PostToolUse"]
    matchers = [rule["matcher"] for rule in post_rules]
    assert "mcp__example_server__.*" in matchers, (
        "build did not wire the example_server guidance hook into PostToolUse; "
        f"matchers present: {matchers}"
    )


def test_the_rendered_entry_announces_when_it_is_ready(runner: CliRunner, tmp_path: Path) -> None:
    """The seeded server says on stderr that it is serving, as the framework's do.

    A launcher cannot otherwise tell a server that is still importing from one
    that is stuck: both are silence on stdout. Every framework server announces
    through ``run_mcp_server``, and the worked example a facility author copies
    has to teach the same contract -- so the marker is asserted here against the
    very pattern the boot smoke's handshake reads.
    """
    target = tmp_path / "my-facility"
    init_result = runner.invoke(init, [str(target), "--preset", "hello-world", "--no-git"])
    assert init_result.exit_code == 0, init_result.output

    build_result = runner.invoke(build, ["--repo", str(target), "--skip-deps", "--skip-lifecycle"])
    assert build_result.exit_code == 0, build_result.output

    entry = json.loads((target / "build" / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "example_server"
    ]
    env = dict(os.environ)
    env.update(entry.get("env") or {})

    proc = subprocess.Popen(  # noqa: S603 - command comes from the generated .mcp.json
        [entry["command"], *entry["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )
    phases: list[str] = []
    lines: Queue[str] = Queue()

    def drain() -> None:
        for line in iter(proc.stderr.readline, ""):
            lines.put(line)

    threading.Thread(target=drain, daemon=True).start()
    try:
        deadline = time.monotonic() + _ANNOUNCE_TIMEOUT
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=_ANNOUNCE_POLL)
            except Empty:
                continue
            match = _STARTUP_TIMING.search(line)
            if match:
                phases.append(match.group(1))
                break
    finally:
        proc.terminate()
        proc.wait(timeout=_ANNOUNCE_TIMEOUT)

    assert "total_startup" in phases, (
        f"the seeded example server never announced total_startup on stderr; phases seen: {phases}"
    )
