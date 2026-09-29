#!/usr/bin/env python3
"""Built-in tool inventory of each Claude Code CLI build OSPREY pins.

The inventory of a build is the list of built-in tools it names in its
``system/init`` stream-json message when started under a clean, offline,
deny-free environment: a fresh ``HOME`` and config directory, a dummy API key,
a base URL nothing listens on, no settings but the (empty) project's, and
``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1``. MCP names are dropped: the
probe loads no MCP server.

That list is the *offline core*, recorded as ``tools``. With remote
configuration reachable a build also lists flag-gated tools (``Monitor``
among them), so each build is probed a second time with nonessential traffic
allowed, and that list is recorded as ``remote_config_tools``. An inventory
that is too small only ever rejects a real tool name, loudly; it can never
accept a name the build lacks, because every recorded name was listed by the
build under the probe that recorded it.

OSPREY pins two builds: the npm build (``_DEFAULT_CLAUDE_CLI_VERSION``),
which the containers install, and the build bundled inside the Agent SDK,
which every SDK path runs. Each is recorded as
``tests/fixtures/cli_tool_inventory/<version>.json``.

Modes:
    --write   probe each pinned build, write its file, delete every other file
    --check   probe each pinned build and fail when a pinned version has no
              file, a file names the wrong version, or a recorded name is no
              longer listed by the build under the same probe
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DIR = REPO_ROOT / "tests" / "fixtures" / "cli_tool_inventory"

PROBE_ENV = {
    "ANTHROPIC_API_KEY": "inventory-probe",
    # Nothing listens on the discard port, so no request ever leaves the host.
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
#: The second probe reaches remote configuration, so flag-gated tools are listed.
REMOTE_CONFIG_PROBE_ENV = {
    k: v for k, v in PROBE_ENV.items() if k != "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"
}
#: Inventory key -> the probe environment that records it.
PROBES = {"tools": PROBE_ENV, "remote_config_tools": REMOTE_CONFIG_PROBE_ENV}
INIT_TIMEOUT_S = 60

REMEDY = "uv run python scripts/cli_tool_inventory.py --write"
_NPM_PACKAGE = "@anthropic-ai/claude-code"


class InventoryError(RuntimeError):
    """A pinned build could not be located, installed or probed."""


def init_tools(lines: Iterable[str]) -> list[str]:
    """Return the sorted non-MCP tools named by the first ``system/init`` line.

    Lines that are not JSON objects are skipped.

    Raises:
        ValueError: No init message is present.
    """
    for line in lines:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        if message.get("type") == "system" and message.get("subtype") == "init":
            return sorted(t for t in message.get("tools", []) if not t.startswith("mcp__"))
    raise ValueError("no system/init message in the CLI output")


def render_inventory(version: str, lists: dict[str, list[str]]) -> str:
    """Return the file text recording each probe's tool list for build ``version``."""
    return json.dumps({"cli_version": version, **lists}, indent=2) + "\n"


def stale_names(recorded: Iterable[str], live: Iterable[str]) -> list[str]:
    """Return the recorded names the live build no longer lists, sorted."""
    return sorted(set(recorded) - set(live))


def probe(cli: Path, probe_env: dict[str, str] = PROBE_ENV) -> list[str]:
    """Start ``cli`` under ``probe_env`` and return its init tool list."""
    with tempfile.TemporaryDirectory(prefix="cli-inventory-") as tmp:
        home = Path(tmp) / "home"
        project = Path(tmp) / "project"
        home.mkdir()
        project.mkdir()
        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            **probe_env,
        }
        argv = [
            str(cli),
            "-p",
            "inventory",
            "--output-format",
            "stream-json",
            "--verbose",
            "--setting-sources",
            "project",
            "--max-turns",
            "1",
        ]
        with subprocess.Popen(
            argv,
            cwd=project,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ) as proc:
            # The process retries the dead endpoint after the init message, so
            # it is killed as soon as the list is read. A timer bounds a silent
            # build, and a read cut short by that timer is reported as a timeout.
            timed_out = threading.Event()

            def _expire() -> None:
                timed_out.set()
                proc.kill()

            timer = threading.Timer(INIT_TIMEOUT_S, _expire)
            timer.start()
            try:
                assert proc.stdout is not None
                return init_tools(proc.stdout)
            except ValueError:
                if timed_out.is_set():
                    raise InventoryError(
                        f"{cli}: no system/init message within {INIT_TIMEOUT_S} s"
                    ) from None
                raise
            finally:
                timer.cancel()
                proc.kill()


def _npm_build(version: str, prefix: Path) -> Path:
    subprocess.run(
        [
            "npm",
            "install",
            "--prefix",
            str(prefix),
            "--no-audit",
            "--no-fund",
            "--no-save",
            f"{_NPM_PACKAGE}@{version}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    binary = prefix / "node_modules" / ".bin" / "claude"
    reported = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not reported.startswith(version + " ") and reported != version:
        raise InventoryError(f"npm installed {reported or 'nothing'}, expected {version}")
    return binary


def pinned_builds(npm_prefix: Path) -> dict[str, Path]:
    """Return ``{version: binary}`` for the SDK-bundled and the npm-pinned build."""
    from osprey.agent_runner.launcher import bundled_cli_path, bundled_cli_version
    from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

    sdk_path = bundled_cli_path()
    sdk_version = bundled_cli_version()
    if sdk_path is None or sdk_version is None:
        raise InventoryError("the Agent SDK's bundled CLI was not found; run `uv sync`")
    npm_version = _DEFAULT_CLAUDE_CLI_VERSION
    return {sdk_version: sdk_path, npm_version: _npm_build(npm_version, npm_prefix)}


def _write(builds: dict[str, Path]) -> int:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    for version, cli in builds.items():
        lists = {key: probe(cli, env) for key, env in PROBES.items()}
        (INVENTORY_DIR / f"{version}.json").write_text(render_inventory(version, lists))
        counts = ", ".join(f"{len(tools)} {key}" for key, tools in lists.items())
        print(f"{version}: {counts}")
    for path in INVENTORY_DIR.glob("*.json"):
        if path.stem not in builds:
            path.unlink()
            print(f"removed {path.name}")
    return 0


def _check(builds: dict[str, Path]) -> int:
    failures: list[str] = []
    for version, cli in builds.items():
        path = INVENTORY_DIR / f"{version}.json"
        if not path.is_file():
            failures.append(f"{version}: no inventory file {path.name}")
            continue
        recorded = json.loads(path.read_text())
        if recorded.get("cli_version") != version:
            failures.append(
                f"{path.name}: records cli_version {recorded.get('cli_version')!r}, "
                f"expected {version!r}"
            )
        for key, env in PROBES.items():
            live = probe(cli, env)
            stale = stale_names(recorded.get(key, []), live)
            if stale:
                failures.append(f"{version}: recorded {key} the build no longer lists: {stale}")
            extra = sorted(set(live) - set(recorded.get(key, [])))
            if extra:
                print(f"note: {version} lists {key} the inventory does not record: {extra}")
    for failure in failures:
        print(f"FAIL {failure}; remedy: {REMEDY}")
    if not failures:
        print(f"ok: {', '.join(sorted(builds))}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="record each pinned build")
    mode.add_argument("--check", action="store_true", help="re-prove the recorded files")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="cli-inventory-npm-") as npm_prefix:
        try:
            builds = pinned_builds(Path(npm_prefix))
            return _write(builds) if args.write else _check(builds)
        except (InventoryError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
