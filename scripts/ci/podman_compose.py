#!/usr/bin/env python3
"""Pin podman-compose as podman's compose provider, and prove what it parses.

``podman compose`` hands the work to whichever external provider it finds
first, and on a GitHub runner that is the docker-compose CLI plugin: the
engine the docker lanes already cover, and one that accepts the
``--project-directory`` OSPREY's podman argv deliberately omits. A podman lane
that quietly ran the plugin would report green having tested nothing new, so
the pin, the force and the assertion are three separate commands here:

``pin [--optional]``
    Install exactly the pinned set (``--no-deps``), name the installed
    binary in ``~/.config/containers/containers.conf`` and in
    ``PODMAN_COMPOSE_PROVIDER`` (a user-level ``containers.conf.d`` drop-in
    would outrank the file; the variable sidesteps the chain), then pay
    rootless storage init with ``podman ps``. ``--optional`` is for a job that
    gates every PR: without podman on the runner it records
    ``OSPREY_PARSE_SKIP=1`` and exits 0. A lane never passes it, so a runner
    that lost podman still reds there.
``assert-provider``
    Probe through OSPREY's own detector that the provider is podman-compose at
    the pinned version, before an expensive boot certifies anything under it.
``parse-check [--dir DIR]``
    Render a deployment, have OSPREY build its compose argv (which writes the
    merged document and ``build/.env.merged``) and run that argv's ``config``:
    the merged topology must parse, and the env chain must interpolate it with
    local-wins precedence. Honours ``OSPREY_PARSE_SKIP``.

``pin`` is stdlib-only and runs under the runner's ``python3``; the other two
import OSPREY and run under ``uv run python``. Every value a later step reads is
appended to ``$GITHUB_ENV``; outside Actions it is printed as an ``export``.
``pin`` writes the user's ``containers.conf``: run it locally only on a machine
whose podman should use podman-compose.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: The install set, exactly: podman-compose and the two dependencies it imports.
PINNED = ("podman-compose==1.0.6", "python-dotenv==1.2.4", "PyYAML==6.0.3")
#: The version the provider must report, read from the pin so the two cannot drift.
PINNED_VERSION = tuple(int(part) for part in PINNED[0].split("==")[1].split("."))
#: Set by ``pin --optional`` on a runner without podman; ``parse-check`` then skips.
SKIP_VARIABLE = "OSPREY_PARSE_SKIP"
#: The services the parse-check deployment renders.
PARSE_SERVICES = ("event-dispatcher", "dispatch-worker-1")
#: The same key in both env-chain members: the local value must win.
LOCAL_SENTINEL = "local-wins"
SHARED_SENTINEL = "shared-loses"


def export(name: str, value: str) -> None:
    """Hand *name* to the job's later steps, or print it outside Actions."""
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"export {name}={value}")


def install_argv(pip: str) -> list[str]:
    """The pip call that installs exactly :data:`PINNED`."""
    return [pip, "install", "--user", "--no-deps", *PINNED]


def containers_conf(provider_bin: str) -> str:
    """A ``containers.conf`` that forces podman's compose provider to *provider_bin*."""
    return f'[engine]\ncompose_providers = ["{provider_bin}"]\n'


def pin(*, optional: bool) -> int:
    if shutil.which("podman") is None:
        if not optional:
            print("❌ podman is not on PATH")
            return 1
        print("⏭️  podman is not on this runner image: the podman-compose parse below will skip.")
        print("    The two-shape-boot-e2e lane is NOT guarded this way and reds if podman is gone.")
        export(SKIP_VARIABLE, "1")
        return 0
    pip = shutil.which("pip") or shutil.which("pip3")
    if pip is None:
        print("❌ no pip on PATH to install podman-compose with")
        return 1
    subprocess.run(install_argv(pip), check=True)
    provider_bin = shutil.which("podman-compose")
    if provider_bin is None:
        print("❌ podman-compose is not on PATH after install")
        return 1
    conf_dir = Path.home() / ".config" / "containers"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "containers.conf").write_text(containers_conf(provider_bin), encoding="utf-8")
    export("PODMAN_COMPOSE_PROVIDER", provider_bin)
    subprocess.run(["podman", "--version"], check=True)
    subprocess.run(["podman", "ps"], check=True, stdout=subprocess.DEVNULL)
    print(f"✅ {PINNED[0]} is podman's compose provider: {provider_bin}")
    return 0


def provider_problems(info: Any) -> list[str]:
    """What is wrong with the probed provider *info* (empty when it is the pin)."""
    from osprey.deployment.runtime_helper import ComposeProvider

    problems = []
    if info.provider is not ComposeProvider.PODMAN_COMPOSE:
        problems.append(f"the provider must be podman-compose; probed {info.provider}")
    if tuple(info.version) != PINNED_VERSION:
        problems.append(f"the provider must be {PINNED[0]}; probed {info.version_text}")
    return problems


def _probe_provider(config: Any = None) -> tuple[list[str], Any | None]:
    """The runtime command and the probed provider, or ``None`` for a provider off the pin."""
    from osprey.deployment.runtime_helper import detect_compose_provider, get_runtime_command

    runtime_cmd = get_runtime_command(config)
    info = detect_compose_provider(runtime_cmd, config)
    print("runtime command:", " ".join(runtime_cmd))
    print("provider banner:", info.banner)
    problems = provider_problems(info)
    for problem in problems:
        print(f"❌ {problem}")
    return runtime_cmd, (None if problems else info)


def assert_provider() -> int:
    subprocess.run(["podman", "compose", "version"], check=True)
    _, info = _probe_provider()
    if info is None:
        return 1
    print(f"✅ compose provider is {PINNED[0]}")
    return 0


def merged_document_problems(document: str) -> list[str]:
    """What ``config``'s rendering of the merged topology fails to show."""
    problems = [
        f"{s} missing from the merged document" for s in PARSE_SERVICES if f"{s}:" not in document
    ]
    if LOCAL_SENTINEL not in document:
        problems.append("build/.env.merged did not interpolate the topology")
    if SHARED_SENTINEL in document:
        problems.append(".env.shared outranked .env in the merge")
    return problems


def check_merged_topology() -> list[str]:
    """Parse the built project's merged topology (the working directory) with
    OSPREY's own argv; the problems found."""
    from osprey.deployment.compose_generator import (
        compose_base_cmd,
        compose_provider_env,
        resolve_repo_root,
    )
    from osprey.deployment.container_lifecycle import _env_file_args, as_built_compose_files
    from osprey_connectors.config import load_config

    config = load_config("build/config.yml")
    root = Path(resolve_repo_root(config)).resolve()
    runtime_cmd, info = _probe_provider(config)
    if info is None:
        return ["the compose provider is not the pinned podman-compose"]
    # Building the argv is what writes .osprey-compose.yml and build/.env.merged.
    argv = compose_base_cmd(
        runtime_cmd,
        as_built_compose_files(config, root),
        root,
        _env_file_args(root, info.provider),
        info.provider,
    )
    print("argv:", " ".join(argv))
    if "--project-directory" in argv:
        return ["podman-compose does not parse --project-directory"]
    env = {**os.environ, **compose_provider_env(info.provider, root)}
    # `config` parses and interpolates; any other subcommand only talks to the runtime.
    parsed = subprocess.run(
        [*argv, "config"], capture_output=True, text=True, env=env, cwd=root, check=False
    )
    sys.stdout.write(parsed.stdout)
    sys.stderr.write(parsed.stderr)
    if parsed.returncode != 0:
        return ["podman-compose could not parse the merged topology"]
    return merged_document_problems(parsed.stdout)


def parse_check(workdir: Path) -> int:
    if os.environ.get(SKIP_VARIABLE) == "1":
        print("⏭️  skipped: podman is absent from this runner (see the pin step).")
        return 0
    project = workdir.resolve()
    subprocess.run(
        [
            "osprey", "init", project.name, "--preset", "hello-world", "--no-git",
            "--set", "dispatch.triggers=tutorial_triggers.yml",
            "--set", "dispatch.worker_count=1",
            "--set", "dispatch.workspace_mode=isolated",
            "--set", "dispatch.network=bridge",
            "--set", "config.deployed_services=[]",
        ],
        check=True,
        cwd=project.parent,
    )  # fmt: skip
    # Same key, different values: the result shows local-wins precedence, not just that a file was read.
    with open(project / ".env.shared", "a", encoding="utf-8") as shared:
        shared.write(f"\nDISPATCH_WORKER_TOKEN={SHARED_SENTINEL}\n")
    (project / ".env").write_text(
        f"EVENT_DISPATCHER_TOKEN={LOCAL_SENTINEL}\nDISPATCH_WORKER_TOKEN={LOCAL_SENTINEL}\n",
        encoding="utf-8",
    )
    subprocess.run(["osprey", "build", "--skip-deps", "--skip-lifecycle"], check=True, cwd=project)
    os.chdir(project)
    problems = check_merged_topology()
    for problem in problems:
        print(f"❌ {problem}")
    if problems:
        return 1
    print(f"✅ merged topology parses under {PINNED[0]}, .env.merged interpolates it")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # `python -OO` strips docstrings.
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0] or None)
    commands = parser.add_subparsers(dest="command", required=True)
    pin_cmd = commands.add_parser("pin", help="install and force the pinned podman-compose")
    pin_cmd.add_argument("--optional", action="store_true", help="exit 0 without podman")
    commands.add_parser("assert-provider", help="assert podman's provider is the pin")
    parse_cmd = commands.add_parser("parse-check", help="parse a merged topology")
    parse_cmd.add_argument("--dir", type=Path, default=Path("parse-check"))
    args = parser.parse_args(argv)
    if args.command == "pin":
        return pin(optional=args.optional)
    if args.command == "assert-provider":
        return assert_provider()
    return parse_check(args.dir)


if __name__ == "__main__":
    sys.exit(main())
