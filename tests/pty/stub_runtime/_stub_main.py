"""The stub container runtime: every conversation the deploy path has with docker.

The pty harness drives a real ``osprey up`` on a real terminal. Everything above
the runtime is the code under test; the runtime itself is the one thing that
cannot be real — a CI lane has no daemon, and a test that pulled images would
measure the network rather than the output hierarchy. So the seam is cut at the
process boundary: a ``docker`` executable earlier on ``PATH`` than any real one,
plus ``CONTAINER_RUNTIME=docker`` so
:func:`osprey.deployment.runtime_helper._runtimes_to_try` pins the probe to it
and never falls through to podman.

This module is the whole of that executable's behaviour. It is importable (the
``docker`` script beside it is a five-line launcher) for one reason: the
coverage gate — ``tests/pty/test_stub_coverage.py`` — reads :data:`HANDLED_PLAIN`
and :data:`HANDLED_COMPOSE` out of it and checks them against the seams the
deploy path actually builds. A stub whose handled set is a private detail could
only be checked by running it, and a subcommand nobody thought to script would
be discovered as a mysterious hang months later.

Three contracts the callers depend on, each load-bearing:

* **Exit 0, always.** Even for an argv nothing here recognizes. A stub that
  failed closed would turn "the harness has not scripted this yet" into "the
  deploy is broken", which is the harder failure to read. Unrecognized calls are
  recorded with ``"handled": false`` instead, and the gate reads that flag.
* **Every invocation is logged**, one JSON object per line, to
  ``$OSPREY_STUB_RUNTIME_LOG`` — argv, cwd, stdin, the resolved seam, and
  whether it was handled. Appended under an exclusive lock, because compose
  invocations overlap.
* **Answers are plausible, and consistent with each other.** ``compose up``
  reads the service names out of the very compose files it was handed and
  records them in ``$OSPREY_STUB_RUNTIME_STATE``, so the ``ps`` that follows
  lists the containers the deploy just started. Image IDs are one constant, so
  an image inspect and a container inspect agree and no caller sees drift it
  would try to reconcile.
"""

from __future__ import annotations

import json
import os
import re
import select
import sys
import time
from pathlib import Path
from typing import Any

#: Environment variable naming the JSONL invocation log. Every call appends one
#: line; the tests read the file back.
LOG_ENV = "OSPREY_STUB_RUNTIME_LOG"

#: Environment variable naming the JSON state file (which services are "up").
#: Absent, the stub is stateless and ``ps`` lists nothing.
STATE_ENV = "OSPREY_STUB_RUNTIME_STATE"

#: The one image ID every inspect answers with. Constant on purpose: image
#: drift is computed by comparing an image inspect against a container inspect,
#: and two different IDs would send the deploy into a recreate it never asked
#: for.
IMAGE_ID = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"

#: Version banners. The compose one is parsed by
#: :func:`~osprey.deployment.runtime_helper.detect_compose_provider`, which
#: fails closed on a banner it cannot place — so this string is a contract, not
#: decoration.
DOCKER_VERSION = "Docker version 27.3.1, build ce12230"
COMPOSE_VERSION = "Docker Compose version v2.30.3"

#: Plain-runtime seams the stub answers. Two-word entries are the namespaced
#: subcommands (``docker image inspect``); the first word alone would not say
#: which call it was.
HANDLED_PLAIN = frozenset(
    {
        "build",
        "container inspect",
        "container rm",
        "container stop",
        "cp",
        "create",
        "events",
        "exec",
        "images",
        "image inspect",
        "image ls",
        "image pull",
        "image rm",
        "info",
        "inspect",
        "kill",
        "login",
        "logout",
        "logs",
        "network create",
        "network inspect",
        "network ls",
        "network rm",
        "port",
        "ps",
        "pull",
        "push",
        "restart",
        "rm",
        "rmi",
        "run",
        "start",
        "stats",
        "stop",
        "system df",
        "tag",
        "top",
        "version",
        "volume create",
        "volume inspect",
        "volume ls",
        "volume rm",
        "wait",
    }
)

#: Compose seams the stub answers, spelled without the ``compose`` prefix.
HANDLED_COMPOSE = frozenset(
    {
        "build",
        "config",
        "cp",
        "create",
        "down",
        "events",
        "exec",
        "images",
        "kill",
        "logs",
        "ls",
        "port",
        "ps",
        "pull",
        "push",
        "restart",
        "rm",
        "run",
        "start",
        "stop",
        "top",
        "up",
        "version",
        "wait",
    }
)

#: Namespaced plain subcommands: the seam is two words, not one.
_NAMESPACES = frozenset({"container", "image", "network", "system", "volume", "builder", "buildx"})

#: Compose's own flags that take a value, so the scanner walking argv for the
#: subcommand skips the value too and does not mistake it for one.
_COMPOSE_VALUE_FLAGS = frozenset(
    {
        "-f",
        "--file",
        "-p",
        "--project-name",
        "--project-directory",
        "--env-file",
        "--profile",
        "--progress",
        "--ansi",
        "--parallel",
        "--compatibility",
    }
)

#: How compose reports each container-touching verb once it is done, so the
#: stub's progress lines read like the ones the renderer will see in anger.
_COMPOSE_PAST_TENSE = {
    "create": "Created",
    "kill": "Killed",
    "restart": "Started",
    "rm": "Removed",
    "start": "Started",
    "stop": "Stopped",
    "wait": "Exited",
}

#: The same, for the plain runtime's global flags.
_PLAIN_VALUE_FLAGS = frozenset(
    {"-H", "--host", "--context", "-c", "--config", "--log-level", "-l", "--tls-verify"}
)


# ---------------------------------------------------------------------------
# argv parsing
# ---------------------------------------------------------------------------


def split_argv(args: list[str]) -> tuple[bool, str, list[str]]:
    """Resolve an argv tail into ``(is_compose, seam, remainder)``.

    Args:
        args: Everything after the executable name.

    Returns:
        Whether this is a ``compose`` invocation, the seam it resolves to
        (``"up"``, ``"image inspect"``, ``""`` when there is no subcommand at
        all), and the arguments that followed the subcommand.
    """
    rest = list(args)
    is_compose = False
    if rest and rest[0] == "compose":
        is_compose = True
        rest = rest[1:]

    value_flags = _COMPOSE_VALUE_FLAGS if is_compose else _PLAIN_VALUE_FLAGS
    index = 0
    while index < len(rest):
        token = rest[index]
        if not token.startswith("-"):
            break
        if token in value_flags:
            index += 2
        elif "=" in token:
            index += 1
        else:
            index += 1
    else:
        return is_compose, "", []

    subcommand = rest[index]
    remainder = rest[index + 1 :]
    if not is_compose and subcommand in _NAMESPACES:
        for candidate in remainder:
            if not candidate.startswith("-"):
                return is_compose, f"{subcommand} {candidate}", remainder[1:]
        return is_compose, subcommand, remainder
    return is_compose, subcommand, remainder


def flag_value(args: list[str], *names: str) -> str | None:
    """The value of the first of *names* present in *args*, or ``None``.

    Handles both spellings a runtime accepts: ``--format json`` and
    ``--format=json``.
    """
    for index, token in enumerate(args):
        for name in names:
            if token == name and index + 1 < len(args):
                return args[index + 1]
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
    return None


def _positional(args: list[str], value_flags: frozenset[str]) -> list[str]:
    """The non-flag arguments, with the values of value-taking flags dropped."""
    out: list[str] = []
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in value_flags
            continue
        out.append(token)
    return out


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def _state_path() -> Path | None:
    raw = os.environ.get(STATE_ENV)
    return Path(raw) if raw else None


def read_state() -> dict[str, Any]:
    """The containers this stub currently considers up; empty when stateless."""
    path = _state_path()
    if path is None or not path.is_file():
        return {"containers": []}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"containers": []}
    return loaded if isinstance(loaded, dict) else {"containers": []}


def write_state(state: dict[str, Any]) -> None:
    """Persist *state*, atomically, when a state file was configured."""
    path = _state_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


#: A ``services:`` key: two spaces of indent under the top-level block. Crude on
#: purpose — the stub reads rendered compose files, which are emitted by
#: OSPREY's own generator in exactly this shape, and a YAML dependency in a
#: script that must run under the bare system interpreter is not worth it.
_SERVICE_KEY = re.compile(r"^ {2}([A-Za-z0-9][A-Za-z0-9_.-]*):\s*$")
_CONTAINER_NAME = re.compile(r"^\s+container_name:\s*['\"]?([^'\"\s]+)['\"]?\s*$")


def services_in(compose_file: Path) -> list[tuple[str, str]]:
    """``(service, container_name)`` pairs declared in one compose file.

    A service with no explicit ``container_name`` is named the way compose
    itself would name it if it had to: the service name stands in, and the
    caller prefixes the project.
    """
    try:
        lines = compose_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    found: list[tuple[str, str]] = []
    in_services = False
    current: str | None = None
    for line in lines:
        if line.startswith("services:"):
            in_services = True
            continue
        if line and not line[0].isspace() and not line.startswith("#"):
            in_services = False
            continue
        if not in_services:
            continue
        match = _SERVICE_KEY.match(line)
        if match:
            current = match.group(1)
            found.append((current, current))
            continue
        named = _CONTAINER_NAME.match(line)
        if named and current is not None and found:
            found[-1] = (current, named.group(1))
    return found


def _compose_files(args: list[str]) -> list[Path]:
    """Every ``-f``/``--file`` path in an argv, in order.

    A relative path is resolved the way compose resolves it — against the
    working directory — and, failing that, against ``--project-directory``. The
    fallback is what keeps a caller that pinned the project directory but
    invoked from elsewhere readable, which is the shape the deploy path uses.
    """
    project_directory = flag_value(args, "--project-directory")
    raw: list[str] = []
    for index, token in enumerate(args):
        if token in ("-f", "--file") and index + 1 < len(args):
            raw.append(args[index + 1])
        elif token.startswith("--file="):
            raw.append(token.split("=", 1)[1])

    files: list[Path] = []
    for entry in raw:
        path = Path(entry)
        if not path.is_file() and not path.is_absolute() and project_directory:
            path = Path(project_directory) / entry
        files.append(path)
    return files


def _project_name(args: list[str]) -> str:
    explicit = flag_value(args, "-p", "--project-name")
    if explicit:
        return explicit
    return os.environ.get("COMPOSE_PROJECT_NAME", "osprey")


def _container_record(project: str, service: str, name: str, index: int) -> dict[str, Any]:
    """One ``ps``-shaped record, with the keys the status renderer reads."""
    qualified = name if name != service else f"{project}-{service}-1"
    return {
        "ID": f"{index:012x}",
        "Names": qualified,
        "Image": f"{project}/{service}:local",
        "Command": '"/entrypoint.sh"',
        "State": "running",
        "Status": "Up 3 seconds",
        "Ports": "",
        "Labels": f"com.docker.compose.project={project},com.docker.compose.service={service}",
        "Networks": f"{project}_default",
        "RunningFor": "3 seconds ago",
        "Service": service,
        "Project": project,
        "Health": "healthy",
    }


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def log_invocation(record: dict[str, Any]) -> None:
    """Append one JSON line to the invocation log, if one was configured.

    Opened per call with ``O_APPEND``: compose runs overlap, and a handle held
    open across them would interleave partial lines.
    """
    raw = os.environ.get(LOG_ENV)
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_stdin(deadline: float = 0.1) -> str:
    """Whatever was piped in, or ``""`` when nothing was.

    Deliberately not ``sys.stdin.read()``. Most runtime calls on the deploy path
    inherit the parent's stdin rather than being handed one, and on the piped
    lane that inherited descriptor is a pipe nobody is going to close — a
    straight read would hang the whole harness on the first ``docker ps``. So
    stdin is drained only while it is *ready*, and abandoned after *deadline*
    seconds: the calls that really do pipe something in (``exec -i``) hand over
    a closed-after-write pipe and read to EOF well inside it.

    Args:
        deadline: Seconds to keep draining before giving up.

    Returns:
        The decoded input, or ``""``.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        fd = sys.stdin.fileno()
    except (OSError, ValueError):
        return ""

    chunks: list[bytes] = []
    give_up_at = time.monotonic() + deadline
    while time.monotonic() < give_up_at:
        try:
            ready, _, _ = select.select([fd], [], [], 0.01)
        except (OSError, ValueError):
            break
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _handle_plain(seam: str, args: list[str], remainder: list[str]) -> bool:
    """Answer a plain-runtime seam. Returns whether it was recognized."""
    out = sys.stdout
    if seam == "version":
        print(DOCKER_VERSION, file=out)
        return True
    if not seam and any(token in ("--version", "-v") for token in args):
        # ``docker --version`` names no subcommand at all, so it arrives here
        # with an empty seam. A probe that got silence back would read the stub
        # as a runtime that is installed but broken, which is the one answer
        # neither the harness nor the code under test knows what to do with.
        print(DOCKER_VERSION, file=out)
        return True
    if seam == "info":
        print("Client:\n Version:    27.3.1\n\nServer:\n Server Version: 27.3.1", file=out)
        return True
    if seam == "ps":
        _handle_ps(args)
        return True
    if seam == "volume ls":
        template = flag_value(args, "--format") or ""
        names = sorted({f"{c['Project']}_{c['Service']}-data" for c in read_state()["containers"]})
        if "{{" in template:
            for name in names:
                print(name, file=out)
        else:
            print("DRIVER    VOLUME NAME", file=out)
            for name in names:
                print(f"local     {name}", file=out)
        return True
    if seam in ("volume rm", "volume create", "volume inspect"):
        for name in _positional(remainder, frozenset({"--format"})):
            print(name, file=out)
        return True
    if seam in ("image inspect", "container inspect", "inspect"):
        _handle_inspect(seam, args, remainder)
        return True
    if seam in ("image rm", "rmi"):
        for tag in _positional(remainder, frozenset()):
            print(f"Untagged: {tag}", file=out)
        return True
    if seam in ("image ls", "images"):
        print("REPOSITORY   TAG       IMAGE ID       CREATED         SIZE", file=out)
        return True
    if seam == "build":
        _handle_build(args)
        return True
    if seam == "exec":
        _handle_exec(remainder)
        return True
    if seam in ("stop", "start", "rm", "kill", "restart", "container rm", "container stop", "wait"):
        for name in _positional(remainder, frozenset()):
            print(name, file=out)
        return True
    if seam in ("pull", "image pull", "push", "tag", "login", "logout", "cp", "port", "top"):
        return True
    if seam in ("network ls", "network create", "network rm", "network inspect"):
        if seam == "network ls":
            print("NETWORK ID     NAME      DRIVER    SCOPE", file=out)
        return True
    if seam in ("logs", "events", "stats", "system df", "create", "run"):
        return True
    return False


def _handle_ps(args: list[str]) -> None:
    """``ps`` in all four shapes the deploy path asks for it in."""
    containers = read_state()["containers"]
    label_filter = flag_value(args, "--filter", "-f")
    if label_filter and label_filter.startswith("label="):
        # Sliced rather than ``removeprefix``: this file runs under whatever
        # ``/usr/bin/env python3`` finds on the host, which is not the version
        # the rest of the repository is allowed to assume.
        wanted = label_filter[len("label=") :]
        containers = [c for c in containers if wanted in c["Labels"]]

    template = flag_value(args, "--format") or ""
    short = any(
        token.startswith("-") and not token.startswith("--") and "q" in token for token in args
    )

    if "json" in template.lower():
        for container in containers:
            print(json.dumps(container), file=sys.stdout)
        return
    if "{{" in template:
        field = "Names"
        if ".ID" in template:
            field = "ID"
        elif ".Image" in template:
            field = "Image"
        for container in containers:
            print(container[field], file=sys.stdout)
        return
    if short:
        for container in containers:
            print(container["ID"], file=sys.stdout)
        return
    print("CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES", file=sys.stdout)
    for container in containers:
        print(
            f"{container['ID']}   {container['Image']}   {container['Command']}   "
            f"3 seconds ago   {container['Status']}      {container['Names']}",
            file=sys.stdout,
        )


def _handle_inspect(seam: str, args: list[str], remainder: list[str]) -> None:
    """``inspect`` for images, containers, and the bare ``--type container`` form.

    Go templates are answered by what they ask for rather than by rendering
    them: the four templates the deploy path uses are a closed set, and a
    template engine in a stub would be a second implementation of docker's.
    """
    template = flag_value(args, "--format") or ""
    if ".Id" in template or ".Image" in template:
        print(f"sha256:{IMAGE_ID}" if ".Id" in template else IMAGE_ID, file=sys.stdout)
        return
    if "Labels" in template:
        # ``{}`` rather than a fabricated label map: an empty map is what a
        # container the stub never really created has, and it is the answer
        # that makes every label-gated destructive path decline to act.
        print("{}", file=sys.stdout)
        return
    if ".Config.Env" in template:
        print("PATH=/usr/local/bin:/usr/bin:/bin", file=sys.stdout)
        return
    names = _positional(remainder, frozenset({"--format", "--type"}))
    payload = [
        {
            "Id": f"sha256:{IMAGE_ID}",
            "Name": f"/{name}",
            "Config": {"Labels": {}, "Env": []},
            "State": {"Status": "running", "Running": True},
        }
        for name in names
    ] or [{"Id": f"sha256:{IMAGE_ID}", "Config": {"Labels": {}, "Env": []}}]
    print(json.dumps(payload), file=sys.stdout)


def _handle_build(args: list[str]) -> None:
    """A buildkit-shaped plain-progress transcript.

    Shaped, not decorative: ``build_progress`` parses exactly these lines out of
    a real build, so the continuation-line rules it was fixed to honour
    (``DONE``, ``CACHED``, ``sha256:``) are exercised by the stub too.
    """
    tag = flag_value(args, "-t", "--tag") or "stub:local"
    for line in (
        "#1 [internal] load build definition from Dockerfile",
        "#1 transferring dockerfile: 2.31kB done",
        "#1 DONE 0.0s",
        "#2 [internal] load metadata for docker.io/library/python:3.11-slim",
        "#2 DONE 0.1s",
        "#3 [1/2] FROM docker.io/library/python:3.11-slim",
        "#3 CACHED",
        "#4 exporting to image",
        f"#4 writing image sha256:{IMAGE_ID} done",
        f"#4 naming to docker.io/library/{tag} done",
        "#4 DONE 0.2s",
    ):
        print(line, file=sys.stderr)


def _handle_exec(remainder: list[str]) -> None:
    """``exec`` answers only what the deploy path actually asks a container."""
    joined = " ".join(remainder)
    if "id -u" in joined:
        print("1000:1000", file=sys.stdout)


def _handle_compose(seam: str, args: list[str], remainder: list[str]) -> bool:
    """Answer a compose seam. Returns whether it was recognized."""
    if seam == "version":
        print(COMPOSE_VERSION, file=sys.stdout)
        return True
    if seam == "up":
        _handle_compose_up(args, remainder)
        return True
    if seam == "down":
        _handle_compose_down(args)
        return True
    if seam == "ps":
        _handle_ps(args)
        return True
    if seam == "config":
        print("services: {}", file=sys.stdout)
        return True
    if seam in ("images", "ls"):
        print("CONTAINER   REPOSITORY   TAG   IMAGE ID   SIZE", file=sys.stdout)
        return True
    if seam == "build":
        _handle_build(args)
        return True
    if seam in _COMPOSE_PAST_TENSE:
        _report_compose_containers(args, remainder, verb=_COMPOSE_PAST_TENSE[seam])
        return True
    if seam in ("pull", "push", "logs", "exec", "run", "cp", "port", "top", "events"):
        return True
    return False


def _selected_services(args: list[str], remainder: list[str]) -> list[tuple[str, str]]:
    """The services this invocation targets: the named ones, else all of them."""
    declared: list[tuple[str, str]] = []
    for compose_file in _compose_files(args):
        declared.extend(services_in(compose_file))
    named = set(_positional(remainder, frozenset({"--scale", "--timeout", "-t"})))
    if named:
        return [pair for pair in declared if pair[0] in named]
    return declared


def _handle_compose_up(args: list[str], remainder: list[str]) -> None:
    """Start the services the compose files declare, and record them as up."""
    project = _project_name(args)
    state = read_state()
    existing = {c["Names"]: c for c in state["containers"]}
    lines: list[str] = []
    for index, (service, name) in enumerate(_selected_services(args, remainder), start=1):
        record = _container_record(project, service, name, index)
        was_running = record["Names"] in existing
        existing[record["Names"]] = record
        lines.append(f" Container {record['Names']}  {'Recreated' if was_running else 'Created'}")
        lines.append(f" Container {record['Names']}  Started")
    state["containers"] = list(existing.values())
    write_state(state)
    # Compose writes its progress to stderr and keeps stdout for data. The
    # renderer under test relies on that split, so the stub must honour it.
    for line in lines:
        print(line, file=sys.stderr)


def _handle_compose_down(args: list[str]) -> None:
    """Stop everything this project has up, and forget it."""
    project = _project_name(args)
    state = read_state()
    kept = []
    for container in state["containers"]:
        if container["Project"] == project:
            print(f" Container {container['Names']}  Removed", file=sys.stderr)
        else:
            kept.append(container)
    print(f" Network {project}_default  Removed", file=sys.stderr)
    state["containers"] = kept
    write_state(state)


def _report_compose_containers(args: list[str], remainder: list[str], *, verb: str) -> None:
    """Echo one compose-shaped progress line per targeted container."""
    project = _project_name(args)
    for _, name in _selected_services(args, remainder):
        qualified = name if "-" in name else f"{project}-{name}-1"
        print(f" Container {qualified}  {verb}", file=sys.stderr)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Answer one runtime invocation; always exits 0.

    Args:
        argv: Full argv including the program name. Defaults to ``sys.argv``.

    Returns:
        Always ``0`` — see the module docstring for why a stub must not fail
        closed on an argv it does not know.
    """
    raw = list(sys.argv if argv is None else argv)
    args = raw[1:]
    stdin_text = _read_stdin()
    is_compose, seam, remainder = split_argv(args)

    handled = False
    try:
        if is_compose:
            handled = _handle_compose(seam, args, remainder)
        else:
            handled = _handle_plain(seam, args, remainder)
    finally:
        log_invocation(
            {
                "argv": raw,
                "args": args,
                "compose": is_compose,
                "seam": f"compose {seam}".strip() if is_compose else seam,
                "handled": handled,
                "cwd": os.getcwd(),
                "stdin": stdin_text,
                "pid": os.getpid(),
                "ts": time.time(),
            }
        )

    if not handled:
        print(f"stub-runtime: unhandled invocation: {' '.join(raw)}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0
