"""Process and container helpers shared by the deploy-backed e2e suites.

Each of these suites drives the real ``osprey`` CLI against a scratch
deployment repo and then reads the result back off the container runtime, so
they all need the same handful of small tools: locate the console script, run
a command with ``CLAUDECODE`` cleared, poll an HTTP or container health gate,
and ask the runtime about a container, volume or image.

The failure-time log capture lives next door in :mod:`tests.e2e._deploy_diagnostics`.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Container runtime selection -- ``docker`` by default, ``podman`` opt-in via
#: OSPREY_E2E_RUNTIME (the CI podman lanes set it), so one env var drives every
#: lane. Any other value fails clearly at collection time rather than silently
#: falling back to docker.
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )


#: The provider key a deploy-backed suite stands in, so a render whose web
#: terminals need one (`osprey up` refuses an ``anthropic`` render without it)
#: deploys. These suites never call the model; the template's own line is
#: commented out, as a real ``.env`` starts.
STAND_IN_PROVIDER_KEY = "ANTHROPIC_API_KEY=e2e-stand-in-not-a-key\n"


@contextmanager
def host_stack_lock(path: Path) -> Iterator[None]:
    """Hold a lane's host-wide stack lock for the duration of the block.

    For a lane with a fixed project name and fixed ports, which parallel
    benchmark cells on one host would otherwise fight over. An advisory
    ``flock``, so the operating system releases it when the holder dies: a
    crashed cell never leaves the host locked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkout_revision() -> str | None:
    """The checked-out commit, or ``None`` when it cannot be read (always rebuild then)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    return head.stdout.strip() or None


def images_built_for(stamp: Path, revision: str | None) -> bool:
    """Whether a lane's images were last built at *revision*."""
    if revision is None or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == revision


def record_images_built(stamp: Path, revision: str | None) -> None:
    """Record that a lane's images now match *revision*; an unknown one records nothing."""
    if revision is None:
        stamp.unlink(missing_ok=True)
        return
    stamp.write_text(revision, encoding="utf-8")


def env_from_example(repo: Path) -> str:
    """The ``.env`` a suite deploys with: the CLI's ``cp .env.example .env``, plus a stand-in key."""
    return f"{(repo / '.env.example').read_text().rstrip()}\n{STAND_IN_PROVIDER_KEY}"


def find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for the ACTIVE interpreter.

    Deliberately interpreter-relative first: a bare ``osprey`` on PATH may belong to a
    different checkout entirely, and in a worktree it usually does.
    """
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError(
        "Could not locate the 'osprey' console script. "
        f"Tried {Path(sys.executable).parent / 'osprey'} and PATH."
    )


def run_command(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run ``cmd`` in ``cwd`` with output captured and ``CLAUDECODE`` cleared."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb with the deploy-side runtime pinned to :func:`container_runtime`.

    Deliberately does NOT set ``OSPREY_PIP_SPEC``: the lifecycle fixtures poison
    it so a stub Dockerfile that ever started consuming it would fail loudly,
    but the real framework installs some suites build must resolve normally.
    """
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": "", "CONTAINER_RUNTIME": RUNTIME},
    )


def format_failure(label: str, result: subprocess.CompletedProcess) -> str:
    """A failure message carrying the exit code and both captured streams."""
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def runtime_cli(
    *args: str, timeout: int = 30, input_text: str | None = None
) -> subprocess.CompletedProcess:
    """Run one runtime CLI call. Deliberately does NOT swallow a timeout: a
    daemon that stops answering is a fact worth surfacing, not a false negative.
    """
    return subprocess.run(
        [RUNTIME, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )


def container_id(name: str) -> str | None:
    """The id of container ``name``, or ``None`` when no such container exists."""
    result = runtime_cli("inspect", "--type", "container", "-f", "{{.Id}}", name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def image_exists(tag: str) -> bool:
    return runtime_cli("inspect", "--type", "image", tag, timeout=15).returncode == 0


def image_label(tag: str, key: str) -> str | None:
    """The value of label ``key`` on image ``tag``, or ``None`` when there is no such image."""
    result = runtime_cli(
        "inspect",
        "--type",
        "image",
        "-f",
        f'{{{{ index .Config.Labels "{key}" }}}}',
        tag,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def volume_exists(name: str) -> bool:
    return runtime_cli("volume", "inspect", name, timeout=15).returncode == 0


def runtime_logs(name: str) -> str:
    """The last 80 log lines of container ``name``, both streams labelled."""
    result = runtime_cli("logs", "--tail", "80", name, timeout=20)
    return f"--- {name} stdout ---\n{result.stdout}\n--- {name} stderr ---\n{result.stderr}"


def docker_inspect(container: str, fmt: str) -> str:
    """``docker inspect --format fmt container``, asserting the container exists.

    Docker by name, not :func:`runtime_cli`: its callers are docker-only suites,
    skipped without a ``docker`` CLI and driving it directly whatever
    OSPREY_E2E_RUNTIME says.
    """
    proc = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
    return proc.stdout.strip()


def drain_output(proc: subprocess.Popen) -> str:
    """Best-effort grab of a subprocess's combined output for failure messages."""
    if proc.stdout is None:
        return "(no captured output)"
    try:
        data = proc.stdout.read1(16384) if hasattr(proc.stdout, "read1") else b""
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return f"--- subprocess output (partial) ---\n{text}" if text else "(no captured output)"


def wait_for_health(
    url: str,
    timeout: float,
    *,
    probe_timeout: float = 3.0,
    proc: subprocess.Popen | None = None,
) -> None:
    """Poll ``url`` every half second until it answers HTTP 200.

    With ``proc``, the subprocess serving ``url``: its early exit fails the wait
    at once, and either failure carries its captured output.
    """
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    with httpx.Client(timeout=probe_timeout, trust_env=False, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise AssertionError(
                    f"subprocess for {url} exited early (rc={proc.returncode}).\n"
                    f"{drain_output(proc)}"
                )
            try:
                status = client.get(url).status_code  # localhost
                if status == 200:
                    return
                last_err = f"HTTP {status}"
            except httpx.HTTPError as exc:
                last_err = str(exc)
            time.sleep(0.5)
    output = "" if proc is None else f"\n{drain_output(proc)}"
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err}){output}"
    )


def wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``docker inspect .State.Health.Status`` until ``healthy``.

    An HTTP-readiness gate can pass while Docker still reports ``starting``
    (the healthcheck runs only on its interval, after ``start_period``), so an
    instant equality assert would be racy -- and a service with no HTTP
    surface has nothing else to wait on at all.
    """
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        last = docker_inspect(container, "{{.State.Health.Status}}")
        if last == "healthy":
            return
        time.sleep(2.0)
    raise AssertionError(
        f"{container} did not reach 'healthy' within {timeout:.0f}s (last status: {last!r})"
    )


def env_value(repo: Path, key: str) -> str:
    """The non-empty value ``osprey deploy`` minted for ``key`` in the repo's ``.env``."""
    from osprey_connectors.dotenv import parse_dotenv_file

    env_path = repo / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — nothing was minted"
    value = parse_dotenv_file(env_path).get(key)
    assert value, f"{key} missing/empty in the deployment repo's .env"
    return value
