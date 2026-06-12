"""E2E: build, boot, and serve an agent from the generated reference Dockerfile.

Renders a hello-world project, runs a real ``docker build`` on the Dockerfile
that ``osprey build`` generated, then exercises the image at two depths:

``test_generated_dockerfile_builds_and_boots`` — static smoke checks:

- ``osprey --version`` works (OSPREY installed and importable),
- the runtime user is the non-root ``osprey`` user,
- ``claude --version`` works as that user (canary for the /root/.local
  permission-traversal chain the native installer requires),
- ``config.yml`` inside the image points at the container path (the
  ``osprey claude regen --runtime-root`` build step healed the host paths —
  the host build here uses ``--skip-deps``, which records the host
  interpreter, so this genuinely exercises the relocation path),
- ``.dockerignore`` kept ``.env`` out of the image.

``test_generated_image_serves_agent_over_http`` — full functional proof:
boots the image's actual ``CMD`` (``osprey web``), waits for ``/health``, and
drives one real agent turn through ``POST /api/chat`` with a live LLM call via
the als-apg provider. This is the only test that proves the *shipped*
entrypoint actually serves an agent — ``claude --version`` proves the binary
launches, not that the assembled image answers a prompt.

Both tests share one image build (module-scoped fixture). The image is built
with ``--set provider=als-apg`` so the in-image ``config.yml`` resolves to the
provider CI can reach; LLM credentials are injected at ``docker run`` time
(never baked into the image).

Set ``OSPREY_E2E_PIP_SPEC`` to override which OSPREY gets installed inside the
image. The image default is the PyPI release, which only works once a release
containing ``regen --runtime-root`` is published — CI pins the PR head SHA
instead so the image tests the branch under review.

The HTTP test additionally needs ``ALS_APG_API_KEY`` (``requires_als_apg``);
it auto-skips without it. The build/boot test has no such requirement.

Skipped entirely when docker is unavailable. Excluded from the main e2e-tests
CI job (runs in its own dockerfile-e2e job).
"""

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_available(), reason="docker binary or daemon not available"),
]

BUILD_TIMEOUT = 1800  # cold image build downloads base layers + pip deps
RUN_TIMEOUT = 300
HEALTH_TIMEOUT = 120  # server process up + first port bind
CHAT_TIMEOUT = 240  # one real LLM round-trip (haiku, single turn)
PROJECT_NAME = "dockere2e"

# A distinctive token the model is unlikely to emit by chance — proves the
# prompt reached a live model and came back, not that *some* text returned.
CHAT_MARKER = "OSPREY-E2E-OK"
CHAT_PROMPT = f"Respond with exactly this token and nothing else: {CHAT_MARKER}"

# OSPREY's dependency chain (accelerator-toolbox) ships linux/amd64 wheels
# only; on arm64 hosts (Apple Silicon) a native build would need a compiler
# the slim base lacks. Build/run amd64 under emulation instead — it matches
# the real deployment targets.
_PLATFORM_ARGS = (
    ["--platform", "linux/amd64"] if platform.machine().lower() in ("arm64", "aarch64") else []
)


def _docker_run(tag: str, *cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "run", "--rm", *_PLATFORM_ARGS, tag, *cmd],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )


def _render_hello_world(out_dir) -> "tuple":
    """Render a hello-world project pinned to the als-apg provider.

    Returns ``(project_path, project_name)``. ``--skip-deps`` records the host
    interpreter on purpose (the image's ``regen --runtime-root`` step heals it).
    """
    result = CliRunner().invoke(
        cli,
        [
            "build",
            PROJECT_NAME,
            "--preset",
            "hello-world",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--skip-deps",
            "--skip-lifecycle",
            "-o",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    project = out_dir / PROJECT_NAME
    assert (project / "Dockerfile").exists()
    return project, PROJECT_NAME


def _build_image(project, tag: str) -> None:
    """Run ``docker build`` on the generated Dockerfile."""
    build_cmd = ["docker", "build", *_PLATFORM_ARGS, "-t", tag]
    pip_spec = os.environ.get("OSPREY_E2E_PIP_SPEC")
    if pip_spec:
        build_cmd += ["--build-arg", f"OSPREY_PIP_SPEC={pip_spec}"]
    build_cmd.append(".")
    build = subprocess.run(
        build_cmd, cwd=project, capture_output=True, text=True, timeout=BUILD_TIMEOUT
    )
    assert build.returncode == 0, (
        f"docker build failed:\n--- stdout ---\n{build.stdout[-4000:]}"
        f"\n--- stderr ---\n{build.stderr[-4000:]}"
    )


@pytest.fixture(scope="module")
def built_image(tmp_path_factory):
    """Render + build the reference image once for the whole module."""
    out_dir = tmp_path_factory.mktemp("dockerfile-e2e")
    project, project_name = _render_hello_world(out_dir)
    tag = f"osprey-dockerfile-e2e:{uuid.uuid4().hex[:8]}"
    try:
        _build_image(project, tag)
        yield tag, project, project_name, out_dir
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def test_generated_dockerfile_builds_and_boots(built_image):
    tag, _project, project_name, out_dir = built_image

    # OSPREY installed and importable
    version = _docker_run(tag, "osprey", "--version")
    assert version.returncode == 0, version.stderr

    # Non-root runtime user
    whoami = _docker_run(tag, "whoami")
    assert whoami.stdout.strip() == "osprey"

    # Claude Code callable as the non-root user (permission-chain canary)
    claude = _docker_run(tag, "claude", "--version")
    assert claude.returncode == 0, (
        f"claude --version failed as non-root user — the /root/.local "
        f"traversal chain is broken:\n{claude.stderr}"
    )

    # regen --runtime-root healed the recorded host paths
    config = _docker_run(tag, "cat", f"/app/{project_name}/config.yml")
    assert config.returncode == 0, config.stderr
    assert f"project_root: /app/{project_name}" in config.stdout
    assert str(out_dir) not in config.stdout, "host build path leaked into the image config"

    # .dockerignore did its job: no secrets/host state in the image
    env_check = _docker_run(tag, "sh", "-c", f"test ! -e /app/{project_name}/.env && echo OK")
    assert "OK" in env_check.stdout, ".env must never enter the image"


# ── Functional: the shipped CMD actually serves an agent ─────────────────────


def _host_port(cid: str) -> int:
    """Resolve the ephemeral host port docker mapped to container :8087."""
    out = subprocess.run(
        ["docker", "port", cid, "8087/tcp"], capture_output=True, text=True, timeout=15
    )
    assert out.returncode == 0, f"docker port failed: {out.stderr}"
    # Output like "127.0.0.1:54321" (possibly multiple lines for v4/v6).
    return int(out.stdout.strip().splitlines()[0].rsplit(":", 1)[1])


def _container_logs(cid: str) -> str:
    logs = subprocess.run(["docker", "logs", "--tail", "60", cid], capture_output=True, text=True)
    return (logs.stdout + logs.stderr)[-4000:]


def _is_running(cid: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", cid], capture_output=True, text=True
    )
    return out.stdout.strip() == "true"


def _wait_for_health(base_url: str, cid: str, timeout: float) -> None:
    """Poll ``GET /health`` until healthy, failing fast if the container dies."""
    deadline = time.monotonic() + timeout
    last_err = "no response"
    while time.monotonic() < deadline:
        if not _is_running(cid):
            pytest.fail(f"container exited before becoming healthy:\n{_container_logs(cid)}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("status") == "healthy":
                    return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    pytest.fail(f"server never became healthy ({last_err}):\n{_container_logs(cid)}")


def _post_chat(base_url: str, prompt: str, timeout: float) -> dict:
    """POST a prompt to the buffered chat endpoint, return the JSON body."""
    req = urllib.request.Request(
        f"{base_url}/api/chat?stream=false",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        assert resp.status == 200, f"chat returned HTTP {resp.status}"
        return json.loads(resp.read())


@pytest.mark.requires_als_apg
def test_generated_image_serves_agent_over_http(built_image):
    """Boot the image's CMD and drive one real LLM turn through osprey web."""
    tag, _project, _project_name, _out_dir = built_image

    # Mirror the production run contract (`docker run --env-file .env`): pass
    # the raw provider secret, never a pre-resolved token. The full osprey web
    # stack resolves ${ALS_APG_API_KEY} from config at runtime and stands up
    # an internal proxy that authenticates upstream with it — injecting a
    # pre-resolved ANTHROPIC_AUTH_TOKEN would bypass that and leave config
    # resolution (and the proxy) without the key.
    api_key = os.environ["ALS_APG_API_KEY"]  # guaranteed present by requires_als_apg
    env_args = ["-e", f"ALS_APG_API_KEY={api_key}"]

    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            *_PLATFORM_ARGS,
            "-p",
            "127.0.0.1:0:8087",
            *env_args,
            tag,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, f"docker run failed: {run.stderr}"
    cid = run.stdout.strip()

    try:
        base_url = f"http://127.0.0.1:{_host_port(cid)}"
        _wait_for_health(base_url, cid, HEALTH_TIMEOUT)

        try:
            data = _post_chat(base_url, CHAT_PROMPT, CHAT_TIMEOUT)
        except urllib.error.HTTPError as exc:
            pytest.fail(f"chat HTTP {exc.code}: {exc.read()[:2000]!r}\n{_container_logs(cid)}")

        assert data.get("is_error") is False, (
            f"agent returned an error: {data.get('error')}\n{_container_logs(cid)}"
        )
        text = (data.get("text") or "").strip()
        assert text, f"agent returned empty text\n{_container_logs(cid)}"
        assert CHAT_MARKER in text, (
            f"expected marker {CHAT_MARKER!r} in agent reply, got: {text[:500]!r}"
        )
        assert (data.get("num_turns") or 0) >= 1, "expected at least one agent turn"
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
