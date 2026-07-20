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

Three further tests cover the fast-dev-rebuild layer split:

- ``test_rebuild_without_changes_is_fully_cached`` — a no-change rebuild runs
  every RUN/COPY step from the layer cache,
- ``test_equal_version_dev_wheel_lands_local_code`` — a staged dev wheel whose
  version equals the deps-layer copy still lands its code (pip's silent
  equal-version skip is defeated by ``--no-deps --force-reinstall``),
- ``test_final_image_has_no_toolchain_and_carries_project_label`` — the C
  toolchain is purged from the final image and the ``com.osprey.project``
  label is stamped.

The smoke/HTTP/cache/hygiene tests share one image build (the module-scoped
``built_image`` fixture); the sentinel test builds its own. The image is built
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
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

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


def _build_image(project, tag: str, *, progress_plain: bool = False) -> subprocess.CompletedProcess:
    """Run ``docker build`` on the generated Dockerfile; return the completed process.

    Passes the same ``com.osprey.project`` label ``osprey deploy`` stamps via
    ``_project_image_build_cmd``, so label assertions here cover the shape the
    real build path produces. ``DOCKER_BUILDKIT=1`` pins the BuildKit builder
    for every build so layer-cache semantics (and ``--progress=plain`` output,
    used by the cache-hit test) are uniform across docker versions.
    """
    build_cmd = [
        "docker",
        "build",
        *_PLATFORM_ARGS,
        "-t",
        tag,
        "--label",
        f"com.osprey.project={PROJECT_NAME}",
    ]
    if progress_plain:
        build_cmd += ["--progress=plain"]
    pip_spec = os.environ.get("OSPREY_E2E_PIP_SPEC")
    if pip_spec:
        build_cmd += ["--build-arg", f"OSPREY_PIP_SPEC={pip_spec}"]
    build_cmd.append(".")
    build = subprocess.run(
        build_cmd,
        cwd=project,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    assert build.returncode == 0, (
        f"docker build failed:\n--- stdout ---\n{build.stdout[-4000:]}"
        f"\n--- stderr ---\n{build.stderr[-4000:]}"
    )
    return build


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


# ── Layer cache, dev-wheel sentinel, and image hygiene ───────────────────────
#
# Regression tests for the fast-dev-rebuild layer split: a cached deps layer
# (toolchain installed and purged in the same RUN) followed by a dev-only
# wheel layer (`COPY .dockerignore *.wh[l]` + `--no-deps --force-reinstall`).


# BuildKit --progress=plain step header for real Dockerfile instructions, e.g.
# "#8 [ 4/10] RUN apt-get update ..." (stage-qualified "[stage-0 4/10]" in
# multi-stage builds). FROM steps are excluded on purpose: BuildKit reports
# base-image resolution as DONE even on a fully cached rebuild.
_STEP_HEADER_RE = re.compile(r"^#(\d+) \[\s*(?:[\w.-]+ +)?\d+/\d+\] +(RUN|COPY) (.*)$", re.M)
_CACHED_STEP_RE = re.compile(r"^#(\d+) CACHED", re.M)


def test_rebuild_without_changes_is_fully_cached(built_image):
    """A no-change rebuild must run every RUN/COPY step from the layer cache.

    This is the payoff of the layer split + `.dockerignore` hygiene (excluding
    the regenerated ``build/`` dir): rebuilding an unchanged project must not
    re-run pip, apt, or the project COPY. The module fixture primed the cache;
    build again with ``--progress=plain`` and assert every Dockerfile RUN/COPY
    step is reported CACHED.
    """
    _tag, project, _project_name, _out_dir = built_image
    rebuild_tag = f"osprey-dockerfile-e2e-cachehit:{uuid.uuid4().hex[:8]}"
    try:
        build = _build_image(project, rebuild_tag, progress_plain=True)
        progress = build.stdout + build.stderr

        steps = {
            sid: f"{kind} {rest.strip()}" for sid, kind, rest in _STEP_HEADER_RE.findall(progress)
        }
        assert steps, (
            f"no RUN/COPY step headers found in --progress=plain output — "
            f"progress format changed?\n{progress[-4000:]}"
        )
        cached = set(_CACHED_STEP_RE.findall(progress))
        uncached = {sid: instr for sid, instr in steps.items() if sid not in cached}
        assert not uncached, (
            "steps re-executed on a no-change rebuild (cache miss):\n"
            + "\n".join(f"  #{sid} {instr[:120]}" for sid, instr in sorted(uncached.items()))
            + f"\n--- progress tail ---\n{progress[-4000:]}"
        )
    finally:
        subprocess.run(["docker", "rmi", "-f", rebuild_tag], capture_output=True)


# Marker baked into the sentinel wheel; asserting it imports in the final
# image proves the staged local wheel really landed despite an equal version.
SENTINEL_MARKER = "fast-dev-rebuild-sentinel"


def _build_sentinel_wheel(version: str, work_dir: Path) -> Path:
    """Build a local osprey wheel pinned to ``version`` containing a sentinel module.

    Copies the minimal build inputs (``pyproject.toml``, ``README.md``,
    ``src/osprey``) to a scratch tree, stamps ``__version__`` to ``version``,
    injects ``osprey/_e2e_sentinel.py``, and runs ``python -m build --wheel``.
    Returns the built wheel path (inside ``work_dir``).
    """
    import osprey as _osprey

    source_root = Path(_osprey.__file__).resolve().parents[2]
    wheel_src = work_dir / "wheel-src"
    dist_dir = work_dir / "dist"
    shutil.copytree(
        source_root / "src" / "osprey",
        wheel_src / "src" / "osprey",
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
    )
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(source_root / name, wheel_src / name)

    # Stamp the version hatchling reads ([tool.hatch.version] -> __init__.py).
    init_path = wheel_src / "src" / "osprey" / "__init__.py"
    stamped, n = re.subn(
        r'__version__ = "[^"]+"', f'__version__ = "{version}"', init_path.read_text(), count=1
    )
    assert n == 1, "could not stamp __version__ in the wheel source copy"
    init_path.write_text(stamped)

    (wheel_src / "src" / "osprey" / "_e2e_sentinel.py").write_text(
        f'SENTINEL = "{SENTINEL_MARKER}"\n'
    )

    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=wheel_src,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, (
        f"sentinel wheel build failed:\n--- stdout ---\n{build.stdout[-3000:]}"
        f"\n--- stderr ---\n{build.stderr[-3000:]}"
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
    return wheels[0]


def test_equal_version_dev_wheel_lands_local_code(tmp_path):
    """Regression: a staged wheel must win even when pip sees an equal version.

    Plain ``pip install`` silently skips a wheel whose version equals the copy
    the deps layer already primed ("Requirement already satisfied") — the
    historical failure mode where ``--dev`` rebuilds shipped stale framework
    code. The wheel layer's ``--no-deps --force-reinstall`` exists to defeat
    that skip. Prove it end to end: build the image with no wheel, read the
    version the deps layer resolved, stage a locally built wheel with that
    EXACT version plus a sentinel module, rebuild, and assert the sentinel is
    importable in the final image.
    """
    project, _project_name = _render_hello_world(tmp_path)
    suffix = uuid.uuid4().hex[:8]
    base_tag = f"osprey-dockerfile-e2e-sentinel-base:{suffix}"
    dev_tag = f"osprey-dockerfile-e2e-sentinel-dev:{suffix}"
    try:
        # 1. Base build (no wheel staged): the installed osprey IS the version
        #    the deps layer resolved from OSPREY_PIP_SPEC. Note: the dist name
        #    is osprey-framework, so read osprey.__version__, not importlib
        #    metadata for "osprey".
        _build_image(project, base_tag)
        primed = _docker_run(base_tag, "python", "-c", "import osprey; print(osprey.__version__)")
        assert primed.returncode == 0, primed.stderr
        primed_version = primed.stdout.strip()
        assert primed_version, "could not read the deps-layer osprey version"

        # 2. Stage a version-equal sentinel wheel in the build context and
        #    rebuild — exactly what `osprey deploy up --dev` does.
        wheel = _build_sentinel_wheel(primed_version, tmp_path)
        shutil.copy2(wheel, project / wheel.name)
        _build_image(project, dev_tag)

        # 3. Local code landed despite the equal version.
        probe = _docker_run(
            dev_tag,
            "python",
            "-c",
            "from osprey._e2e_sentinel import SENTINEL; print(SENTINEL)",
        )
        assert probe.returncode == 0, (
            f"sentinel module missing — the staged wheel was silently skipped "
            f"(pip equal-version skip regression):\n{probe.stderr}"
        )
        assert SENTINEL_MARKER in probe.stdout

        # The version really was equal — i.e. plain pip WOULD have skipped it.
        version_after = _docker_run(
            dev_tag, "python", "-c", "import osprey; print(osprey.__version__)"
        )
        assert version_after.stdout.strip() == primed_version
    finally:
        subprocess.run(["docker", "rmi", "-f", base_tag], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", dev_tag], capture_output=True)


def test_final_image_has_no_toolchain_and_carries_project_label(built_image):
    """The deps layer's purge-in-same-RUN kept the C toolchain out of the image,
    and the build carries the ``com.osprey.project`` label the deploy path
    stamps (``_build_image`` passes it the way ``_project_image_build_cmd``
    does, so ``osprey deploy nuke`` can identify the image)."""
    tag, _project, _project_name, _out_dir = built_image

    for pkg in ("build-essential", "python3-dev"):
        check = _docker_run(tag, "sh", "-c", f"dpkg -s {pkg}")
        assert check.returncode != 0, (
            f"{pkg} is installed in the final image — the deps layer must purge "
            f"the toolchain inside the same RUN:\n{check.stdout[-1000:]}"
        )

    inspect = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "-f",
            '{{ index .Config.Labels "com.osprey.project" }}',
            tag,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert inspect.returncode == 0, inspect.stderr
    assert inspect.stdout.strip() == PROJECT_NAME, (
        f"com.osprey.project label missing or wrong: {inspect.stdout.strip()!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
