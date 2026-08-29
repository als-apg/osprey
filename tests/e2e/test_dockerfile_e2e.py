"""E2E: build, boot, and serve an agent from the generated reference Dockerfile.

Initializes a deployment repo, builds it, runs a real ``docker build`` on the
Dockerfile that ``osprey build`` generated, then exercises the image at two
depths.

What the image has to be is the whole premise here: **a deployment repo at
``/app/<project>``**. The framework inside a container resolves paths exactly
the way it does on a host — there is no container branch — so the image needs a
``profile.yml`` at the project root (``osprey web``, this image's ``CMD``,
resolves the deployment it serves by walking up to one), the render below it in
``build/`` (which is where ``registry/mcp.py`` tells every MCP server the config
lives), and that render's ``project_root`` pointing at the container path rather
than at whichever host built it. Those three are one contract, and
``test_every_config_path_in_the_image_resolves`` is the acceptance gate for it:
no MCP server may name a file the image does not have.

``test_generated_dockerfile_builds_and_boots`` — static smoke checks:

- ``osprey --version`` works (OSPREY installed and importable),
- the runtime user is the non-root ``osprey`` user,
- ``claude --version`` works as that user (canary for the /root/.local
  permission-traversal chain the native installer requires),
- ``.dockerignore`` kept ``.env`` out of the image.

``test_generated_image_serves_agent_over_http`` — full functional proof:
boots the image's actual ``CMD`` (``osprey web``), waits for ``/health``, and
drives one real agent turn through ``POST /api/chat`` with a live LLM call via
the als-apg provider. This is the only test that proves the *shipped*
entrypoint actually serves an agent — ``claude --version`` proves the binary
launches, not that the assembled image answers a prompt.

``test_node_runtime_satisfies_the_cli_engine_floor`` — the Node runtime the
image installs from its own Debian release clears the CLI's engines floor
(node >= 18, never an exact major), npm is present for the ``npx`` launch path,
and the image's size is reported so a provenance change's size delta is on the
record.

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
image. The image default is the PyPI release; CI pins the PR head SHA instead so
the image tests the branch under review.

The HTTP test additionally needs ``ALS_APG_API_KEY`` (``requires_als_apg``);
it auto-skips without it. The build/boot test has no such requirement.

Skipped entirely when docker is unavailable. Excluded from the main e2e-tests
CI job (runs in its own dockerfile-e2e job).
"""

import http.cookiejar
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.port_layout import default_port
from osprey.utils.workspace import container_image_context


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


#: The port the project image serves on INSIDE the container: the ``web`` slot
#: of the layout, which ``Dockerfile.j2`` renders into its ``EXPOSE`` line. The
#: image is built from a config that never moves ``deployment.port_base``, so
#: the layout's default base is the right one to derive it at.
_WEB_SLOT = default_port("web")

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


def _init_and_build_repo(out_dir: Path) -> Path:
    """Init a hello-world deployment repo pinned to als-apg, and build it.

    ``--no-git`` because nothing here reads history and the tmp tree may sit
    inside another repository; ``--skip-deps`` because the image installs its
    own dependencies from ``OSPREY_PIP_SPEC`` and the host venv would only slow
    the render down. Returns the repo root.
    """
    repo = out_dir / PROJECT_NAME
    runner = CliRunner()
    init = runner.invoke(
        cli,
        [
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--no-git",
        ],
    )
    assert init.exit_code == 0, init.output
    build = runner.invoke(cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert build.exit_code == 0, build.output
    return repo


def _image_context(repo: Path) -> Path:
    """The directory ``docker build`` runs against for this repo's image.

    Resolved through the production helper, not spelled here: the context is
    what lands in the image, so it is what decides the container's layout, and a
    test that picked its own would prove the layout of something the deploy path
    never builds. It is a deployment repo the build rendered against the
    ``/app/<project>`` path a container sees, so its ``Dockerfile`` is one level
    down, in the render.
    """
    context = container_image_context(repo, PROJECT_NAME)
    assert (context / "build" / "Dockerfile").is_file(), f"no rendered Dockerfile under {context}"
    return context


def _build_image(context, tag: str, *, progress_plain: bool = False) -> subprocess.CompletedProcess:
    """Run ``docker build`` on the generated Dockerfile; return the completed process.

    Passes the same ``com.osprey.project`` label ``osprey up`` stamps via
    ``_project_image_build_cmd``, so label assertions here cover the shape the
    real build path produces. ``DOCKER_BUILDKIT=1`` pins the BuildKit builder
    for every build so layer-cache semantics (and ``--progress=plain`` output,
    used by the cache-hit test) are uniform across docker versions.
    """
    build_cmd = [
        "docker",
        "build",
        *_PLATFORM_ARGS,
        "-f",
        str(Path(context) / "build" / "Dockerfile"),
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
        cwd=context,
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
    """Build the repo + its reference image once for the whole module."""
    out_dir = tmp_path_factory.mktemp("dockerfile-e2e")
    repo = _init_and_build_repo(out_dir)
    tag = f"osprey-dockerfile-e2e:{uuid.uuid4().hex[:8]}"
    try:
        _build_image(_image_context(repo), tag)
        yield tag, repo, PROJECT_NAME, out_dir
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def test_generated_dockerfile_builds_and_boots(built_image):
    tag, _repo, project_name, _out_dir = built_image

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

    # .dockerignore did its job: no secrets/host state in the image
    env_check = _docker_run(tag, "sh", "-c", f"find /app/{project_name} -name '.env' | head -1")
    assert not env_check.stdout.strip(), (
        f".env must never enter the image, found: {env_check.stdout.strip()}"
    )


def _report_image_size(tag: str) -> int:
    """Report the built image's size, and return it in bytes.

    Node now comes from the base image's own Debian release instead of a
    third-party apt repo, which moves the image size — the number belongs in
    the change that moves it. The lane runs pytest without ``-s``, so stdout
    alone would only surface on a failure; ``$GITHUB_STEP_SUMMARY`` is the
    channel that shows it on a green run, and is simply absent locally.
    """
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Size}}", tag],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    size = int(out.stdout.strip())
    line = f"generated image size: {size} bytes ({size / 1e9:.2f} GB)"
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"- {line}\n")
    return size


def test_node_runtime_satisfies_the_cli_engine_floor(built_image):
    """Node and npm work in the image, and Node clears the CLI's engines floor.

    The floor is the contract — ``@anthropic-ai/claude-code`` requires node >=
    18 — and the version above it is whatever the base image's Debian release
    ships. Asserting an exact major would be asserting a pin this image
    deliberately does not have: it would fail the day the base image moves,
    for a change that is not a regression.

    npm is checked separately because it is a *runtime* dependency here: the
    agent is launched via ``npx``, so an image with node but no npm boots and
    then fails on the first turn.
    """
    tag, _repo, _project_name, _out_dir = built_image

    node = _docker_run(tag, "node", "-v")
    assert node.returncode == 0, f"node is not runnable in the image:\n{node.stderr}"
    raw = node.stdout.strip()
    match = re.match(r"v?(\d+)\.", raw)
    assert match, f"unparseable `node -v` output: {raw!r}"
    assert int(match.group(1)) >= 18, (
        f"node {raw} is below the Claude Code CLI's engines floor (node >= 18)"
    )

    npm = _docker_run(tag, "npm", "--version")
    assert npm.returncode == 0, (
        f"npm is not runnable in the image — the agent is launched via `npx`, "
        f"so this fails at the first turn, not at build time:\n{npm.stderr}"
    )
    assert npm.stdout.strip(), "npm --version printed nothing"

    _report_image_size(tag)


# ── The container layout contract ────────────────────────────────────────────


def test_every_config_path_in_the_image_resolves(built_image):
    """Acceptance gate: no MCP server may name a file the image does not have.

    Reads the ``.mcp.json`` the image actually ships and resolves every config
    path it hands its servers, inside the image. This is the property the whole
    layout exists to deliver — an unresolvable path here means every framework
    MCP server in every container built from this image starts against a
    missing config.
    """
    tag, _repo, project_name, _out_dir = built_image
    project_root = f"/app/{project_name}"

    mcp = _docker_run(tag, "cat", f"{project_root}/build/.mcp.json")
    assert mcp.returncode == 0, (
        f"no .mcp.json under {project_root}/build — the render is not where "
        f"registry/mcp.py says it is:\n{mcp.stderr}"
    )
    servers = json.loads(mcp.stdout)["mcpServers"]

    named = {
        f"{server}.{key}": value
        for server, spec in servers.items()
        for key, value in (spec.get("env") or {}).items()
        if key in ("OSPREY_CONFIG", "CONFIG_FILE")
    }
    assert named, "no server declares a config path — the registry render changed"

    missing = {
        where: path
        for where, path in named.items()
        if _docker_run(tag, "test", "-f", path).returncode != 0
    }
    assert not missing, f"config paths named in .mcp.json but absent from the image: {missing}"


def test_image_project_root_is_a_deployment_repo(built_image):
    """``osprey web`` — this image's CMD — must be able to resolve its deployment.

    Repo discovery is the one rule, containers included: a ``profile.yml`` at
    the project root, the render in ``build/`` below it, and a ``project_root``
    that names the container path rather than the host that built the image.
    """
    tag, _repo, project_name, out_dir = built_image
    project_root = f"/app/{project_name}"

    marker = _docker_run(tag, "test", "-f", f"{project_root}/profile.yml")
    assert marker.returncode == 0, (
        f"no profile.yml at {project_root} — `osprey web` cannot resolve the "
        "deployment this image ships"
    )

    config = _docker_run(tag, "cat", f"{project_root}/build/config.yml")
    assert config.returncode == 0, config.stderr
    assert f"project_root: {project_root}" in config.stdout
    assert str(out_dir) not in config.stdout, "host build path leaked into the image config"


# ── Functional: the shipped CMD actually serves an agent ─────────────────────


def _host_port(cid: str) -> int:
    """Resolve the ephemeral host port docker mapped to the image's web slot."""
    out = subprocess.run(
        ["docker", "port", cid, f"{_WEB_SLOT}/tcp"],
        capture_output=True,
        text=True,
        timeout=15,
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


# The launcher prints its one-time login URL as `Open: http://<host>:<port>/?token=<secret>`
# (mint_and_announce). The container mints its OWN operator secret, so the token
# can only be recovered from its logs — the whole reason this parses the printed
# URL rather than assuming a bare, credential-less base URL.
_OPEN_URL_RE = re.compile(r"Open:\s*(http://\S+\?token=\S+)")


def _login_opener(base_url: str, cid: str) -> urllib.request.OpenerDirector:
    """Exchange the container's printed ``?token=`` login URL for a session cookie.

    Every ``/api/*`` route on the deployed image is now behind the web-auth gate,
    so an unauthenticated ``POST /api/chat`` is refused ``401``. A browser gets in
    by following the ``Open: …?token=…`` URL the launcher prints, which
    ``GET``s to a ``303`` that sets an ``HttpOnly`` session cookie; this does the
    same, returning a cookie-jar-backed opener that carries that session on every
    subsequent request.

    The token is read from the container's FULL logs (not the tailed view), since
    the ``Open:`` line is printed once at startup and would scroll out of a short
    tail by the time the agent turn runs.
    """
    logs = subprocess.run(["docker", "logs", cid], capture_output=True, text=True, timeout=15)
    match = _OPEN_URL_RE.search(logs.stdout + logs.stderr)
    assert match, f"container never printed an 'Open: …?token=…' login URL:\n{_container_logs(cid)}"
    login_url = match.group(1)
    # The token identifies the container's secret; the exchange itself must be
    # driven at the host-mapped base_url, so swap the authority the container
    # announced (its internal web slot) for the reachable one.
    token_query = urllib.parse.urlparse(login_url).query
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    with opener.open(f"{base_url}/?{token_query}", timeout=15) as resp:
        assert resp.status == 200, f"token exchange did not resolve to 200 (got {resp.status})"
    return opener


def _post_chat(
    base_url: str, opener: urllib.request.OpenerDirector, prompt: str, timeout: float
) -> dict:
    """POST a prompt to the buffered chat endpoint, return the JSON body.

    Sent through the cookie-bearing *opener* from :func:`_login_opener` with a
    same-origin ``Origin`` header: ``/api/chat`` is a mutating cookie-authenticated
    route, so the web-auth gate refuses it ``403`` without an ``Origin`` matching
    the app's external origin (derived from the request ``Host`` = *base_url*).
    """
    # chat_id is a required field — it keys the server-side ChatSessionPool.
    # A single fixed id is enough for this one-turn smoke test.
    req = urllib.request.Request(
        f"{base_url}/api/chat?stream=false",
        data=json.dumps({"prompt": prompt, "chat_id": "e2e"}).encode(),
        headers={"Content-Type": "application/json", "Origin": base_url},
        method="POST",
    )
    with opener.open(req, timeout=timeout) as resp:
        assert resp.status == 200, f"chat returned HTTP {resp.status}"
        return json.loads(resp.read())


@pytest.mark.requires_als_apg
def test_generated_image_serves_agent_over_http(built_image):
    """Boot the image's CMD and drive one real LLM turn through osprey web."""
    tag, _repo, _project_name, _out_dir = built_image

    # Mirror the production run contract (`docker run --env-file .env`): pass
    # the raw provider secret, never a pre-resolved token. The full osprey web
    # stack resolves ${ALS_APG_API_KEY} from config at runtime and stands up
    # an internal proxy that authenticates upstream with it — injecting a
    # pre-resolved ANTHROPIC_AUTH_TOKEN would bypass that and leave config
    # resolution (and the proxy) without the key.
    api_key = os.environ["ALS_APG_API_KEY"]  # guaranteed present by requires_als_apg
    env_args = ["-e", f"ALS_APG_API_KEY={api_key}"]

    # The endpoint override rides the same contract. It is deliberately absent
    # from the image (build-time rendering never bakes the builder's
    # environment), so a run pointed at a non-default gateway must pass it in
    # here exactly as a deployment passes it via env_file — otherwise the
    # container dials the provider's built-in default instead.
    base_url = os.environ.get("ALS_APG_BASE_URL")
    if base_url:
        env_args += ["-e", f"ALS_APG_BASE_URL={base_url}"]

    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            *_PLATFORM_ARGS,
            "-p",
            f"127.0.0.1:0:{_WEB_SLOT}",
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
            opener = _login_opener(base_url, cid)
            data = _post_chat(base_url, opener, CHAT_PROMPT, CHAT_TIMEOUT)
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
        # The buffered chat payload is reduced to {text, events, is_error} —
        # turn counts are deliberately withheld from the chat client — so a
        # completed turn is evidenced by its terminal result event.
        events = data.get("events") or []
        assert any(e.get("type") == "result" for e in events), (
            f"expected a completed agent turn, got events: {events}"
        )
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
    _tag, repo, _project_name, _out_dir = built_image
    rebuild_tag = f"osprey-dockerfile-e2e-cachehit:{uuid.uuid4().hex[:8]}"
    try:
        build = _build_image(_image_context(repo), rebuild_tag, progress_plain=True)
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
    ``src/osprey``) to a scratch tree, injects ``osprey/_e2e_sentinel.py``, and
    runs ``python -m build --wheel``. Returns the built wheel path (inside
    ``work_dir``).

    The scratch tree has no ``.git``, and the version is derived from git rather
    than from a literal, so the build is pinned with
    ``SETUPTOOLS_SCM_PRETEND_VERSION``. The per-package
    ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*`` form does not take here — do not
    substitute it without re-verifying against the ``osprey-framework`` dist name.
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

    (wheel_src / "src" / "osprey" / "_e2e_sentinel.py").write_text(
        f'SENTINEL = "{SENTINEL_MARKER}"\n'
    )

    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=wheel_src,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": version},
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
    context = _image_context(_init_and_build_repo(tmp_path))
    suffix = uuid.uuid4().hex[:8]
    base_tag = f"osprey-dockerfile-e2e-sentinel-base:{suffix}"
    dev_tag = f"osprey-dockerfile-e2e-sentinel-dev:{suffix}"
    try:
        # 1. Base build (no wheel staged): the installed osprey IS the version
        #    the deps layer resolved from OSPREY_PIP_SPEC. Note: the dist name
        #    is osprey-framework, so read osprey.__version__, not importlib
        #    metadata for "osprey".
        _build_image(context, base_tag)
        primed = _docker_run(base_tag, "python", "-c", "import osprey; print(osprey.__version__)")
        assert primed.returncode == 0, primed.stderr
        primed_version = primed.stdout.strip()
        assert primed_version, "could not read the deps-layer osprey version"

        # 2. Stage a version-equal sentinel wheel in the build context and
        #    rebuild — exactly what `osprey up --dev` does.
        wheel = _build_sentinel_wheel(primed_version, tmp_path)
        shutil.copy2(wheel, context / wheel.name)
        _build_image(context, dev_tag)

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
    does, so ``osprey reset`` can identify the image)."""
    tag, _repo, _project_name, _out_dir = built_image

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
