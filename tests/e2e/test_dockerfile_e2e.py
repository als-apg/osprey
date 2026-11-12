"""E2E: build and boot the generated reference Dockerfile, without a model.

Initializes a deployment repo, builds it, runs a real ``docker build`` on the
Dockerfile that ``osprey build`` generated (``_dockerfile_image``), then
inspects the image.

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

The one test that drives an agent turn through the image's ``CMD`` spends
model tokens, so it lives in ``test_dockerfile_agent_e2e.py`` (its own
model-spending CI lane); this module calls no model.

``test_node_runtime_satisfies_the_cli_engine_floor`` — the Node runtime the
image copies from the official Node image clears the CLI's engines floor
(node >= 22, never an exact major), npm is present for the ``npx`` launch path,
and the image's size is reported so a provenance change's size delta is on the
record.

The ``test_whoami_*`` tests cover the entrypoint's control-identity step, with
this tree's ``control_identity`` script and its catalog closure staged into
one directory and bound read-only where compose binds it:
an applied identity is what ``whoami`` prints and the only thing on stdout; a
person identity under ``--user`` refuses to start; an ``osprey-*`` identity
under ``--user`` starts with a warning; a root start without the module dies,
for a service identity as for a person.

Three further tests cover the fast-dev-rebuild layer split:

- ``test_rebuild_without_changes_is_fully_cached`` — a no-change rebuild runs
  every RUN/COPY step from the layer cache,
- ``test_equal_version_dev_wheel_lands_local_code`` — a staged dev wheel whose
  version equals the deps-layer copy still lands its code (pip's silent
  equal-version skip is defeated by ``--no-deps --force-reinstall``),
- ``test_final_image_has_no_toolchain_and_carries_project_label`` — the C
  toolchain is purged from the final image and the ``com.osprey.project``
  label is stamped.

The smoke/cache/hygiene tests share one image build (the module-scoped
``built_image`` fixture); the sentinel test builds its own. How the image is
built, and ``OSPREY_E2E_PIP_SPEC``, are described in ``_dockerfile_image``.

Skipped entirely when docker is unavailable. Excluded from the main e2e-tests
CI job (runs in its own dockerfile-e2e job, which holds no model key).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from osprey.agent_runner.build_artifacts import BuildArtifactCatalog
from osprey.cli.templates.manager import TemplateManager
from osprey.deployment.control_identity import (
    CONTROL_IDENTITY_CONTAINER_DIR,
    CONTROL_IDENTITY_CONTAINER_PATH,
)
from tests.e2e import _dockerfile_image as image
from tests.e2e._dockerfile_image import (
    PLATFORM_ARGS,
    PROJECT_NAME,
    RUN_TIMEOUT,
    build_image,
    docker_run,
    image_context,
    init_and_build_repo,
)

pytestmark = image.MODULE_MARKS


@pytest.fixture(scope="module")
def built_image(tmp_path_factory):
    """Build the repo + its reference image once for the whole module."""
    yield from image.built_image(tmp_path_factory)


def test_generated_dockerfile_builds_and_boots(built_image):
    tag, _repo, project_name, _out_dir = built_image

    # OSPREY installed and importable
    version = docker_run(tag, "osprey", "--version")
    assert version.returncode == 0, version.stderr

    # The command runs as uid 1000, whose name is one of two answers: `osprey`
    # when OSPREY_CONTROL_IDENTITY is unset (this run), or the name that
    # variable carries once the entrypoint has applied it (the test_whoami_*
    # tests below). Never root either way.
    whoami = docker_run(tag, "whoami")
    assert whoami.stdout.strip() == "osprey"

    # Claude Code callable as the non-root user (permission-chain canary)
    claude = docker_run(tag, "claude", "--version")
    assert claude.returncode == 0, (
        f"claude --version failed as non-root user — the /root/.local "
        f"traversal chain is broken:\n{claude.stderr}"
    )

    # .dockerignore did its job: no secrets/host state in the image
    env_check = docker_run(tag, "sh", "-c", f"find /app/{project_name} -name '.env' | head -1")
    assert not env_check.stdout.strip(), (
        f".env must never enter the image, found: {env_check.stdout.strip()}"
    )


def _docker_run_with(tag: str, docker_args: list[str], *cmd: str) -> subprocess.CompletedProcess:
    """``docker_run`` with ``docker run`` options (``-e``, ``-v``, ``--user``) before the tag."""
    return subprocess.run(
        ["docker", "run", "--rm", *PLATFORM_ARGS, *docker_args, tag, *cmd],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )


def _identity_env(name: str) -> list[str]:
    """The two variables compose sets on an identity-carrying service."""
    return [
        "-e",
        f"OSPREY_CONTROL_IDENTITY={name}",
        "-e",
        f"OSPREY_CONTROL_IDENTITY_MODULE={CONTROL_IDENTITY_CONTAINER_PATH}",
    ]


@pytest.fixture(scope="module")
def identity_mount(tmp_path_factory) -> list[str]:
    """Stage this tree's identity script with its catalog closure, and bind it read-only.

    The image installs OSPREY from PyPI by default, so the files under test are
    this tree's, staged into one directory and bound the way a deployment binds
    its staged directory: the script's sibling import resolves from that
    directory alone.
    """
    staged = tmp_path_factory.mktemp("control-identity")
    files = BuildArtifactCatalog.default().files_beside(
        "control-identity", TemplateManager().template_root
    )
    for name, source in files.items():
        (staged / name).write_bytes(source.read_bytes())
    return ["-v", f"{staged}:{CONTROL_IDENTITY_CONTAINER_DIR}:ro"]


def _streams(proc: subprocess.CompletedProcess) -> str:
    return f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"


def test_whoami_prints_the_applied_identity(built_image, identity_mount):
    tag = built_image[0]
    run = _docker_run_with(tag, [*_identity_env("alice"), *identity_mount], "whoami")
    assert run.returncode == 0, _streams(run)
    assert run.stdout.strip() == "alice", _streams(run)
    assert "control identity: uid 1000 is 'alice'" in run.stderr, _streams(run)


def test_whoami_stdout_carries_only_the_name(built_image, identity_mount):
    """Every entrypoint diagnostic, the module's own output included, is on stderr."""
    tag = built_image[0]
    run = _docker_run_with(tag, [*_identity_env("alice"), *identity_mount], "whoami")
    assert run.returncode == 0, _streams(run)
    assert run.stdout == "alice\n", _streams(run)
    assert "[osprey-entrypoint]" in run.stderr, _streams(run)


def test_whoami_person_identity_under_user_refuses_to_start(built_image, identity_mount):
    tag = built_image[0]
    run = _docker_run_with(
        tag,
        ["--user", "1000:1000", *_identity_env("alice"), *identity_mount],
        "whoami",
    )
    assert run.returncode != 0, _streams(run)
    assert run.stdout == "", _streams(run)
    assert "needs a root start" in run.stderr, _streams(run)


def test_whoami_service_identity_under_user_starts_with_a_warning(built_image, identity_mount):
    tag = built_image[0]
    run = _docker_run_with(
        tag,
        ["--user", "1000:1000", *_identity_env("osprey-dispatch-1"), *identity_mount],
        "sh",
        "-c",
        'whoami; printf "%s\\n" "$OSPREY_CONTROL_IDENTITY_SKIPPED"',
    )
    assert run.returncode == 0, _streams(run)
    # Not applied: uid 1000 keeps the image's name, and the skip is exported
    # for /health to report.
    assert run.stdout == "osprey\nnon-root-start\n", _streams(run)
    assert "WARNING: control identity 'osprey-dispatch-1' needs a root start" in run.stderr, (
        _streams(run)
    )
    assert "control identity 'osprey-dispatch-1' not applied, continuing" in run.stderr, _streams(
        run
    )


def test_whoami_person_identity_root_start_without_the_module_dies(built_image):
    tag = built_image[0]
    run = _docker_run_with(tag, _identity_env("alice"), "whoami")
    assert run.returncode != 0, _streams(run)
    assert run.stdout == "", _streams(run)
    assert "FATAL" in run.stderr, _streams(run)
    assert f"{CONTROL_IDENTITY_CONTAINER_PATH} is missing or unreadable" in run.stderr, _streams(
        run
    )


def test_whoami_service_identity_root_start_without_the_module_starts_with_a_warning(
    built_image,
):
    tag = built_image[0]
    run = _docker_run_with(
        tag,
        _identity_env("osprey-dispatch-1"),
        "sh",
        "-c",
        'whoami; printf "%s\\n" "$OSPREY_CONTROL_IDENTITY_SKIPPED"',
    )
    assert run.returncode == 0, _streams(run)
    assert run.stdout == "osprey\napply-failed\n", _streams(run)
    assert f"{CONTROL_IDENTITY_CONTAINER_PATH} is missing or unreadable" in run.stderr, _streams(
        run
    )
    assert "control identity 'osprey-dispatch-1' not applied, continuing" in run.stderr, _streams(
        run
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

    The floor is the contract — the pinned ``@anthropic-ai/claude-code``
    declares node >= 22 — and the version above it is whatever the Node image
    the recipe copies from ships. Asserting an exact major would fail the day
    that image moves, for a change that is not a regression.

    npm is checked separately because it is a *runtime* dependency here: the
    agent is launched via ``npx``, so an image with node but no npm boots and
    then fails on the first turn.
    """
    tag, _repo, _project_name, _out_dir = built_image

    node = docker_run(tag, "node", "-v")
    assert node.returncode == 0, f"node is not runnable in the image:\n{node.stderr}"
    raw = node.stdout.strip()
    match = re.match(r"v?(\d+)\.", raw)
    assert match, f"unparseable `node -v` output: {raw!r}"
    assert int(match.group(1)) >= 22, (
        f"node {raw} is below the Claude Code CLI's engines floor (node >= 22)"
    )

    npm = docker_run(tag, "npm", "--version")
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

    mcp = docker_run(tag, "cat", f"{project_root}/build/.mcp.json")
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
        if docker_run(tag, "test", "-f", path).returncode != 0
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

    marker = docker_run(tag, "test", "-f", f"{project_root}/profile.yml")
    assert marker.returncode == 0, (
        f"no profile.yml at {project_root} — `osprey web` cannot resolve the "
        "deployment this image ships"
    )

    config = docker_run(tag, "cat", f"{project_root}/build/config.yml")
    assert config.returncode == 0, config.stderr
    assert f"project_root: {project_root}" in config.stdout
    assert str(out_dir) not in config.stdout, "host build path leaked into the image config"


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
        build = build_image(image_context(repo), rebuild_tag, progress_plain=True)
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
        subprocess.run(["docker", "rmi", "-f", rebuild_tag], capture_output=True, timeout=120)


# Marker baked into the sentinel wheel; asserting it imports in the final
# image proves the staged local wheel really landed despite an equal version.
SENTINEL_MARKER = "fast-dev-rebuild-sentinel"


def _build_sentinel_wheel(version: str, work_dir: Path) -> Path:
    """Build a local osprey wheel pinned to ``version`` containing a sentinel module.

    Copies the minimal build inputs (``pyproject.toml``, ``README.md`` and
    ``src/osprey``) to a scratch tree,
    injects ``osprey/_e2e_sentinel.py``, and runs ``python -m build --wheel``.
    Returns the built wheel path (inside ``work_dir``).

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
    context = image_context(init_and_build_repo(tmp_path))
    suffix = uuid.uuid4().hex[:8]
    base_tag = f"osprey-dockerfile-e2e-sentinel-base:{suffix}"
    dev_tag = f"osprey-dockerfile-e2e-sentinel-dev:{suffix}"
    try:
        # 1. Base build (no wheel staged): the installed osprey IS the version
        #    the deps layer resolved from OSPREY_PIP_SPEC. Note: the dist name
        #    is osprey-framework, so read osprey.__version__, not importlib
        #    metadata for "osprey".
        build_image(context, base_tag)
        primed = docker_run(base_tag, "python", "-c", "import osprey; print(osprey.__version__)")
        assert primed.returncode == 0, primed.stderr
        primed_version = primed.stdout.strip()
        assert primed_version, "could not read the deps-layer osprey version"

        # 2. Stage a version-equal sentinel wheel in the build context and
        #    rebuild — exactly what `osprey up --dev` does.
        wheel = _build_sentinel_wheel(primed_version, tmp_path)
        shutil.copy2(wheel, context / wheel.name)
        build_image(context, dev_tag)

        # 3. Local code landed despite the equal version.
        probe = docker_run(
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
        version_after = docker_run(
            dev_tag, "python", "-c", "import osprey; print(osprey.__version__)"
        )
        assert version_after.stdout.strip() == primed_version
    finally:
        subprocess.run(["docker", "rmi", "-f", base_tag], capture_output=True, timeout=120)
        subprocess.run(["docker", "rmi", "-f", dev_tag], capture_output=True, timeout=120)


def test_final_image_has_no_toolchain_and_carries_project_label(built_image):
    """The deps layer's purge-in-same-RUN kept the C toolchain out of the image,
    and the build carries the ``com.osprey.project`` label the deploy path
    stamps (``_build_image`` passes it the way ``_project_image_build_cmd``
    does, so ``osprey reset`` can identify the image)."""
    tag, _repo, _project_name, _out_dir = built_image

    for pkg in ("build-essential", "python3-dev"):
        check = docker_run(tag, "sh", "-c", f"dpkg -s {pkg}")
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
