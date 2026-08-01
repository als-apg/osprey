"""E2E: a built project's Python environment is the same on the host and in the image.

This is the instrument for the two bugs the honest-execution-config work exists
to fix, and it is deliberately built so that either bug reappearing fails a test
rather than passing quietly.

**Drift.** A profile declares ``environment.python`` pointing at an existing
venv plus one ``environment.packages`` addition. ``uv venv --python
<venv>/bin/python`` does *not* inherit the base venv's packages, so
``freeze_base_environment`` pins them into the project's dependency record
instead. The build therefore has to carry two distinct things into the project —
a package the base venv happened to hold, and a package the profile explicitly
asked for — and both have to survive the trip into the container image. The
reported symptom was a package importable on the host and missing inside a
deployed web-terminal container; :func:`test_container_environment_matches_host`
is that symptom's direct test.

**Execution.** Agent-authored Python ran against a Jupyter kernel gateway OSPREY
never shipped, so ``mcp__python__execute`` failed out of the box.
:func:`test_execute_succeeds_on_host_build` and
:func:`test_execute_succeeds_in_deployed_container` drive the real tool function
— safety checks, adapter, ``ExecutionWrapper`` subprocess, response builder — on
both sides of the host/container boundary and demand a successful run of
``print(2+2)``.

Every assertion here is built to be able to fail:

- ``MARKER_ABSENT`` is a real, installable PyPI distribution that the profile
  never mentions. It must be *un*-importable on both sides, which is what proves
  the probe reports absence rather than always reporting presence — a parity
  test that cannot see a missing package would certify drift instead of
  catching it.
- The host-side assertions run first, so the container comparison can never pass
  by comparing two empty sets.
- The frozen marker's *version* is compared across the boundary, not just its
  name, so a pin that is dropped and re-resolved is still caught.
- Each execution test also runs code that must fail, so "success" cannot be a
  driver that reports success unconditionally.

Requires a container runtime and an editable OSPREY checkout (the image is built
against a wheel staged from this working tree — the same mechanism ``osprey
deploy up --dev`` uses — because an image running the released PyPI build would
measure released code, not this branch). Skips cleanly without either.
"""

import json
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

# ── Markers ──────────────────────────────────────────────────────────────────


def _docker_installed() -> bool:
    """Whether a docker binary exists at all — the only skip-worthy condition.

    Deliberately NOT a ``docker info`` probe. A daemon that is present but
    erroring, or merely slow on a loaded runner, would read as "docker not
    available" and turn infrastructure breakage into a green skip. The only CI
    lane for this file is a docker-guaranteed ubuntu-latest job, where a broken
    daemon is a fault to surface rather than an absence to tolerate — so let the
    first ``docker build`` fail loudly and print its log instead.
    """
    return shutil.which("docker") is not None


# harness_benchmark, not agentic_benchmark: no model runs here at all. Every
# assertion is model-independent OSPREY plumbing (see tests/e2e/README.md).
pytestmark = [
    pytest.mark.harness_benchmark,
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_installed(), reason="no docker binary on PATH"),
]

# ── Package markers ──────────────────────────────────────────────────────────
#
# All three are tiny, pure-Python, zero-dependency, index-resolvable, and absent
# from OSPREY's dependency tree — so each one's presence is attributable to
# exactly one mechanism and nothing else can install it by accident.
#
# lmfit (the package from the original bug report) was rejected: it pulls numpy
# and scipy, both of which OSPREY pins. `freeze_base_environment` fails a build
# whose base venv conflicts with those pins, so lmfit would make the fixture's
# success depend on transitive-resolution luck rather than on the mechanism
# NONE OF THE THREE MAY BE ADDED TO OSPREY'S OWN DEV DEPENDENCIES. Under its
# litellm-quarantine fallback, `_create_project_venv` writes an
# `_osprey_build_env.pth` into the project venv pointing at the BUILD
# environment's site-packages (build_environment.py:940-950), so when that
# branch fires "importable in the project venv" is a union, not "installed
# here". The markers are absent from the repo venv today (verified), but a
# marker that entered OSPREY's dev deps would make the frozen-marker assertion
# below silently stop testing the freeze.
#
# under test. What the test needs is a package whose *only* possible route into
# the project is the freeze; a leaf package is the honest choice.

MARKER_FROZEN = "inflection"
"""Installed ONLY into the fixture's base venv. Reaches the project via the
freeze — nothing else can put it there."""

MARKER_ADDED = "humanize"
"""Declared ONLY in ``environment.packages``. Reaches the project via the
profile — it is absent from the base venv."""

MARKER_ABSENT = "pyfiglet"
"""Declared nowhere. The negative control: it must be missing on BOTH sides, so
a probe that always answered "importable" fails instead of certifying parity."""

# (distribution name, module name) — identical for all three markers; osprey is
# the anchor whose distribution name differs from its import name.
PROBE_TARGETS = [
    ("osprey-framework", "osprey"),
    (MARKER_FROZEN, MARKER_FROZEN),
    (MARKER_ADDED, MARKER_ADDED),
    (MARKER_ABSENT, MARKER_ABSENT),
]

PROJECT_NAME = "parityproj"
CONTAINER_PROJECT_ROOT = f"/app/{PROJECT_NAME}"

# Must stay comfortably BELOW the CI job's `timeout-minutes`, which also has to
# cover checkout, `uv sync --extra dev`, the wheel build and four container
# runs. If GitHub's cap fires first the job dies with no output; this timeout
# fires into `assert build.returncode == 0`, which prints the docker log —
# exactly when the diagnostic is needed most.
BUILD_TIMEOUT = 1200
PROBE_TIMEOUT = 300
EXEC_TIMEOUT = 600

SENTINEL = "###OSPREY-PARITY###"

# Prefix the agent-code probe stamps its interpreter with. The wrapper prints
# its own preamble lines, so the value is picked out by prefix rather than by
# position.
AGENT_PYTHON_PREFIX = "OSPREY-AGENT-PYTHON="

# OSPREY's dependency chain (accelerator-toolbox) ships linux/amd64 wheels only;
# on arm64 hosts a native build would need a compiler the slim base lacks. Build
# and run amd64 under emulation instead — it matches the real deploy targets.
_PLATFORM_ARGS = (
    ["--platform", "linux/amd64"] if platform.machine().lower() in ("arm64", "aarch64") else []
)

# ── Probes (run under a target interpreter, host or container) ───────────────

_PACKAGE_PROBE = f"""
import importlib
import importlib.metadata as md
import json
import sys

out = {{}}
for spec in sys.argv[1:]:
    dist, _, module = spec.partition(":")
    try:
        version = md.version(dist)
    except Exception:
        version = None
    if module:
        try:
            importlib.import_module(module)
            importable = True
        except Exception:
            importable = False
    else:
        # No import name supplied — distribution presence is the whole question
        # (a recorded dependency's import name need not match its dist name).
        importable = None
    out[dist] = {{"importable": importable, "version": version}}
print({SENTINEL!r} + json.dumps(out) + {SENTINEL!r})
"""

# Drives the real mcp__python__execute path: the tool function itself (safety
# checks -> adapter -> ExecutionWrapper subprocess -> response builder), plus a
# direct adapter call so the raw `ExecutionResult.success` flag is observable,
# plus code that MUST fail so a driver that reports success unconditionally
# cannot pass.
_EXEC_DRIVER = f"""
import asyncio
import json


async def main():
    from osprey.mcp_server.python_executor.executor import execute_code
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    out = {{}}

    ok = await execute_code(
        code="print(2+2)", execution_mode="readonly", description="parity: arithmetic"
    )
    out["adapter"] = {{
        "success": ok.success,
        "stdout": ok.stdout,
        "execution_method_used": ok.execution_method_used,
        "error_message": ok.error_message,
    }}

    # The seam between the two bugs: agent code must be able to import the
    # packages the build put in the environment. `print(2+2)` cannot see this
    # — the wrapper self-heals sys.path and try/excepts numpy/pandas/matplotlib
    # and the registry, so it succeeds under essentially any Python 3. Importing
    # the markers, and reporting the interpreter that DID the importing, is what
    # ties the package assertions to the execution assertions.
    agent_env = await execute_code(
        code=(
            "import sys, {MARKER_FROZEN}, {MARKER_ADDED}\\n"
            "print('{AGENT_PYTHON_PREFIX}' + sys.executable)"
        ),
        execution_mode="readonly",
        description="parity: agent-visible packages",
    )
    out["agent_env"] = {{
        "success": agent_env.success,
        "stdout": agent_env.stdout,
        "stderr": (agent_env.stderr or "")[-600:],
        "error_message": agent_env.error_message,
    }}

    canary = await execute_code(
        code="raise RuntimeError('parity-canary')",
        execution_mode="readonly",
        description="parity: canary",
    )
    out["canary"] = {{"success": canary.success, "stderr": (canary.stderr or "")[-400:]}}

    fn = getattr(execute, "fn", execute)
    result = await fn(code="print(2+2)", description="parity: tool")
    payload = None
    if isinstance(result, str):
        payload = json.loads(result)
    else:
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                payload = json.loads(text)
                break
            except ValueError:
                continue
    out["tool"] = payload

    print({SENTINEL!r} + json.dumps(out, default=str) + {SENTINEL!r})


asyncio.run(main())
"""


def _parse_sentinel(completed: subprocess.CompletedProcess, what: str) -> dict:
    """Pull the sentinel-delimited JSON payload out of a driver's stdout."""
    combined = completed.stdout or ""
    parts = combined.split(SENTINEL)
    assert len(parts) >= 3, (
        f"{what}: no sentinel payload (rc={completed.returncode})\n"
        f"--- stdout ---\n{combined[-3000:]}\n--- stderr ---\n{(completed.stderr or '')[-3000:]}"
    )
    return json.loads(parts[1])


def _probe_packages(
    argv: list[str], *, specs: list[str] | None = None, env: dict | None = None, what: str
) -> dict:
    """Run the package probe via *argv* (host interpreter or ``docker run``)."""
    if specs is None:
        specs = [f"{dist}:{module}" for dist, module in PROBE_TARGETS]
    completed = subprocess.run(
        [*argv, "-c", _PACKAGE_PROBE, *specs],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT,
        env=env,
    )
    return _parse_sentinel(completed, what)


def _importable(probe: dict) -> set[str]:
    return {name for name, info in probe.items() if info["importable"]}


def _recorded_dependencies(project: Path) -> dict[str, str | None]:
    """Distributions the build recorded in ``pyproject.toml`` → pinned version or None.

    This is the set the build promises the project's environment contains, and
    therefore the set an image built from the project has to reproduce. Reading
    it back (rather than hard-coding the markers) keeps the parity assertion
    honest for any profile: whatever the build recorded, the container must have.
    """
    import tomllib

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    data = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    recorded: dict[str, str | None] = {}
    for raw in data["project"]["dependencies"]:
        req = Requirement(raw)
        pinned = None
        for spec in req.specifier:
            if spec.operator == "==":
                pinned = spec.version
        recorded[canonicalize_name(req.name)] = pinned
    return recorded


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_venv(tmp_path_factory) -> Path:
    """A venv holding exactly one distribution — the frozen marker.

    Exactly one, because ``uv venv`` seeds no pip/setuptools: the freeze's
    output is then fully determined, and no incidental package can collide with
    OSPREY's own pins and fail the build for reasons unrelated to this test.
    """
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the fixture's base venv")

    venv = tmp_path_factory.mktemp("parity-base") / "venv"
    create = subprocess.run(
        [uv, "venv", str(venv), "--quiet"], capture_output=True, text=True, timeout=120
    )
    assert create.returncode == 0, f"uv venv failed: {create.stdout}{create.stderr}"

    interpreter = venv / "bin" / "python"
    install = subprocess.run(
        [uv, "pip", "install", "--python", str(interpreter), "--quiet", MARKER_FROZEN],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert install.returncode == 0, f"seeding base venv failed: {install.stdout}{install.stderr}"

    # The fixture's own premise, asserted rather than assumed: the marker is in
    # the BASE venv. Everything downstream is about whether it travels.
    probe = _probe_packages([str(interpreter)], what="base venv")
    assert probe[MARKER_FROZEN]["importable"], "base venv did not receive the frozen marker"
    assert not probe[MARKER_ADDED]["importable"], (
        f"{MARKER_ADDED} must NOT be in the base venv — it is supposed to arrive "
        "via environment.packages, and this test could not tell the two routes apart"
    )
    return interpreter


@pytest.fixture(scope="module")
def built_project(base_venv, tmp_path_factory) -> Path:
    """A real ``osprey build`` whose profile declares the venv base + one addition."""
    out_dir = tmp_path_factory.mktemp("parity-build")
    override = out_dir / "environment-override.yml"
    override.write_text(
        "environment:\n"
        f"  python: {base_venv}\n"
        "  packages:\n"
        f"    - {MARKER_ADDED}\n"
        "config:\n"
        # The gallery auto-launches a web server on a fixed host port on first
        # artifact save. Nothing here inspects the gallery, and a shared CI or
        # dev host may already hold that port — turn it off on BOTH sides so the
        # comparison stays about packages and execution.
        "  artifact_server.auto_launch: false\n",
        encoding="utf-8",
    )

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
            "-O",
            str(override),
            "--skip-lifecycle",
            "-o",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    project = out_dir / PROJECT_NAME
    assert (project / ".venv" / "bin" / "python").exists(), "build produced no project venv"
    assert (project / "Dockerfile").exists()
    return project


@pytest.fixture(scope="module")
def host_probe(built_project) -> dict:
    """Package probe run under the project venv — the host side of the comparison."""
    return _probe_packages(
        [str(built_project / ".venv" / "bin" / "python")], what="host project venv"
    )


@pytest.fixture(scope="module")
def built_image(built_project):
    """Build the project's own generated Dockerfile into a throwaway image.

    The image is built against a wheel staged from this working tree (the
    mechanism ``osprey deploy up --dev`` uses). Without it the deps layer would
    install the released PyPI build and every container assertion below would
    describe released code rather than the branch under test.

    Skipping is confined to the environment preconditions ``preflight_dev_mode``
    checks — osprey not installed editable, no source root, no ``build``
    package. Those describe a host that cannot run this test at all. Everything
    after that point (the wheel build itself, the manifest write, the image
    build) is a real breakage and must FAIL: a broken wheel build reported as a
    skip would be a green run that proved nothing, the same category of failure
    as a parity check that cannot detect drift.
    """
    from osprey.deployment.errors import DevModeUnavailableError
    from osprey.deployment.wheel_build import (
        LOCAL_REQUIREMENTS_FILENAME,
        _copy_local_framework_for_override,
        preflight_dev_mode,
    )

    try:
        preflight_dev_mode()
    except DevModeUnavailableError as exc:
        pytest.skip(f"host cannot stage a local osprey wheel: {exc}")

    # Deliberately NOT wrapped: _copy_local_framework_for_override raises
    # DevModeUnavailableError for staging failures too (a failed wheel build,
    # an unwritable context), and those must surface as errors, not skips.
    _copy_local_framework_for_override(str(built_project))

    staged = [built_project / LOCAL_REQUIREMENTS_FILENAME, *built_project.glob("*.whl")]
    tag = f"osprey-parity-e2e:{uuid.uuid4().hex[:8]}"
    try:
        build = subprocess.run(
            [
                "docker",
                "build",
                *_PLATFORM_ARGS,
                "-t",
                tag,
                "--label",
                f"com.osprey.project={PROJECT_NAME}",
                "--build-arg",
                "OSPREY_DEV=1",
                ".",
            ],
            cwd=built_project,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
        )
        assert build.returncode == 0, (
            f"docker build failed:\n--- stdout ---\n{build.stdout[-4000:]}"
            f"\n--- stderr ---\n{build.stderr[-4000:]}"
        )
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
        for artifact in staged:
            # Staged dev artifacts must not outlive the build that needed them:
            # a leftover wheel would silently reinstall osprey in any later
            # image built from this context.
            try:
                artifact.unlink()
            except OSError:
                pass


def _docker_run_argv(tag: str) -> list[str]:
    """``docker run`` argv that lands in the project root with config resolved."""
    return [
        "docker",
        "run",
        "--rm",
        *_PLATFORM_ARGS,
        "-w",
        CONTAINER_PROJECT_ROOT,
        "-e",
        f"OSPREY_CONFIG={CONTAINER_PROJECT_ROOT}/config.yml",
        tag,
        "python",
    ]


# ── 1-2. The project venv carries both the frozen base and the addition ──────


def test_project_venv_imports_frozen_base_and_added_package(built_project, host_probe):
    """The built venv holds the base venv's package AND the profile's addition."""
    assert host_probe[MARKER_FROZEN]["importable"], (
        f"{MARKER_FROZEN} was installed in the declared base venv but is not "
        f"importable in the project venv — the freeze did not carry it "
        f"({host_probe[MARKER_FROZEN]})"
    )
    assert host_probe[MARKER_ADDED]["importable"], (
        f"{MARKER_ADDED} was declared in environment.packages but is not "
        f"importable in the project venv ({host_probe[MARKER_ADDED]})"
    )
    assert host_probe["osprey-framework"]["importable"], "project venv cannot import osprey"

    # The negative control, on the side where we control the inputs completely:
    # if this were importable, the probe would be reporting presence for a
    # package nothing installed, and every parity assertion below would be void.
    assert not host_probe[MARKER_ABSENT]["importable"], (
        f"{MARKER_ABSENT} is importable in the project venv but nothing declares "
        "it — the probe cannot distinguish present from absent"
    )

    # The freeze records a pin, not a bare name. Assert the recorded artifact
    # says so, so a rebuild elsewhere resolves the same version.
    pyproject = (built_project / "pyproject.toml").read_text(encoding="utf-8")
    frozen_version = host_probe[MARKER_FROZEN]["version"]
    assert f'"{MARKER_FROZEN}=={frozen_version}"' in pyproject, (
        f"pyproject.toml does not pin {MARKER_FROZEN}=={frozen_version}:\n{pyproject}"
    )
    assert MARKER_ADDED in pyproject, f"pyproject.toml does not record {MARKER_ADDED}"


# ── 3. The image's environment matches the host's ────────────────────────────


def test_container_environment_matches_host(built_project, built_image, host_probe):
    """The image reproduces the project venv's distribution set.

    The drift bug's direct test: the reported failure was a package importable
    on the host and missing inside the deployed container.

    Two comparisons, deliberately both:

    1. the markers, by *import* — the mechanism-level check (did the freeze and
       ``environment.packages`` reach the image at all);
    2. every distribution the build recorded in ``pyproject.toml``, by
       distribution presence and by pinned version — the contract-level check,
       read back from the record rather than hard-coded, so it holds for any
       profile and not just this fixture's.
    """
    container_probe = _probe_packages(_docker_run_argv(built_image), what="container")

    # Non-vacuity, restated at the point of comparison: the host side must
    # actually carry the markers, so this can never be an equality of empties.
    host_importable = _importable(host_probe)
    assert {MARKER_FROZEN, MARKER_ADDED, "osprey-framework"} <= host_importable, (
        f"host probe is missing markers, nothing to compare: {host_probe}"
    )

    # And the container probe must be able to report absence.
    assert not container_probe[MARKER_ABSENT]["importable"], (
        f"{MARKER_ABSENT} is importable in the container but nothing installs it "
        "— the in-container probe would report any package as present, so it "
        f"could not detect drift: {container_probe}"
    )

    container_importable = _importable(container_probe)
    missing = host_importable - container_importable
    extra = container_importable - host_importable
    assert not missing and not extra, (
        "container Python environment drifted from the host project venv\n"
        f"  importable on host, MISSING in container: {sorted(missing) or 'none'}\n"
        f"  importable in container, absent on host:  {sorted(extra) or 'none'}\n"
        f"  host:      {host_probe}\n"
        f"  container: {container_probe}"
    )

    # ── The recorded dependency set, in full ──
    recorded = _recorded_dependencies(built_project)
    assert MARKER_FROZEN in recorded and MARKER_ADDED in recorded, (
        f"the build did not record the markers, so this comparison would be "
        f"trivially satisfiable: {recorded}"
    )
    recorded_probe = _probe_packages(
        _docker_run_argv(built_image), specs=sorted(recorded), what="container recorded deps"
    )
    absent = sorted(name for name in recorded if recorded_probe[name]["version"] is None)
    assert not absent, (
        f"{len(absent)} recorded project dependenc(ies) are not installed in the "
        f"image: {absent}\n"
        "The build recorded them in pyproject.toml as the environment it "
        "created; an image that lacks them is the drift this test exists to "
        f"catch.\n  recorded: {recorded}\n  in image: {recorded_probe}"
    )

    # Names matching is not enough: a pin that got dropped and re-resolved to a
    # different release is still drift.
    mismatched = {
        name: (pin, recorded_probe[name]["version"])
        for name, pin in recorded.items()
        if pin is not None and recorded_probe[name]["version"] != pin
    }
    assert not mismatched, (
        f"pinned dependencies resolved to a different version in the image "
        f"(name -> (recorded pin, in image)): {mismatched}"
    )


# ── 4. mcp__python__execute succeeds on both sides ───────────────────────────


def _assert_execution_healthy(payload: dict, where: str) -> str:
    """Assert the execution path is healthy; return the AGENT interpreter path.

    The returned path is reported by the agent-authored code itself, so it is
    the interpreter ``resolve_agent_interpreter`` actually selected — not a
    value this test chose.
    """
    adapter = payload["adapter"]
    assert adapter["success"] is True, (
        f"{where}: mcp__python__execute could not run print(2+2) — "
        f"{adapter['error_message']!r}\n{adapter}"
    )
    assert "4" in adapter["stdout"], f"{where}: expected 4 in stdout, got {adapter['stdout']!r}"
    assert adapter["execution_method_used"] == "subprocess", (
        f"{where}: expected the subprocess backend, got "
        f"{adapter['execution_method_used']!r} — the only backend OSPREY ships"
    )

    # The driver reports failure when execution fails: "success" above is a
    # measurement, not a constant.
    canary = payload["canary"]
    assert canary["success"] is False, (
        f"{where}: code that raises was reported successful — the execution "
        f"result is not tracking what the subprocess actually did: {canary}"
    )
    assert "parity-canary" in canary["stderr"], (
        f"{where}: the canary's traceback did not reach the caller: {canary}"
    )

    tool = payload["tool"]
    assert tool is not None, f"{where}: the execute tool returned no JSON payload"
    summary = tool.get("summary", tool)
    assert summary.get("has_errors") is False, f"{where}: tool reported errors: {tool}"
    assert summary.get("status") == "Success", f"{where}: tool status not Success: {tool}"
    assert "4" in (summary.get("output") or ""), f"{where}: tool output missing 4: {tool}"

    # The seam: agent code imported the build's declared packages and told us
    # which interpreter did it. A green run without this could still be an
    # interpreter that has osprey but not the operator's packages — the drift
    # bug exactly as an operator meets it, an ImportError from their own Python.
    agent_env = payload["agent_env"]
    assert agent_env["success"] is True, (
        f"{where}: agent code could not import the packages this build declared "
        f"({MARKER_FROZEN}, {MARKER_ADDED}) — the interpreter running agent code "
        f"is not the environment the build populated.\n"
        f"  error: {agent_env['error_message']!r}\n  stderr: {agent_env['stderr']}"
    )
    stamped = [
        line[len(AGENT_PYTHON_PREFIX) :].strip()
        for line in (agent_env["stdout"] or "").splitlines()
        if line.startswith(AGENT_PYTHON_PREFIX)
    ]
    assert len(stamped) == 1, (
        f"{where}: expected exactly one {AGENT_PYTHON_PREFIX} stamp in agent "
        f"stdout, got {stamped}\n  stdout: {agent_env['stdout']!r}"
    )
    assert stamped[0], f"{where}: agent code reported an empty interpreter path"
    return stamped[0]


def test_execute_succeeds_on_host_build(built_project):
    """A freshly built project executes agent code without any further setup."""
    completed = subprocess.run(
        [str(built_project / ".venv" / "bin" / "python"), "-c", _EXEC_DRIVER],
        cwd=built_project,
        capture_output=True,
        text=True,
        timeout=EXEC_TIMEOUT,
        env={**os.environ, "OSPREY_CONFIG": str(built_project / "config.yml")},
    )
    payload = _parse_sentinel(completed, "host execution")
    agent_python = _assert_execution_healthy(payload, "host")

    # Agent code ran in the project venv, not in whatever interpreter happened
    # to launch OSPREY — resolve_agent_interpreter's "project venv exists"
    # branch. The path comes from the executed code, so this measures the
    # resolution rather than restating the interpreter the test launched.
    assert agent_python == str(built_project / ".venv" / "bin" / "python"), (
        f"agent code ran under {agent_python}, expected the project venv"
    )


def test_execute_succeeds_in_deployed_container(built_image):
    """The same execution path works inside the image, with no project venv."""
    completed = subprocess.run(
        [*_docker_run_argv(built_image), "-c", _EXEC_DRIVER],
        capture_output=True,
        text=True,
        timeout=EXEC_TIMEOUT,
    )
    payload = _parse_sentinel(completed, "container execution")
    agent_python = _assert_execution_healthy(payload, "container")

    # The image ships no .venv (.dockerignore excludes it), so agent code falls
    # back to the interpreter running OSPREY — resolve_agent_interpreter's other
    # branch, exercised only here. Again the path is what the executed code
    # reported, so the two branches are genuinely distinguished.
    assert not agent_python.startswith(f"{CONTAINER_PROJECT_ROOT}/.venv"), (
        f"container resolved a project venv that should not exist: {agent_python}"
    )
