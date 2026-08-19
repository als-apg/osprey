"""Real ``linux/arm64`` build of the project image's dependency layer, with p4p.

``p4p`` is a core dependency of ``osprey-connectors`` (the PVA transport of the
EPICS connector). It ships binary wheels for macOS universal2, linux x86_64 and
windows — but **not** for linux aarch64, and neither do the two native libraries
underneath it (``pvxslibs``, ``epicscorelibs``). On an arm64 Linux image all
three therefore build from source, which needs a C toolchain *and* a setuptools
older than 84.0.0: 84.0.0 broke ``setuptools_dso``'s compile-probe error
handling, so a naive arm64 build of the EPICS stack fails (upstream p4p#156).

The project image's deps layer already carries both halves of that — it stages
``build-essential``/``python3-dev``, writes a ``setuptools<84`` file and exports
``PIP_CONSTRAINT`` at it (a plain requirement pin would not reach pip's isolated
build environments), then purges the toolchain in the same ``RUN`` so it never
lands in the shipped image. Every one of those is a string in a template that no
x86_64 build can falsify, which is what this module is for.

Two tests, deliberately split by cost:

``test_deps_layer_stages_a_toolchain_under_the_setuptools_constraint`` reads the
Dockerfile ``osprey build`` renders and asserts the mechanism is present —
toolchain in, constraint exported at a ``setuptools<84`` file, toolchain purged.
It needs no container runtime and runs wherever the suite runs, so deleting the
constraint from the template is caught immediately rather than only on a host
that can build arm64.

``test_deps_layer_builds_p4p_from_source_on_arm64`` is the real proof: it
derives a **deps-layer-only** Dockerfile from that same render — the base image,
its ``ARG``s, the apt-mirror preamble, the ``/tmp/deps-ctx`` ``COPY`` and the
deps ``RUN``, all verbatim, with the node/Claude-CLI layers and everything after
the deps layer dropped — builds it for ``linux/arm64`` against the real image
context, and inspects the result. ``OSPREY_PIP_SPEC`` is pointed at the p4p
requirement ``packages/osprey-connectors/pyproject.toml`` actually declares, so
the layer installs exactly the dependency under test through its own unmodified
install/purge chain, rather than pulling the whole framework from PyPI (whose
released connectors may predate the p4p dependency entirely).

What the built image is then asked to prove, in one ``docker run``:

- ``import p4p`` works, including the ``_p4p`` C extension and the client/NT
  surface the connector uses — the source build produced a *working* module, not
  just files on disk;
- each of the three distributions records ``Tag: …-linux_aarch64`` in its
  ``WHEEL`` metadata, which only a local source build writes (a downloaded wheel
  would say ``manylinux…``). This is what pins the test to the source-build path
  and survives a fully cached rebuild, unlike scraping the build log;
- their ``WHEEL`` generator is a setuptools below 84 — direct evidence that
  ``PIP_CONSTRAINT`` reached the isolated build environment, the one thing the
  constraint exists to do;
- ``gcc`` is gone, so the same-``RUN`` purge still holds on the platform that
  actually uses the toolchain.

Cost and skipping. This is an expensive test: cold, it compiles EPICS Base plus
two Python extensions. On a native arm64 host (Apple Silicon) that is a few
minutes; under QEMU on an x86_64 host it is far longer, so ``BUILD_TIMEOUT`` is
generous and the tag is stable, letting BuildKit's layer cache make every rerun
near-instant. It skips — never silently, with a reason naming what was missing —
when docker is absent, when the builder cannot produce ``linux/arm64`` at all
(no qemu/binfmt registered), or when a CI runner could only build it under
emulation — a cold emulated compile would outlast the job timeout, so it is
skipped by name unless ``OSPREY_ARM64_EMULATED_BUILD=1`` asks for it. That is
the same module-level guard convention
``test_nginx_validate.py`` and ``test_dockerfile_e2e.py`` use; the precedent for
real-docker tests living in this suite rather than in ``tests/e2e/`` is
``web_terminals/test_env_digest_recreate_proof.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.utils.workspace import container_image_context

REPO_ROOT = Path(__file__).resolve().parents[2]
CONNECTORS_PYPROJECT = REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml"

PROJECT_NAME = "arm64deps"
IMAGE_TAG = "osprey-arm64-p4p-deps:test"
TARGET_PLATFORM = "linux/arm64"

# Cold, this compiles EPICS Base + pvxslibs + p4p. Native arm64 does it in
# minutes; emulated arm64 on an x86_64 runner takes far longer, and a build that
# is slow but progressing should not be mistaken for a wedged one.
BUILD_TIMEOUT = 5400
RUN_TIMEOUT = 300

# The distributions with no linux-aarch64 wheel: each one must show up in the
# image as a locally built artifact, not a download.
SOURCE_BUILT_DISTRIBUTIONS = ("p4p", "pvxslibs", "epicscorelibs")

# setuptools 84.0.0 is the release the deps layer's constraint exists to exclude.
MAX_BUILD_SETUPTOOLS_MAJOR = 84

# Opt-in for running the build under emulation on CI; see _arm64_skip_reason.
EMULATED_BUILD_OPT_IN = "OSPREY_ARM64_EMULATED_BUILD"


@cache
def _docker_available() -> bool:
    """Whether a usable docker daemon is reachable (memoised, probed once)."""
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@cache
def _buildx_platforms() -> tuple[str, ...]:
    """The active builder's advertised platforms, its own native one first.

    Native on Apple Silicon, qemu/binfmt-emulated elsewhere — and simply absent
    on a runner where no binfmt handler was registered. ``docker buildx
    inspect`` reports the answer for the active builder either way, so it is
    asked rather than inferred from the host architecture. Order carries the
    distinction the flat list does not spell out: buildx prints the builder's
    own platform first and the emulated ones after it, which is how the CI
    guard below tells a native arm64 machine from an emulating x86_64 one.

    Both this probe and ``_docker_available`` are memoised because this module
    is evaluated at import time in every collecting process; without the cache
    the same two ``docker`` calls run again on each re-evaluation.
    """
    if not _docker_available():
        return ()
    try:
        probe = subprocess.run(
            ["docker", "buildx", "inspect"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if probe.returncode != 0:
        return ()
    for line in probe.stdout.splitlines():
        if line.startswith("Platforms:"):
            listed = line.split(":", 1)[1].split(",")
            return tuple(entry.strip().rstrip("*").strip() for entry in listed if entry.strip())
    return ()


@cache
def _arm64_skip_reason() -> str | None:
    """Why the arm64 build cannot run here, or ``None`` when it can.

    Two distinct refusals, each named rather than folded into one boolean:

    - the builder cannot produce ``linux/arm64`` at all;
    - the builder *can*, but only under emulation, and this is CI. A cold
      emulated run compiles EPICS Base plus two extensions under QEMU, which
      is tens of minutes — long enough to blow a job timeout and surface as an
      unattributed red rather than as this test. Stock runners cannot build
      arm64 today and so skip on the first branch anyway; this second one is
      what keeps a future binfmt-registered shared runner from silently
      inheriting that compile into the unit lane. A native arm64 host is
      unaffected (its own platform is the first one buildx lists), and an
      emulated CI run stays reachable on purpose via
      ``OSPREY_ARM64_EMULATED_BUILD=1``.
    """
    platforms = _buildx_platforms()
    if TARGET_PLATFORM not in platforms:
        return (
            f"no builder for {TARGET_PLATFORM} (docker missing, or no qemu/binfmt "
            "handler registered for arm64)"
        )
    if os.environ.get("CI") and platforms[0] != TARGET_PLATFORM:
        if not os.environ.get(EMULATED_BUILD_OPT_IN):
            return (
                f"CI runner builds {TARGET_PLATFORM} only under emulation (native "
                f"platform is {platforms[0]}); a cold emulated EPICS compile would "
                f"outlast the job timeout — set {EMULATED_BUILD_OPT_IN}=1 to run it "
                "anyway"
            )
    return None


_ARM64_SKIP_REASON = _arm64_skip_reason()

# The marker is module-level so `-m dockerbuild` selects this file; the skip is
# on the build test alone, so the template assertions below still run on hosts
# with no docker and no arm64 emulation.
pytestmark = [pytest.mark.dockerbuild, pytest.mark.slow]

requires_arm64_builder = pytest.mark.skipif(
    _ARM64_SKIP_REASON is not None,
    reason=_ARM64_SKIP_REASON or "",
)


# --------------------------------------------------------------------------
# The rendered Dockerfile
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def image_context(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built hello-world deployment repo's image build context.

    Resolved through the production helper rather than spelled here, exactly as
    ``tests/e2e/test_dockerfile_e2e.py`` does: the context is what a real image
    build runs against, so a test that picked its own directory would prove
    nothing about the shipped build path. ``--no-git`` because nothing here
    reads history, ``--skip-deps``/``--skip-lifecycle`` because only the render
    is wanted — that combination is docker-free, so this fixture works on hosts
    where the build test below skips.
    """
    repo = tmp_path_factory.mktemp("arm64_p4p_image_build") / PROJECT_NAME
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

    context = container_image_context(repo, PROJECT_NAME)
    assert (context / "build" / "Dockerfile").is_file(), f"no rendered Dockerfile under {context}"
    return context


@pytest.fixture(scope="module")
def rendered_dockerfile(image_context: Path) -> str:
    return (image_context / "build" / "Dockerfile").read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> list[str]:
    """The Dockerfile's logical instructions, line continuations joined.

    Standalone comments and blank lines between instructions are dropped;
    everything inside a continued instruction is kept verbatim, because the
    derived Dockerfile below re-emits these strings unchanged and a dropped
    continuation would silently change what gets built.
    """
    instructions: list[str] = []
    current: list[str] = []
    for line in dockerfile.splitlines():
        if not current:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
        current.append(line)
        if not line.rstrip().endswith("\\"):
            instructions.append("\n".join(current))
            current = []
    if current:
        instructions.append("\n".join(current))
    return instructions


def _only_instruction(instructions: list[str], keyword: str, needle: str, what: str) -> str:
    """The single *keyword* instruction containing *needle*, or a failure naming *what*.

    Both halves matter: the deps ``RUN`` writes its constraint file into
    ``/tmp/deps-ctx`` too, so the directory alone does not identify the ``COPY``
    that creates it.
    """
    matches = [text for text in instructions if text.startswith(f"{keyword} ") and needle in text]
    assert len(matches) == 1, (
        f"expected exactly one {what} instruction ({keyword} matching {needle!r}) in "
        f"the rendered Dockerfile, found {len(matches)}"
    )
    return matches[0]


def _deps_run(instructions: list[str]) -> str:
    return _only_instruction(instructions, "RUN", "PIP_CONSTRAINT", "dependency-layer RUN")


def _deps_copy(instructions: list[str]) -> str:
    return _only_instruction(instructions, "COPY", "/tmp/deps-ctx/", "deps-context COPY")


# --------------------------------------------------------------------------
# The mechanism, asserted on the template render (no container runtime)
# --------------------------------------------------------------------------


def test_deps_layer_stages_a_toolchain_under_the_setuptools_constraint(
    rendered_dockerfile: str,
) -> None:
    """The deps layer stages a compiler, constrains setuptools, and purges again.

    Guarding all three here — not only inside the arm64 build — is deliberate:
    the build test skips on every host without an arm64 builder, and a template
    that quietly lost its ``PIP_CONSTRAINT`` line would otherwise reach a
    release with nothing red anywhere.
    """
    instructions = _instructions(rendered_dockerfile)
    deps_run = _deps_run(instructions)

    assert "build-essential" in deps_run and "python3-dev" in deps_run, (
        "deps layer no longer stages a C toolchain; p4p/pvxslibs/epicscorelibs "
        f"cannot build from source on linux/arm64 without one:\n{deps_run}"
    )
    assert "setuptools<84" in deps_run, (
        "deps layer no longer pins setuptools below 84.0.0, which breaks "
        f"setuptools_dso's compile probe and fails every EPICS source build:\n{deps_run}"
    )
    assert "PIP_CONSTRAINT" in deps_run, (
        "the setuptools pin must be exported as PIP_CONSTRAINT — a plain "
        f"requirement pin does not reach pip's isolated build environments:\n{deps_run}"
    )
    assert "apt-get purge -y build-essential python3-dev" in deps_run, (
        f"the toolchain must be purged in the same RUN, or it ships in the final image:\n{deps_run}"
    )

    # The constraint file is written into /tmp/deps-ctx, so the COPY that
    # creates that directory has to come first.
    copy_index = instructions.index(_deps_copy(instructions))
    assert copy_index < instructions.index(deps_run), (
        "the /tmp/deps-ctx COPY must precede the deps RUN that writes the "
        "constraint file into that directory"
    )


def test_connectors_declares_the_p4p_dependency_this_module_builds() -> None:
    """p4p is a declared core dependency, and that is the spec built below.

    The build test installs whatever this returns. If p4p ever stopped being a
    connectors dependency, the build would keep passing against a hard-coded
    requirement while the image no longer shipped p4p at all.
    """
    assert _p4p_requirement().startswith("p4p")


def _p4p_requirement() -> str:
    """The p4p requirement string ``osprey-connectors`` declares."""
    manifest = tomllib.loads(CONNECTORS_PYPROJECT.read_text(encoding="utf-8"))
    for dependency in manifest["project"]["dependencies"]:
        if re.match(r"^p4p\b", dependency.strip()):
            return dependency.strip()
    pytest.fail(
        f"no p4p requirement in {CONNECTORS_PYPROJECT} — the PVA transport's "
        "dependency was removed or renamed"
    )


# --------------------------------------------------------------------------
# The real arm64 build
# --------------------------------------------------------------------------


def _deps_layer_dockerfile(rendered: str) -> str:
    """A Dockerfile that stops after the project image's dependency layer.

    Every instruction is copied verbatim out of the render — nothing is
    re-spelled here, so a change to the template's toolchain staging, its
    constraint or its install chain lands in what this builds. Only the
    node/Claude-CLI layers (irrelevant to a C source build, and the most
    expensive thing to emulate) and everything after the deps ``RUN`` are left
    out; that omission is what makes this "the deps layer only".
    """
    instructions = _instructions(rendered)

    from_lines = [text for text in instructions if text.startswith("FROM ")]
    assert len(from_lines) == 1, (
        f"expected exactly one FROM instruction in the rendered Dockerfile, found "
        f"{len(from_lines)}: this helper derives the deps layer by taking the single "
        "stage's base image, which is only correct while the render is single-stage. "
        "If the image template became multi-stage, re-derive the deps stage here "
        "explicitly (name the stage the deps RUN belongs to and carry its FROM plus "
        "any stages it copies from) rather than letting this take the first FROM"
    )
    from_line = from_lines[0]
    args = [text for text in instructions if text.startswith("ARG ")]
    apt_preamble = _only_instruction(instructions, "RUN", "Acquire::Retries", "apt preamble RUN")
    deps_copy = _deps_copy(instructions)
    deps_run = _deps_run(instructions)

    return (
        "# syntax=docker/dockerfile:1\n"
        + "\n".join([from_line, *args, apt_preamble, deps_copy, deps_run])
        + "\n"
    )


@pytest.fixture(scope="module")
def _remove_image_tag() -> Iterator[None]:
    """Drop the tag afterwards; BuildKit's layer cache is untouched by that.

    Keeping the cache is the point — a rerun of this module then costs a
    container start rather than another EPICS compile.
    """
    yield
    subprocess.run(["docker", "image", "rm", "-f", IMAGE_TAG], capture_output=True, timeout=120)


# What the built image is asked, in one run. Printed as JSON so a partial
# failure still reports every fact rather than only the first one to blow up.
_PROBE = """
import importlib.metadata as md, json, shutil
import p4p, p4p._p4p
from p4p.client.thread import Context
from p4p.nt import NTNDArray, NTScalar
facts = {"imported": [p4p.__name__, Context.__name__, NTNDArray.__name__, NTScalar.__name__]}
facts["wheels"] = {
    name: dict(
        line.split(": ", 1)
        for line in (md.distribution(name).read_text("WHEEL") or "").splitlines()
        if ": " in line
    )
    | {"version": md.version(name)}
    for name in %(dists)r
}
facts["gcc"] = shutil.which("gcc")
print(json.dumps(facts))
"""


@requires_arm64_builder
def test_deps_layer_builds_p4p_from_source_on_arm64(
    image_context: Path,
    rendered_dockerfile: str,
    tmp_path: Path,
    _remove_image_tag: None,
) -> None:
    """The deps layer builds, and the image it produces has a working p4p.

    This is the platform with no p4p wheel, so a green here means the source
    build the connectors package's dependency comment promises actually happens
    inside the shipped image recipe.
    """
    dockerfile = tmp_path / "Dockerfile.deps"
    dockerfile.write_text(_deps_layer_dockerfile(rendered_dockerfile), encoding="utf-8")

    build = subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            TARGET_PLATFORM,
            "-f",
            str(dockerfile),
            "-t",
            IMAGE_TAG,
            "--build-arg",
            f"OSPREY_PIP_SPEC={_p4p_requirement()}",
            ".",
        ],
        cwd=image_context,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        # BuildKit pinned on, as tests/e2e/test_dockerfile_e2e.py does, so
        # layer-cache semantics are uniform across docker versions.
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    assert build.returncode == 0, (
        f"{TARGET_PLATFORM} deps-layer build failed — the EPICS source build is "
        f"broken on the platform with no wheel:\n{build.stdout[-8000:]}\n{build.stderr[-8000:]}"
    )

    probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            TARGET_PLATFORM,
            IMAGE_TAG,
            "python",
            "-c",
            _PROBE % {"dists": SOURCE_BUILT_DISTRIBUTIONS},
        ],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )
    assert probe.returncode == 0, (
        "p4p does not import inside the arm64 image — the source build produced "
        f"an unusable module:\n{probe.stdout}\n{probe.stderr}"
    )
    facts = json.loads(probe.stdout.strip().splitlines()[-1])

    for name in SOURCE_BUILT_DISTRIBUTIONS:
        wheel = facts["wheels"][name]
        assert wheel["Tag"].endswith("linux_aarch64"), (
            f"{name} {wheel['version']} came from a prebuilt wheel ({wheel['Tag']}), "
            "so this run never exercised the source-build path — if upstream "
            "started shipping linux-aarch64 wheels, the toolchain and the "
            "setuptools constraint are no longer what makes arm64 work"
        )
        generator = wheel["Generator"]
        built_by = re.search(r"setuptools \((\d+)\.", generator)
        assert built_by is not None, f"{name} was not built by setuptools: {generator}"
        assert int(built_by.group(1)) < MAX_BUILD_SETUPTOOLS_MAJOR, (
            f"{name} was built by {generator}: PIP_CONSTRAINT did not reach pip's "
            "isolated build environment, which is the one thing it exists to do"
        )

    assert facts["gcc"] is None, (
        f"the C toolchain survived the deps layer (gcc at {facts['gcc']}) — the "
        "same-RUN purge stopped working on arm64"
    )
