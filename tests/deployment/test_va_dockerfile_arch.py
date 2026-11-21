"""The VA service image builds natively on linux/amd64 and linux/arm64.

PyPI has every framework wheel the image needs on amd64. On arm64 it has none
for pcaspy (the Channel Access server) at the image's interpreter, so the
recipe's native-wheels stage's aarch64 branch builds it from its pinned sdist
against EPICS Base and PCAS compiled in the same stage, and hands the wheel to
the final stage through a find-links directory. Everything else without an arm64
wheel (p4p, the EPICS libraries under it, accelerator-toolbox) is a core
dependency and builds from source in the deps layer every Python service image
shares.

These are static pins on the template: what a native arm64 build depends on
is strings no amd64 build can falsify. The real arm64 build is proved by
building the image on an arm64 host and serving a channel from it.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tests.deployment._pip_probe import run_body, stubbed_env
from tests.deployment._proxy_idiom import run_instructions

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "src/osprey/templates/services/virtual_accelerator/Dockerfile"

#: The floor lume-pva-apg's `ca` extra carries: 0.8.0's `SimplePV.writeNotify`
#: aborts the server on any asynchronous write.
PCASPY_FLOOR = SpecifierSet(">=0.8.1")
#: Every pinned download the native-wheels stage makes on aarch64, as `<NAME>_VERSION` and
#: `<NAME>_SHA256` ARG pairs.
PINNED_DOWNLOADS = ("EPICS_BASE", "PCAS", "PCASPY")
FIND_LINKS_DIR = "/opt/osprey-native-wheels"
SERVING_IMPORT = 'python -c "import pcaspy, p4p, lume_pva_apg.runner"'
#: The install that adds the extra, whose pcaspy is the one native wheel.
EXTRA_INSTALL = '"osprey-framework[virtual-accelerator]==$OSPREY_VERSION" --only-binary pcaspy'


def _stages(text: str) -> dict[str, str]:
    """Each named build stage's body, keyed by its `AS` name ('' for the last)."""
    parts = re.split(r"^FROM ", text, flags=re.M)[1:]
    stages = {}
    for part in parts:
        head, _, body = part.partition("\n")
        match = re.search(r"\sAS\s+(\S+)\s*$", head)
        stages[match.group(1) if match else ""] = f"FROM {head}\n{body}"
    return stages


def _aarch64_branch(stage: str) -> str:
    """The text of the native-wheels RUN's ``aarch64)`` branch."""
    return stage.split("    aarch64) \\\n", 1)[1].split("\n    *) \\\n", 1)[0]


@pytest.fixture
def text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_no_stage_pins_a_platform(text: str) -> None:
    """A constant `--platform` on a FROM builds that platform everywhere, which
    on the other architecture means emulation."""
    assert not re.search(r"^FROM\s+--platform", text, flags=re.M), (
        "a FROM pins a platform; the VA image builds for the build's own platform"
    )
    assert "dpkg --print-architecture" not in text, "an architecture refusal is back"


def test_no_stage_pins_a_platform__mutation_pins_amd64(text: str) -> None:
    mutated = text.replace(
        "FROM ${OSPREY_BASE_REGISTRY}/library/python:3.11-slim AS va-base",
        "FROM --platform=linux/amd64 ${OSPREY_BASE_REGISTRY}/library/python:3.11-slim AS va-base",
    )
    assert mutated != text, "mutation is stale"
    with pytest.raises(AssertionError, match="pins a platform"):
        test_no_stage_pins_a_platform(mutated)


#: Automatic platform ARGs some builders (docker 24's embedded BuildKit) leave
#: empty when they expand a stage name.
PLATFORM_ARG = re.compile(r"^ARG\s+(TARGET|BUILD)\w*", flags=re.M)


def test_no_stage_is_selected_by_a_build_arg(text: str) -> None:
    """Every FROM that does not pull an image names an earlier stage literally,
    and no platform ARG is declared, so the build reads no platform build arg."""
    preamble = text.split("\nFROM ", 1)[0]
    assert not PLATFORM_ARG.search(preamble), (
        "a platform ARG is declared before the first FROM; a stage is selected by a build arg"
    )
    stages: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("FROM "):
            continue
        words = [w for w in line.split()[1:] if not w.startswith("--")]
        base = words[0]
        if not base.startswith("${OSPREY_BASE_REGISTRY}/"):
            assert "$" not in base and base in stages, (
                f"{line!r}: a stage selected by a build arg, or not defined earlier"
            )
        if len(words) >= 3 and words[1].upper() == "AS":
            stages.add(words[2])


def test_no_stage_is_selected_by_a_build_arg__mutation_selects_by_targetarch(text: str) -> None:
    mutated = text.replace(
        'ARG OSPREY_BASE_REGISTRY="docker.io"',
        'ARG TARGETARCH\nARG OSPREY_BASE_REGISTRY="docker.io"',
        1,
    ).replace(
        "FROM va-base AS native-wheels\n", "FROM native-wheels-${TARGETARCH} AS native-wheels\n"
    )
    assert mutated.count("TARGETARCH") == 2, "mutation is stale"
    with pytest.raises(AssertionError, match="selected by a build arg"):
        test_no_stage_is_selected_by_a_build_arg(mutated)


def test_each_architecture_has_a_native_wheels_stage(text: str) -> None:
    """The final stage's wheels come from the one native-wheels stage, which
    reads the build's own architecture."""
    stages = _stages(text)
    assert "native-wheels" in stages, (
        f"missing the native-wheels stage; stages are {sorted(stages)}"
    )
    wheels = stages["native-wheels"]
    assert wheels.startswith("FROM va-base AS native-wheels\n")
    (run,) = run_instructions(wheels)
    assert '"$(uname -m)"' in run, "the native-wheels stage does not read the build's architecture"
    for branch in ("x86_64)", "aarch64)", "*)"):
        assert branch in run, f"the native-wheels stage has no {branch} branch"
    assert "exit 1" in run.split("*)", 1)[1], "an unknown architecture does not fail the build"
    final = stages[""]
    assert f"COPY --from=native-wheels /wheels/ {FIND_LINKS_DIR}/" in final, (
        "the final stage does not take the native wheels"
    )
    for var in ("PIP_FIND_LINKS", "UV_FIND_LINKS"):
        assert f"{var}={FIND_LINKS_DIR}" in final, f"{var} does not name the native wheels"
    assert final.index(FIND_LINKS_DIR) < final.index(EXTRA_INSTALL), (
        "the native wheels must be in place before the extra resolves"
    )


def test_each_architecture_has_a_native_wheels_stage__mutation_drops_uv_links(text: str) -> None:
    mutated = text.replace(f"    UV_FIND_LINKS={FIND_LINKS_DIR}", "    UV_UNUSED=1")
    assert mutated != text, "mutation is stale"
    with pytest.raises(AssertionError, match="UV_FIND_LINKS does not name"):
        test_each_architecture_has_a_native_wheels_stage(mutated)


#: Tools the native-wheels RUN calls that the dispatch test replaces with a
#: stub logging its argv.
_LOGGED_TOOLS = ("apt-get", "curl", "sha256sum", "make", "perl", "cp", "nproc")


def _dispatch_stubs(log: Path, arch: str) -> dict[str, str]:
    """Stubs for the native-wheels RUN: ``uname`` prints *arch*, every other
    tool logs its argv; ``tar`` also leaves the ``configure`` directory its
    extraction would, and ``pip wheel`` leaves an aarch64 pcaspy wheel in its
    ``-w`` directory."""
    logged = f'printf "%s %s\\n" "$(basename "$0")" "$*" >> {shlex.quote(str(log))}\n'
    stubs = dict.fromkeys(_LOGGED_TOOLS, f"#!/bin/sh\n{logged}exit 0\n")
    stubs["uname"] = f"#!/bin/sh\necho {arch}\n"
    stubs["tar"] = (
        f"#!/bin/sh\n{logged}"
        'prev=""\nfor a in "$@"; do [ "$prev" = -C ] && mkdir -p "$a/configure"; prev="$a"; done\n'
        "exit 0\n"
    )
    stubs["pip"] = (
        f"#!/bin/sh\n{logged}"
        '[ "$1" = wheel ] || exit 0\n'
        'prev=""\nfor a in "$@"; do\n'
        '  [ "$prev" = -w ] && touch "$a/pcaspy-0.8.1-cp311-cp311-linux_aarch64.whl"\n'
        '  prev="$a"\ndone\n'
        "exit 0\n"
    )
    return stubs


@pytest.mark.parametrize(
    ("arch", "targetarch"),
    [("x86_64", None), ("x86_64", ""), ("aarch64", None), ("riscv64", None)],
    ids=["x86_64", "x86_64-empty-TARGETARCH", "aarch64", "riscv64"],
)
def test_the_native_wheels_stage_dispatches_on_the_build_architecture(
    text: str, tmp_path: Path, arch: str, targetarch: str | None
) -> None:
    """The stage's RUN picks its branch from ``uname -m`` alone: with
    ``TARGETARCH`` absent or empty it builds what that architecture needs, and
    an unknown architecture fails naming itself."""
    (run,) = run_instructions(_stages(text)["native-wheels"])
    wheels, native = tmp_path / "wheels", tmp_path / "native"
    body = run.removeprefix("RUN ").replace("/tmp/native", str(native))
    body = body.replace("/wheels", str(wheels))
    log = tmp_path / "calls.log"
    env = stubbed_env(tmp_path, _dispatch_stubs(log, arch))
    env.pop("TARGETARCH", None)
    if targetarch is not None:
        env["TARGETARCH"] = targetarch
    result = run_body(body, env)
    calls = [line.split(" ", 1) for line in log.read_text().splitlines()] if log.exists() else []
    tools = [tool for tool, *_ in calls]
    if arch == "x86_64":
        assert result.returncode == 0, f"{result.stdout}{result.stderr}"
        assert wheels.is_dir() and not any(wheels.iterdir()), "x86_64 must leave /wheels empty"
        assert not {"apt-get", "curl", "make"} & set(tools), f"x86_64 built something: {calls}"
    elif arch == "aarch64":
        assert result.returncode == 0, f"{result.stdout}{result.stderr}"
        assert tools.count("curl") == 2 and tools.count("make") == 2, calls
        pip_wheels = [argv for tool, argv in calls if tool == "pip" and argv.startswith("wheel ")]
        assert len(pip_wheels) == 1 and "--require-hashes" in pip_wheels[0], calls
    else:
        assert result.returncode != 0, "an unknown architecture must fail the build"
        assert arch in result.stderr, f"the failure does not name {arch}: {result.stderr!r}"


def test_the_arm64_stage_pulls_only_pinned_checked_downloads(text: str) -> None:
    """Every tarball and sdist is pinned by version and checked by sha256."""
    stage = _stages(text)["native-wheels"]
    arm = _aarch64_branch(stage)
    for name in PINNED_DOWNLOADS:
        assert re.search(rf"^ARG {name}_VERSION=\S+$", stage, flags=re.M), (
            f"{name} has no pinned version"
        )
        assert re.search(rf"^ARG {name}_SHA256=[0-9a-f]{{64}}$", stage, flags=re.M), (
            f"{name} has no sha256 pin"
        )
    assert arm.count("| sha256sum -c -") == 2, "EPICS Base and PCAS must each be checked"
    for name in ("EPICS_BASE", "PCAS"):
        assert f"${{{name}_VERSION}}" in arm, f"the {name} download ignores its pin"
    assert "--require-hashes" in arm and "--no-binary pcaspy " in arm, (
        "the pcaspy sdist must be built from source under pip's hash mode"
    )
    pcaspy = re.search(r"^ARG PCASPY_VERSION=(\S+)$", stage, flags=re.M)
    assert pcaspy and Version(pcaspy.group(1)) in PCASPY_FLOOR, (
        f"pcaspy must keep the {PCASPY_FLOOR} floor"
    )


def test_the_arm64_stage_pulls_only_pinned__mutation_unchecked_tarball(text: str) -> None:
    mutated = text.replace(
        '    echo "${PCAS_SHA256}  /tmp/native/pcas.tar.gz" | sha256sum -c -; \\\n', "", 1
    )
    assert mutated != text, "mutation is stale"
    with pytest.raises(AssertionError, match="each be checked"):
        test_the_arm64_stage_pulls_only_pinned_checked_downloads(mutated)


def test_the_arm64_stage_pulls_only_pinned__mutation_pcaspy_below_floor(text: str) -> None:
    mutated = re.sub(r"^ARG PCASPY_VERSION=\S+$", "ARG PCASPY_VERSION=0.8.0", text, flags=re.M)
    with pytest.raises(AssertionError, match="floor"):
        test_the_arm64_stage_pulls_only_pinned_checked_downloads(mutated)


def test_the_image_refuses_to_build_without_its_serving_stack(text: str) -> None:
    """A marker that does not admit the platform installs nothing and raises
    nothing, so the build itself must import the servers it ships."""
    final = _stages(text)[""]
    assert SERVING_IMPORT in final, "the image never checks its serving stack imports"
    assert final.index("--force-reinstall") < final.index(SERVING_IMPORT) < final.index("CMD "), (
        "the serving-stack check must follow the wheel overlay and precede the CMD"
    )


def test_the_image_refuses_to_build_without__mutation_drops_the_check(text: str) -> None:
    mutated = text.replace(SERVING_IMPORT, 'python -c "import json"')
    with pytest.raises(AssertionError, match="never checks"):
        test_the_image_refuses_to_build_without_its_serving_stack(mutated)
