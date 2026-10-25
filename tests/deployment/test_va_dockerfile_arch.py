"""The VA service image builds natively on linux/amd64 and linux/arm64.

PyPI has every framework wheel the image needs on amd64. On arm64 it has none
for pcaspy (the Channel Access server) or for accelerator-toolbox at the
image's interpreter, so the recipe's ``native-wheels-arm64`` stage builds both
from pinned sdists, pcaspy against EPICS Base and PCAS compiled in the same
stage, and hands the wheels to the final stage through a find-links directory.

These are static pins on the template: what a native arm64 build depends on
is strings no amd64 build can falsify. The real arm64 build is proved by
building the image on an arm64 host and serving a channel from it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "src/osprey/templates/services/virtual_accelerator/Dockerfile"

#: The floor lume-pva-apg's `ca` extra carries: 0.8.0's `SimplePV.writeNotify`
#: aborts the server on any asynchronous write.
PCASPY_FLOOR = SpecifierSet(">=0.8.1")
#: Every pinned download the arm64 stage makes, as `<NAME>_VERSION` and
#: `<NAME>_SHA256` ARG pairs.
PINNED_DOWNLOADS = ("EPICS_BASE", "PCAS", "PCASPY", "PYAT")
FIND_LINKS_DIR = "/opt/osprey-native-wheels"
SERVING_IMPORT = 'python -c "import pcaspy, p4p, lume_pva_apg.runner"'


def _stages(text: str) -> dict[str, str]:
    """Each named build stage's body, keyed by its `AS` name ('' for the last)."""
    parts = re.split(r"^FROM ", text, flags=re.M)[1:]
    stages = {}
    for part in parts:
        head, _, body = part.partition("\n")
        match = re.search(r"\sAS\s+(\S+)\s*$", head)
        stages[match.group(1) if match else ""] = f"FROM {head}\n{body}"
    return stages


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
        "FROM python:3.11-slim AS va-base",
        "FROM --platform=linux/amd64 python:3.11-slim AS va-base",
    )
    assert mutated != text, "mutation is stale"
    with pytest.raises(AssertionError, match="pins a platform"):
        test_no_stage_pins_a_platform(mutated)


def test_each_architecture_has_a_native_wheels_stage(text: str) -> None:
    """The final stage's wheels come from the stage its TARGETARCH names."""
    stages = _stages(text)
    assert {"native-wheels-amd64", "native-wheels-arm64", "native-wheels"} <= stages.keys(), (
        f"missing a native-wheels stage; stages are {sorted(stages)}"
    )
    assert re.search(r"^ARG TARGETARCH$", text.split("\nFROM ", 1)[0], flags=re.M), (
        "TARGETARCH must be declared before the first FROM to select a stage"
    )
    assert stages["native-wheels"].startswith("FROM native-wheels-${TARGETARCH} AS native-wheels")
    final = stages[""]
    assert f"COPY --from=native-wheels /wheels/ {FIND_LINKS_DIR}/" in final, (
        "the final stage does not take the native wheels"
    )
    for var in ("PIP_FIND_LINKS", "UV_FIND_LINKS"):
        assert f"{var}={FIND_LINKS_DIR}" in final, f"{var} does not name the native wheels"
    assert final.index(FIND_LINKS_DIR) < final.index("uv pip install"), (
        "the native wheels must be in place before the deps layer resolves"
    )


def test_each_architecture_has_a_native_wheels_stage__mutation_drops_uv_links(text: str) -> None:
    mutated = text.replace(f"    UV_FIND_LINKS={FIND_LINKS_DIR}", "    UV_UNUSED=1")
    assert mutated != text, "mutation is stale"
    with pytest.raises(AssertionError, match="UV_FIND_LINKS does not name"):
        test_each_architecture_has_a_native_wheels_stage(mutated)


def test_the_arm64_stage_pulls_only_pinned_checked_downloads(text: str) -> None:
    """Every tarball and sdist is pinned by version and checked by sha256."""
    arm = _stages(text)["native-wheels-arm64"]
    for name in PINNED_DOWNLOADS:
        assert re.search(rf"^ARG {name}_VERSION=\S+$", arm, flags=re.M), (
            f"{name} has no pinned version"
        )
        assert re.search(rf"^ARG {name}_SHA256=[0-9a-f]{{64}}$", arm, flags=re.M), (
            f"{name} has no sha256 pin"
        )
    assert arm.count("| sha256sum -c -") == 2, "EPICS Base and PCAS must each be checked"
    for name in ("EPICS_BASE", "PCAS"):
        assert f"${{{name}_VERSION}}" in arm, f"the {name} download ignores its pin"
    assert "--require-hashes" in arm and "--no-binary pcaspy,accelerator-toolbox" in arm, (
        "the sdists must be built from source under pip's hash mode"
    )
    pcaspy = re.search(r"^ARG PCASPY_VERSION=(\S+)$", arm, flags=re.M)
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
