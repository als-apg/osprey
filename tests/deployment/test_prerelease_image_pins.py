"""A pre-release framework pin reaches every image build as a pre-release resolve.

osprey-framework and osprey-connectors ship as a pair from one tag, so a beta
of one exists only beside a beta of the other. The framework's own connectors
requirement names no pre-release, and plain ``pip`` — what every shipped image
recipe installs with — never picks one for such a requirement. A beta pin
therefore resolves to nothing inside an image unless the build says the whole
resolve may admit pre-releases. The ``osprey build`` venv already does this for
``uv`` (tests/cli/test_build_prerelease_pin.py); these tests pin the same
decision on the image side: the version helpers that make it, and the two
build commands that hand it to a Dockerfile as ``OSPREY_PIP_PRE=1``.
"""

from __future__ import annotations

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.web_terminals import persona_images
from osprey.version import is_prerelease, pins_prerelease

pytestmark = pytest.mark.unit


class TestVersionHelpers:
    @pytest.mark.parametrize(
        "version", ["2026.9.0b2", "2026.9.0a1", "2026.9.0rc1", "2026.9.0.dev3"]
    )
    def test_a_prerelease_version(self, version):
        assert is_prerelease(version) is True

    @pytest.mark.parametrize("version", ["2026.9.0", "2026.6.2.post783+g83fda5e60", ""])
    def test_a_stable_or_unparseable_version(self, version):
        assert is_prerelease(version) is False

    def test_a_requirement_pinning_a_prerelease(self):
        assert pins_prerelease("osprey-framework==2026.9.0b2") is True
        assert pins_prerelease("osprey-framework[teams]==2026.9.0rc1") is True

    @pytest.mark.parametrize(
        "requirement",
        [
            "osprey-framework==2026.9.0",
            "osprey-framework",
            "osprey-framework>=2026.6.2,!=2026.6.2a0",
            "git+https://example.invalid/osprey@1b2c3d4",
            "osprey-framework @ file:///tmp/osprey",
        ],
    )
    def test_a_requirement_that_pins_none(self, requirement):
        assert pins_prerelease(requirement) is False


def _build_args(cmd: list[str]) -> dict[str, str]:
    return dict(
        arg.split("=", 1) for flag, arg in zip(cmd, cmd[1:], strict=False) if flag == "--build-arg"
    )


class TestProjectImageBuildCmd:
    def test_a_prerelease_pin_admits_prereleases(self, monkeypatch):
        monkeypatch.setenv("OSPREY_PIP_SPEC", "osprey-framework==2026.9.0b2")
        cmd = container_lifecycle._project_image_build_cmd({"project_name": "p"}, "docker", "/proj")

        args = _build_args(cmd)
        assert args["OSPREY_PIP_SPEC"] == "osprey-framework==2026.9.0b2"
        assert args["OSPREY_PIP_PRE"] == "1"
        assert cmd[-1] == "/proj"

    def test_a_stable_pin_stays_strict(self, monkeypatch):
        monkeypatch.setenv("OSPREY_PIP_SPEC", "osprey-framework==2026.9.0")
        cmd = container_lifecycle._project_image_build_cmd({"project_name": "p"}, "docker", "/proj")

        assert "OSPREY_PIP_PRE" not in _build_args(cmd)


class TestPersonaImageBuildCmd:
    def test_a_prerelease_pin_admits_prereleases(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSPREY_PIP_SPEC", "osprey-framework==2026.9.0b2")
        cmd = persona_images._persona_image_build_cmd(
            "docker", str(tmp_path), "p-alice:local", "p", {"project_name": "p"}
        )

        args = _build_args(cmd)
        assert args["OSPREY_PIP_SPEC"] == "osprey-framework==2026.9.0b2"
        assert args["OSPREY_PIP_PRE"] == "1"
        assert cmd[-1] == str(tmp_path)

    def test_a_stable_pin_stays_strict(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSPREY_PIP_SPEC", "osprey-framework==2026.9.0")
        cmd = persona_images._persona_image_build_cmd(
            "docker", str(tmp_path), "p-alice:local", "p", {"project_name": "p"}
        )

        assert "OSPREY_PIP_PRE" not in _build_args(cmd)
