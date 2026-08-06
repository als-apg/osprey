"""The generated Dockerfile installs exactly what the project venv installed.

A built project has two environments: the ``.venv`` on the build host and the
image built from the project directory. They are the same project, so they must
carry the same packages — a package present on the host and missing in the
container is the drift this feature exists to close.

The property under test is therefore an *equality*, not a containment: the
requirement list rendered into the Dockerfile's install line is the same list
the project's ``pyproject.toml`` records (minus osprey itself, which the image
installs from its own pinned ``OSPREY_PIP_SPEC`` build arg). Asserting that the
image merely *contains* some expected package would leave room for the two
lists to diverge in everything else.

Both are read back from the rendered artifacts rather than from the build
context, so the assertions cover the whole path — merge, install, record,
template — and not just the dictionary in the middle of it.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli import build_environment
from osprey.cli.main import cli

pytestmark = pytest.mark.unit

OSPREY_SPEC = "osprey-framework==1.2.3"

# The Dockerfile's primer install: the pinned spec, then the project's own
# requirements. Only the tail is under test here; `tests/cli/test_dockerfile_template.py`
# owns the shape of the surrounding RUN.
_INSTALL_LINE = re.compile(r'pip install --no-cache-dir "\$OSPREY_PIP_SPEC"(?P<args>[^\\\n]*)')


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _record(name: str, version: str) -> dict:
    """A probe record as :func:`_probe_base_distributions` returns them."""
    return {"name": name, "version": version, "direct_url": None}


@pytest.fixture
def base_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Declare a venv base and control the distributions it reports.

    The interpreter is a real (non-executable-Python) file because the profile
    validator stats it; what it *contains* never matters, since both probes
    that would run it are patched out.
    """

    def declare(*records: dict) -> str:
        venv_root = tmp_path / "base-venv"
        (venv_root / "bin").mkdir(parents=True, exist_ok=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        python = venv_root / "bin" / "python"
        python.touch()
        python.chmod(0o755)
        monkeypatch.setattr(build_environment, "_base_is_venv", lambda _python: True)
        monkeypatch.setattr(
            build_environment, "_probe_base_distributions", lambda _python: list(records)
        )
        return str(python)

    return declare


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run ``osprey build`` with the venv install stubbed out.

    Every ``subprocess.run`` inside :mod:`osprey.cli.build_environment` reports
    success without running anything, so the venv is never really created — but
    the requirement list is assembled, recorded and rendered for real, which is
    what these tests read back.
    """

    def run(profile_yaml: str, *, skip_deps: bool = False) -> Path:
        monkeypatch.setattr(
            build_environment.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        profile = out / "profile.yml"
        profile.write_text(profile_yaml, encoding="utf-8")
        args = ["build", "proj", str(profile), "--skip-lifecycle", "-o", str(out)]
        if skip_deps:
            args.append("--skip-deps")
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
        return out / "proj"

    return run


def _profile_yaml(
    *,
    python: str | None = None,
    dependencies: list[str] | None = None,
    packages: list[str] | None = None,
) -> str:
    """A hello-world-derived profile carrying the declarations under test."""
    lines = ["extends: hello-world", f"osprey_install: {OSPREY_SPEC}"]
    if dependencies:
        lines += ["dependencies:", *(f"  - {json.dumps(dep)}" for dep in dependencies)]
    if python or packages:
        lines.append("environment:")
        if python:
            lines.append(f"  python: {json.dumps(python)}")
        if packages:
            lines += ["  packages:", *(f"    - {json.dumps(pkg)}" for pkg in packages)]
    return "\n".join(lines) + "\n"


def image_deps(project_path: Path) -> list[str]:
    """The requirements the generated Dockerfile installs, un-shell-quoted."""
    text = (project_path / "Dockerfile").read_text(encoding="utf-8")
    matches = _INSTALL_LINE.finditer(text)
    args = [shlex.split(match.group("args")) for match in matches]
    assert args, 'no `pip install "$OSPREY_PIP_SPEC"` line in the rendered Dockerfile'
    # The template renders the same args twice (primer + the OSPREY_DEV
    # fallback). Both must carry the project's requirements, or a dev build
    # would silently produce a different image than a release build.
    assert all(rendered == args[0] for rendered in args), "install lines disagree"
    return args[0]


def recorded_deps(project_path: Path) -> list[str]:
    """The project's recorded non-osprey requirements, in recorded order."""
    data = tomllib.loads((project_path / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert deps[0] == OSPREY_SPEC, "osprey's own requirement is recorded first"
    return deps[1:]


# --------------------------------------------------------------------------
# The equality
# --------------------------------------------------------------------------


class TestImageMatchesRecord:
    """One list reaches the venv, the record and the image."""

    def test_image_installs_the_recorded_set(self, build, base_venv):
        """The whole property, in one assertion: same requirements, same order."""
        python = base_venv(_record("lmfit", "1.3.2"), _record("cothread", "2.20"))

        project = build(
            _profile_yaml(python=python, dependencies=["numpy>=1.24"], packages=["humanize"])
        )

        assert image_deps(project) == recorded_deps(project)
        assert image_deps(project) == [
            "cothread==2.20",
            "lmfit==1.3.2",
            "numpy>=1.24",
            "humanize",
        ]

    def test_frozen_base_package_reaches_the_image(self, build, base_venv):
        """The reported drift, by name: on the host venv, absent from the image."""
        python = base_venv(_record("lmfit", "1.3.2"))

        project = build(_profile_yaml(python=python))

        assert "lmfit==1.3.2" in image_deps(project)

    def test_environment_packages_reach_the_image(self, build, base_venv):
        """`environment.packages` is installed on the host, so the image needs it."""
        python = base_venv()

        project = build(_profile_yaml(python=python, packages=["humanize"]))

        assert image_deps(project) == recorded_deps(project) == ["humanize"]

    def test_osprey_is_not_installed_twice(self, build, base_venv):
        """The image installs osprey from `$OSPREY_PIP_SPEC`, not from this list."""
        python = base_venv(_record("lmfit", "1.3.2"))

        project = build(_profile_yaml(python=python, dependencies=["osprey-framework==9.9.9"]))

        assert not [dep for dep in image_deps(project) if dep.startswith("osprey-framework")]

    def test_deduplicated_once_for_both_consumers(self, build, base_venv):
        """A pin that displaced a bare name must be displaced in the image too."""
        python = base_venv(_record("numpy", "2.2.6"))

        project = build(_profile_yaml(python=python, dependencies=["numpy"]))

        assert image_deps(project) == recorded_deps(project) == ["numpy==2.2.6"]

    def test_constrained_requirement_survives_shell_quoting(self, build, base_venv):
        """`>=` unquoted would redirect the RUN's output into a file named `=1.24`."""
        python = base_venv()
        project = build(_profile_yaml(python=python, dependencies=["numpy>=1.24"]))

        text = (project / "Dockerfile").read_text(encoding="utf-8")

        assert "'numpy>=1.24'" in text
        assert image_deps(project) == ["numpy>=1.24"]


# --------------------------------------------------------------------------
# Cases with nothing to freeze
# --------------------------------------------------------------------------


class TestNoFrozenBase:
    """An undeclared base is not enumerated, so the image is what it always was."""

    def test_undeclared_base_installs_profile_dependencies(self, build):
        project = build(_profile_yaml(dependencies=["numpy>=1.24", "humanize"]))

        assert image_deps(project) == ["numpy>=1.24", "humanize"]
        assert image_deps(project) == recorded_deps(project)

    def test_undeclared_base_carries_no_build_host_packages(self, build):
        """Nothing of osprey's own dev environment may leak into the image."""
        project = build(_profile_yaml(dependencies=["numpy>=1.24"]))

        assert image_deps(project) == ["numpy>=1.24"]

    def test_no_dependencies_renders_an_empty_install_tail(self, build):
        project = build(_profile_yaml())

        assert image_deps(project) == []
        assert recorded_deps(project) == []


class TestSkipDeps:
    """`--skip-deps` builds no venv, so there is no assembled list to carry."""

    def test_falls_back_to_profile_dependencies(self, build, base_venv):
        """The declared base is never probed and its packages never appear."""
        python = base_venv(_record("lmfit", "1.3.2"))

        project = build(
            _profile_yaml(python=python, dependencies=["numpy>=1.24"], packages=["humanize"]),
            skip_deps=True,
        )

        assert image_deps(project) == ["numpy>=1.24"]

    def test_writes_no_dependency_record(self, build):
        """Nothing was installed, so nothing is recorded — and nothing can diverge."""
        project = build(_profile_yaml(dependencies=["numpy>=1.24"]), skip_deps=True)

        assert not (project / "pyproject.toml").exists()
