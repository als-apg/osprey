"""The CANOPY panel's image checks, as an image build runs them.

The panel names two commands in the registry (``SidecarPanel.image_checks``),
each a run of :mod:`osprey.integrations.canopy.image_check`, and
``Dockerfile.j2`` renders each as ``RUN cd /app/<project> && <command>`` for
every image whose profile shows the panel. Each command has to decide by
itself, because the Dockerfile adds nothing around it:

* an image that carries CANOPY has its library checked against the library's
  write-binding report, and its facility collections checked against CANOPY's
  rules and that library; a failing check fails the build;
* a deployment without a collections directory has none to check, says so,
  and builds;
* an image without CANOPY serves neither, says so, and builds;
* nothing else passes: a probe that could not answer is a failed build, not a
  missing package.

These run the rendered line, not a restatement of it, under interpreters made
here: a virtual environment that sees OSPREY's sources but cannot see
``canopy_mcp`` whatever the test environment has installed, and one that also
holds a stand-in for it. So those cases do not depend on CANOPY being installed
or absent where the tests run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

import osprey
from osprey.cli.build_cmd import _profile_image_checks_context
from osprey.cli.templates.manager import TemplateManager
from osprey.integrations.canopy import image_check
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.profiles.sidecar_panels import SIDECAR_PANELS

PROJECT = "demo"
PREFIX = f"RUN cd /app/{PROJECT} && "

#: The two rendered commands.
LIBRARY_COMMAND = "python -I -m osprey.integrations.canopy.image_check library"
COLLECTIONS_COMMAND = "python -I -m osprey.integrations.canopy.image_check collections"

#: What each check says when it skips. Fixed text, and none is another's, so a
#: build log tells the skips apart.
NOT_INSTALLED = "canopy not installed, library not served"
COLLECTIONS_NOT_INSTALLED = "canopy not installed, collections not served"
NO_COLLECTIONS = "no facility collections, none checked"

#: The directory that holds the ``osprey`` package: an interpreter made here sees
#: OSPREY through it, and nothing else this environment has installed.
OSPREY_ROOT = Path(osprey.__file__).resolve().parent.parent

#: The stand-in for a ``canopy_mcp`` module: it prints its name and arguments
#: and exits with the code it is told to.
_STAND_IN_TEMPLATE = """\
import os
import sys

if __name__ == "__main__":
    print("{name} " + " ".join(sys.argv[1:]))
    sys.exit(int(os.environ["STAND_IN_EXIT"]))
"""
_STAND_IN = _STAND_IN_TEMPLATE.format(name="report")
_COLLECTIONS_STAND_IN = _STAND_IN_TEMPLATE.format(name="collections")


class _Profile:
    """The one attribute of a resolved profile the image context reads."""

    def __init__(self, *web_panels: str) -> None:
        self.web_panels = list(web_panels)


def _dockerfile(*web_panels: str) -> str:
    """``Dockerfile.j2`` as rendered for a profile that selects *web_panels*."""
    template = TemplateManager().jinja_env.get_template("project/Dockerfile.j2")
    return template.render(
        project_name=PROJECT,
        claude_code_cli_version="1.2.3",
        port_base=DEFAULT_PORT_BASE,
        osprey_ports=layout_ports(DEFAULT_PORT_BASE),
        **_profile_image_checks_context(_Profile(*web_panels)),
    )


def _check_lines(dockerfile: str) -> list[str]:
    return [line for line in dockerfile.splitlines() if line.startswith("RUN cd /app/")]


@pytest.fixture(scope="module")
def rendered_checks() -> list[str]:
    """The shell commands of the rendered check lines, without their ``cd``."""
    lines = _check_lines(_dockerfile("artifacts", "canopy", "jupyter"))
    assert len(lines) == 2, lines
    assert all(line.startswith(PREFIX) for line in lines)
    return [line.removeprefix(PREFIX) for line in lines]


@pytest.fixture(scope="module")
def rendered_check(rendered_checks) -> str:
    """The library check's command."""
    return rendered_checks[0]


@pytest.fixture(scope="module")
def collections_check(rendered_checks) -> str:
    """The collections check's command."""
    return rendered_checks[1]


def _make_venv(root: Path) -> Path:
    """A virtual environment whose only package is OSPREY's; returns its ``bin``.

    The check module imports nothing outside the standard library and its own
    package, so the sources alone are enough to run it.
    """
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(root)
    bin_dir = root / "bin"
    (_site_packages(bin_dir) / "osprey-sources.pth").write_text(
        f"{OSPREY_ROOT}\n", encoding="utf-8"
    )
    return bin_dir


def _site_packages(bin_dir: Path) -> Path:
    answer = subprocess.run(
        [
            str(bin_dir / "python"),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(answer.stdout.strip())


@pytest.fixture(scope="module")
def bare_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An interpreter that sees OSPREY but not ``canopy_mcp``, wherever these tests run."""
    bin_dir = _make_venv(tmp_path_factory.mktemp("image-check-bare") / "venv")
    seen = subprocess.run(
        [
            str(bin_dir / "python"),
            "-I",
            "-c",
            "import importlib.util as u; print(u.find_spec('canopy_mcp'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert seen.stdout.strip() == "None"
    return bin_dir


@pytest.fixture(scope="module")
def stand_in_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An interpreter whose ``canopy_mcp.report`` is the stand-in."""
    bin_dir = _make_venv(tmp_path_factory.mktemp("image-check-stand-in") / "venv")
    package = _site_packages(bin_dir) / "canopy_mcp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "report.py").write_text(_STAND_IN, encoding="utf-8")
    (package / "collections_cli.py").write_text(_COLLECTIONS_STAND_IN, encoding="utf-8")
    return bin_dir


def _run(
    check: str, bin_dir: Path, deployment: Path, **extra_env: str
) -> subprocess.CompletedProcess[str]:
    """Run the rendered line as the build does: ``cd`` in, then the command.

    ``python`` is whatever *bin_dir* holds, as it is whatever the image holds.
    ``PYTHONPATH`` names the test interpreter's own packages on purpose: the
    check has to ignore it.
    """
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
        **extra_env,
    }
    return subprocess.run(
        ["/bin/sh", "-c", f'cd "{deployment}" && {check}'],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# -- the registry entry and the rendered line --------------------------------------


class TestTheRenderedLine:
    def test_the_panel_names_two_checks(self):
        assert SIDECAR_PANELS["canopy"].image_checks == (LIBRARY_COMMAND, COLLECTIONS_COMMAND)

    def test_they_render_verbatim_behind_the_deployment_directory(self, rendered_checks):
        assert rendered_checks == list(SIDECAR_PANELS["canopy"].image_checks)
        assert all("\n" not in check for check in rendered_checks)

    def test_they_run_after_the_deployment_is_copied_in(self):
        lines = _dockerfile("canopy").splitlines()
        copied = lines.index(f"COPY . /app/{PROJECT}/")
        checks = [i for i, line in enumerate(lines) if line.startswith(PREFIX)]
        handed_over = next(i for i, line in enumerate(lines) if "groupadd --gid" in line)
        assert len(checks) == 2
        assert copied < checks[0] < checks[1] < handed_over

    def test_every_skip_says_so_in_its_own_words(self):
        """A build log names which check skipped, and why."""
        skips = [
            image_check.LIBRARY_NOT_INSTALLED,
            image_check.COLLECTIONS_NOT_INSTALLED,
            image_check.NO_COLLECTIONS,
        ]
        assert skips == [NOT_INSTALLED, COLLECTIONS_NOT_INSTALLED, NO_COLLECTIONS]
        assert not any(a in b for a in skips for b in skips if a != b)

    def test_an_unknown_check_is_a_usage_error(self, capsys):
        assert image_check.main(["everything"]) == 2
        assert image_check.main([]) == 2
        assert "library|collections" in capsys.readouterr().err

    def test_a_profile_without_the_panel_renders_no_check(self):
        assert _check_lines(_dockerfile("artifacts", "jupyter")) == []
        assert _check_lines(_dockerfile()) == []


# -- without CANOPY ----------------------------------------------------------------


class TestWithoutCanopy:
    def test_it_passes_and_says_the_library_is_not_served(self, rendered_check, bare_bin, tmp_path):
        result = _run(rendered_check, bare_bin, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{NOT_INSTALLED}\n"
        assert result.stderr == ""

    def test_a_library_in_the_tree_changes_nothing(self, rendered_check, bare_bin, tmp_path):
        (tmp_path / "data" / "canopy" / "library" / "screens").mkdir(parents=True)
        result = _run(rendered_check, bare_bin, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{NOT_INSTALLED}\n"

    def test_a_package_of_that_name_in_the_deployment_is_not_canopy(
        self, rendered_check, bare_bin, tmp_path
    ):
        """The deployment tree is the working directory, and is not importable."""
        shadow = tmp_path / "canopy_mcp"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("", encoding="utf-8")
        (shadow / "report.py").write_text(_STAND_IN, encoding="utf-8")
        result = _run(rendered_check, bare_bin, tmp_path, STAND_IN_EXIT="9")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{NOT_INSTALLED}\n"


# -- with a CANOPY ------------------------------------------------------------------


class TestWithCanopy:
    def test_a_passing_check_passes(self, rendered_check, stand_in_bin, tmp_path):
        result = _run(rendered_check, stand_in_bin, tmp_path, STAND_IN_EXIT="0")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "report data/canopy/library --served-at library --check\n"

    @pytest.mark.parametrize("code", [1, 2, 3])
    def test_a_failing_check_fails_the_build(self, rendered_check, stand_in_bin, tmp_path, code):
        """Any code, the probe's own "not installed" code included."""
        result = _run(rendered_check, stand_in_bin, tmp_path, STAND_IN_EXIT=str(code))
        assert result.returncode == code
        assert NOT_INSTALLED not in result.stdout


# -- a probe that could not answer -------------------------------------------------


class TestAProbeThatCouldNotAnswer:
    @pytest.mark.parametrize("code", [1, 2, 127])
    def test_it_fails_the_build(self, rendered_check, tmp_path, code):
        """Only the probe's own answer means "not installed"."""
        broken = tmp_path / "bin"
        broken.mkdir()
        python = broken / "python"
        python.write_text(f"#!/bin/sh\nexit {code}\n", encoding="utf-8")
        python.chmod(0o755)
        deployment = tmp_path / "deployment"
        deployment.mkdir()
        result = _run(rendered_check, broken, deployment)
        assert result.returncode != 0
        assert NOT_INSTALLED not in result.stdout

    def test_no_interpreter_at_all_fails_the_build(self, rendered_check, tmp_path):
        empty = tmp_path / "bin"
        empty.mkdir()
        result = _run(rendered_check, empty, tmp_path)
        assert result.returncode != 0
        assert NOT_INSTALLED not in result.stdout


# -- the real report check, on a stale report --------------------------------------


def _with_collections(deployment: Path) -> Path:
    directory = deployment / "data" / "canopy" / "collections"
    directory.mkdir(parents=True)
    return directory


class TestTheCollectionsCheckWithoutCanopy:
    def test_it_passes_and_says_the_collections_are_not_served(
        self, collections_check, bare_bin, tmp_path
    ):
        _with_collections(tmp_path)
        result = _run(collections_check, bare_bin, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{COLLECTIONS_NOT_INSTALLED}\n"
        assert result.stderr == ""

    def test_no_directory_either_still_says_canopy_is_missing(
        self, collections_check, bare_bin, tmp_path
    ):
        result = _run(collections_check, bare_bin, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{COLLECTIONS_NOT_INSTALLED}\n"

    def test_a_package_of_that_name_in_the_deployment_is_not_canopy(
        self, collections_check, bare_bin, tmp_path
    ):
        _with_collections(tmp_path)
        shadow = tmp_path / "canopy_mcp"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("", encoding="utf-8")
        (shadow / "collections_cli.py").write_text(_COLLECTIONS_STAND_IN, encoding="utf-8")
        result = _run(collections_check, bare_bin, tmp_path, STAND_IN_EXIT="9")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{COLLECTIONS_NOT_INSTALLED}\n"


class TestTheCollectionsCheckWithCanopy:
    def test_no_directory_passes_and_says_none_were_checked(
        self, collections_check, stand_in_bin, tmp_path
    ):
        """The stand-in would fail: it is never started."""
        result = _run(collections_check, stand_in_bin, tmp_path, STAND_IN_EXIT="9")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{NO_COLLECTIONS}\n"
        assert result.stderr == ""

    def test_a_file_where_the_directory_belongs_is_not_a_directory(
        self, collections_check, stand_in_bin, tmp_path
    ):
        (tmp_path / "data" / "canopy").mkdir(parents=True)
        (tmp_path / "data" / "canopy" / "collections").write_text("", encoding="utf-8")
        result = _run(collections_check, stand_in_bin, tmp_path, STAND_IN_EXIT="9")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{NO_COLLECTIONS}\n"

    def test_a_passing_check_passes(self, collections_check, stand_in_bin, tmp_path):
        _with_collections(tmp_path)
        result = _run(collections_check, stand_in_bin, tmp_path, STAND_IN_EXIT="0")
        assert result.returncode == 0, result.stderr
        assert result.stdout == (
            "collections data/canopy/collections --library data/canopy/library\n"
        )

    @pytest.mark.parametrize("code", [1, 2, 3])
    def test_a_failing_check_fails_the_build(self, collections_check, stand_in_bin, tmp_path, code):
        """Any code: 3 is a refused collection here, never "not installed"."""
        _with_collections(tmp_path)
        result = _run(collections_check, stand_in_bin, tmp_path, STAND_IN_EXIT=str(code))
        assert result.returncode == code
        assert COLLECTIONS_NOT_INSTALLED not in result.stdout
        assert NO_COLLECTIONS not in result.stdout


class TestTheCollectionsCheckWhenTheProbeCannotAnswer:
    @pytest.mark.parametrize("code", [1, 2, 127])
    def test_it_fails_the_build(self, collections_check, tmp_path, code):
        broken = tmp_path / "bin"
        broken.mkdir()
        python = broken / "python"
        python.write_text(f"#!/bin/sh\nexit {code}\n", encoding="utf-8")
        python.chmod(0o755)
        deployment = tmp_path / "deployment"
        _with_collections(deployment)
        result = _run(collections_check, broken, deployment)
        assert result.returncode != 0
        assert COLLECTIONS_NOT_INSTALLED not in result.stdout
        assert NO_COLLECTIONS not in result.stdout

    def test_no_interpreter_at_all_fails_the_build(self, collections_check, tmp_path):
        empty = tmp_path / "bin"
        empty.mkdir()
        _with_collections(tmp_path)
        result = _run(collections_check, empty, tmp_path)
        assert result.returncode != 0
        assert COLLECTIONS_NOT_INSTALLED not in result.stdout
