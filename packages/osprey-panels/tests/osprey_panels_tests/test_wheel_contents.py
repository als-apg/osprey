"""The built osprey-panels wheel ships the engine's data and nothing of the repository.

The engine validates every screen it writes against the Screen-IR schema it
vendors, and measures text with the font-metrics table ``metrics.py``. Both are
read from the installed package, so they must be inside the wheel at the path
the engine resolves, not only in the source tree. The font-metrics generator,
the font it measured, the package's tests and its checks module are repository
workflow and stay out. The wheel declares the ``osprey-panels`` command.

The tests read one built wheel: the one ``OSPREY_PANELS_WHEEL`` names (the
package's standalone CI job builds it once and sets the variable), or, when the
variable is unset, one they build themselves into a temporary directory.
"""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from osprey_panels_tests._harness._paths import PACKAGE_ROOT, PACKAGE_SRC, REPO_ROOT

pytestmark = pytest.mark.wheel_build

#: The variable naming an already built wheel.
WHEEL_VARIABLE = "OSPREY_PANELS_WHEEL"

#: The command that builds the wheel when the variable is unset.
BUILD = ("uv", "build", "--package", "osprey-panels", "--wheel")

#: The engine's sources.
ENGINE = PACKAGE_SRC / "osprey_panels"

#: Package data the engine reads at run time, as wheel members.
SHIPPED = ("osprey_panels/screen-ir.schema.json", "osprey_panels/metrics.py")

#: The console script the wheel declares, as ``entry_points.txt`` spells it.
CONSOLE_SCRIPT = ("osprey-panels", "osprey_panels.cli:main")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The built wheel: the one the environment names, or one built here."""
    named = os.environ.get(WHEEL_VARIABLE)
    if named:
        path = Path(named)
        assert path.is_file(), f"{WHEEL_VARIABLE}={named} names no file"
        return path
    uv = shutil.which("uv")
    assert uv, f"{WHEEL_VARIABLE} is unset and uv is not on PATH to build the wheel"
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [uv, *BUILD[1:], "--out-dir", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    built = sorted(out.glob("osprey_panels-*.whl"))
    assert len(built) == 1, f"{' '.join(BUILD)} wrote {[p.name for p in built]}"
    return built[0]


@pytest.fixture(scope="module")
def members(wheel: Path) -> dict[str, bytes]:
    """Every file in the wheel, by its archive path."""
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _dist_info(members: dict[str, bytes], name: str) -> bytes:
    """The one ``*.dist-info/<name>`` member of the wheel."""
    found = [
        member
        for member in members
        if PurePosixPath(member).name == name
        and PurePosixPath(member).parent.suffix == ".dist-info"
    ]
    assert len(found) == 1, f"the wheel holds {found} for {name}"
    return members[found[0]]


@pytest.mark.parametrize("member", SHIPPED)
def test_the_wheel_ships_the_engine_data(members: dict[str, bytes], member: str) -> None:
    assert member in members, f"the wheel ships no {member}"
    source = ENGINE / PurePosixPath(member).name
    assert members[member] == source.read_bytes(), f"the wheel's {member} differs from {source}"


def test_the_wheel_ships_only_the_engine_package(members: dict[str, bytes]) -> None:
    tops = {PurePosixPath(name).parts[0] for name in members}
    assert {top for top in tops if not top.endswith(".dist-info")} == {"osprey_panels"}


@pytest.mark.parametrize(
    ("what", "refused"),
    [
        pytest.param("the font-metrics generator", lambda name: "panel_font_metrics" in name),
        pytest.param("the tools directory", lambda name: "tools/" in name),
        pytest.param(
            "the measured font and its licence",
            lambda name: "panel_font" in name or name.endswith((".woff", ".woff2", ".ttf")),
        ),
        pytest.param(
            "the package's tests",
            lambda name: "osprey_panels_tests" in name or "/tests/" in f"/{name}",
        ),
        pytest.param("the package's checks module", lambda name: name.endswith("repo_checks.py")),
        pytest.param("a changelog fragment", lambda name: "changelog.d/" in name),
    ],
    ids=["generator", "tools", "font", "tests", "repo-checks", "changelog-fragments"],
)
def test_the_wheel_ships_nothing_of_the_repository(
    members: dict[str, bytes], what: str, refused
) -> None:
    shipped = sorted(name for name in members if refused(name))
    assert shipped == [], f"the wheel ships {what}: {shipped}"


def test_the_excluded_files_exist_in_the_source_tree() -> None:
    """Control: the files the wheel must leave out are there to be left out."""
    assert (PACKAGE_ROOT / "tools" / "panel_font_metrics.py").is_file()
    assert any((PACKAGE_ROOT / "tools" / "panel_font").glob("*.woff"))
    assert (PACKAGE_ROOT / "repo_checks.py").is_file()


def test_the_wheel_declares_the_console_script(members: dict[str, bytes]) -> None:
    entry_points = configparser.ConfigParser()
    entry_points.read_string(_dist_info(members, "entry_points.txt").decode("utf-8"))
    name, target = CONSOLE_SCRIPT
    assert entry_points.has_section("console_scripts"), "the wheel declares no console script"
    assert dict(entry_points["console_scripts"]) == {name: target}


def test_the_wheel_names_jsonschema_as_its_only_dependency(members: dict[str, bytes]) -> None:
    metadata = _dist_info(members, "METADATA").decode("utf-8")
    requires = [
        line.removeprefix("Requires-Dist:").strip()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    assert [req for req in requires if "extra ==" not in req] == ["jsonschema<5,>=4.26"]
