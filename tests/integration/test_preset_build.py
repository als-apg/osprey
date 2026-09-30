"""Every shipped preset initialises and builds.

``osprey build --skip-deps`` runs on a fresh ``osprey init`` of each preset under
``src/osprey/profiles/presets/`` and must exit 0 with the facility file at the
build root. The build renders without the project venv, so a preset whose
sources stop the facility build, or whose profile no longer renders, is caught
here without an install.

The unit lane ignores this module by name and the Tier 0 static job runs it by
name: nine real builds are worth one reading per run, not one per matrix cell.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.facility.render import FACILITY_FILE
from osprey.utils.workspace import BUILD_DIR_NAME

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"

#: The presets that ship.
PRESETS = sorted(path.stem for path in PRESETS_DIR.glob("*.yml"))


def test_nine_presets_ship() -> None:
    assert len(PRESETS) == 9, PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_builds(preset: str, tmp_path: Path) -> None:
    repo = tmp_path / "demo"
    result = CliRunner().invoke(init, [str(repo), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps"])

    assert result.exit_code == 0, result.output
    assert (repo / BUILD_DIR_NAME / FACILITY_FILE).is_file()
