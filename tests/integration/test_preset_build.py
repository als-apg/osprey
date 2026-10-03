"""Every shipped preset initialises and builds.

``osprey build --skip-deps`` runs on a fresh ``osprey init`` of each preset under
``src/osprey/profiles/presets/`` and must exit 0 with the facility file at the
build root. The build renders without the project venv, so a preset whose
sources stop the facility build, or whose profile no longer renders, is caught
here without an install.

A control-assistant render that selects the middle-layer pipeline answers
``run_sql`` from the DuckDB index the build writes.

The unit lane ignores this module by name and the Tier 0 static job runs it by
name: nine real builds are worth one reading per run, not one per matrix cell.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.facility.render import FACILITY_FILE
from osprey.utils.workspace import BUILD_DIR_NAME
from tests._builds import init_project, run_build

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"

#: The presets that ship.
PRESETS = sorted(path.stem for path in PRESETS_DIR.glob("*.yml"))


def test_nine_presets_ship() -> None:
    assert len(PRESETS) == 9, PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_builds(preset: str, tmp_path: Path) -> None:
    repo = init_project(tmp_path, preset, "demo")

    result = run_build(repo)

    assert result.exit_code == 0, result.output
    assert (repo / BUILD_DIR_NAME / FACILITY_FILE).is_file()


@pytest.fixture
def _fresh_middle_layer_server() -> Iterator[None]:
    """A middle-layer server context and config cache built for this test only."""
    import osprey.utils.config as config
    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        reset_cf_ml_context,
    )
    from osprey.utils.workspace import reset_config_cache

    def reset() -> None:
        reset_cf_ml_context()
        reset_config_cache()
        config._default_config = None
        config._default_configurable = None
        config._config_cache.clear()

    reset()
    yield
    reset()


@pytest.mark.usefixtures("_fresh_middle_layer_server")
def test_a_middle_layer_render_answers_run_sql_with_its_channel_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from click.testing import CliRunner

    from osprey.cli.init_cmd import init
    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        initialize_cf_ml_context,
    )
    from osprey.mcp_server.channel_finder_middle_layer.tools.run_sql import run_sql

    repo = tmp_path / "demo"
    result = CliRunner().invoke(
        init,
        [
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "channel_finder_mode=middle_layer",
        ],
    )
    assert result.exit_code == 0, result.output
    built = run_build(repo)
    assert built.exit_code == 0, built.output

    monkeypatch.setenv("OSPREY_CONFIG", str(repo / BUILD_DIR_NAME / "config.yml"))
    context = initialize_cf_ml_context()
    answer = json.loads(
        getattr(run_sql, "fn", run_sql)(
            sql="SELECT count(DISTINCT channel_name) AS n FROM channels"
        )
    )

    assert context.duckdb_path is not None
    assert answer["rows"] == [{"n": len(context.database.channel_map)}]
    assert len(context.database.channel_map) > 0
