"""A middle-layer build binds ``duckdb_path`` exactly when the profile ships the file.

``channel_finder.pipelines`` is build-derived, so no profile can spell
``duckdb_path`` itself. The build therefore decides it from the one fact it can
see: whether the profile's data tree — the tree the render copies — holds
``channel_databases/middle_layer.duckdb``. With the file the rendered config
names it; without it the key is absent (never an empty string), which is what
the health check and ``run_sql`` read as "not configured".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

#: The posture every deployment states in its own ``config:`` block.
_POSTURE_CONFIG: dict = {
    "control_system.type": "mock",
    "archiver.type": "mock",
    "claude_code.telemetry.enabled": False,
    "hooks.debug": False,
}

_DUCKDB_RELATIVE = Path("channel_databases") / "middle_layer.duckdb"


def _packaged_data_root() -> Path:
    import osprey

    return Path(osprey.__file__).parent / "templates" / "apps" / "control_assistant" / "data"


def _middle_layer_repo(tmp_path: Path, *, with_duckdb: bool, data_dir: str = "data") -> Path:
    """A hello-world-shaped deployment repo pinned to the middle-layer paradigm."""
    repo = tmp_path / ("with-duckdb" if with_duckdb else "without-duckdb")
    repo.mkdir(parents=True)
    profile = {
        "name": "Middle Layer DuckDB",
        "data": data_dir,
        "provider": "cborg",
        "model": "claude-haiku-4-5",
        "channel_finder_mode": "middle_layer",
        "config": dict(_POSTURE_CONFIG),
    }
    (repo / "profile.yml").write_text(yaml.dump(profile, default_flow_style=False))
    data_root = repo / data_dir
    shutil.copytree(_packaged_data_root(), data_root, dirs_exist_ok=True)
    (data_root / "facility_knowledge").mkdir(parents=True, exist_ok=True)
    duckdb = data_root / _DUCKDB_RELATIVE
    if with_duckdb:
        duckdb.parent.mkdir(parents=True, exist_ok=True)
        duckdb.write_bytes(b"not-a-real-duckdb")
    else:
        duckdb.unlink(missing_ok=True)
    return repo


def _build(repo: Path) -> dict:
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n{result.output}\n{result.exception}"
    )
    return yaml.safe_load((repo / "build" / "config.yml").read_text())


def _database_block(config: dict) -> dict:
    return config["channel_finder"]["pipelines"]["middle_layer"]["database"]


def test_duckdb_path_is_rendered_when_the_profile_ships_the_file(tmp_path: Path) -> None:
    config = _build(_middle_layer_repo(tmp_path, with_duckdb=True))
    database = _database_block(config)
    assert database["duckdb_path"] == "data/channel_databases/middle_layer.duckdb"
    assert database["path"] == "data/channel_databases/middle_layer.json"


def test_duckdb_path_is_absent_when_the_profile_ships_no_file(tmp_path: Path) -> None:
    config = _build(_middle_layer_repo(tmp_path, with_duckdb=False))
    database = _database_block(config)
    assert "duckdb_path" not in database
    assert database["path"] == "data/channel_databases/middle_layer.json"


def test_the_file_is_looked_for_under_the_profiles_own_data_root(tmp_path: Path) -> None:
    """A ``data:`` tree that is not ``<repo>/data`` is the one consulted."""
    config = _build(_middle_layer_repo(tmp_path, with_duckdb=True, data_dir="site-data"))
    assert _database_block(config)["duckdb_path"] == "data/channel_databases/middle_layer.duckdb"


@pytest.mark.parametrize("flag", [True, False])
def test_repo_render_context_flag_tracks_the_data_root(tmp_path: Path, flag: bool) -> None:
    from osprey.cli.build_cmd import _repo_render_context
    from osprey.cli.build_profile_model import BuildProfile

    repo = tmp_path / "ctx"
    target = repo / "elsewhere" / _DUCKDB_RELATIVE
    target.parent.mkdir(parents=True)
    if flag:
        target.write_bytes(b"x")
    profile = BuildProfile(name="ctx", data="elsewhere")
    context = _repo_render_context(
        profile,
        repo_root=repo,
        build_dir=repo / "build",
        runtime_root=None,
        project_deps=[],
        skip_deps=True,
    )
    assert context["middle_layer_duckdb"] is flag
