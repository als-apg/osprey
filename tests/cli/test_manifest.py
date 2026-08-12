"""``data/`` is build-owned: the manifest checksums it, a rebuild clears it.

Everything under a rendered ``data/`` tree (machine models, channel databases,
benchmark query sets) comes from the profile or preset, so it belongs in the
manifest's checksum set — a difference there is real drift, not user state.

Runtime state is not in the checksum set because it is not in the render at
all: it lives in the repo's state zone (``var/agent_data``, the value of
:data:`osprey.utils.config.RUNTIME_STATE_DIR`), a sibling of ``build/`` that a
rebuild never walks. :func:`calculate_file_checksums` additionally filters the
name ``_agent_data`` — the older in-project spelling — which is what the
fixture below plants and the exclusion tests exercise; it is a name filter in
the checksummer, not a claim about where runtime writers put things today.

The config-validation warning covers the way the two can still be confused: a
runtime-write path deliberately pointed back into ``data/``.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manifest import calculate_file_checksums
from osprey.utils.config import (
    RUNTIME_STATE_DIR,
    RUNTIME_WRITE_PATH_KEYS,
    ConfigBuilder,
    find_runtime_write_paths_under_data,
)


def _project(tmp_path: Path) -> Path:
    """A project tree with one file in each ownership class."""
    project = tmp_path / "proj"
    (project / "data" / "simulation").mkdir(parents=True)
    (project / "data" / "channel_databases").mkdir(parents=True)
    (project / "_agent_data" / "simulation").mkdir(parents=True)
    (project / ".git").mkdir()

    (project / "config.yml").write_text("project_root: .\n")
    (project / "data" / "simulation" / "machine.json").write_text('{"name": "sim"}')
    (project / "data" / "channel_databases" / "in_context.json").write_text('{"channels": []}')
    (project / "_agent_data" / "simulation" / "active_scenarios").write_text("nominal\n")
    (project / ".env").write_text("SECRET=1\n")
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (project / ".osprey-manifest.json").write_text("{}")
    return project


class TestChecksumScope:
    def test_data_tree_is_checksummed(self, tmp_path):
        checksums = calculate_file_checksums(_project(tmp_path))

        assert "data/simulation/machine.json" in checksums
        assert "data/channel_databases/in_context.json" in checksums
        assert checksums["data/simulation/machine.json"].startswith("sha256:")

    def test_runtime_and_secret_trees_stay_excluded(self, tmp_path):
        checksums = calculate_file_checksums(_project(tmp_path))

        assert "_agent_data/simulation/active_scenarios" not in checksums
        assert ".env" not in checksums
        assert ".git/HEAD" not in checksums
        assert ".osprey-manifest.json" not in checksums

    def test_a_write_into_data_changes_the_checksums(self, tmp_path):
        """The point of including ``data/``: runtime writes there are visible."""
        project = _project(tmp_path)
        before = calculate_file_checksums(project)

        (project / "data" / "simulation" / "active_scenarios").write_text("vacuum-burst\n")

        after = calculate_file_checksums(project)
        assert after != before
        assert "data/simulation/active_scenarios" in after

    def test_a_write_into_agent_data_does_not(self, tmp_path):
        project = _project(tmp_path)
        before = calculate_file_checksums(project)

        (project / "_agent_data" / "simulation" / "active_scenarios").write_text("burst\n")

        assert calculate_file_checksums(project) == before


class TestRuntimeWritePathValidation:
    def test_clean_config_reports_nothing(self, tmp_path):
        config = {
            "simulation": {"state_dir": "_agent_data/simulation"},
            "services": {
                "channel_finder": {
                    "pipelines": {
                        "hierarchical": {
                            "feedback": {"store_path": "_agent_data/feedback/store.json"}
                        }
                    }
                }
            },
        }
        assert find_runtime_write_paths_under_data(config, tmp_path) == []

    def test_unset_keys_report_nothing(self, tmp_path):
        assert find_runtime_write_paths_under_data({}, tmp_path) == []

    @pytest.mark.parametrize("key", RUNTIME_WRITE_PATH_KEYS)
    def test_each_key_is_checked(self, tmp_path, key):
        config: dict = {}
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = "data/somewhere/state.json"

        assert find_runtime_write_paths_under_data(config, tmp_path) == [
            (key, "data/somewhere/state.json")
        ]

    def test_absolute_path_under_data_is_caught(self, tmp_path):
        config = {"simulation": {"state_dir": str(tmp_path / "data" / "simulation")}}

        offenders = find_runtime_write_paths_under_data(config, tmp_path)
        assert [key for key, _ in offenders] == ["simulation.state_dir"]

    def test_path_outside_the_project_is_not_caught(self, tmp_path):
        config = {"simulation": {"state_dir": str(tmp_path / "elsewhere" / "data")}}

        assert find_runtime_write_paths_under_data(config, tmp_path) == []

    def test_config_load_warns(self, tmp_path, caplog):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "config.yml").write_text(
            yaml.safe_dump(
                {
                    "project_root": str(project),
                    "simulation": {"state_dir": "data/simulation"},
                }
            )
        )

        with caplog.at_level("WARNING", logger="CONFIG"):
            ConfigBuilder(str(project / "config.yml"), load_env=False)

        assert "simulation.state_dir" in caplog.text
        # The advisory names the durable state zone to point at — pinned through
        # the constant it renders, so the message and the default cannot drift.
        assert RUNTIME_STATE_DIR in caplog.text

    def test_config_load_is_quiet_when_clean(self, tmp_path, caplog):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "config.yml").write_text(
            yaml.safe_dump(
                {
                    "project_root": str(project),
                    "simulation": {"state_dir": "_agent_data/simulation"},
                }
            )
        )

        with caplog.at_level("WARNING", logger="CONFIG"):
            ConfigBuilder(str(project / "config.yml"), load_env=False)

        assert "state_dir" not in caplog.text
