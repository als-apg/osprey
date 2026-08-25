"""Tests for config updater functions."""

import yaml

from osprey.utils.config_writer import (
    CONFIG_BACKUP_DIRNAME,
    config_backup_path,
    update_yaml_file,
)
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

# =============================================================================
# update_yaml_file Tests (moved from test_config_builder.py)
# =============================================================================


class TestUpdateYamlFile:
    """Test comment-preserving YAML file updates."""

    def test_update_preserves_comments(self, tmp_path):
        """Test that comments are preserved when updating YAML."""
        config_file = tmp_path / "config.yml"
        original_content = """# Header comment
project_name: test  # inline comment

# Section comment
control_system:
  type: mock  # type comment
  port: 5064
"""
        config_file.write_text(original_content)

        update_yaml_file(
            config_file,
            {"control_system.type": "epics"},
            create_backup=False,
        )

        updated_content = config_file.read_text()

        # Comments preserved
        assert "# Header comment" in updated_content
        assert "# inline comment" in updated_content
        assert "# Section comment" in updated_content
        assert "# type comment" in updated_content

        # Value updated
        assert "type: epics" in updated_content
        assert "type: mock" not in updated_content

    def test_update_preserves_blank_lines(self, tmp_path):
        """Test that blank lines are preserved when updating YAML."""
        config_file = tmp_path / "config.yml"
        original_content = """project_name: test

control_system:
  type: mock

models:
  name: test
"""
        config_file.write_text(original_content)

        update_yaml_file(
            config_file,
            {"control_system.type": "epics"},
            create_backup=False,
        )

        updated_content = config_file.read_text()

        # Structure should be preserved with blank line separators
        assert "project_name: test" in updated_content
        assert "type: epics" in updated_content

    def test_update_creates_nested_path(self, tmp_path):
        """Test that nested paths are created when they don't exist."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        update_yaml_file(
            config_file,
            {"control_system.connector.epics.port": 5064},
            create_backup=False,
        )

        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["control_system"]["connector"]["epics"]["port"] == 5064

    def test_update_creates_backup(self, tmp_path):
        """Test that backup file is created when requested.

        The backup lands in the agent-data state zone, not beside the config.
        The zone is anchored on the deployment repo the config sits in, which
        for a flat project is the config's own directory.
        """
        config_file = tmp_path / "config.yml"
        original_content = "project_name: original\n"
        config_file.write_text(original_content)

        backup_path = update_yaml_file(
            config_file,
            {"project_name": "updated"},
            create_backup=True,
        )

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text() == original_content
        assert backup_path == config_backup_path(config_file)
        assert backup_path.parent.name == CONFIG_BACKUP_DIRNAME
        # The point of the move: nothing new appears next to the config itself.
        assert not (tmp_path / "config.yml.bak").exists()
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".bak"] == []

    def test_update_backup_anchors_on_the_repo_root_not_the_render(self, tmp_path):
        """The anchor is the repo, not the directory config.yml happens to be in.

        ``agent_data.base_dir`` is rendered and documented relative to the
        deployment repo ROOT -- the tree holding ``profile.yml`` and the durable
        ``var/`` -- while ``config.yml`` lives one level down in the render that
        every build re-creates. Anchoring on the config's own parent would put
        the backup inside that disposable zone, and inside a container it would
        put it inside the ROOT-OWNED one: the mkdir of ``<render>/var`` raises
        ``PermissionError`` before a byte of the save is written. This is the
        real container layout, ``/app/<project>/build/config.yml``, with the
        state zone the image creates and chowns one level up.
        """
        repo = tmp_path / "app" / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        original_content = "project_name: original\n"
        config_file.write_text(original_content)

        backup_path = update_yaml_file(config_file, {"project_name": "updated"})

        assert (
            backup_path
            == repo / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )
        assert backup_path.read_text() == original_content
        # Nothing at all was created in the render zone.
        assert not (render / "var").exists()
        assert [p.name for p in render.iterdir()] == ["config.yml"]

    def test_update_backup_of_a_relocated_render_config_still_follows_the_config(self, tmp_path):
        """Both rules at once: repo-anchored, and the repo's own zone is read.

        A render-zone config that relocates ``agent_data.base_dir`` to a
        relative path must resolve it against the repo root as well -- the pair
        a caller could get individually right and jointly wrong.
        """
        repo = tmp_path / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        config_file.write_text("agent_data:\n  base_dir: state/data\n", encoding="utf-8")

        backup_path = config_backup_path(config_file)

        assert backup_path == repo / "state" / "data" / CONFIG_BACKUP_DIRNAME / "config.yml.bak"

    def test_update_backup_follows_a_relocated_agent_data_root(self, tmp_path):
        """The zone is read from the config being written, never assumed.

        The case a hard-coded ``var/agent_data`` would pass and this scheme must
        not: the root is moved somewhere no default would guess, and the backup
        has to go with it. Read off the file itself because this entry point has
        no other source of truth for the project it is editing.
        """
        relocated = tmp_path / "elsewhere" / "state"
        config_file = tmp_path / "config.yml"
        original_content = f"agent_data:\n  base_dir: {relocated}\nproject_name: original\n"
        config_file.write_text(original_content)

        backup_path = update_yaml_file(config_file, {"project_name": "updated"})

        assert backup_path == relocated / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        assert backup_path.read_text() == original_content
        assert not (tmp_path / "var").exists()

    def test_update_backup_follows_the_configs_own_project_root_key(self, tmp_path):
        """The anchor mirrors ``resolve_project_root``, key first.

        The shape the two rules disagree on, and it is a real one: a project
        directory literally NAMED ``build`` reads as a render to
        ``repo_root_for_config`` — its documented non-goal — so the tail rule
        alone would put the backup one level ABOVE the project while every
        runtime reader of the state zone looked inside it. The config's own
        ``project_root`` key is what settles it, exactly as it does at runtime.
        """
        project = tmp_path / "build"
        project.mkdir()
        config_file = project / "config.yml"
        config_file.write_text(f"project_root: {project}\n", encoding="utf-8")

        backup_path = config_backup_path(config_file)

        assert (
            backup_path
            == project / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )
        assert not (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).exists()

    def test_update_backup_ignores_a_project_root_that_is_not_here(self, tmp_path):
        """A build recorded on another machine names a path that is not here.

        Same qualifier ``resolve_project_root`` applies: the key wins only when
        it names a directory that exists, so a container reading a config staged
        on the host does not anchor its state zone on the host's path.
        """
        repo = tmp_path / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        config_file.write_text("project_root: /build/machine/psplit\n", encoding="utf-8")

        assert (
            config_backup_path(config_file)
            == repo / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )

    def test_update_backup_of_an_unreadable_config_uses_the_default_zone(self, tmp_path):
        """A config too broken to name its own zone is the one most worth copying."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system: [unterminated\n")

        backup_path = config_backup_path(config_file)

        assert (
            backup_path
            == tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )

    def test_update_no_backup_when_disabled(self, tmp_path):
        """Test that no backup is created when disabled."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        backup_path = update_yaml_file(
            config_file,
            {"project_name": "updated"},
            create_backup=False,
        )

        assert backup_path is None
        assert not (tmp_path / "config.yml.bak").exists()
        assert not config_backup_path(config_file).exists()

    def test_update_with_nested_dict(self, tmp_path):
        """Test updating with nested dictionary structure."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        update_yaml_file(
            config_file,
            {
                "simulation": {
                    "ioc": {"name": "test_ioc", "port": 5064},
                    "backend": {"type": "mock"},
                }
            },
            create_backup=False,
        )

        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["simulation"]["ioc"]["name"] == "test_ioc"
        assert updated_config["simulation"]["ioc"]["port"] == 5064
        assert updated_config["simulation"]["backend"]["type"] == "mock"

    def test_update_merges_nested_dicts(self, tmp_path):
        """Test that nested dicts are merged, not replaced."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """control_system:
  type: mock
  connector:
    epics:
      timeout: 30
"""
        )

        update_yaml_file(
            config_file,
            {"control_system": {"type": "epics", "connector": {"epics": {"port": 5064}}}},
            create_backup=False,
        )

        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        # Updated value
        assert updated_config["control_system"]["type"] == "epics"
        assert updated_config["control_system"]["connector"]["epics"]["port"] == 5064
        # Original value preserved
        assert updated_config["control_system"]["connector"]["epics"]["timeout"] == 30

    def test_update_adds_section_comment_for_new_key(self, tmp_path):
        """Test that section comments are added for new top-level keys."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """project_name: test
control_system:
  type: mock
"""
        )

        update_yaml_file(
            config_file,
            {"simulation": {"ioc": {"name": "test_ioc", "port": 5064}}},
            create_backup=False,
            section_comments={"simulation": "SIMULATION CONFIGURATION"},
        )

        updated_content = config_file.read_text()

        # Section comment should be present in boxed format
        assert "# ====" in updated_content  # Separator line
        assert "# SIMULATION CONFIGURATION" in updated_content
        # Content should be there
        assert "simulation:" in updated_content
        assert "test_ioc" in updated_content

    def test_update_no_comment_for_existing_key(self, tmp_path):
        """Test that section comments are NOT added for existing keys."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """project_name: test
simulation:
  old_key: old_value
"""
        )

        update_yaml_file(
            config_file,
            {"simulation": {"new_key": "new_value"}},
            create_backup=False,
            section_comments={"simulation": "Simulation Configuration"},
        )

        # Section comment should NOT be added since simulation already existed
        # (comment is only for NEW keys)
        # The merge happens, new_key is added
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert config["simulation"]["new_key"] == "new_value"
