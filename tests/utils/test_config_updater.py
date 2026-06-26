"""Tests for config updater functions."""

import yaml

from osprey.utils.config_writer import update_yaml_file

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
        """Test that backup file is created when requested."""
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
