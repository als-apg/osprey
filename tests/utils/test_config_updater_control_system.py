"""Tests for control system type configuration in config_updater."""

from textwrap import dedent

import pytest
import yaml

from osprey.utils.config_writer import (
    get_control_system_type,
    set_control_system_type,
)

MONGODB_SETTINGS = {
    "archiver.mongodb_archiver.host": "localhost",
    "archiver.mongodb_archiver.port": 27017,
    "archiver.mongodb_archiver.name": "osprey_archiver",
    "archiver.mongodb_archiver.collection": "pv_history",
    "archiver.mongodb_archiver.auth": "admin",
    "archiver.mongodb_archiver.username": "osprey",
    "archiver.mongodb_archiver.password_env": "MONGO_ROOT_PASSWORD",
    "archiver.mongodb_archiver.timeout": 5,
}


@pytest.fixture
def sample_config_content():
    """Sample config.yml content for testing."""
    return dedent(
        """
        control_system:
          type: mock
          writes_enabled: false

        archiver:
          type: mock_archiver
    """
    )


def test_get_control_system_type(tmp_path, sample_config_content):
    """Test reading control system type from config."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    control_type = get_control_system_type(config_path)
    assert control_type == "mock"

    archiver_type = get_control_system_type(config_path, key="archiver.type")
    assert archiver_type == "mock_archiver"


def test_set_control_system_type_to_epics(tmp_path, sample_config_content):
    """Test switching from mock to EPICS."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, preview = set_control_system_type(config_path, "epics", "epics_archiver")

    # Verify preview
    assert "epics" in preview
    assert "epics_archiver" in preview

    # Verify content was updated
    assert "type: epics" in new_content
    assert "type: epics_archiver" in new_content

    # Verify by re-reading
    config_path.write_text(new_content)
    assert get_control_system_type(config_path) == "epics"
    assert get_control_system_type(config_path, key="archiver.type") == "epics_archiver"


def test_set_control_system_type_to_mock(tmp_path):
    """Test switching from EPICS back to mock."""
    content = dedent(
        """
        control_system:
          type: epics
          writes_enabled: true

        archiver:
          type: epics_archiver
    """
    )

    config_path = tmp_path / "config.yml"
    config_path.write_text(content)

    new_content, preview = set_control_system_type(config_path, "mock", "mock_archiver")

    # Verify content was updated
    assert "type: mock" in new_content
    assert "type: mock_archiver" in new_content

    # Verify by re-reading
    config_path.write_text(new_content)
    assert get_control_system_type(config_path) == "mock"
    assert get_control_system_type(config_path, key="archiver.type") == "mock_archiver"


def test_set_control_system_only(tmp_path, sample_config_content):
    """Test updating control system without changing archiver."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, preview = set_control_system_type(
        config_path,
        "epics",
        None,  # Don't update archiver
    )

    # Control system should be updated
    assert "type: epics" in new_content

    # Archiver should remain mock_archiver
    config_path.write_text(new_content)
    assert get_control_system_type(config_path) == "epics"
    assert get_control_system_type(config_path, key="archiver.type") == "mock_archiver"


def test_archiver_settings_land_as_nested_sections(tmp_path, sample_config_content):
    """An archiver that needs settings gets them written under its own section.

    Nested, not as top-level dotted lines: config.yml is read as nested
    sections, so a flat `archiver.mongodb_archiver.host:` key would configure
    nothing while looking like it configured something.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, _ = set_control_system_type(
        config_path,
        "virtual_accelerator",
        "mongodb_archiver",
        archiver_settings=MONGODB_SETTINGS,
    )

    data = yaml.safe_load(new_content)
    assert [key for key in data if "." in key] == []
    assert data["archiver"]["type"] == "mongodb_archiver"
    assert data["archiver"]["mongodb_archiver"] == {
        "host": "localhost",
        "port": 27017,
        "name": "osprey_archiver",
        "collection": "pv_history",
        "auth": "admin",
        "username": "osprey",
        "password_env": "MONGO_ROOT_PASSWORD",
        "timeout": 5,
    }


def test_archiver_settings_appear_in_the_preview(tmp_path, sample_config_content):
    """What is about to be written is what the confirmation prompt shows."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    _, preview = set_control_system_type(
        config_path,
        "virtual_accelerator",
        "mongodb_archiver",
        archiver_settings=MONGODB_SETTINGS,
    )

    assert "archiver.mongodb_archiver.host: localhost" in preview
    assert "archiver.mongodb_archiver.password_env: MONGO_ROOT_PASSWORD" in preview


def test_archiver_settings_preserve_surrounding_comments(tmp_path):
    """The connection block is added without flattening the file around it."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        dedent(
            """
            # Control system
            control_system:
              type: mock
              writes_enabled: false  # safety

            archiver:
              type: mock_archiver
            """
        )
    )

    new_content, _ = set_control_system_type(
        config_path,
        "virtual_accelerator",
        "mongodb_archiver",
        archiver_settings=MONGODB_SETTINGS,
    )

    assert "# Control system" in new_content
    assert "# safety" in new_content


def test_no_archiver_settings_leaves_the_section_alone(tmp_path, sample_config_content):
    """Omitting the settings writes only the types, as it always has."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, _ = set_control_system_type(config_path, "epics", "epics_archiver")

    data = yaml.safe_load(new_content)
    assert set(data["archiver"]) == {"type"}
