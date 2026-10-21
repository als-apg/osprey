"""Tests for control system type configuration in config_updater."""

from textwrap import dedent

import pytest
import yaml

from osprey.utils.config_writer import (
    get_control_system_type,
    set_control_system_type,
)


@pytest.fixture
def sample_config_content():
    """Sample config.yml content for testing."""
    return dedent(
        """
        control_system:
          type: virtual_accelerator
          connector:
            virtual_accelerator:
              serving: in_process
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
    assert control_type == "virtual_accelerator"

    archiver_type = get_control_system_type(config_path, key="archiver.type")
    assert archiver_type == "mock_archiver"


def test_a_non_string_type_reads_as_not_found(tmp_path):
    """A type that is not a string is reported as not found."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("control_system:\n  type: 5\n")

    assert get_control_system_type(config_path) is None


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


def test_set_control_system_type_to_the_simulator(tmp_path):
    """Test switching from EPICS back to the simulator."""
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

    new_content, preview = set_control_system_type(
        config_path, "virtual_accelerator", "mock_archiver"
    )

    # Verify content was updated
    assert "type: virtual_accelerator" in new_content
    assert "type: mock_archiver" in new_content

    # Verify by re-reading
    config_path.write_text(new_content)
    assert get_control_system_type(config_path) == "virtual_accelerator"
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


def test_only_the_types_are_written(tmp_path, sample_config_content):
    """The writer touches the two type keys and nothing else in the section."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, _ = set_control_system_type(config_path, "epics", "epics_archiver")

    data = yaml.safe_load(new_content)
    assert set(data["archiver"]) == {"type"}
