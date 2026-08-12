"""The ``virtual_accelerator`` control-system type, at the config writer.

``set_control_system_type`` is the one function that writes the type, shared by
every path that sets it. The CLI surface above it is pinned where those verbs
live: ``osprey set connector=virtual_accelerator`` in tests/cli/test_set_verb.py,
and the shorthand's validation in tests/cli/test_connector_shorthand.py.
"""

from osprey.utils.config_writer import get_control_system_type, set_control_system_type

#: A project already reading a real archive, so switching it onto the virtual
#: accelerator is a legal switch rather than the refused pairing.
ARCHIVED_PROJECT = (
    "control_system:\n  type: mock\n\narchiver:\n  type: mongodb_archiver\n"
    "  mongodb_archiver:\n    host: localhost\n"
)

#: The same project with the archiver that synthesizes its history — the one
#: the virtual accelerator may not be paired with.
STORELESS_PROJECT = "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"


class TestConfigWriterAcceptsVirtualAccelerator:
    """utils/config_writer.set_control_system_type handles the VA type directly."""

    def test_set_control_system_type_to_virtual_accelerator(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(
            "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
        )

        new_content, preview = set_control_system_type(
            config_path, "virtual_accelerator", "epics_archiver"
        )

        assert "virtual_accelerator" in preview
        assert "type: virtual_accelerator" in new_content
        assert "type: epics_archiver" in new_content

        config_path.write_text(new_content)
        assert get_control_system_type(config_path) == "virtual_accelerator"
        assert get_control_system_type(config_path, key="archiver.type") == "epics_archiver"
