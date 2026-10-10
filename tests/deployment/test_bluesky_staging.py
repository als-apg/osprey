"""A virtual-accelerator build stages the Bluesky devices view unchanged.

``osprey build`` of the control-assistant preset writes
``build/data/bluesky_devices.yml`` from the facility file, and the compose
generator copies it byte for byte into ``build/services/bluesky/``, the file
the worker's compose fragment mounts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = [pytest.mark.slow]

VIEW = "data/bluesky_devices.yml"
STAGED = "services/bluesky/bluesky_devices.yml"


def test_the_build_writes_the_view_with_its_schema(built_control_assistant: BuiltProject) -> None:
    text = (built_control_assistant.build_dir / VIEW).read_text(encoding="utf-8")

    assert text.split("\n", 1)[0] == "schema: osprey.facility.bluesky_devices/1"


def test_the_build_stages_a_byte_equal_copy(built_control_assistant: BuiltProject) -> None:
    build_dir = built_control_assistant.build_dir

    assert (build_dir / STAGED).read_bytes() == (build_dir / VIEW).read_bytes()
    assert (build_dir / STAGED).stat().st_mode & 0o777 == 0o644


def test_the_compose_fragment_mounts_the_staged_copy(
    built_control_assistant: BuiltProject,
) -> None:
    compose = built_control_assistant.build_dir / "services" / "bluesky" / "docker-compose.yml"

    assert "./build/services/bluesky/bluesky_devices.yml:" in compose.read_text(encoding="utf-8")
