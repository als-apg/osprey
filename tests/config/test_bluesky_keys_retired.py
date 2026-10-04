"""The Bluesky worker's device file is the build's view, named by no key.

The build writes the Bluesky devices view from the facility file and stages it
unchanged, so neither a profile key nor a rendered config key can point the
worker at a second device file. The resurrection guard lists the retired
config key and proves no shipped surface spells it again.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_profile_load import _parse_profile
from osprey.cli.build_profile_schema import BlueskyConfig
from osprey.errors import BuildProfileError

if TYPE_CHECKING:
    from tests._builds import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"
RETIRED_CONFIG_KEY = "services.bluesky.devices_file"


def test_the_retired_key_is_on_the_resurrection_list() -> None:
    """The retired key is listed as deleted and has no live row."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert RETIRED_CONFIG_KEY in manifest["deleted"]
    assert RETIRED_CONFIG_KEY not in manifest["keys"]


def test_no_shipped_source_spells_the_retired_key() -> None:
    """No writer, profile parse or reference row under src/osprey names the key.

    The key's sites postdate the guard's back-test baseline, so no orphan
    regex can be proven to fire there; this scan holds the same line.
    """
    spelling = re.compile(r'bluesky\.devices_file|"devices_file"|devices_file=')
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in sorted((REPO_ROOT / "src" / "osprey").rglob("*"))
        if path.is_file() and path != MANIFEST and path.suffix not in {".pyc"}
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        )
        if spelling.search(line)
    ]

    assert offenders == []


def test_the_resurrection_guard_is_green() -> None:
    """No preset, template, reader or reference page spells a retired key."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_config_keys.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_profile_naming_the_devices_file_is_refused() -> None:
    """The device file is the build's view; a profile cannot name another."""
    with pytest.raises(BuildProfileError, match="'devices_file'"):
        _parse_profile({"name": "x", "bluesky": {"devices_file": "x.yml"}})


def test_the_bluesky_block_has_no_devices_file_field() -> None:
    """The profile schema carries no device-file path."""
    assert "devices_file" not in BlueskyConfig.__dataclass_fields__


@pytest.mark.slow
def test_the_demo_render_names_no_devices_file(built_control_assistant: BuiltProject) -> None:
    """No service block of the demo's rendered config.yml names a device file."""
    with open(built_control_assistant.build_dir / "config.yml") as fh:
        services = YAML().load(fh)["services"]

    assert "bluesky" in services
    for name, block in services.items():
        assert "devices_file" not in (block or {}), name
