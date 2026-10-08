"""The simulator is the build's view, so no key names its noise.

The mock connector and the Virtual Accelerator serve the simulator view the
build writes from the facility file. The noise a reading carries belongs to
that view, so it has no config key and no container variable. The resurrection
guard lists each retired key and proves no shipped surface spells it again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"
PRESETS = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
VA_COMPOSE = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "services"
    / "virtual_accelerator"
    / "docker-compose.yml.j2"
)
RETIRED_CONFIG_KEYS = (
    "control_system.connector.mock.noise_level",
    "control_system.connector.virtual_accelerator.noise_level",
)


@pytest.mark.parametrize("key", RETIRED_CONFIG_KEYS)
def test_each_retired_key_is_on_the_resurrection_list(key: str) -> None:
    """A retired key is listed as deleted and has no live row."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert key in manifest["deleted"]
    assert key not in manifest["keys"]


@pytest.mark.parametrize("key", RETIRED_CONFIG_KEYS)
def test_each_retired_key_has_an_orphan_site(key: str) -> None:
    """Every retired key is guarded by at least one orphan regex."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["orphan_sites"].get(key), f"{key} has no orphan site"


def test_no_preset_spells_a_retired_key() -> None:
    """No shipped preset states or comments a noise level."""
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(PRESETS.glob("*.yml"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "noise_level" in line
    ]

    assert offenders == []


def test_the_va_compose_carries_no_noise_variable() -> None:
    """The simulator's container takes no noise level from the host."""
    assert "VA_NOISE_LEVEL" not in VA_COMPOSE.read_text(encoding="utf-8")


def test_the_compose_generator_resolves_no_noise_level() -> None:
    """The generator hands the VA template no noise level to render."""
    from osprey.deployment import compose_generator

    assert not hasattr(compose_generator, "_va_noise_level")


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
