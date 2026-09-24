"""Tests for the ``virtual_accelerator.pva_port:`` refusals in ``BuildProfile.validate``.

Instance 1 of the virtual accelerator publishes its model surface on a pvAccess
host port that ``deployment.port_base`` does not move, so a second deployment on
one host names its own with ``virtual_accelerator.pva_port``. The key is the one
way to move it: the rendered ``services.virtual_accelerator.pva_port`` is the
build's to write and is refused in ``config:`` by name.

Each refusal is pinned by its exact message, because the message is the whole
deliverable — a refusal an author cannot act on is a build that fails twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import BuildProfile, _parse_profile
from osprey.errors import BuildProfileError

#: The refusal a ``config:`` spelling of the rendered key earns.
CONFIG_REFUSAL = (
    "config: sets services.virtual_accelerator.pva_port, which the build writes from "
    "virtual_accelerator.pva_port and would overwrite. Set virtual_accelerator.pva_port instead."
)


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


@pytest.fixture(autouse=True)
def _facility_data_tree(tmp_path: Path) -> None:
    """The tree every profile's ``data:`` key names, beside the profile.

    ``data:`` is required of a repo profile and must resolve to a real
    directory, so without this each profile below would report one extra
    failure about a key none of these tests is about.
    """
    (tmp_path / "data").mkdir(exist_ok=True)


def _va_profile(va: dict[str, Any] | None = None, **extra: Any) -> BuildProfile:
    """A minimal profile deploying a virtual accelerator with block ``va``."""
    raw: dict[str, Any] = {
        "name": "pva",
        "data": "data",
        "virtual_accelerator": {"port": 5064, **(va or {})},
        **extra,
    }
    return _parse_profile(raw)


def test_pva_port_is_a_known_virtual_accelerator_key() -> None:
    """The profile surface accepts the key and carries it onto the VA block."""
    profile = _va_profile({"pva_port": 15075})
    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.pva_port == 15075


def test_a_profile_without_pva_port_validates(tmp_path: Path) -> None:
    """Unset is the shipped default and adds no refusal."""
    profile = _va_profile()
    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.pva_port is None
    profile.validate(tmp_path)


def test_pva_port_out_of_range_is_refused(tmp_path: Path) -> None:
    """The pvAccess port must be a usable TCP port."""
    assert _errors(_va_profile({"pva_port": 70000}), tmp_path) == [
        "virtual_accelerator.pva_port must be in 1..65535 (got 70000)"
    ]


@pytest.mark.parametrize("value", ["5085", True])
def test_pva_port_that_is_not_a_number_is_refused(tmp_path: Path, value: Any) -> None:
    """A string or a boolean names no port, and is refused as such."""
    assert _errors(_va_profile({"pva_port": value}), tmp_path) == [
        f"virtual_accelerator.pva_port must be a port number (got {value!r})"
    ]


def test_pva_port_on_the_channel_access_port_is_refused(tmp_path: Path) -> None:
    """Both servers bind TCP in one container, so they cannot share a number."""
    assert _errors(_va_profile({"pva_port": 5064}), tmp_path) == [
        "virtual_accelerator.pva_port must differ from virtual_accelerator.port (both 5064)"
    ]


def test_pva_port_on_the_standin_port_is_refused_once(tmp_path: Path) -> None:
    """A pvAccess/stand-in collision is one fault, reported by one rule."""
    profile = _va_profile({"pva_port": 5074, "live_standin": 5074})
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.pva_port must differ from virtual_accelerator.live_standin (both 5074)"
    ]


def test_pva_port_on_another_blocks_port_is_refused(tmp_path: Path) -> None:
    """A port another block spends is taken, whichever block asks second."""
    profile = _va_profile({"pva_port": 15075}, bluesky={"port": 15075})
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.pva_port (15075) collides with bluesky.port (15075)"
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"services.virtual_accelerator.pva_port": 15075},
        {"services": {"virtual_accelerator": {"pva_port": 15075}}},
    ],
    ids=["dotted", "nested"],
)
@pytest.mark.parametrize("va", [{}, {"pva_port": 16075}], ids=["unset", "set"])
def test_pva_port_in_config_is_refused_by_name(
    tmp_path: Path, config: dict[str, Any], va: dict[str, Any]
) -> None:
    """The rendered key is the build's to write, so ``config:`` may not spell it."""
    assert _errors(_va_profile(va, config=config), tmp_path) == [CONFIG_REFUSAL]
