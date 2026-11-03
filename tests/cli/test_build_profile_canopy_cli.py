"""`osprey build` and `osprey validate` refuse a CANOPY home screen the library does not hold.

A temporary profile turns the CANOPY panel on and names a home screen its
library lacks; both commands refuse it before anything is written. The checks
over a shipped library belong with that library.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A hello-world materialization with the CANOPY panel on and a one-screen library."""
    root = tmp_path_factory.mktemp("canopy-home") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    profile = root / "profile.yml"
    body = profile.read_text()
    assert body.count("\nweb_panels: []") == 1 and body.count("\nconfig:\n") == 1
    body = body.replace("\nweb_panels: []", "\nweb_panels: [canopy]")
    body = body.replace(
        "\nconfig:\n", "\nconfig:\n  web.panels.canopy.home_screen: facility/nope\n"
    )
    profile.write_text(body)
    screen = root / "data" / "canopy" / "library" / "screens" / "facility" / "overview.screen.json"
    screen.parent.mkdir(parents=True)
    screen.write_text(json.dumps({"kind": "screen", "widgets": []}))
    return root


def test_validate_refuses_a_home_screen_the_library_lacks(materialized: Path):
    result = CliRunner().invoke(validate, ["--repo", str(materialized), "--drift", "warn"])
    assert result.exit_code == 2, result.output
    assert "web.panels.canopy.home_screen" in result.output
    assert "'facility/nope'" in result.output
    assert "'facility/overview'" in result.output


def test_build_refuses_it_before_writing_anything(
    materialized: Path, monkeypatch: pytest.MonkeyPatch
):
    # The panel stays in the render only when a payload is staged or staging
    # is skipped; with no payload named it leaves, and its key with it.
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD_CHECK", "0")
    result = CliRunner().invoke(
        build, ["--repo", str(materialized), "--skip-deps", "--skip-lifecycle"]
    )
    assert result.exit_code == 2, result.output
    assert "Profile validation failed" in result.output
    assert "'facility/nope'" in result.output
    assert not (materialized / "build" / "config.yml").exists()
