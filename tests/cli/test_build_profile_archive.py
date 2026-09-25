"""End-to-end ``osprey build`` coverage for the bundled ``archive`` service.

The template suite renders the packaged ``.j2`` against hand-built contexts; this
module runs the real build on a profile that selects the service and asserts on
what it leaves on disk: the rendered compose file, the service's registration in
``deployed_services``, and the archive root provisioned operator-private.

The build runs ``--skip-deps --skip-lifecycle``: the virtualenv install and the
profile's shell phases render nothing this module reads.
"""

from __future__ import annotations

import stat
from pathlib import Path

import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build

_PROFILE = """\
extends: hello-world
name: Archive Fixture
data: data
services:
  archive:
    template: osprey.archive
"""


def test_declared_service_renders_its_compose_and_registers(tmp_path: Path) -> None:
    repo = tmp_path / "archive-fixture"
    repo.mkdir()
    (repo / "data").mkdir()
    (repo / "profile.yml").write_text(_PROFILE, encoding="utf-8")

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    compose = yaml.safe_load(
        (repo / "build" / "services" / "archive" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert compose["services"]["archive"]["entrypoint"] == ["osprey", "archive", "--watch"]
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    assert "archive" in config["deployed_services"]
    archive_root = repo / "var" / "archive"
    assert archive_root.is_dir()
    assert stat.S_IMODE(archive_root.stat().st_mode) == 0o700
