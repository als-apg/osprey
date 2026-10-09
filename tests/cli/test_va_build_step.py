"""End-to-end wiring of the simulator view the virtual accelerator serves.

Only a real ``osprey build`` can show that the view lands in the directory the
container's bind mount resolves to: the build writes
``build/data/simulator/addresses.json`` and the VA compose service mounts
``build/data`` at ``/data``, where the entrypoint reads ``simulator/``. Break
either link and the container serves nothing of the facility's or refuses to
boot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.facility.views.simulator import ADDRESSES_FILE
from tests._preset_data import copy_bundle_data

#: What a hand-written profile has to state to build at all.
#:
#: A preset ships these, but this suite writes its profile by hand rather than
#: materializing one — it exercises the VA build step in isolation, and
#: inheriting a preset would drag in that preset's own ``virtual_accelerator:``
#: block, which is exactly the variable under test here.
#:
#: Two groups: the posture floor, which refuses a build that leaves any of its
#: keys to a reader's fallback, and the store behind the ARIEL server, which is
#: on by framework default and whose client would otherwise dial a
#: ``postgresql`` port with nothing listening behind it.
POSTURE_FLOOR = {
    "control_system.type": "mock",
    "archiver.type": "mock_archiver",
    "approval.enabled": True,
    "approval.default_policy": "always",
    "claude_code.telemetry.enabled": False,
    "hooks.debug": False,
    "services.postgresql.path": "./services/postgresql",
    "deployed_services": ["postgresql"],
    "system.timezone": "UTC",
    # A target-switch block the build could write an acknowledgment into. The
    # stand-in tests assert that it never does, and an absent block would make
    # that assertion pass for the wrong reason.
    "control_system.target_switch.drain_timeout_s": 5,
}


def _write_profile(
    repo_dir: Path,
    *,
    deploy_va: bool = False,
    live_standin: int | None = None,
) -> Path:
    """A profile sourcing its own copy of the bundled control-assistant tree.

    ``hierarchical`` resolves to tier 3, the tier whose three paradigm
    databases agree — the shape the generator can actually build from.

    Written directly at the deployment repo's root, exactly where
    ``osprey build`` looks for it — this suite exercises the build step in
    isolation and has no need for ``osprey init``'s preset machinery.

    ``deploy_va`` adds the ``virtual_accelerator:`` block, which is what makes
    the build render the service's compose file. Off by default: the view is
    written whether or not the IOC is deployed, and rendering a service costs
    each test a template copy.

    ``live_standin`` adds that block's stand-in port, which deploys a SECOND
    soft-IOC and gives the deployment a THIRD control target, ``standin``. It
    implies ``deploy_va``.
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    copy_bundle_data(repo_dir / "data")

    profile: dict = {
        "name": "VA Build Step Test",
        "project_name": "repo",
        "data": "data",
        "provider": "cborg",
        "model": "claude-haiku-4-5",
        "channel_finder_mode": "hierarchical",
        "config": dict(POSTURE_FLOOR),
    }
    if deploy_va or live_standin is not None:
        profile["virtual_accelerator"] = {"port": 5064}
        if live_standin is not None:
            profile["virtual_accelerator"]["live_standin"] = live_standin
    path = repo_dir / "profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    return path


def _invoke_build(repo_dir: Path):
    from click.testing import CliRunner

    from osprey.cli.main import cli

    return CliRunner().invoke(
        cli,
        ["build", "--repo", str(repo_dir), "--skip-deps", "--skip-lifecycle"],
    )


def _build(repo_dir: Path) -> Path:
    result = _invoke_build(repo_dir)
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception}"
    )
    return repo_dir / "build"


@pytest.fixture(autouse=True)
def detected_provider_key(monkeypatch):
    """Keep the build's provider-credential summary from warning about a miss."""
    monkeypatch.setenv("CBORG_API_KEY", "test-key")


def _served_addresses(project_dir: Path) -> set[str]:
    document = json.loads((project_dir / "data" / "simulator" / ADDRESSES_FILE).read_text())
    return set(document["channels"])


class TestGeneratedFromProfileData:
    def test_the_compose_mount_resolves_to_the_view_written(self, tmp_path):
        """The mount and the view have to name one tree.

        The entrypoint reads ``simulator/addresses.json`` under ``/data``.
        Compose resolves a relative bind source against the pinned project
        directory, the repo root, so the served view is the mount source plus
        ``simulator``, and it is the view the build wrote only when the mount is
        spelled against the output zone.
        """
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, deploy_va=True)

        project_dir = _build(repo_dir)

        compose = (
            project_dir / "services" / "virtual_accelerator" / "docker-compose.yml"
        ).read_text()
        mount = next(line.strip() for line in compose.splitlines() if ":/data:" in line)
        source = mount.removeprefix("- ").split(":/data:")[0]
        served = (repo_dir / source / "simulator" / ADDRESSES_FILE).resolve()
        assert served == (project_dir / "data" / "simulator" / ADDRESSES_FILE).resolve()
        assert served.is_file()

    def test_facility_data_edit_reaches_the_served_channel_set(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        records = repo_dir / "data" / "facility" / "records" / "channels.yaml"
        channels = yaml.safe_load(records.read_text())
        gauge = next(c for c in channels if c["id"] == "SR:VAC:GAUGE:SR01:PRESSURE:RB")
        channels.append(
            {
                **gauge,
                "id": "SR:VAC:GAUGE:SR01:PRESSURE:RB2",
                "names": ["StorageRing_VacGauge_SR01_Pressure_Readback_2"],
                "description": "Facility-added gauge readback",
            }
        )
        records.write_text(yaml.safe_dump(channels, sort_keys=False))

        project_dir = _build(repo_dir)

        assert "SR:VAC:GAUGE:SR01:PRESSURE:RB2" in _served_addresses(project_dir)

    def test_the_build_writes_nothing_into_the_repo_env(self, tmp_path):
        """The view is the container's whole input; ``.env`` carries no pointer."""
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        (repo_dir / ".env").write_text("OTHER=x\n")

        _build(repo_dir)

        assert (repo_dir / ".env").read_text() == "OTHER=x\n"


class TestLiveStandinReachesTheRender:
    """``virtual_accelerator.live_standin`` has to survive a whole real build.

    tests/cli/test_inject_va_gateways.py covers the injector against a template
    render; only a real ``osprey build`` shows that nothing between the profile
    parser and the written ``config.yml`` drops the second instance — and that
    the compose file the operator ends up with describes two containers rather
    than one.

    The stand-in is a THIRD control target, ``standin``, not a rewrite of
    ``live``: the build gives it its own
    ``control_system.connector.live_standin`` block and touches nothing that
    describes the facility's machine. So the acknowledgment gating
    ``control_target_set live`` stays the profile's to state, and a whole build
    must be shown NOT to write it.
    """

    STANDIN_PORT = 5074

    def test_the_render_carries_both_instances_and_no_acknowledgment(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, live_standin=self.STANDIN_PORT)

        config = yaml.safe_load((_build(repo_dir) / "config.yml").read_text())

        assert config["services"]["live_standin"] == {
            "path": "./services/virtual_accelerator",
            "port": self.STANDIN_PORT,
        }
        assert "live_standin" in config["deployed_services"]
        # The stand-in is reached as `standin`, so nothing here speaks for the
        # operator about `live`: the acknowledgment stays the profile's to say,
        # exactly as on a deployment that asked for no stand-in.
        assert "live_gateway_acknowledged" not in config["control_system"]["target_switch"]

    def test_the_compose_file_describes_a_second_container(self, tmp_path):
        """One template directory, two soft-IOCs, named for what each serves."""
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, live_standin=self.STANDIN_PORT)

        project_dir = _build(repo_dir)

        compose = (
            project_dir / "services" / "virtual_accelerator" / "docker-compose.yml"
        ).read_text()
        assert "live-standin" in compose
        assert f"{self.STANDIN_PORT}" in compose
        # And no second copy of the template tree was staged for it.
        assert not (project_dir / "services" / "live_standin").exists()

    def test_compose_is_handed_the_shared_file_once(self, tmp_path):
        """Two instances, one compose file, one ``-f``.

        Both service blocks carry the same ``path``, and the file that path
        resolves to describes both containers already. The lookup walks
        ``deployed_services``, so it reports that file once per instance — the
        same shape a two-lane Bluesky deployment produces, and the reason
        ``as_built_compose_files`` dedupes what it returns. Pinned here through
        the function ``osprey up`` actually builds its ``-f`` list from, because
        that is where the claim "compose sees it once" is true.
        """
        from osprey.deployment.container_lifecycle import as_built_compose_files

        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, live_standin=self.STANDIN_PORT)
        config = yaml.safe_load((_build(repo_dir) / "config.yml").read_text())

        compose_files = as_built_compose_files(config, repo_dir)

        va_files = [path for path in compose_files if "virtual_accelerator" in path]
        assert len(va_files) == 1, compose_files
        assert len(compose_files) == len(set(compose_files))

    def test_a_build_without_the_key_deploys_one_instance(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, deploy_va=True)

        config = yaml.safe_load((_build(repo_dir) / "config.yml").read_text())

        assert "live_standin" not in config["services"]
        assert "live_standin" not in config["deployed_services"]
        # Unchanged by the stand-in becoming a target of its own: this build
        # never wrote the key, and now neither does the one above.
        assert "live_gateway_acknowledged" not in config["control_system"]["target_switch"]
