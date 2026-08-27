"""End-to-end wiring of the build's virtual-accelerator manifest step.

``tests/va/test_build_time_manifest.py`` covers the generator itself; this
module covers the things only a real ``osprey build`` can show: that the
generated files land in the directory the container's bind mount actually
resolves to, that the ``.env`` keys pointing at them appear (and stay away)
together with the files, and that neither of those two facts can drift from the
other without a test failing.

The whole chain is one property, and it has three links: the manifest is
written into ``build/data/simulation/``, the VA compose service mounts THAT
directory at ``/data/simulation``, and the repo-root ``.env`` — the only file
``--env-file`` gives compose — carries the ``VA_CHANNELS_FILE`` pointer at it.
Break any one and the container either serves the framework's bundled channel
namespace while the operator believes they are driving their own facility's, or
refuses to boot: the entrypoint's manifest load raises on an absent file.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.services.virtual_accelerator.manifest.build import (
    LIMITS_FILENAME,
    MANIFEST_FILENAME,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_manifest_file
from osprey.utils.dotenv import parse_dotenv_file


def _bundle_data_dir() -> Path:
    import osprey

    return Path(osprey.__file__).parent / "templates" / "apps" / "control_assistant" / "data"


def _write_profile(
    repo_dir: Path,
    *,
    deploy_va: bool = False,
    live_standin: int | None = None,
    config: dict | None = None,
) -> Path:
    """A profile sourcing its own copy of the bundled control-assistant tree.

    ``hierarchical`` resolves to tier 3, the tier whose three paradigm
    databases agree — the shape the generator can actually build from.

    Written directly at the deployment repo's root, exactly where
    ``osprey build`` looks for it — this suite exercises the build step in
    isolation and has no need for ``osprey init``'s preset machinery.

    ``deploy_va`` adds the ``virtual_accelerator:`` block, which is what makes
    the build render the service's compose file. Off by default: every test
    here but the mount one is about the files and the ``.env`` keys, both of
    which the build produces whether or not the IOC is deployed, and rendering
    a service costs each of them a template copy.

    ``live_standin`` adds that block's stand-in port, which deploys a SECOND
    soft-IOC as the deployment's ``live`` target. It implies ``deploy_va``.

    ``config`` carries dotted overrides into the rendered ``config.yml`` — the
    profile's own escape hatch for a value the bundle's template pins, used
    here by the one test that has to build a deployment holding no
    channel-limits database.
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_bundle_data_dir(), repo_dir / "data")

    profile = {
        "name": "VA Build Step Test",
        "data_bundle": "control_assistant",
        "data": "data",
        "provider": "cborg",
        "model": "haiku",
        "channel_finder_mode": "hierarchical",
    }
    if deploy_va or live_standin is not None:
        profile["virtual_accelerator"] = {"port": 5064}
        if live_standin is not None:
            profile["virtual_accelerator"]["live_standin"] = live_standin
    if config:
        profile["config"] = config
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


def _repo_env(repo_dir: Path) -> dict[str, str]:
    """The deployment's one secret store, or ``{}`` when it has none yet.

    The repo-root ``.env`` — the file the pinned compose invocation passes as
    ``--env-file``, and therefore the only place a ``${VA_CHANNELS_FILE}`` in a
    compose template can be substituted from. A repo that has never been
    deployed and holds no provider key does not have one at all, which is not
    the same fact as holding one without the VA keys; both answers read the
    same here on purpose, because both mean "the container gets nothing".
    """
    env_path = repo_dir / ".env"
    return parse_dotenv_file(env_path) if env_path.is_file() else {}


@pytest.fixture(autouse=True)
def detected_provider_key(monkeypatch):
    """Keep the build's provider-credential summary from warning about a miss."""
    monkeypatch.setenv("CBORG_API_KEY", "test-key")


class TestGeneratedFromProfileData:
    def test_manifest_and_limits_land_in_the_mounted_directory(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        project_dir = _build(repo_dir)

        # Both land in the render's own data/simulation/, which is the
        # directory the VA compose file mounts — pinned as its own property by
        # test_the_compose_mount_resolves_to_the_directory_written below, so
        # this test can be about the files alone.
        simulation = project_dir / "data" / "simulation"
        assert (simulation / MANIFEST_FILENAME).is_file()
        assert (simulation / LIMITS_FILENAME).is_file()

    def test_manifest_loads_through_the_container_reader(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        project_dir = _build(repo_dir)

        channels = load_manifest_file(project_dir / "data" / "simulation" / MANIFEST_FILENAME)
        assert channels, "generated manifest served no channels"

    def test_env_points_the_container_at_the_generated_manifest(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        _build(repo_dir)

        # The repo root, not the render: the render's own `.env` was retired
        # with the one-repo secret store, and compose is pinned to
        # `--env-file <repo>/.env`.
        env = _repo_env(repo_dir)
        assert env["VA_CHANNELS_FILE"] == MANIFEST_FILENAME
        # Without this the entrypoint defaults a file-backed source to
        # lattice "none" and the PyAT physics bridge is never built.
        assert env["VA_LATTICE"] == "builtin"

    def test_the_compose_mount_resolves_to_the_directory_written(self, tmp_path):
        """The pointer and the mount have to name one directory.

        ``VA_CHANNELS_FILE`` is a name resolved against the container's data
        dir, so the value the test above pins is only meaningful if the mount
        behind ``/data/simulation`` is the directory the manifest was written
        into. Compose resolves a relative bind source against the pinned
        project directory — the repo root — so the two are the same directory
        only when the mount is spelled against ``build/``.
        """
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, deploy_va=True)

        project_dir = _build(repo_dir)

        compose = (
            project_dir / "services" / "virtual_accelerator" / "docker-compose.yml"
        ).read_text()
        mount = next(line.strip() for line in compose.splitlines() if ":/data/simulation:" in line)
        source = mount.removeprefix("- ").split(":/data/simulation:")[0]
        assert (repo_dir / source).resolve() == (project_dir / "data" / "simulation").resolve(), (
            f"the VA data mount resolves to {(repo_dir / source).resolve()}, which is not "
            f"where the build wrote the manifest"
        )

    def test_a_value_already_on_file_is_never_replaced(self, tmp_path, caplog):
        """The build reports a conflict; it does not resolve one.

        The repo ``.env`` is the deployment's whole secret store, hand-edited
        and written back to by ``osprey up``. A build appending to it must
        behave like every other writer of that file — append-only, existing
        value wins — or a rebuild would silently repoint a running IOC at a
        different channel set than the one it booted with.
        """
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        (repo_dir / ".env").write_text("VA_CHANNELS_FILE=my-own-manifest.json\n")

        with caplog.at_level(logging.WARNING):
            _build(repo_dir)

        env = _repo_env(repo_dir)
        assert env["VA_CHANNELS_FILE"] == "my-own-manifest.json"
        # The key the build DID own and the file did not still lands.
        assert env["VA_LATTICE"] == "builtin"
        assert "VA_CHANNELS_FILE" in caplog.text

    def test_rebuilding_neither_duplicates_nor_rewrites(self, tmp_path):
        """A second build over its own output is a no-op on the store."""
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        _build(repo_dir)
        first = (repo_dir / ".env").read_text()
        _build(repo_dir)

        assert (repo_dir / ".env").read_text() == first

    def test_the_store_is_written_with_secret_permissions(self, tmp_path):
        """It is the file holding the facility's provider keys."""
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        _build(repo_dir)

        assert (repo_dir / ".env").stat().st_mode & 0o777 == 0o600

    def test_facility_data_edit_reaches_the_served_channel_set(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        machine_json = repo_dir / "data" / "simulation" / "machine.json"
        machine = json.loads(machine_json.read_text())
        machine["channels"]["SR:VAC:GAUGE:SR99:PRESSURE:RB"] = {
            "value": 1e-9,
            "units": "Torr",
            "description": "Facility-added gauge",
        }
        machine_json.write_text(json.dumps(machine, indent=2))

        project_dir = _build(repo_dir)

        channels = load_manifest_file(project_dir / "data" / "simulation" / MANIFEST_FILENAME)
        assert "SR:VAC:GAUGE:SR99:PRESSURE:RB" in {c["address"] for c in channels}

    def test_limits_copy_matches_the_project_limits(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)

        project_dir = _build(repo_dir)

        # The entrypoint reads drive limits from beside the manifest; they
        # must be the same limits the project ships, or the accelerator
        # enforces bounds the project never declared.
        assert (project_dir / "data" / "simulation" / LIMITS_FILENAME).read_bytes() == (
            project_dir / "data" / LIMITS_FILENAME
        ).read_bytes()


class TestSkippedWithoutParadigmDatabases:
    """A tree that can't back a manifest wires nothing at all."""

    def test_neither_files_nor_env_keys_appear(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        shutil.rmtree(repo_dir / "data" / "channel_databases" / "tiers")

        project_dir = _build(repo_dir)

        env = _repo_env(repo_dir)
        assert "VA_CHANNELS_FILE" not in env
        assert "VA_LATTICE" not in env
        assert not (project_dir / "data" / "simulation" / MANIFEST_FILENAME).exists()

    def test_a_pointer_left_over_from_a_working_build_is_reported(self, tmp_path, caplog):
        """The one case the append-only rule cannot fix by itself.

        A repo that built a manifest once and then lost the databases behind it
        keeps the pointer — the build does not own the operator's copy of that
        file and will not edit a value out of it. What it must not do is stay
        quiet: the pointer now names a file the mounted directory no longer
        holds, and the entrypoint raises on an absent manifest rather than
        falling back, so the next `osprey up` gets a container that will not
        boot. Naming it is the whole remedy the operator needs.
        """
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        (repo_dir / ".env").write_text(f"VA_CHANNELS_FILE={MANIFEST_FILENAME}\n")
        shutil.rmtree(repo_dir / "data" / "channel_databases" / "tiers")

        with caplog.at_level(logging.WARNING):
            _build(repo_dir)

        assert "VA_CHANNELS_FILE" in caplog.text
        assert _repo_env(repo_dir)["VA_CHANNELS_FILE"] == MANIFEST_FILENAME

    def test_the_build_still_succeeds(self, tmp_path):
        repo_dir = tmp_path / "repo"
        # The limits database is what this tree loses, so the deployment is
        # read-only: writes ON with no limits file to enforce is a refusal of
        # its own (`resolve_limits_mount`), and letting that fire here would
        # decide this test on a fact it is not about.
        _write_profile(repo_dir, config={"control_system.writes_enabled": False})
        (repo_dir / "data" / LIMITS_FILENAME).unlink()

        # Profile data is user-owned: a tree the generator can't use is not a
        # build failure, it just leaves the container's package fallback.
        project_dir = _build(repo_dir)
        assert (project_dir / "config.yml").is_file()


class TestCorruptFacilityDataFailsTheBuild:
    """Absent databases fall back; unreadable ones must not.

    Falling back on a corrupt tree would hand the operator a virtual
    accelerator serving the framework's bundled channel namespace while they
    believe they are driving their own facility's — for a control system that
    is a worse outcome than a failed build.
    """

    def test_error_names_the_offending_file(self, tmp_path, caplog):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir)
        corrupt = repo_dir / "data" / "channel_databases" / "tiers" / "tier3" / "in_context.json"
        corrupt.write_text('{"channels": [ truncated')

        with caplog.at_level(logging.ERROR):
            result = _invoke_build(repo_dir)

        assert result.exit_code != 0
        # The operator gets the path to go fix, not a bare KeyError from
        # inside a database parser. Asserted on the log record rather than
        # result.output, which rich line-wraps long paths in.
        assert str(corrupt) in caplog.text
        assert "Build error" in caplog.text
        assert "Unexpected error" not in caplog.text


class TestLiveStandinReachesTheRender:
    """``virtual_accelerator.live_standin`` has to survive a whole real build.

    tests/cli/test_inject_va_gateways.py covers the injector against a template
    render; only a real ``osprey build`` shows that nothing between the profile
    parser and the written ``config.yml`` drops the second instance — and that
    the compose file the operator ends up with describes two containers rather
    than one.
    """

    STANDIN_PORT = 5074

    def test_the_render_carries_both_instances_and_the_acknowledgment(self, tmp_path):
        repo_dir = tmp_path / "repo"
        _write_profile(repo_dir, live_standin=self.STANDIN_PORT)

        config = yaml.safe_load((_build(repo_dir) / "config.yml").read_text())

        assert config["services"]["live_standin"] == {
            "path": "./services/virtual_accelerator",
            "port": self.STANDIN_PORT,
        }
        assert "live_standin" in config["deployed_services"]
        # Without this the deployment's own `live` target refuses itself.
        assert config["control_system"]["target_switch"]["live_gateway_acknowledged"] == (
            f"localhost:{self.STANDIN_PORT}"
        )

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
        assert "live_gateway_acknowledged" not in config["control_system"]["target_switch"]
