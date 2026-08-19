"""Channel-snapshot injection into the compose render paths.

The compose render must hand every consumer one decision: the
``channel_snapshot`` render-context key, the ``channels.json`` artifact in the
bluesky_web build context, and (Task 2.2) the compose fragment's conditional
mount all derive from a single :func:`compute_channel_snapshot` call. These
tests pin the full (:func:`setup_build_dir`) and incremental
(:func:`_incremental_setup_build_dir`) paths to identical behaviour — same
context value, same artifact bytes, same absence.
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

import pytest
import yaml

BLUESKY_WEB_TEMPLATE = "services/bluesky_web/docker-compose.yml.j2"
OUT_DIR = Path("build") / "services" / "bluesky_web"
SNAPSHOT = OUT_DIR / "channels.json"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal deployment repo holding the packaged bluesky_web service.

    The render helpers resolve every path against the working directory by
    contract (the build runs them from the repo root), so the fixture chdirs in.
    """
    import osprey

    repo = tmp_path / "repo"
    packaged = resources.files(osprey).joinpath("templates", "services")
    shutil.copytree(str(packaged / "bluesky_web"), repo / "services" / "bluesky_web")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def rendered_contexts(monkeypatch):
    """Record the context dict each render_template call receives.

    Observing the context directly pins the injection itself, independent of
    what the template does with it; the mount tests below parse the rendered
    YAML instead.
    """
    from osprey.deployment import compose_generator

    contexts: list[dict] = []
    real = compose_generator.render_template

    def recording(template_path, config, out_dir):
        contexts.append(config)
        return real(template_path, config, out_dir)

    monkeypatch.setattr(compose_generator, "render_template", recording)
    return contexts


def _database(repo: Path) -> Path:
    """Write a small flat channel database into the repo."""
    path = repo / "channels_db.json"
    path.write_text(
        json.dumps(
            [
                {"channel": "StorageRing_BPM_02_X", "address": "SR:BPM:02:X"},
                {"channel": "StorageRing_BPM_01_X", "address": "SR:BPM:01:X"},
            ]
        )
    )
    return path


def _config(repo: Path, db_path: Path | None = None) -> dict:
    """The slice of project config the render paths and the decision read."""
    config: dict = {
        "project_name": "demo",
        "project_root": str(repo),
        "build_dir": "./build",
        "services": {"bluesky_web": {}, "bluesky": {}},
        "deployment": {},
        "system": {"timezone": "UTC"},
        "deployed_services": [],
    }
    if db_path is not None:
        config["channel_finder"] = {
            "pipelines": {"in_context": {"database": {"path": str(db_path), "type": "flat"}}}
        }
    return config


def _run_full(config: dict) -> None:
    from osprey.deployment.compose_generator import setup_build_dir

    setup_build_dir(BLUESKY_WEB_TEMPLATE, config, {})


def _run_incremental(config: dict) -> None:
    from osprey.deployment.compose_generator import _incremental_setup_build_dir

    _incremental_setup_build_dir(BLUESKY_WEB_TEMPLATE, config, {}, str(OUT_DIR))


def test_paths_agree_when_the_database_emits(repo, rendered_contexts):
    """Full and incremental renders inject the same flag and identical bytes."""
    config = _config(repo, _database(repo))

    _run_full(config)
    assert rendered_contexts[0]["channel_snapshot"] is True
    full_bytes = SNAPSHOT.read_bytes()

    shutil.rmtree("build")
    _run_incremental(config)
    assert rendered_contexts[1]["channel_snapshot"] is True
    assert SNAPSHOT.read_bytes() == full_bytes

    # The artifact is what the sidecar's /channels route serves: the sorted
    # addresses, not the human-facing channel names.
    assert json.loads(full_bytes) == ["SR:BPM:01:X", "SR:BPM:02:X"]


def test_paths_agree_when_no_database_is_configured(repo, rendered_contexts):
    """Without a channel database, both paths render exactly as today."""
    config = _config(repo)

    _run_full(config)
    assert rendered_contexts[0]["channel_snapshot"] is False
    assert not SNAPSHOT.exists()

    shutil.rmtree("build")
    _run_incremental(config)
    assert rendered_contexts[1]["channel_snapshot"] is False
    assert not SNAPSHOT.exists()


def test_incremental_rebuild_drops_a_stale_snapshot(repo, rendered_contexts):
    """A reused build directory cannot keep a snapshot the decision revoked."""
    OUT_DIR.mkdir(parents=True)
    SNAPSHOT.write_text('["SR:OLD:CHANNEL"]')

    _run_incremental(_config(repo))

    assert rendered_contexts[0]["channel_snapshot"] is False
    assert not SNAPSHOT.exists()


def test_other_services_never_compute_the_snapshot(repo, rendered_contexts, monkeypatch):
    """Only the bluesky_web render pays for the database load."""
    from osprey.deployment import compose_generator

    def unexpected(config):
        raise AssertionError("compute_channel_snapshot ran for a non-bluesky_web service")

    monkeypatch.setattr(compose_generator, "compute_channel_snapshot", unexpected)

    other = repo / "services" / "other"
    other.mkdir()
    (other / "docker-compose.yml.j2").write_text("services: {}\n")

    from osprey.deployment.compose_generator import setup_build_dir

    setup_build_dir("services/other/docker-compose.yml.j2", _config(repo, _database(repo)), {})

    assert rendered_contexts[0]["channel_snapshot"] is False
    assert not (Path("build") / "services" / "other" / "channels.json").exists()


RENDERED_COMPOSE = OUT_DIR / "docker-compose.yml"
CONFIG_MOUNT = "./build/services/bluesky_web/config.yml:/app/project/config.yml:ro"
SNAPSHOT_MOUNT = "./build/services/bluesky_web/channels.json:/app/project/channels.json:ro"


def _rendered_volumes() -> list[str]:
    """The bluesky-web service's volumes list in the rendered compose file."""
    document = yaml.safe_load(RENDERED_COMPOSE.read_text())
    return document["services"]["bluesky-web"]["volumes"]


def test_mount_present_when_the_snapshot_emits(repo):
    """An emitted snapshot gets its read-only bind mount beside config.yml."""
    _run_full(_config(repo, _database(repo)))

    volumes = _rendered_volumes()
    assert SNAPSHOT_MOUNT in volumes
    assert CONFIG_MOUNT in volumes


@pytest.mark.parametrize(
    ("web_block", "with_database"),
    [
        pytest.param({"channel_suggestions": {"enabled": False}}, True, id="disabled"),
        pytest.param({"channel_suggestions": {"max_channels": 1}}, True, id="over-guard"),
        pytest.param(None, False, id="no-database"),
    ],
)
def test_mount_absent_when_the_decision_is_false(repo, web_block, with_database):
    """Every non-emitting branch renders valid YAML without the snapshot mount."""
    config = _config(repo, _database(repo) if with_database else None)
    if web_block is not None:
        config["web"] = web_block

    _run_full(config)

    volumes = _rendered_volumes()
    assert not any("channels.json" in volume for volume in volumes)
    assert CONFIG_MOUNT in volumes
