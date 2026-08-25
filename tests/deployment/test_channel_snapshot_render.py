"""Staged-artifact injection into the compose render paths.

Two artifacts are staged by the render and gated the same way — the
bluesky_web sidecar's ``channels.json`` behind ``channel_snapshot``, and the
queueserver worker's ``bluesky_devices.yml`` behind ``bluesky_devices``. Each
is one decision reaching three consumers: a render-context boolean, a file in
the service's build context, and the compose fragment's conditional mount.

Both are pinned across BOTH render paths — the full :func:`setup_build_dir` and
the incremental :func:`_incremental_setup_build_dir` — because a project's
first build takes one and every rebuild takes the other, and a divergence
between them is a deployment whose mounted file depends on which command last
ran. Same context value, same artifact bytes, same absence, and the same
removal of a stale file the decision has revoked.
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
    for service in ("bluesky_web", "bluesky"):
        shutil.copytree(str(packaged / service), repo / "services" / service)
    # The shared macro partials travel with the template that imports them, by
    # the same glob the build copies them with (_copy_shared_service_partials) —
    # naming one here would leave the next partial to fail as a TemplateNotFound.
    for partial in sorted(Path(str(packaged)).glob("_*.j2")):
        shutil.copy2(partial, repo / "services" / partial.name)
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


# ---------------------------------------------------------------------------
# The queueserver worker's device file: the same paired shape, one artifact
# over.
#
# `_stage_bluesky_devices` answers a richer question than the snapshot's
# (authored / derived / mock / nothing), but the property that has to hold
# across the two renderers is identical: one decision, one boolean in the
# context, one set of bytes in the build context, and nothing left behind when
# the decision goes the other way.
# ---------------------------------------------------------------------------

BLUESKY_TEMPLATE = "services/bluesky/docker-compose.yml.j2"
DEVICES_OUT_DIR = Path("build") / "services" / "bluesky"
DEVICES = DEVICES_OUT_DIR / "bluesky_devices.yml"
DEVICES_RELPATH = "data/bluesky_devices.yml"
DEVICES_MOUNT = (
    "./build/services/bluesky/bluesky_devices.yml:/app/project/data/bluesky_devices.yml:ro"
)

#: A document the worker's own loader accepts in full. A file with problems is
#: a refusal rather than a staging decision (the worker skips malformed entries
#: with a warning, so a bad file would deploy healthy and silently short a
#: device), and that refusal is not what this module is pinning.
DEVICES_DOCUMENT = """\
settables:
  - name: COR:H:01
    setpoint: COR:H:01:SP
    readback: COR:H:01:RB
readables:
  - name: BPM:01:X
    pv: BPM:01:X
"""


def _devices_config(repo: Path, *, authored: bool, mock: bool = False) -> dict:
    """The config slice the device staging reads.

    ``services.bluesky.devices_file`` is where the build injector writes the
    path on every lane, so that is where the staging looks; the control-system
    type is read first, because a mock deployment drives no channels and is
    browse-only whatever file is lying around.
    """
    config = _config(repo)
    config["control_system"] = {"type": "mock" if mock else "epics", "writes_enabled": False}
    if authored:
        document = repo / DEVICES_RELPATH
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(DEVICES_DOCUMENT, encoding="utf-8")
        config["services"]["bluesky"]["devices_file"] = DEVICES_RELPATH
    return config


def _run_full_bluesky(config: dict) -> None:
    from osprey.deployment.compose_generator import setup_build_dir

    setup_build_dir(BLUESKY_TEMPLATE, config, {})


def _staged_device_mounts() -> list[str]:
    """The device binds the rendered bluesky compose file declares, if any."""
    document = yaml.safe_load((DEVICES_OUT_DIR / "docker-compose.yml").read_text())
    return [
        volume
        for service in document["services"].values()
        for volume in service.get("volumes") or []
        if "bluesky_devices" in str(volume)
    ]


def _run_incremental_bluesky(config: dict) -> None:
    from osprey.deployment.compose_generator import _incremental_setup_build_dir

    _incremental_setup_build_dir(BLUESKY_TEMPLATE, config, {}, str(DEVICES_OUT_DIR))


def test_device_paths_agree_when_a_file_is_authored(repo, rendered_contexts):
    """Full and incremental renders stage the same flag and identical bytes.

    Byte equality is the load-bearing half: a two-lane deploy renders this one
    directory twice and a running deployment holds the staged file as a bind
    mount, so a rebuild that produced different bytes would swap a live
    worker's device set underneath it.
    """
    config = _devices_config(repo, authored=True)

    _run_full_bluesky(config)
    assert rendered_contexts[0]["bluesky_devices"] is True
    full_bytes = DEVICES.read_bytes()

    shutil.rmtree("build")
    _run_incremental_bluesky(config)
    assert rendered_contexts[1]["bluesky_devices"] is True
    assert DEVICES.read_bytes() == full_bytes

    # The staged file is a copy of what the operator authored, under the
    # basename the staging step owns — the authored filename never travels.
    assert full_bytes == DEVICES_DOCUMENT.encode("utf-8")

    # And the third consumer of the same decision: the mount that puts the
    # staged file where the worker's env var says to look for it.
    assert _staged_device_mounts() == [DEVICES_MOUNT]


def test_device_paths_agree_when_no_file_is_configured(repo, rendered_contexts):
    """With nothing to stage, both paths render a browse-only worker.

    Fail-closed: no file staged means no mount and no env var, so the worker
    comes up able to browse plans and run none — rather than naming a file the
    mount does not carry, which fails its own load instead.
    """
    config = _devices_config(repo, authored=False)

    _run_full_bluesky(config)
    assert rendered_contexts[0]["bluesky_devices"] is False
    assert not DEVICES.exists()

    shutil.rmtree("build")
    _run_incremental_bluesky(config)
    assert rendered_contexts[1]["bluesky_devices"] is False
    assert not DEVICES.exists()
    assert _staged_device_mounts() == []


def test_a_mock_deployment_stages_no_devices_on_either_path(repo, rendered_contexts):
    """A mock control system drives no channels, authored file or not.

    Decided before the file is even looked at, so an authored document cannot
    make a mock deployment look like it can steer anything.
    """
    config = _devices_config(repo, authored=True, mock=True)

    _run_full_bluesky(config)
    assert rendered_contexts[0]["bluesky_devices"] is False
    assert not DEVICES.exists()

    shutil.rmtree("build")
    _run_incremental_bluesky(config)
    assert rendered_contexts[1]["bluesky_devices"] is False
    assert not DEVICES.exists()


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(_run_full_bluesky, id="full"),
        pytest.param(_run_incremental_bluesky, id="incremental"),
    ],
)
def test_a_rebuild_drops_a_stale_device_file(repo, rendered_contexts, run):
    """Neither renderer may leave a device set the deployment no longer has.

    The incremental path reuses the directory, and the full path can be handed
    one a previous build wrote — so a deployment that stops having devices (the
    authored file deleted, the control system switched to mock) must stop
    mounting the previous render's, on either path.
    """
    DEVICES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICES.write_text("settables: [{name: STALE, setpoint: STALE:SP}]\n", encoding="utf-8")

    run(_devices_config(repo, authored=False))

    assert rendered_contexts[0]["bluesky_devices"] is False
    assert not DEVICES.exists()


def test_other_services_never_stage_a_device_file(repo, rendered_contexts):
    """Only the bluesky service directory gets a device file.

    The staging is keyed on the service directory name, so a sibling service
    rendered from the same config must come away with the flag false and
    nothing written into its build context.
    """
    _run_full(_devices_config(repo, authored=True))

    assert rendered_contexts[0]["bluesky_devices"] is False
    assert not (Path("build") / "services" / "bluesky_web" / "bluesky_devices.yml").exists()
