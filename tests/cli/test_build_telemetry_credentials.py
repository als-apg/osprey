"""What ``osprey build`` does about the telemetry store's credential files.

Three things, all before the compose files are rendered: it records who host
launches send and record as, it mints every credential file the compose files
are about to name, and it does both at the real repo root — the files are
durable secrets and must never land in the disposable render or in an image.
"""

from __future__ import annotations

import fnmatch
import inspect
from pathlib import Path

import click
import pytest
import yaml

from osprey.cli import build_cmd
from osprey.deployment import telemetry_credentials as creds

STORE = {"deployed_services": ["telemetry_store", "telemetry_gateway"]}

#: Every shape of credential file name.
CREDENTIAL_NAMES = [
    creds.ADMIN_FILE,
    creds.INGEST_FILE,
    creds.HTPASSWD_FILE,
    creds.ARCHIVE_FILE,
    creds.HISTORY_FILE,
    creds.HISTORY_GATE_FILE,
    creds.sender_file("alice"),
    creds.sender_file("dispatch-worker-1"),
    creds.reader_file("ops"),
    f"{creds.reader_file('ops')}.withdrawn-2026-03-04",
]


def _render(tmp_path: Path, config: dict) -> Path:
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return render


def _config(render: Path) -> dict:
    return yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))


# --- deployment.host_identity ------------------------------------------------


def test_the_build_records_the_building_account(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "builder")
    render = _render(tmp_path, dict(STORE))

    assert build_cmd._record_host_identity(render, _config(render)) is True

    assert _config(render)["deployment"]["host_identity"] == "builder"


def test_a_stated_host_identity_is_left_as_stated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "builder")
    render = _render(tmp_path, {**STORE, "deployment": {"host_identity": "ops"}})

    assert build_cmd._record_host_identity(render, _config(render)) is False

    assert _config(render)["deployment"]["host_identity"] == "ops"


def test_a_deployment_without_the_store_records_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "builder")
    render = _render(tmp_path, {"deployed_services": []})
    before = (render / "config.yml").read_bytes()

    assert build_cmd._record_host_identity(render, _config(render)) is False

    assert (render / "config.yml").read_bytes() == before


def test_an_account_with_no_name_records_nothing(tmp_path: Path, monkeypatch) -> None:
    def no_name() -> str:
        raise KeyError("uid not found")

    monkeypatch.setattr("getpass.getuser", no_name)
    render = _render(tmp_path, dict(STORE))

    assert build_cmd._record_host_identity(render, _config(render)) is False


# --- the mint ----------------------------------------------------------------


def test_the_build_mints_at_the_repo_root_and_no_reader_key(tmp_path: Path) -> None:
    rendered = {
        **STORE,
        "deployment": {"host_identity": "builder"},
        "services": {"telemetry_store": {"readers": [{"name": "ops"}]}},
    }
    stage = tmp_path / "build" / ".stage" / "p"
    stage.mkdir(parents=True)

    build_cmd._ensure_telemetry_credentials(tmp_path, rendered)

    at_root = {path.name for path in tmp_path.iterdir() if path.is_file()}
    assert at_root == {
        creds.ADMIN_FILE,
        creds.INGEST_FILE,
        creds.ARCHIVE_FILE,
        creds.HTPASSWD_FILE,
        creds.sender_file("builder"),
    }
    assert not [path for path in (tmp_path / "build").rglob("*") if path.is_file()]
    # The host identity's login is in the sender list before the gateway starts.
    assert (tmp_path / creds.HTPASSWD_FILE).read_text().startswith("builder:{SHA}")


def test_an_identity_that_cannot_be_a_login_fails_the_build(tmp_path: Path) -> None:
    rendered = {**STORE, "deployment": {"host_identity": "DOMAIN\\user"}}

    with pytest.raises(click.UsageError, match="deployment.host_identity"):
        build_cmd._ensure_telemetry_credentials(tmp_path, rendered)


def test_the_mint_sits_between_the_persona_renders_and_the_compose_files() -> None:
    """After the personas, which settle the roster; before the compose files,
    which name the files as ``env_file``. And never under ``--runtime-root``."""
    source = inspect.getsource(build_cmd)
    personas = source.index("persona_renders = _render_persona_projects(")
    mint = source.index("_ensure_telemetry_credentials(zones.repo_root, rendered)")
    compose = source.index("config = _render_compose_files(zones, runtime_root")

    assert personas < mint < compose
    assert "if not runtime_root:" in source[personas:mint]


# --- no credential file reaches an image context -----------------------------


def _seed(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in CREDENTIAL_NAMES:
        (directory / name).write_text("secret\n", encoding="utf-8")
    (directory / ".env.example").write_text("KEY=\n", encoding="utf-8")
    (directory / "profile.yml").write_text("project_name: p\n", encoding="utf-8")


def test_the_source_zone_copy_takes_no_credential_file(tmp_path: Path) -> None:
    repo, image = tmp_path / "repo", tmp_path / "image"
    _seed(repo)
    _seed(repo / "data")
    image.mkdir()

    build_cmd._stage_source_zone(repo, image)

    copied = {path.name for path in image.rglob("*") if path.is_file()}
    assert "profile.yml" in copied
    assert not copied & set(CREDENTIAL_NAMES)


def test_the_secret_strip_removes_every_credential_file_at_any_depth(tmp_path: Path) -> None:
    image = tmp_path / "image"
    _seed(image)
    _seed(image / "build")
    _seed(image / "build" / "services" / "x")

    build_cmd._strip_secrets(image)

    left = {path.name for path in image.rglob("*") if path.is_file()}
    assert left == {".env.example", "profile.yml"}


def test_the_image_context_ignore_file_excludes_every_credential_file(tmp_path: Path) -> None:
    image = tmp_path / "image"
    _seed(image)
    _seed(image / "build")
    # The render's own ignore file, which the context's is derived from.
    shipped = _project_template("dockerignore").read_text(encoding="utf-8")
    (image / "build" / ".dockerignore").write_text(shipped, encoding="utf-8")

    patterns = build_cmd._write_image_context_dockerignore(image)
    build_cmd._prune_ignored_entries(image, patterns)

    left = {path.name for path in image.rglob("*") if path.is_file()}
    assert "profile.yml" in left
    assert not left & set(CREDENTIAL_NAMES)


def _project_template(name: str) -> Path:
    import osprey

    return Path(inspect.getfile(osprey)).parent / "templates" / "project" / name


def _ignored(patterns: list[str], name: str) -> bool:
    """Whether a root-level file called *name* is ignored, last match winning."""
    verdict = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if fnmatch.fnmatch(name, pattern.lstrip("!").lstrip("/")):
            verdict = not negated
    return verdict


def _patterns(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_git_and_docker_ignore_every_credential_file(name: str) -> None:
    from osprey.cli import init_cmd

    assert _ignored(_patterns(_project_template("gitignore").read_text()), name)
    assert _ignored(_patterns(_project_template("dockerignore").read_text()), name)
    # The repo's own .gitignore, as `osprey init` writes it.
    assert "\n/.env*\n" in inspect.getsource(init_cmd)
    assert _ignored(["/.env*", "!/.env.example", "!/.env.shared"], name)


def test_no_credential_file_is_a_member_of_the_env_chain() -> None:
    from osprey_connectors.dotenv import ENV_CHAIN_FILENAMES

    assert not [name for name in ENV_CHAIN_FILENAMES if creds.is_credential_file(name)]
    assert all(creds.is_credential_file(name) for name in CREDENTIAL_NAMES[:-1])
