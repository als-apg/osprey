"""Deploy-time write-back of service secrets into the owning profile.

A deploy is where several of a stack's secrets first come into existence: the
fail-closed service tokens, the passwords a docker volume is initialized with.
If they only ever land in the *project* ``.env``, the profile the project was
built from cannot reproduce the stack — a rebuild mints a second set of tokens
the running containers do not trust, and a ``build --force`` wipe destroys the
only copy of the password its volume is pinned to.

So ``_ensure_service_tokens`` persists the *effective* value of every secret
back into the profile ``.env``, located through the project's own manifest. The
rules these tests pin:

* append-only — a key already in the profile keeps its value, and a
  disagreement is reported by variable *name* (never by value), because a
  minted secret is pinned by the volume and containers that adopted it;
* the project ``.env`` is then re-derived from the profile in ``deploy`` mode:
  overlay only, so a key the deploy has never heard of survives untouched;
* when there is no reachable profile the secrets stay where they were minted,
  a warning names the path, and ``secrets_synced_to_profile: false`` goes into
  the manifest for ``build --force`` to surface before wiping;
* only *secrets* travel: the bluesky substrate wiring beside them is derived
  configuration a rebuild regenerates, so pinning it in the profile would
  freeze a device list that must re-derive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.cli.templates.manifest import (
    SECRETS_SYNCED_KEY,
    read_secrets_synced_to_profile,
)
from osprey.deployment import container_lifecycle
from osprey.utils.dotenv import (
    DEPLOY_MINTED_BANNER,
    PROFILE_CARRIED_BANNER,
    parse_dotenv_text,
)

_DISPATCH_VARS = ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN")

_CONFIG = {"deployed_services": ["event_dispatcher"]}


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """No exported token may win over the ``.env`` values under test."""
    for var in (*_DISPATCH_VARS, "ARIEL_DSN"):
        monkeypatch.delenv(var, raising=False)


def _profile(tmp_path: Path, env_text: str | None = None) -> Path:
    """A profile tree (``profile.yml`` + optional ``.env``); returns its root."""
    root = tmp_path / "facility-profile"
    root.mkdir()
    (root / "profile.yml").write_text("name: facility\n", encoding="utf-8")
    if env_text is not None:
        (root / ".env").write_text(env_text, encoding="utf-8")
    return root


def _project(tmp_path: Path, profile_root: Path | None, env_text: str = "") -> Path:
    """A built project dir with a manifest pointing at ``profile_root``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(env_text, encoding="utf-8")
    build_args: dict[str, str] = {"project_name": "project", "source": "preset"}
    if profile_root is not None:
        build_args = {
            "project_name": "project",
            "source": "profile",
            "profile_path": "facility-profile/profile.yml",
            "profile_path_abs": str(profile_root / "profile.yml"),
        }
    (project / ".osprey-manifest.json").write_text(
        json.dumps({"schema_version": "1.2.0", "build_args": build_args}), encoding="utf-8"
    )
    return project


def _env_of(path: Path) -> dict[str, str]:
    return parse_dotenv_text(path.read_text(encoding="utf-8"))


def test_minted_tokens_land_in_the_profile_env(tmp_path):
    """The profile gains the tokens the deploy minted, under the deploy banner."""
    profile_root = _profile(tmp_path, env_text="ANTHROPIC_API_KEY=sk-facility\n")
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    profile_env = _env_of(profile_root / ".env")
    project_env = _env_of(project / ".env")
    for var in _DISPATCH_VARS:
        assert profile_env[var], f"{var} was not persisted to the profile"
        assert project_env[var] == profile_env[var]
    assert DEPLOY_MINTED_BANNER in (profile_root / ".env").read_text(encoding="utf-8")
    # The profile's own facility secret reaches the project through the
    # deploy-mode derivation, which is what makes the write-back a round trip.
    assert project_env["ANTHROPIC_API_KEY"] == "sk-facility"
    assert read_secrets_synced_to_profile(project) is True


def test_tokens_already_in_the_project_env_are_synced_too(tmp_path):
    """Write-back covers effective values, not only freshly minted ones.

    This is the recovery path: a project that minted its tokens while the
    profile was unreachable syncs them on the next deploy that can reach it.
    """
    profile_root = _profile(tmp_path)
    existing = "".join(f"{var}=already-minted-{var.lower()}\n" for var in _DISPATCH_VARS)
    project = _project(tmp_path, profile_root, env_text=existing)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    profile_env = _env_of(profile_root / ".env")
    for var in _DISPATCH_VARS:
        assert profile_env[var] == f"already-minted-{var.lower()}"
    assert read_secrets_synced_to_profile(project) is True


def test_profile_value_wins_and_the_conflict_is_named(tmp_path, caplog):
    """A disagreement warns by variable name and never overwrites the profile."""
    profile_root = _profile(tmp_path, env_text="EVENT_DISPATCHER_TOKEN=profile-value\n")
    project = _project(tmp_path, profile_root, env_text="EVENT_DISPATCHER_TOKEN=project-value\n")

    with caplog.at_level(logging.WARNING):
        container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert _env_of(profile_root / ".env")["EVENT_DISPATCHER_TOKEN"] == "profile-value"
    assert "EVENT_DISPATCHER_TOKEN" in caplog.text
    # Names, never values: a warning is not a place to print secrets.
    assert "project-value" not in caplog.text
    assert "profile-value" not in caplog.text
    # The project still carries a secret the profile does not, so a wipe would
    # lose it — the flag has to say so.
    assert read_secrets_synced_to_profile(project) is False


def test_conflicting_key_keeps_its_project_value(tmp_path):
    """The running stack keeps the secret its containers already trust.

    ``EVENT_DISPATCHER_TOKEN`` is a runtime-writer key, so the deploy-mode
    derivation must not pull the profile's differing value down over it.
    """
    profile_root = _profile(tmp_path, env_text="EVENT_DISPATCHER_TOKEN=profile-value\n")
    project = _project(tmp_path, profile_root, env_text="EVENT_DISPATCHER_TOKEN=project-value\n")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert _env_of(project / ".env")["EVENT_DISPATCHER_TOKEN"] == "project-value"


def test_unknown_project_key_survives_the_deploy_derivation(tmp_path):
    """Deploy mode is an overlay: a key it has never heard of is left alone."""
    profile_root = _profile(tmp_path, env_text="ANTHROPIC_API_KEY=sk-facility\n")
    project = _project(
        tmp_path,
        profile_root,
        env_text="# operator's own note\nSOME_OPERATOR_KEY=hand-written\n",
    )

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    text = (project / ".env").read_text(encoding="utf-8")
    assert parse_dotenv_text(text)["SOME_OPERATOR_KEY"] == "hand-written"
    assert "# operator's own note" in text


def test_missing_profile_dir_degrades_loudly_and_flags_the_manifest(tmp_path, caplog):
    """A profile that has moved or been deleted: mint locally, warn, flag."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    (profile_root / "profile.yml").unlink()
    profile_root.rmdir()

    with caplog.at_level(logging.WARNING):
        container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    project_env = _env_of(project / ".env")
    for var in _DISPATCH_VARS:
        assert project_env[var], f"{var} must still be minted into the project .env"
    assert str(profile_root / ".env") in caplog.text
    assert read_secrets_synced_to_profile(project) is False
    manifest = json.loads((project / ".osprey-manifest.json").read_text(encoding="utf-8"))
    assert manifest[SECRETS_SYNCED_KEY] is False
    # The degraded path must not disturb the rest of the manifest.
    assert manifest["schema_version"] == "1.2.0"


def test_preset_built_project_has_no_profile_to_sync_to(tmp_path):
    """No ``profile_path_abs``: nothing to write to, and the flag says so."""
    project = _project(tmp_path, None)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    project_env = _env_of(project / ".env")
    for var in _DISPATCH_VARS:
        assert project_env[var]
    assert read_secrets_synced_to_profile(project) is False


def test_no_manifest_at_all_still_deploys(tmp_path):
    """A project without a manifest mints as before and raises nothing."""
    project = tmp_path / "bare"
    project.mkdir()
    (project / ".env").write_text("", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    project_env = _env_of(project / ".env")
    for var in _DISPATCH_VARS:
        assert project_env[var]
    assert read_secrets_synced_to_profile(project) is None


def test_persona_delta_syncs_to_its_profile_root(tmp_path):
    """A persona shares the root profile's ``.env`` rather than keeping its own."""
    profile_root = _profile(tmp_path)
    personas = profile_root / "personas"
    personas.mkdir()
    (personas / "readonly.yml").write_text("name: readonly\n", encoding="utf-8")
    project = _project(tmp_path, profile_root)
    manifest_path = project / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_args"]["profile_path_abs"] = str(personas / "readonly.yml")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert not (personas / ".env").exists()
    profile_env = _env_of(profile_root / ".env")
    for var in _DISPATCH_VARS:
        assert profile_env[var]
    assert read_secrets_synced_to_profile(project) is True


def test_repeated_deploys_are_idempotent(tmp_path):
    """A second deploy re-syncs the same values without a second banner."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")
    first_profile = (profile_root / ".env").read_text(encoding="utf-8")
    first_project = (project / ".env").read_text(encoding="utf-8")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert (profile_root / ".env").read_text(encoding="utf-8") == first_profile
    assert (project / ".env").read_text(encoding="utf-8") == first_project
    assert first_profile.count(DEPLOY_MINTED_BANNER) == 1


def test_banner_does_not_stack_when_the_profile_gains_a_key(tmp_path):
    """A second carried key joins the existing section, it does not open a new one.

    Deploy-mode derivation re-reads the file it last wrote, so an unguarded
    header would stack one banner per deploy that carries something new.
    """
    profile_root = _profile(tmp_path, env_text="ANTHROPIC_API_KEY=sk-facility\n")
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")
    with (profile_root / ".env").open("a", encoding="utf-8") as handle:
        handle.write("OPENAI_API_KEY=sk-second\n")
    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    text = (project / ".env").read_text(encoding="utf-8")
    assert text.count(PROFILE_CARRIED_BANNER) == 1
    carried = parse_dotenv_text(text)
    assert carried["ANTHROPIC_API_KEY"] == "sk-facility"
    assert carried["OPENAI_API_KEY"] == "sk-second"


def test_profile_env_is_created_private(tmp_path):
    """A profile ``.env`` this deploy creates is born 0600 — it holds secrets."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert (profile_root / ".env").stat().st_mode & 0o777 == 0o600
    assert (project / ".env").stat().st_mode & 0o777 == 0o600


def test_no_token_service_deployed_touches_nothing(tmp_path):
    """A deploy with no token-requiring service writes no profile ``.env``."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["agent"]}, False, env_path=project / ".env"
    )

    assert not (profile_root / ".env").exists()
    assert read_secrets_synced_to_profile(project) is None


def test_substrate_wiring_never_reaches_the_profile(tmp_path, monkeypatch):
    """The bluesky substrate wiring stays in the project ``.env``.

    It is derived configuration, not a secret: a rebuild regenerates it from
    the project's own ``channel_limits.json``. Persisting it into the profile
    would freeze a device list that must re-derive — the profile's copy would
    win on the next build, and ``_ensure_bluesky_substrate_env`` would then
    skip because the key is already set.
    """
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    derived = {
        "BLUESKY_EPICS_SUBSTRATE": "epics",
        "BLUESKY_EPICS_MOTORS": "SR:C01:COR:X",
        "BLUESKY_EPICS_DETECTORS": "SR:C01:BPM:X",
    }
    for var in derived:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "osprey.services.bluesky_bridge.substrate_devices.derive_substrate_env",
        lambda project_dir: dict(derived),
    )
    # Deploy the tokens first, so the profile .env exists and the assertion
    # below is about what the substrate writer did — not about a missing file.
    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    container_lifecycle._ensure_bluesky_substrate_env(
        {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "epics"},
        },
        env_path=project / ".env",
    )

    profile_env = _env_of(profile_root / ".env")
    assert not (set(derived) & set(profile_env)), "substrate wiring must not be pinned in profile"
    assert profile_env["EVENT_DISPATCHER_TOKEN"], "the token write-back still has to happen"
    project_env = _env_of(project / ".env")
    assert {k: project_env[k] for k in derived} == derived
