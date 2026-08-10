"""Pre-``up`` warning for a service volume older than the profile's secret.

Some of the secrets a deploy mints are adopted exactly once, by a docker volume,
when the service first initializes: openobserve's root password, the ARIEL
database password. After that the volume is the authority — the container never
re-reads the variable, and a deploy supplying a different value simply cannot
open the data.

The state worth warning about is therefore an *existing* volume beside a profile
that does not carry the secret it was opened with. That is what a migration
leaves behind: a stack first deployed before the profile write-back existed, or
one whose write-back degraded while its containers kept running. It is not
broken today — the project ``.env`` still has the value — but the next rebuild
from the profile mints a secret the volume rejects.

What these tests pin is how *shallow* the check stays: volume existence is the
entire not-first-deploy signal (no volume is read or inspected), the profile is
checked by variable name only (never by value), and the result is a warning the
deploy carries on past.

No container runtime is touched: the volume-existence probe is mocked in every
test here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle

_OO_VAR = "ZO_ROOT_USER_PASSWORD"
_PG_VAR = "ARIEL_DB_PASSWORD"


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """No exported value may stand in for what the profile carries."""
    for var in (_OO_VAR, _PG_VAR, "EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def volumes(monkeypatch):
    """Control what the runtime reports without going near a real one.

    Yields a setter; the default is a runtime that reports no volumes at all.
    """
    state: dict[str, set[str] | None] = {"names": set()}

    def _fake_volume_names(config):
        return state["names"]

    monkeypatch.setattr(container_lifecycle, "_existing_volume_names", _fake_volume_names)

    def _set(names):
        state["names"] = names

    return _set


def _config(*services: str) -> dict:
    return {"project_name": "proj", "deployed_services": list(services)}


def _profile(tmp_path: Path, env_text: str | None = None) -> Path:
    """A profile tree with an optional ``.env``; returns its root."""
    root = tmp_path / "facility-profile"
    root.mkdir()
    (root / "profile.yml").write_text("name: facility\n", encoding="utf-8")
    if env_text is not None:
        (root / ".env").write_text(env_text, encoding="utf-8")
    return root


def _project(tmp_path: Path, profile_root: Path | None, env_text: str = "") -> Path:
    """A built project dir whose manifest points at ``profile_root``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(env_text, encoding="utf-8")
    build_args: dict[str, str] = {"project_name": "proj", "source": "preset"}
    if profile_root is not None:
        build_args = {
            "project_name": "proj",
            "source": "profile",
            "profile_path_abs": str(profile_root / "profile.yml"),
        }
    (project / ".osprey-manifest.json").write_text(
        json.dumps({"schema_version": "1.2.0", "build_args": build_args}), encoding="utf-8"
    )
    return project


def test_existing_volume_without_the_secret_in_the_profile_warns(tmp_path, caplog, volumes):
    """The migration case: state that predates the profile's copy of the secret."""
    profile_root = _profile(tmp_path, env_text="ANTHROPIC_API_KEY=sk-facility\n")
    project = _project(tmp_path, profile_root, env_text=f"{_OO_VAR}=only-here\n")
    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert _OO_VAR in caplog.text, "the warning must name the variable that is missing"
    assert "openobserve" in caplog.text, "the warning must name the service"
    assert "proj_openobserve_data" in caplog.text, "the warning must name the volume"
    assert str(profile_root / ".env") in caplog.text, "the warning must name the profile .env"
    # The value is never the point — only which name is absent.
    assert "only-here" not in caplog.text


def test_a_fresh_volume_is_silent(tmp_path, caplog, volumes):
    """No volume yet means this deploy's secret is the one being adopted."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root, env_text=f"{_OO_VAR}=minted\n")
    volumes(set())

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_a_profile_that_carries_the_secret_is_silent(tmp_path, caplog, volumes):
    """Existing volume, but the profile can reproduce it — nothing to say."""
    profile_root = _profile(tmp_path, env_text=f"{_OO_VAR}=carried\n")
    project = _project(tmp_path, profile_root, env_text=f"{_OO_VAR}=carried\n")
    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_presence_is_by_name_not_by_value(tmp_path, caplog, volumes):
    """A profile value that disagrees is still *present*, so this stays quiet.

    Whether the profile's value is the one the volume adopted is unknowable
    here — the volume will not say — and answering it would mean comparing
    secrets. The disagreement is the write-back's business, not this check's.
    """
    profile_root = _profile(tmp_path, env_text=f"{_OO_VAR}=a-different-value\n")
    project = _project(tmp_path, profile_root, env_text=f"{_OO_VAR}=the-live-one\n")
    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_another_projects_volume_does_not_count(tmp_path, caplog, volumes):
    """Volumes are namespaced per compose project; a sibling stack is not us."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes({"someone-else_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_a_numbered_dispatch_workspace_counts_as_existing(tmp_path, caplog, volumes):
    """Per-worker mode declares the workspace once per worker, with a suffix."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes({"proj_dispatch_workspace_1", "proj_dispatch_workspace_2"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("dispatch_worker"), env_path=project / ".env"
        )

    assert "DISPATCH_WORKER_TOKEN" in caplog.text
    assert "proj_dispatch_workspace_1" in caplog.text


def test_a_service_with_no_named_volume_is_never_flagged(tmp_path, caplog, volumes):
    """``event_dispatcher`` keeps no state, so it has no continuity to break."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("event_dispatcher"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_an_undeployed_service_is_not_inspected(tmp_path, caplog, volumes):
    """A volume left behind by a service this deploy no longer runs is not ours."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes({"proj_openobserve_data", "proj_ariel_postgres_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("postgresql"), env_path=project / ".env"
        )

    assert _PG_VAR in caplog.text
    assert _OO_VAR not in caplog.text, "openobserve is not deployed here"


def test_an_unavailable_runtime_says_nothing(tmp_path, caplog, volumes):
    """Unknown is not the same as fresh — a failed query must not cry wolf."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes(None)

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_a_project_with_no_profile_is_left_to_the_writeback(tmp_path, caplog, volumes):
    """A preset-built project has no profile to be missing the secret from.

    ``_sync_secrets_to_profile`` already reports that the project ``.env`` is the
    only copy; repeating it here in different words would only dilute it.
    """
    project = _project(tmp_path, None)
    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_a_stale_manifest_is_left_to_the_writeback(tmp_path, caplog, volumes):
    """Same for a profile that has moved: one warning about the break, not two."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    (profile_root / "profile.yml").unlink()

    volumes({"proj_openobserve_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("openobserve"), env_path=project / ".env"
        )

    assert caplog.text == ""


def test_a_profile_with_no_env_at_all_still_warns(tmp_path, caplog, volumes):
    """The oldest migration shape: a profile that has never been written back to."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root, env_text=f"{_PG_VAR}=only-here\n")
    volumes({"proj_ariel_postgres_data"})

    with caplog.at_level(logging.WARNING):
        container_lifecycle._preflight_token_continuity(
            _config("postgresql"), env_path=project / ".env"
        )

    assert _PG_VAR in caplog.text
    assert not (profile_root / ".env").exists(), "the preflight must not write anything"


def test_the_preflight_never_raises_into_a_deploy(tmp_path, monkeypatch, volumes):
    """An advisory that cannot answer stays quiet rather than aborting the up."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)
    volumes({"proj_openobserve_data"})

    def _boom(config):
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(container_lifecycle, "_existing_volume_names", _boom)

    container_lifecycle._preflight_token_continuity(
        _config("openobserve"), env_path=project / ".env"
    )


def test_every_state_volume_belongs_to_a_token_service():
    """The two maps are read together, so a name in one must exist in the other."""
    assert set(container_lifecycle._SERVICE_STATE_VOLUMES) <= set(
        container_lifecycle._SERVICE_TOKEN_VARS
    )
