"""Persona projects, driven against a REAL three-zone build.

Two halves of one promise, and each is an end-to-end property because each
depends on what a build actually writes rather than on what a unit test can
stage.

``osprey build`` renders a persona project per delta in ``personas/``, into the
same ``build/`` it renders the deployment into. ``osprey up`` renders none: it
checks the renders are there and refuses when one is not, naming ``osprey
build``. That split is what keeps ``build/`` a complete account of what a deploy
will run — a start that rendered the missing personas would have stood a fresh
persona beside a deployment rendered from an older profile, which is the one
state an operator cannot see and cannot debug.

The third property is what re-arms the render: a persona delta is source, so
editing one is drift and the drift gate refuses the next start until a rebuild.
Without that fold, a superseded delta would keep shipping into its persona's
image forever, because nothing but a build replaces a render.

These are slow by unit-test standards — every test here builds the exemplar for
real — and that is the point: nothing is stubbed except the clock the operator
would have spent. No container runtime is touched. The renders are
``--skip-deps`` file writes, and image builds are a later step this module never
reaches.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build as build_command
from osprey.deployment.staleness import DriftState, check_drift
from osprey.deployment.web_terminals.persona_images import verify_persona_renders
from osprey.deployment.web_terminals.personas import resolve_personas
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: Every test here renders for real — seconds each, not milliseconds — which is
#: the property that makes them worth having and the reason they are marked.
pytestmark = pytest.mark.slow

#: A build with no venv and no lifecycle hooks. Neither is what these tests are
#: about, and a real dependency install would dominate their runtime.
CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: The exemplar's two personas, and the ``control_system.writes_enabled`` value
#: each delta pins. That key IS the tier boundary between them, which makes it
#: the one field that proves a render came from the right delta.
PERSONA_WRITES = {"readonly": False, "readwrite": True}


def _run_build(repo: Path) -> None:
    """Render ``build/`` the way an operator standing in the repo would."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, CI_FLAGS)
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output


def _as_built_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


def _resolved_users(config: dict) -> list[dict]:
    """The roster ``up`` resolves before it checks or builds anything."""
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    return resolve_personas(
        web_terminals,
        config.get("registry") or {},
        (config.get("facility") or {}).get("prefix") or "",
        strict=True,
    )


def _persona_project(repo: Path, persona: str) -> Path:
    """Where a build of *repo* renders *persona* — the catalog's own spelling."""
    return repo / "build" / f"{EXEMPLAR_DIRNAME}-{persona}"


@pytest.fixture
def built_repo(tmp_path: Path) -> Path:
    """The exemplar deployment repo, seeded and with a real ``build/`` in it.

    ``seed_env=True`` because the secrets are the half of a deployment that
    lives outside ``build/`` entirely: a repo with no ``.env`` could not show
    that the persona renders leave them where they are.
    """
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME, seed_env=True)
    _run_build(repo)
    return repo


def test_a_real_build_stamps_the_deployments_manifest_inside_build(built_repo):
    """The filesystem fact every other assertion here reads against.

    Asserted on its own, because a test that fabricates the repo-root manifest
    is indistinguishable from one that does not until this is stated: the file
    simply is not there, so anything anchored on it can only ever fail.
    """
    assert (built_repo / "build" / ".osprey-manifest.json").is_file()
    assert not (built_repo / ".osprey-manifest.json").exists()


def test_build_renders_a_project_for_every_persona_delta(built_repo):
    """The demo promise, whole: one `osprey build` renders the entire stack.

    Each rendered project has to be the merge of BOTH halves — the root
    profile's material and the persona's own delta — because that is what makes
    a persona share its deployment's data tree and conventions while still
    differing on the one axis it exists to differ on.
    """
    for persona, writes_enabled in PERSONA_WRITES.items():
        project = _persona_project(built_repo, persona)
        assert project.is_dir(), f"{persona} was never rendered"
        # The two files the image build uses as its context.
        assert (project / "config.yml").is_file()
        assert (project / "Dockerfile").is_file()

        rendered = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert rendered["project_name"] == f"{EXEMPLAR_DIRNAME}-{persona}"
        # From the delta: the tier boundary, and the web tier this render must
        # not try to stand up a second copy of.
        assert rendered["control_system"]["writes_enabled"] is writes_enabled
        assert rendered["modules"]["web_terminals"]["enabled"] is False
        # From the root profile: the facility identity, the connector the
        # deployment runs, the data tree the agent reads, its conventions.
        assert rendered["facility"]["prefix"] == "ca"
        assert rendered["control_system"]["type"] == "virtual_accelerator"
        assert (project / "data" / "channel_limits.json").is_file()
        assert (project / ".claude" / "rules" / "facility.md").is_file()
        # And the repo root, not the render, is what every path in it anchors
        # on — a persona shares its host's durable state, it does not get its
        # own copy inside the disposable zone.
        assert rendered["project_root"] == str(built_repo)


def test_a_persona_render_records_the_delta_it_came_from(built_repo):
    """Its own manifest, naming its own source.

    A rendered project is asked what it was built from by the artifact
    machinery that runs inside its container, so a persona needs a manifest of
    its own rather than borrowing the deployment's. What it must NOT carry is a
    second drift fingerprint: the drift verdict reads one manifest per repo, and
    the deltas are already folded into that one.
    """
    for persona in PERSONA_WRITES:
        manifest = json.loads(
            (_persona_project(built_repo, persona) / ".osprey-manifest.json").read_text("utf-8")
        )
        assert manifest["build_args"]["profile_path_abs"] == str(
            built_repo / "personas" / f"{persona}.yml"
        )
        # The repo IS the invocation: one command reproduces every render in it.
        assert manifest["reproducible_command"] == "osprey build"
        assert "profile_key_digests" not in manifest["creation"]

    deployment = json.loads((built_repo / "build" / ".osprey-manifest.json").read_text("utf-8"))
    assert "profile_key_digests" in deployment["creation"]


def test_no_render_carries_the_repos_secrets_into_the_output_zone(built_repo):
    """`rm -rf build/` has to stay a safe thing to say, personas included.

    A persona project is a container build context sitting inside the zone the
    docs promise is disposable, so the rule that keeps the facility's keys at
    the repo root has to hold one level deeper too.

    The render now writes no ``.env`` at all, so the check is for the file's
    absence rather than for the secret's absence from it — a stronger statement,
    and the one that actually holds: a render that emitted an ``.env`` without
    this key would still be a second secret store for `osprey up` to mint into.
    """
    secret = "ANTHROPIC_API_KEY=sk-ant-exemplar-not-a-real-key"
    assert secret in (built_repo / ".env").read_text(encoding="utf-8")

    for rendered in [
        built_repo / "build",
        *(_persona_project(built_repo, p) for p in PERSONA_WRITES),
    ]:
        assert not (rendered / ".env").exists(), rendered


def test_a_start_accepts_the_renders_the_build_left(built_repo, monkeypatch):
    """The whole of what a start does about personas: read them and agree."""
    config = _as_built_config(built_repo)
    # As the start verbs do: `up` chdirs into the repo before provisioning, and
    # the catalog's `project_path` values are repo-relative (`build/<name>`).
    monkeypatch.chdir(built_repo)

    verify_persona_renders(config, _resolved_users(config), repo_root=built_repo)


def test_a_start_refuses_a_missing_persona_render_and_names_the_build(built_repo, monkeypatch):
    """A start renders nothing, so an absent project is a refusal with a remedy.

    Removing one by hand is the fast way to reach the state a stale or partial
    build reaches on its own. What matters is that the operator is sent to the
    one command that can produce the directory, and not left with a start that
    quietly fills it in.
    """
    config = _as_built_config(built_repo)
    monkeypatch.chdir(built_repo)
    removed = _persona_project(built_repo, "readonly")
    shutil.rmtree(removed)

    with pytest.raises(ValueError) as excinfo:
        verify_persona_renders(config, _resolved_users(config), repo_root=built_repo)

    message = str(excinfo.value)
    assert "readonly" in message
    assert "osprey build" in message
    assert str(removed) in message
    # Still gone: a refusal that had rendered it would pass this test too.
    assert not removed.exists()


def test_a_rebuild_restores_the_render_a_start_would_not(built_repo, monkeypatch):
    """The other end of the refusal: the remedy it names actually works."""
    monkeypatch.chdir(built_repo)
    removed = _persona_project(built_repo, "readonly")
    shutil.rmtree(removed)

    _run_build(built_repo)

    assert removed.is_dir()
    config = _as_built_config(built_repo)
    verify_persona_renders(config, _resolved_users(config), repo_root=built_repo)


def test_editing_a_persona_delta_is_drift_the_next_start_refuses(built_repo):
    """A delta edit must force the rebuild that puts it into force.

    ``personas/`` is source zone, and the render it produces is only ever
    replaced by a build. If the deltas were outside the fingerprint, editing one
    would leave the drift check calling this build clean — so ``up`` would
    start, nothing would re-render, and the superseded delta would keep shipping
    into that persona's image.
    """
    assert check_drift(built_repo).state is DriftState.CLEAN

    delta = built_repo / "personas" / "readonly.yml"
    delta.write_text(
        delta.read_text(encoding="utf-8").replace(
            "control_system.writes_enabled: false",
            "control_system.writes_enabled: false\n  system.timezone: America/Los_Angeles",
        ),
        encoding="utf-8",
    )

    report = check_drift(built_repo)

    assert report.state is DriftState.DRIFT
    assert report.refuses
    assert "osprey build" in report.remedy

    _run_build(built_repo)

    assert check_drift(built_repo).state is DriftState.CLEAN
    # And the rebuild is what carried the edit into the render.
    rendered = yaml.safe_load(
        (_persona_project(built_repo, "readonly") / "config.yml").read_text(encoding="utf-8")
    )
    assert rendered["system"]["timezone"] == "America/Los_Angeles"


def test_a_persona_that_will_not_resolve_leaves_the_previous_build_untouched(built_repo):
    """The atomicity the whole render loop rests on, at its weakest point.

    Persona projects are rendered LAST, after the deployment's own project and
    after the compose files — so they are the failure that has the most already
    written beside it. A build that dies there must still leave ``build/``
    byte-for-byte as the last good build left it, because that build is what
    ``osprey down`` needs to stop the stack it started.

    Compared by content AND mtime: a re-render that happened to produce
    identical bytes would pass a content-only check while having replaced the
    directory the operator's running stack is pinned to.
    """
    before = {
        path.relative_to(built_repo): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (built_repo / "build").rglob("*")
        if path.is_file()
    }
    assert before, "fixture produced no build to protect"

    (built_repo / "personas" / "broken.yml").write_text(
        "name: Broken\ndata_bundle: [this is not a bundle name]\n", encoding="utf-8"
    )

    previous = Path.cwd()
    os.chdir(built_repo)
    try:
        result = CliRunner().invoke(build_command, CI_FLAGS)
    finally:
        os.chdir(previous)
    assert result.exit_code != 0, result.output

    after = {
        path.relative_to(built_repo): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (built_repo / "build").rglob("*")
        if path.is_file()
    }
    assert after == before
    # And nothing half-written landed beside it.
    assert not (built_repo / "build" / f"{EXEMPLAR_DIRNAME}-broken").exists()
    assert not (built_repo / "build" / ".tmp").exists()


def test_a_deleted_delta_leaves_no_ghost_render_behind(built_repo):
    """``build/`` is derived in full, so a persona that no longer has a delta
    must not survive as a directory the next deploy could still mount.

    A render loop that only ADDED would leave the old project in place — and it
    would keep passing the start's presence check, so a persona removed from the
    source zone would go on shipping its stale image indefinitely.
    """
    ghost = _persona_project(built_repo, "readwrite")
    assert ghost.is_dir()

    (built_repo / "personas" / "readwrite.yml").unlink()
    _run_build(built_repo)

    assert not ghost.exists()
    # The persona that still has a delta is untouched by the other's removal.
    assert _persona_project(built_repo, "readonly").is_dir()


def test_a_catalog_entry_whose_delta_was_deleted_is_not_told_to_rewrite_itself(
    built_repo, monkeypatch
):
    """The refusal must not be circular.

    A delta removed while its catalog entry stays behind already holds the
    ``build_profile`` value the generic remedy recommends, so "set it to
    personas/readwrite.yml" would be advice to write the value it already has.
    What is actually true is that the source zone lost a file the catalog still
    references, and the two ways out are restoring it or dropping the entry.
    """
    config = _as_built_config(built_repo)
    monkeypatch.chdir(built_repo)
    (built_repo / "personas" / "readwrite.yml").unlink()
    shutil.rmtree(_persona_project(built_repo, "readwrite"))

    with pytest.raises(ValueError) as excinfo:
        verify_persona_renders(config, _resolved_users(config), repo_root=built_repo)

    message = str(excinfo.value)
    assert "Restore" in message
    assert str(built_repo / "personas" / "readwrite.yml") in message
    assert "drop" in message
    assert "Set modules.web_terminals.personas.readwrite.build_profile" not in message


def test_a_repo_with_no_personas_renders_only_its_own_project(tmp_path: Path):
    """The absent-``personas/`` path, which must stay exactly as it was.

    Every repo that has no personas has to render what it always rendered —
    which is also what keeps the persona fold out of its build hash.
    """
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME, seed_env=True)
    shutil.rmtree(repo / "personas")

    _run_build(repo)

    build_dir = repo / "build"
    assert (build_dir / "config.yml").is_file()
    assert not [entry for entry in build_dir.iterdir() if entry.name.startswith(f"{repo.name}-")]
