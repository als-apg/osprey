"""Render projects, driven against a REAL three-zone build.

Two halves of one promise, and each is an end-to-end property because each
depends on what a build actually writes rather than on what a unit test can
stage.

``osprey build`` renders a render project per delta in ``tiers/``, into the
same ``build/`` it renders the deployment into. ``osprey up`` renders none: it
checks the renders are there and refuses when one is not, naming ``osprey
build``. That split is what keeps ``build/`` a complete account of what a deploy
will run — a start that rendered the missing renders would have stood a fresh
render beside a deployment rendered from an older profile, which is the one
state an operator cannot see and cannot debug.

The third property is what re-arms the render: a layer is source, so
editing one is drift and the drift gate refuses the next start until a rebuild.
Without that fold, a superseded delta would keep shipping into its render's
image forever, because nothing but a build replaces a render.

These are slow by unit-test standards — every test here builds the exemplar for
real — and that is the point: nothing is stubbed except the clock the operator
would have spent. No container runtime is touched. The renders are
``--skip-deps`` file writes, and image builds are a later step this module never
reaches.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.cli.repo_resolver import PROFILE_FILENAME
from osprey.deployment.staleness import DriftState, check_drift
from osprey.deployment.web_terminals.render_images import verify_renders
from osprey.deployment.web_terminals.roster import RENDER_CATALOG_REQUIRED, resolve_cards
from osprey.utils.workspace import APPS_DIRNAME, TIERS_DIRNAME
from tests.cli._lifecycle_build import invoke_build_in, rendered_config
from tests.fixtures.lifecycle_repo import (
    EXEMPLAR_DIRNAME,
    build_exemplar_repo,
    preserved_environ,
)

#: Every test here renders for real — seconds each, not milliseconds — which is
#: the property that makes them worth having and the reason they are marked.
pytestmark = pytest.mark.slow

#: The exemplar's two renders, and the simulator's own
#: ``control_system.connector.virtual_accelerator.writes_enabled`` value each
#: delta pins. That leaf IS the tier boundary between them — both pin the flat
#: key off — which makes it the one field that proves a render came from the
#: right delta.
RENDER_WRITES = {"readonly": "never", "readwrite": "asked"}


def _run_build(repo: Path) -> None:
    """Render ``build/`` the way an operator standing in the repo would."""
    with preserved_environ():
        result = invoke_build_in(repo)
    assert result.exit_code == 0, result.output


def _resolved_users(config: dict) -> list[dict]:
    """The roster ``up`` resolves before it checks or builds anything."""
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    return resolve_cards(
        web_terminals,
        config.get("registry") or {},
        strict=True,
    )


def _render_project(repo: Path, render: str) -> Path:
    """Where a build of *repo* renders *render* — the catalog's own spelling."""
    return repo / "build" / f"{EXEMPLAR_DIRNAME}-{render}"


@pytest.fixture(scope="module")
def _pristine_build(tmp_path_factory) -> tuple[Path, Path]:
    """One real build, shared by the module: ``(repo, pristine copy of it)``.

    ``seed_env=True`` because the secrets are the half of a deployment that
    lives outside ``build/`` entirely: a repo with no ``.env`` could not show
    that the renders leave them where they are.
    """
    base = tmp_path_factory.mktemp("render-render")
    repo = build_exemplar_repo(base / EXEMPLAR_DIRNAME, seed_env=True)
    _run_build(repo)
    pristine = base / "pristine"
    shutil.copytree(repo, pristine, symlinks=True)
    return repo, pristine


@pytest.fixture
def built_repo(_pristine_build) -> Path:
    """The exemplar deployment repo, seeded and with a real ``build/`` in it.

    Restored byte-for-byte (mtimes included) from the module's one build, at
    the path it was built at, so every absolute path the build recorded holds.
    """
    repo, pristine = _pristine_build
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(pristine, repo, symlinks=True)
    return repo


def test_a_real_build_stamps_the_deployments_manifest_inside_build(built_repo):
    """The filesystem fact every other assertion here reads against.

    Asserted on its own, because a test that fabricates the repo-root manifest
    is indistinguishable from one that does not until this is stated: the file
    simply is not there, so anything anchored on it can only ever fail.
    """
    assert (built_repo / "build" / ".osprey-manifest.json").is_file()
    assert not (built_repo / ".osprey-manifest.json").exists()


def test_build_renders_a_project_for_every_layer_delta(built_repo):
    """The demo promise, whole: one `osprey build` renders the entire stack.

    Each rendered project has to be the merge of BOTH halves — the root
    profile's material and the render's own delta — because that is what makes
    a render share its deployment's data tree and conventions while still
    differing on the one axis it exists to differ on.
    """
    for render, writes in RENDER_WRITES.items():
        project = _render_project(built_repo, render)
        assert project.is_dir(), f"{render} was never rendered"
        # The two files the image build uses as its context.
        assert (project / "config.yml").is_file()
        assert (project / "Dockerfile").is_file()

        rendered = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert rendered["project_name"] == f"{EXEMPLAR_DIRNAME}-{render}"
        # From the delta: the tier boundary, and the web tier this render must
        # not try to stand up a second copy of.
        assert rendered["control_system"]["writes"] == writes
        assert "writes_enabled" not in rendered["control_system"]
        connector = rendered["control_system"]["connector"]
        assert "writes_enabled" not in connector["virtual_accelerator"]
        assert rendered["modules"]["web_terminals"]["enabled"] is False
        # From the root profile: the connector the deployment runs, the data
        # tree the agent reads, its conventions.
        assert rendered["control_system"]["type"] == "virtual_accelerator"
        assert (project / "data" / "channel_limits.json").is_file()
        assert (project / ".claude" / "rules" / "facility.md").is_file()
        # And the repo root, not the render, is what every path in it anchors
        # on — a render shares its host's durable state, it does not get its
        # own copy inside the disposable zone.
        assert rendered["project_root"] == str(built_repo)


def test_a_render_records_the_delta_it_came_from(built_repo):
    """Its own manifest, naming its own source.

    A rendered project is asked what it was built from by the artifact
    machinery that runs inside its container, so a render needs a manifest of
    its own rather than borrowing the deployment's. What it must NOT carry is a
    second drift fingerprint: the drift verdict reads one manifest per repo, and
    the deltas are already folded into that one.
    """
    for render in RENDER_WRITES:
        manifest = json.loads(
            (_render_project(built_repo, render) / ".osprey-manifest.json").read_text("utf-8")
        )
        assert manifest["build_args"]["profile_path_abs"] == str(
            built_repo / TIERS_DIRNAME / f"{render}.yml"
        )
        # The repo IS the invocation: one command reproduces every render in it.
        assert manifest["reproducible_command"] == "osprey build"
        assert "profile_key_digests" not in manifest["creation"]

    deployment = json.loads((built_repo / "build" / ".osprey-manifest.json").read_text("utf-8"))
    assert "profile_key_digests" in deployment["creation"]


def test_no_render_carries_the_repos_secrets_into_the_output_zone(built_repo):
    """`rm -rf build/` has to stay a safe thing to say, renders included.

    A render project is a container build context sitting inside the zone the
    docs promise is disposable, so the rule that keeps the facility's keys at
    the repo root has to hold one level deeper too.

    The render writes no ``.env`` at all, so the check is for the file's
    absence rather than for the secret's absence from it — a stronger statement,
    and the one that actually holds: a render that emitted an ``.env`` without
    this key would still be a second secret store for `osprey up` to mint into.
    """
    secret = "ANTHROPIC_API_KEY=sk-ant-exemplar-not-a-real-key"
    assert secret in (built_repo / ".env").read_text(encoding="utf-8")

    for rendered in [
        built_repo / "build",
        *(_render_project(built_repo, p) for p in RENDER_WRITES),
    ]:
        assert not (rendered / ".env").exists(), rendered


def test_a_start_accepts_the_renders_the_build_left(built_repo, monkeypatch):
    """The whole of what a start does about renders: read them and agree."""
    config = rendered_config(built_repo)
    # As the start verbs do: `up` chdirs into the repo before provisioning, and
    # the catalog's `project_path` values are repo-relative (`build/<name>`).
    monkeypatch.chdir(built_repo)

    verify_renders(config, _resolved_users(config), repo_root=built_repo)


def test_a_start_refuses_a_missing_render_and_names_the_build(built_repo, monkeypatch):
    """A start renders nothing, so an absent project is a refusal with a remedy.

    Removing one by hand is the fast way to reach the state a stale or partial
    build reaches on its own. What matters is that the operator is sent to the
    one command that can produce the directory, and not left with a start that
    quietly fills it in.
    """
    config = rendered_config(built_repo)
    monkeypatch.chdir(built_repo)
    removed = _render_project(built_repo, "readonly")
    shutil.rmtree(removed)

    with pytest.raises(ValueError) as excinfo:
        verify_renders(config, _resolved_users(config), repo_root=built_repo)

    message = str(excinfo.value)
    assert "readonly" in message
    assert "osprey build" in message
    assert str(removed) in message
    # Still gone: a refusal that had rendered it would pass this test too.
    assert not removed.exists()


def test_a_rebuild_restores_the_render_a_start_would_not(built_repo, monkeypatch):
    """The other end of the refusal: the remedy it names actually works."""
    monkeypatch.chdir(built_repo)
    removed = _render_project(built_repo, "readonly")
    shutil.rmtree(removed)

    _run_build(built_repo)

    assert removed.is_dir()
    config = rendered_config(built_repo)
    verify_renders(config, _resolved_users(config), repo_root=built_repo)


def test_editing_a_layer_delta_is_drift_the_next_start_refuses(built_repo):
    """A delta edit must force the rebuild that puts it into force.

    ``tiers/`` is source zone, and the render it produces is only ever
    replaced by a build. If the deltas were outside the fingerprint, editing one
    would leave the drift check calling this build clean — so ``up`` would
    start, nothing would re-render, and the superseded delta would keep shipping
    into that render's image.
    """
    assert check_drift(built_repo).state is DriftState.CLEAN

    delta = built_repo / TIERS_DIRNAME / "readonly.yml"
    anchor = "control_system.writes: never"
    original = delta.read_text(encoding="utf-8")
    assert original.count(anchor) == 1, "the edit must land, or the drift check is never asked"
    delta.write_text(
        original.replace(anchor, f"{anchor}\n  system.timezone: America/Los_Angeles"),
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
        (_render_project(built_repo, "readonly") / "config.yml").read_text(encoding="utf-8")
    )
    assert rendered["system"]["timezone"] == "America/Los_Angeles"


def test_a_render_that_will_not_resolve_leaves_the_previous_build_untouched(built_repo):
    """The atomicity the whole render loop rests on, at its weakest point.

    Render projects are rendered LAST, after the deployment's own project and
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

    (built_repo / TIERS_DIRNAME / "broken.yml").write_text(
        "name: Broken\ndata: [this is not a path]\n", encoding="utf-8"
    )

    with preserved_environ():
        result = invoke_build_in(built_repo)
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
    """``build/`` is derived in full, so a render that no longer has a delta
    must not survive as a directory the next deploy could still mount.

    A render loop that only ADDED would leave the old project in place — and it
    would keep passing the start's presence check, so a render removed from the
    source zone would go on shipping its stale image indefinitely.
    """
    ghost = _render_project(built_repo, "readwrite")
    assert ghost.is_dir()

    # Retiring a tier is its file and the roster's references to it together:
    # a reference to a tier with no file is refused by the build.
    (built_repo / TIERS_DIRNAME / "readwrite.yml").unlink()
    profile_path = built_repo / PROFILE_FILENAME
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    for user in profile["config"]["modules.web_terminals"]["users"]:
        if user.get("tier") == "readwrite":
            user["tier"] = "readonly"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    _run_build(built_repo)

    assert not ghost.exists()
    # The render that still has a delta is untouched by the other's removal.
    assert _render_project(built_repo, "readonly").is_dir()


def test_a_catalog_entry_whose_delta_was_deleted_is_not_told_to_rewrite_itself(
    built_repo, monkeypatch
):
    """The refusal must not be circular.

    A delta removed while its catalog entry stays behind already holds the
    ``build_profile`` value the generic remedy recommends, so "set it to
    tiers/readwrite.yml" would be advice to write the value it already has.
    What is actually true is that the source zone lost a file the catalog still
    references, and the two ways out are restoring it or removing the tier from
    the roster that references it.
    """
    config = rendered_config(built_repo)
    monkeypatch.chdir(built_repo)
    (built_repo / TIERS_DIRNAME / "readwrite.yml").unlink()
    shutil.rmtree(_render_project(built_repo, "readwrite"))

    with pytest.raises(ValueError) as excinfo:
        verify_renders(config, _resolved_users(config), repo_root=built_repo)

    message = str(excinfo.value)
    assert "Restore" in message
    assert str(built_repo / TIERS_DIRNAME / "readwrite.yml") in message
    assert "remove 'readwrite' from the roster" in message
    assert "Set modules.web_terminals.renders.readwrite.build_profile" not in message


def _remove_render_layer(repo: Path) -> None:
    """Turn the exemplar into a deployment with no render catalog.

    The build derives the catalog from the tier and app files, so that is the
    files AND every key that names one: ``default_tier``, the roster's
    ``tier:`` keys and ``shared:``, because they are one feature. Deleting only
    the files leaves references to renders that are not there — and a render
    nobody can read is refused where its privileges would decide something
    (the ``default_tier`` every unlabelled user inherits, the shared card
    every login opens), which is the guard doing its job on a broken catalog,
    not the catalog-less shape this test refuses.
    """
    shutil.rmtree(repo / TIERS_DIRNAME)
    shutil.rmtree(repo / APPS_DIRNAME)
    profile_path = repo / PROFILE_FILENAME
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    web_terminals = profile["config"]["modules.web_terminals"]
    del web_terminals["default_tier"]
    del web_terminals["shared"]
    # `image_source: local` means "one image per catalog entry, built here",
    # so it cannot outlive the catalog; unset, the deployment pulls one image,
    # and a pulled image is named under the registry it comes from.
    del web_terminals["image_source"]
    profile["config"]["registry"] = {"url": "registry.example.org/demo"}
    for user in web_terminals["users"]:
        user.pop("tier", None)
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def test_a_repo_with_no_render_catalog_is_refused_by_the_build(tmp_path: Path):
    """A web-terminal deployment with no catalog names no project for its
    terminals, so the build refuses it before it renders anything."""
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME, seed_env=True)
    _remove_render_layer(repo)

    with preserved_environ():
        result = invoke_build_in(repo)

    assert result.exit_code != 0
    assert RENDER_CATALOG_REQUIRED in " ".join(result.output.split())
    assert not (repo / "build" / "config.yml").exists()
